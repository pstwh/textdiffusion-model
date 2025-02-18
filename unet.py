import argparse
import json
import logging
import math
import os
import random
from contextlib import nullcontext
from glob import glob
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from PIL import Image, ImageDraw
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from charbert import CharBERT
from tokenizer import Tokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_transform(image_size: int):
    return A.Compose(
        [
            A.SmallestMaxSize(max_size=image_size),
            A.RandomCrop(height=image_size, width=image_size),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )


def generate_mask(im_shape, x1, x2, y1, y2):
    mask = Image.new("L", im_shape, 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([x1, y1, x2, y2], fill=255)
    return mask


class MaskTextDataset(Dataset):
    def __init__(
        self, glob_path="data/processed/*/*.jpg", transform=None, tokenizer=None
    ):
        self.transform = transform
        self.tokenizer = tokenizer
        self.image_paths = glob(glob_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        ocr_path = image_path.replace(".jpg", ".json")
        image = Image.open(image_path).convert("RGB")
        with open(ocr_path, "r") as f:
            ocr = json.load(f)
        random_ocr = random.choice(ocr)
        text = random_ocr["word"]
        x_min, x_max, y_min, y_max = (
            random_ocr["x1"],
            random_ocr["x2"],
            random_ocr["y1"],
            random_ocr["y2"],
        )
        bboxes = [[x_min, y_min, x_max, y_max]]
        labels = [0]
        transformed = self.transform(
            image=np.array(image), bboxes=bboxes, labels=labels
        )
        bbox_trans = transformed["bboxes"][0]
        x_min, y_min, x_max, y_max = (
            bbox_trans[0],
            bbox_trans[1],
            bbox_trans[2],
            bbox_trans[3],
        )
        mask = generate_mask(
            (transformed["image"].shape[1], transformed["image"].shape[2]),
            x_min,
            x_max,
            y_min,
            y_max,
        )
        mask = torch.tensor(np.array(mask), dtype=torch.float)
        image = transformed["image"]
        text_encoded = self.tokenizer.encode(text)
        if len(text_encoded) > self.tokenizer.block_size:
            text_encoded = text_encoded[: self.tokenizer.block_size]
        elif len(text_encoded) < self.tokenizer.block_size:
            text_encoded = text_encoded + [self.tokenizer.ignore_index] * (
                self.tokenizer.block_size - len(text_encoded)
            )
        text_tensor = torch.tensor(text_encoded, dtype=torch.long)
        return image, mask, text_tensor


def parse_args():
    parser = argparse.ArgumentParser(description="Train UNet on MaskTextDataset")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument(
        "--pretrained_model", type=str, default="runwayml/stable-diffusion-v1-5"
    )
    parser.add_argument(
        "--vae_checkpoint", type=str, default="./output/checkpoint-48500"
    )
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--resume_from_checkpoint", action="store_true", default=True)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--checkpoint_steps", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"]
    )
    parser.add_argument("--glob_path", type=str, default="data/processed/*/*.jpg")
    parser.add_argument("--image_size", type=int, default=256)
    return parser.parse_args()


def main(args):
    output_dir = args.output_dir
    pretrained_model = args.pretrained_model
    vae_checkpoint = Path(args.vae_checkpoint)
    gradient_checkpointing = args.gradient_checkpointing
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    lr_scheduler_name = args.lr_scheduler
    gradient_accumulation_steps = args.gradient_accumulation_steps
    num_epochs = args.num_epochs
    resume_from_checkpoint = args.resume_from_checkpoint
    max_grad_norm = args.max_grad_norm
    checkpoint_steps = args.checkpoint_steps
    num_workers = args.num_workers
    image_size = args.image_size
    glob_path = args.glob_path

    os.makedirs(output_dir, exist_ok=True)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    noise_scheduler = DDPMScheduler.from_pretrained(
        pretrained_model, subfolder="scheduler"
    )

    state_dict = load_file(vae_checkpoint / "model.safetensors")
    vae = AutoencoderKL.from_config(vae_checkpoint / "config.json")
    vae.load_state_dict(state_dict)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    vae.to(device, dtype=weight_dtype)

    tokenizer = Tokenizer()
    text_encoder = CharBERT(vocab_size=tokenizer.vocab_size)
    text_encoder.to(device)

    unet = UNet2DConditionModel.from_pretrained(pretrained_model, subfolder="unet")
    if gradient_checkpointing and hasattr(unet, "enable_gradient_checkpointing"):
        unet.enable_gradient_checkpointing()
    unet.to(device)

    optimizer = torch.optim.AdamW(unet.parameters(), lr=learning_rate)

    transform = get_transform(image_size)
    dataset = MaskTextDataset(
        glob_path=glob_path, transform=transform, tokenizer=tokenizer
    )
    train_dataloader = DataLoader(
        dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps
    )
    max_train_steps = num_epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(lr_scheduler_name, optimizer=optimizer)

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    num_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    logger.info(f"Num examples = {len(dataset)}")
    logger.info(f"Num epochs = {num_epochs}")
    logger.info(f"Batch size = {batch_size}")

    global_step = 0
    first_epoch = 0

    progress_bar = tqdm(range(global_step, max_train_steps))
    progress_bar.set_description("Steps")

    scaler = None
    if device.type == "cuda" and args.mixed_precision in ["fp16", "bf16"]:
        scaler = torch.amp.GradScaler("cuda")

    accumulation_counter = 0
    for epoch in range(first_epoch, num_epochs):
        unet.train()
        for step, (image, mask, text) in enumerate(train_dataloader):
            image = image.to(device)
            mask = mask.to(device)
            text = text.to(device)
            text_embeddings = text_encoder(text)
            with torch.amp.autocast("cuda") if scaler is not None else nullcontext():
                latents = vae.encode(image.to(weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                noise = torch.randn_like(latents)
                latent_h, latent_w = latents.shape[-2], latents.shape[-1]
                if mask.dim() == 3:
                    mask = mask.unsqueeze(1)
                mask_resized = torch.nn.functional.interpolate(
                    mask, size=(latent_h, latent_w), mode="nearest"
                )
                mask_resized = mask_resized.to(weight_dtype)
                mask_resized = mask_resized.repeat(1, 3, 1, 1)  # check
                masked_image_latents = vae.encode(mask_resized).latent_dist.sample()
                masked_image_latents = masked_image_latents * vae.config.scaling_factor
                masked_image_latents = torch.nn.functional.interpolate(
                    masked_image_latents, size=(latent_h, latent_w), mode="nearest"
                )  # check
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                    )
                model_input_latents = torch.cat(
                    [noisy_latents, mask_resized, masked_image_latents], dim=1
                )
                model_output = unet(model_input_latents, timesteps, text_embeddings)
                model_pred = model_output.sample
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            loss = loss / gradient_accumulation_steps
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            accumulation_counter += 1
            if accumulation_counter % gradient_accumulation_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(unet.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(unet.parameters(), max_grad_norm)
                    optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix(
                    {
                        "step_loss": loss.item() * gradient_accumulation_steps,
                        "lr": lr_scheduler.get_last_lr()[0],
                    }
                )
                if global_step % checkpoint_steps == 0:
                    save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                    torch.save(
                        {
                            "unet": unet.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                            "global_step": global_step,
                        },
                        save_path,
                    )
                    logger.info(f"Saved state to {save_path}")
            if global_step >= max_train_steps:
                break
        if global_step >= max_train_steps:
            break


if __name__ == "__main__":
    args = parse_args()
    main(args)
