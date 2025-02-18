import argparse
import logging
import math
import os
from contextlib import nullcontext
from glob import glob

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from diffusers import AutoencoderKL
from diffusers.optimization import get_scheduler
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
torch.cuda.empty_cache()


def get_transform(image_size: int):
    return A.Compose(
        [
            A.RandomCrop(height=image_size, width=image_size, p=1),
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ]
    )


class VaeDataset(Dataset):
    def __init__(self, glob_path: str, size: int, transform=None):
        self.size = size
        self.transform = transform
        self.image_paths = glob(glob_path)
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {glob_path}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
        instance_image = image.copy()
        h, w, _ = instance_image.shape
        short_side = min(h, w)
        if short_side < self.size:
            scale_factor = int((self.size * 2) / short_side)
            new_h, new_w = h * scale_factor, w * scale_factor
            instance_image = cv2.resize(
                instance_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR
            )
        augmented = self.transform(image=instance_image)
        return augmented["image"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train VAE")
    parser.add_argument("--glob_path", type=str, default="data/processed/OCR_1/*.jpg")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument(
        "--pretrained_model", type=str, default="runwayml/stable-diffusion-v1-5"
    )
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--resume_from_checkpoint", action="store_true")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--checkpoint_steps", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Arguments: {args}")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model, subfolder="vae")
    if args.gradient_checkpointing and hasattr(vae, "enable_gradient_checkpointing"):
        vae.enable_gradient_checkpointing()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    vae.to(device)
    if hasattr(torch, "compile"):
        vae = torch.compile(vae)
    optimizer = torch.optim.AdamW(vae.parameters(), lr=args.learning_rate)
    transform = get_transform(args.image_size)
    dataset = VaeDataset(glob_path=args.glob_path, size=args.size, transform=transform)
    train_dataloader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True
    )
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    max_train_steps = args.num_epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=max_train_steps,
    )
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    logger.info(f"Number of training examples: {len(dataset)}")
    global_step = 0
    first_epoch = 0
    if args.resume_from_checkpoint:
        checkpoint_dirs = [
            d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")
        ]
        checkpoint_dirs = sorted(checkpoint_dirs, key=lambda x: int(x.split("-")[1]))
        last_checkpoint = checkpoint_dirs[-1] if checkpoint_dirs else None
        if last_checkpoint is not None:
            print(f"Resuming from checkpoint {last_checkpoint}")
            checkpoint_path = os.path.join(args.output_dir, last_checkpoint)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            vae.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            global_step = checkpoint["global_step"]
            first_epoch = global_step // num_update_steps_per_epoch
    progress_bar = tqdm(range(global_step, max_train_steps))
    cumulative_loss = 0.0
    step_count = 0
    for epoch in range(first_epoch, args.num_epochs):
        logger.info(f"Epoch {epoch + 1}/{args.num_epochs}")
        vae.train()
        optimizer.zero_grad()
        for i, batch in enumerate(train_dataloader):
            batch = batch.to(device)
            with torch.amp.autocast("cuda") if scaler is not None else nullcontext():
                outputs = vae(batch)
                reconstructed = outputs["sample"]
                loss = F.mse_loss(
                    reconstructed.float(), batch.float(), reduction="mean"
                )
                loss = loss / args.gradient_accumulation_steps
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if (i + 1) % args.gradient_accumulation_steps == 0 or (i + 1) == len(
                train_dataloader
            ):
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(vae.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(vae.parameters(), args.max_grad_norm)
                    optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                step_loss = loss.detach().item() * args.gradient_accumulation_steps
                cumulative_loss += step_loss
                step_count += 1
                avg_loss = cumulative_loss / step_count
                progress_bar.update(1)
                if global_step % args.checkpoint_steps == 0:
                    save_path = os.path.join(
                        args.output_dir, f"checkpoint-{global_step}"
                    )
                    torch.save(
                        {
                            "model": vae.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                            "global_step": global_step,
                        },
                        save_path,
                    )
                    logger.info(f"Saved checkpoint to {save_path}")
                progress_bar.set_postfix(
                    {
                        "step_loss": step_loss,
                        "lr": lr_scheduler.get_last_lr()[0],
                        "avg_loss": avg_loss,
                    }
                )
                if global_step >= max_train_steps:
                    break
            if global_step >= max_train_steps:
                break
        if global_step >= max_train_steps:
            break


if __name__ == "__main__":
    args = parse_args()
    main(args)
