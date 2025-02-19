import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert embed_dim % num_heads == 0, (
            "Embedding dimension must be divisible by the number of heads"
        )

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim**0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, embed_dim)
        )

        return self.fc_out(attn_output)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x):
        attn_out = self.attention(x)
        x = self.norm1(x + attn_out)

        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)

        return x


class CharBERT(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=64,
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        max_len=64,
        num_classes=2,
    ):
        super(CharBERT, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Embedding(max_len, embed_dim)

        self.transformer_layers = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, hidden_dim)
                for _ in range(num_layers)
            ]
        )

        self.output_proj = nn.Linear(embed_dim, 768) # TODO: Should be better than this

    def forward(self, x):
        batch_size, seq_len = x.shape
        positions = (
            torch.arange(0, seq_len, device=x.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

        x = self.embedding(x) + self.positional_embedding(positions)

        for layer in self.transformer_layers:
            x = layer(x)

        x = x.mean(dim=1)
        x = self.output_proj(x)
        return x
