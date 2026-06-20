try:
    from .attention import CausalSelfAttention
except ImportError:
    from attention import CausalSelfAttention

import torch
import torch.nn as nn
import torch.nn.functional as F

# SwiGELU (as used in Meta's Llama)
class SwiGLU(nn.Module):
    def __init__(self, n_embd):
        super().__init__()

        self.fc = nn.Linear(n_embd, 4 * n_embd * 2)
        self.proj = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        x = self.fc(x)
        x1, x2 = x.chunk(2, dim=-1)
        x = x1 * F.silu(x2)
        return self.proj(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()

        self.ln1 = nn.RMSNorm(n_embd)
        self.ln2 = nn.RMSNorm(n_embd)

        self.attn = CausalSelfAttention(n_embd, n_head)

        # self.mlp = SwiGLU(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x