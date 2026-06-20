import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module): # casual means that the attention is only applied to the left of the current token, not the right
    def __init__(self, n_embd, n_head):
        super().__init__()

        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention( # fast attention implementation in PyTorch 2.0
            q,
            k,
            v,
            is_causal=True
        )

        y = y.transpose(1, 2).contiguous()
        y = y.view(B, T, C)

        return self.proj(y)