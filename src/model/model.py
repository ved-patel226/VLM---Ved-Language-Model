try:
    from .block import Block
except ImportError:
    from block import Block

import torch.nn as nn
import torch


class VLM(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.wte = nn.Embedding(config["vocab_size"], config["n_embd"])
        self.wpe = nn.Embedding(config["block_size"], config["n_embd"])

        self.blocks = nn.ModuleList([
            Block(config["n_embd"], config["n_head"])
            for _ in range(config["n_layer"])
        ])

        self.ln_f = nn.RMSNorm(config["n_embd"])

        self.lm_head = nn.Linear(config["n_embd"], config["vocab_size"], bias=False)

        self.lm_head.weight = self.wte.weight

    def forward(self, idx):
        # for block in self.blocks:
        #     x = block(x)
        # return x

        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)[None, :, :]

        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

if __name__ == "__main__":
    try:
        from src.config import CONFIG
    except ModuleNotFoundError:
        import sys
        from pathlib import Path

        project_root = Path(__file__).resolve().parents[2]

        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from src.config import CONFIG

    import torchinfo

    model = VLM(CONFIG)
    example_input = torch.zeros(
        (1, CONFIG["block_size"]),
        dtype=torch.long,
        device=next(model.parameters()).device,
    )
    torchinfo.summary(model, input_data=example_input)