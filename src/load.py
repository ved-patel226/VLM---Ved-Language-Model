import torch
from transformers import AutoTokenizer

from config import CONFIG
from model.model import VLM


class ModelLoader:
    def __init__(self, checkpoint_path, device=None):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            "gpt2",
            use_fast=True
        )

        self.model = VLM(CONFIG).to(self.device)

        ckpt = torch.load(
            checkpoint_path,
            map_location=self.device
        )

        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompt,
        max_new_tokens=100,
        temperature=0.8,
        top_k=50,
    ):
        ids = self.tokenizer.encode(
            prompt,
            add_special_tokens=False
        )

        idx = torch.tensor(
            [ids],
            dtype=torch.long,
            device=self.device
        )

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -CONFIG["block_size"]:]

            logits = self.model(idx_cond)
            logits = logits[:, -1, :]

            logits /= max(temperature, 1e-6)

            if top_k is not None:
                values, _ = torch.topk(
                    logits,
                    min(top_k, logits.size(-1))
                )

                logits[logits < values[:, [-1]]] = -float("inf")

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(
                probs,
                num_samples=1
            )

            idx = torch.cat(
                [idx, next_token],
                dim=1
            )

        return self.tokenizer.decode(
            idx[0].tolist(),
            skip_special_tokens=True
        )

@torch.no_grad()
def top_tokens(model, tokenizer, prompt, device):
    ids = tokenizer.encode(prompt, add_special_tokens=False)

    idx = torch.tensor(
        [ids],
        dtype=torch.long,
        device=device
    )

    logits = model(idx)
    probs = torch.softmax(logits[:, -1, :], dim=-1)

    values, indices = torch.topk(probs, 20)

    for p, i in zip(values[0], indices[0]):
        token = tokenizer.decode([i.item()])
        print(f"{token!r:20} {p.item():.6f}")

if __name__ == "__main__":
    model = ModelLoader("/home/ved/Code/Train My Own LLM/runs/sft/20260621_123307/checkpoints/sft_ckpt_49000.pt")

    while True:
        prompt = input("\nPrompt> ")

        if prompt.lower() in {"quit", "exit"}:
            break

        print("\nCompletion:")
        print(
            model.generate(
                prompt,
                max_new_tokens=100,
                temperature=0.5,
                top_k=50,
            )
        )

        print("\nTop tokens:")
        top_tokens(model.model, model.tokenizer, prompt, model.device)