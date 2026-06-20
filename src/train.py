import torch
import torch.nn.functional as F
import math
from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

try:
    from .dataset_loader import BinDataset
except ImportError:
    from dataset_loader import BinDataset

try:
    from .model.model import VLM
except ImportError:
    from model.model import VLM

from transformers import AutoTokenizer


def get_lr(step, max_lr, min_lr, warmup, max_steps):
    if step < warmup:
        return max_lr * (step + 1) / max(1, warmup)

    decay_ratio = (step - warmup) / max(1, max_steps - warmup)
    decay_ratio = min(max(decay_ratio, 0.0), 1.0)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * decay_ratio))


def generate(model, tokenizer, prompt, device, max_new_tokens=40, temperature=0.2, top_k=10):
    model.eval()

    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    idx = torch.tensor([input_ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config["block_size"]:]
        logits = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-6)

        if top_k is not None:
            values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = torch.where(
                logits < values[:, [-1]],
                torch.full_like(logits, float("-inf")),
                logits
            )

        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)

    return tokenizer.decode(idx[0].tolist(), skip_special_tokens=True)


def log_sample_prompts(writer, model, tokenizer, device, step, prompts):
    for i, prompt in enumerate(prompts):
        generated = generate(model, tokenizer, prompt, device)
        writer.add_text(
            f"samples/{i}",
            f"### Prompt\n{prompt}\n\n### Completion\n{generated}",
            step
        )


def train(model, train_data, val_data, config, device):
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = Path(config["tensorboard_dir"]) / run_name
    ckpt_dir = run_dir / "checkpoints"

    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(run_dir))
    sample_prompts = config["sample_prompts"]
    sample_every = config["sample_every"]

    max_lr = config["max_lr"]
    min_lr = config["min_lr"]
    warmup = config["warmup"]
    batch_size = config["batch_size"]
    grad_accum = config["grad_accum"]
    max_steps = config["max_steps"]

    weight_decay = config["weight_decay"]
    grad_clip = config["grad_clip"]

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=max_lr,
        weight_decay=weight_decay
    )
    optimizer_step = 0

    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    optimizer.zero_grad(set_to_none=True)

    for step in range(max_steps):
        lr = get_lr(optimizer_step, max_lr, min_lr, warmup, max_steps // grad_accum)

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        x, y = train_data.get_batch(batch_size)
        x, y = x.to(device), y.to(device)

        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            logits = model(x)

            raw_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )

        loss = raw_loss / grad_accum
        scaler.scale(loss).backward()

        if (step + 1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            optimizer_step += 1

        if step % 100 == 0:
            print(f"step {step} | loss {raw_loss.item():.4f} | lr {lr:.2e}")
            writer.add_scalar("train/loss", raw_loss.item(), step)
            writer.add_scalar("train/lr", lr, step)

        if step % 1000 == 0:
            model.eval()
            with torch.no_grad():
                x, y = val_data.get_batch(batch_size)
                x, y = x.to(device), y.to(device)

                logits = model(x)
                val_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1)
                )

            print(f"[VAL] step {step} | loss {val_loss.item():.4f}")
            writer.add_scalar("val/loss", val_loss.item(), step)

            if step % sample_every == 0:
                log_sample_prompts(writer, model, tokenizer, device, step, sample_prompts)

            model.train()

        if step % 5000 == 0 and step > 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "step": step
                },
                ckpt_dir / f"ckpt_{step}.pt"
            )

    writer.close()


if __name__ == "__main__":
    import torch
    from model.model import VLM
    from config import CONFIG

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = VLM(CONFIG)

    train_data = BinDataset("pretrain_train.bin", CONFIG["block_size"])
    val_data = BinDataset("pretrain_val.bin", CONFIG["block_size"])

    train(model, train_data, val_data, CONFIG, device)