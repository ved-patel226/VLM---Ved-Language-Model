import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

try:
    from .dataset_loader import SFTBinDataset
except ImportError:
    from dataset_loader import SFTBinDataset

try:
    from .model.model import VLM
except ImportError:
    from model.model import VLM

try:
    from .train import get_lr, log_sample_prompts
except ImportError:
    from train import get_lr, log_sample_prompts

from transformers import AutoTokenizer


def masked_loss(logits, y, mask):
    loss_per_token = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        y.view(-1),
        reduction="none"
    )
    mask = mask.view(-1)
    return (loss_per_token * mask).sum() / mask.sum().clamp(min=1)


def train_sft(model, train_data, val_data, config, device):
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

        x, y, mask = train_data.get_batch(batch_size)
        x, y, mask = x.to(device), y.to(device), mask.to(device)

        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            logits = model(x)
            raw_loss = masked_loss(logits, y, mask)

        loss = raw_loss / grad_accum
        scaler.scale(loss).backward()

        if (step + 1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            optimizer_step += 1

        if step % 50 == 0:
            print(f"step {step} | loss {raw_loss.item():.4f} | lr {lr:.2e}")
            writer.add_scalar("sft/train_loss", raw_loss.item(), step)
            writer.add_scalar("sft/lr", lr, step)

        if step % 200 == 0:
            model.eval()
            with torch.no_grad():
                x, y, mask = val_data.get_batch(batch_size)
                x, y, mask = x.to(device), y.to(device), mask.to(device)

                logits = model(x)
                val_loss = masked_loss(logits, y, mask)

            print(f"[VAL] step {step} | loss {val_loss.item():.4f}")
            writer.add_scalar("sft/val_loss", val_loss.item(), step)

            if step % sample_every == 0:
                log_sample_prompts(writer, model, tokenizer, device, step, sample_prompts)

            model.train()

        if step % 1000 == 0 and step > 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "step": step
                },
                ckpt_dir / f"sft_ckpt_{step}.pt"
            )

    writer.close()


if __name__ == "__main__":
    try:
        from .config import CONFIG, SFT_CONFIG
    except ImportError:
        from config import CONFIG, SFT_CONFIG

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = VLM(CONFIG)

    ckpt = torch.load(SFT_CONFIG["pretrained_ckpt"], map_location=device)
    model.load_state_dict(ckpt["model"])

    train_data = SFTBinDataset(
        SFT_CONFIG["train_bin"], SFT_CONFIG["train_mask_bin"], CONFIG["block_size"]
    )
    val_data = SFTBinDataset(
        SFT_CONFIG["val_bin"], SFT_CONFIG["val_mask_bin"], CONFIG["block_size"]
    )

    train_sft(model, train_data, val_data, SFT_CONFIG, device)
