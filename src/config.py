CONFIG = {
    "vocab_size": 50257,
    "n_layer": 8,
    "n_head": 8,
    "n_embd": 512,
    "block_size": 1024,
    "context_length": 1024,

    # training defaults
    "tensorboard_dir": "runs/train",
    "sample_prompts": [
        "The Capital of France is",
        "Once upon a time",
        "Artificial intelligence is",
        "The name 'Ved' is derived from",
        "The meaning of life is",
        "The VLM, Ved Language Model, is a", # interesting test prompt... because VLM is usually used to refer to Vision-Language Models, but here it stands for Ved Language Model
    ],
    "sample_every": 1000,

    "max_lr": 3e-4,
    "min_lr": 3e-5,
    "warmup": 1000,
    "batch_size": 3,
    "grad_accum": 32,
    "max_steps": 1_000_000,

    "weight_decay": 0.1,
    "grad_clip": 1.0,
}

SFT_CONFIG = {
    "pretrained_ckpt": "runs/train/20260620_130243/checkpoints/ckpt_995000.pt",

    "train_bin": "sft_train.bin",
    "train_mask_bin": "sft_train_mask.bin",
    "val_bin": "sft_val.bin",
    "val_mask_bin": "sft_val_mask.bin",

    "tensorboard_dir": "runs/sft",
    "sample_prompts": CONFIG["sample_prompts"],
    "sample_every": 200,

    "max_lr": 2e-5,
    "min_lr": 2e-6,
    "warmup": 100,
    "batch_size": 2,
    "grad_accum": 32,
    "max_steps": 50_000,

    "weight_decay": 0.1,
    "grad_clip": 1.0,
}