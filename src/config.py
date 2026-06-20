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