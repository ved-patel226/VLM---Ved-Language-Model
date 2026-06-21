from transformers import AutoTokenizer
from abc import ABC, abstractmethod
from datasets import load_dataset
import numpy as np
from tqdm import tqdm


class TokenDatasetBuilder(ABC):
    def __init__(
        self,
        tokenizer_name="gpt2",
        train_file="train.bin",
        val_file="val.bin",
        val_split=0.01,
        seed=42
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            use_fast=True
        )

        self.eos_id = self.tokenizer.eos_token_id
        self.train_file = train_file
        self.val_file = val_file

        self.rng = np.random.default_rng(seed)
        self.val_split = val_split

        self.train_tokens = 0
        self.val_tokens = 0

    @abstractmethod
    def examples(self):
        pass

    @abstractmethod
    def example_to_text(self, example):
        pass

    def tokenize(self, text):
        ids = self.tokenizer.encode(
            text,
            add_special_tokens=False
        )

        ids.append(self.eos_id)

        return np.array(ids, dtype=np.uint16)

    def build(self):
        with open(self.train_file, "wb") as train_f, \
             open(self.val_file, "wb") as val_f:

            pbar = tqdm(
                self.examples(),
                desc="Tokenizing",
                unit=" examples",
                dynamic_ncols=True
            )

            for i, example in enumerate(pbar, 1):
                text = self.example_to_text(example)
                tokens = self.tokenize(text)

                if self.rng.random() < self.val_split:
                    tokens.tofile(val_f)
                    self.val_tokens += len(tokens)
                else:
                    tokens.tofile(train_f)
                    self.train_tokens += len(tokens)

                if i % 1000 == 0:
                    pbar.set_postfix(
                        train_tokens=f"{self.train_tokens:,}",
                        val_tokens=f"{self.val_tokens:,}"
                    )

        print(f"Train tokens: {self.train_tokens:,}")
        print(f"Val tokens:   {self.val_tokens:,}")

class FineWebBuilder(TokenDatasetBuilder):
    def __init__(
        self,
        target_tokens,
        dataset_name="sample-10BT",
        **kwargs
    ):
        super().__init__(**kwargs)

        self.target_tokens = target_tokens
        self.dataset_name = dataset_name

    def examples(self):
        total = 0

        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name=self.dataset_name,
            split="train",
            streaming=True
        )

        for row in ds:
            yield row

            total += len(
                self.tokenizer.encode(
                    row["text"],
                    add_special_tokens=False
                )
            )

            if total >= self.target_tokens:
                break

    def example_to_text(self, example):
        return example["text"]

class UltraChatBuilder(TokenDatasetBuilder):
    """Tokenizes chat conversations and also writes a parallel loss-mask file
    (1 = assistant content, train on it; 0 = role headers / other turns, ignore it)."""
    def __init__(
        self,
        mask_train_file="sft_train_mask.bin",
        mask_val_file="sft_val_mask.bin",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.mask_train_file = mask_train_file
        self.mask_val_file = mask_val_file

    def examples(self):
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")

        for row in ds:
            yield row["messages"]

    def example_to_text(self, messages):
        return "\n".join(
            f"<|{msg['role']}|>\n{msg['content']}"
            for msg in messages
        )

    def tokenize_turns(self, messages):
        token_parts = []
        mask_parts = []

        for i, msg in enumerate(messages):
            header = f"<|{msg['role']}|>\n" if i == 0 else f"\n<|{msg['role']}|>\n"
            header_ids = self.tokenizer.encode(header, add_special_tokens=False)
            content_ids = self.tokenizer.encode(msg["content"], add_special_tokens=False)

            token_parts.append(header_ids)
            mask_parts.append([0] * len(header_ids))

            is_assistant = msg["role"] == "assistant"
            token_parts.append(content_ids)
            mask_parts.append([1 if is_assistant else 0] * len(content_ids))

        token_parts.append([self.eos_id])
        mask_parts.append([1])

        tokens = np.array([t for part in token_parts for t in part], dtype=np.uint16)
        mask = np.array([m for part in mask_parts for m in part], dtype=np.uint8)
        return tokens, mask

    def build(self):
        with open(self.train_file, "wb") as train_f, \
             open(self.val_file, "wb") as val_f, \
             open(self.mask_train_file, "wb") as train_mask_f, \
             open(self.mask_val_file, "wb") as val_mask_f:

            pbar = tqdm(
                self.examples(),
                desc="Tokenizing",
                unit=" conversations",
                dynamic_ncols=True
            )

            for i, messages in enumerate(pbar, 1):
                tokens, mask = self.tokenize_turns(messages)

                if self.rng.random() < self.val_split:
                    tokens.tofile(val_f)
                    mask.tofile(val_mask_f)
                    self.val_tokens += len(tokens)
                else:
                    tokens.tofile(train_f)
                    mask.tofile(train_mask_f)
                    self.train_tokens += len(tokens)

                if i % 1000 == 0:
                    pbar.set_postfix(
                        train_tokens=f"{self.train_tokens:,}",
                        val_tokens=f"{self.val_tokens:,}"
                    )

        print(f"Train tokens: {self.train_tokens:,}")
        print(f"Val tokens:   {self.val_tokens:,}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["fineweb", "ultrachat"], default="fineweb")
    args = parser.parse_args()

    if args.dataset == "fineweb":
        builder = FineWebBuilder(
            target_tokens=2_000_000_000, # 2B tokens
            train_file="pretrain_train.bin",
            val_file="pretrain_val.bin"
        )
    else:
        builder = UltraChatBuilder(
            train_file="sft_train.bin",
            val_file="sft_val.bin"
        )

    builder.build()