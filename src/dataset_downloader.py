from transformers import AutoTokenizer
from abc import ABC, abstractmethod
from datasets import load_dataset
from pathlib import Path
import pandas as pd
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
            "HuggingFaceFW/fineweb",
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
    def __init__(self, data_dir, **kwargs):
        super().__init__(**kwargs)
        self.data_dir = Path(data_dir)

    def examples(self):
        files = sorted(
            self.data_dir.glob("train_sft-*.parquet")
        )

        for file in files:
            df = pd.read_parquet(file)

            for messages in df["messages"]:
                yield messages

    def example_to_text(self, messages):
        return "\n".join(
            f"<|{msg['role']}|>\n{msg['content']}"
            for msg in messages
        )

if __name__ == "__main__":
    builder = FineWebBuilder(
        target_tokens=2_000_000_000, # 2B tokens
        train_file="pretrain_train.bin",
        val_file="pretrain_val.bin"
    )

    builder.build()