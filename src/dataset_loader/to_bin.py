from pathlib import Path
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("gpt2")

def conversation_to_text(messages):
    return "\n".join(
        f"<|{msg['role']}|>\n{msg['content']}"
        for msg in messages
    )

files = sorted(Path("data/data").glob("train_sft-*.parquet"))

total_examples = 0
total_tokens = 0

with open("tokens.bin", "wb") as out_file:
    for file in files:
        print(f"Processing {file.name}")

        df = pd.read_parquet(file)

        for messages in tqdm(df["messages"], desc=f"{file.name}", unit="example"):
            text = conversation_to_text(messages)

            token_ids = tokenizer.encode(
                text,
                add_special_tokens=False
            )

            np.array(
                token_ids,
                dtype=np.uint16
            ).tofile(out_file)

            total_examples += 1
            total_tokens += len(token_ids)

print("\nFinished!")
print(f"Examples: {total_examples:,}")
print(f"Tokens:   {total_tokens:,}")
print("Saved to tokens.bin")