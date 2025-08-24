from datasets import load_dataset
from transformers import AutoTokenizer

# 1. Load dataset
print("Loading OpenWebText...")
raw_datasets = load_dataset("openwebtext")

# 2. Load fast GPT-2 tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)

# GPT2 doesn't have pad token → set one for batching (won’t affect training loss)
tokenizer.pad_token = tokenizer.eos_token


# 3. Tokenization function
def tokenize_fn(examples):
    return tokenizer(examples["text"])


# 4. Tokenize entire dataset in parallel
print("Tokenizing dataset...")
tokenized = raw_datasets.map(
    tokenize_fn,
    batched=True,
    batch_size=2000,  # how many texts per batch
    num_proc=16,  # adjust to # of CPU cores
    remove_columns=["text"],
    desc="Tokenizing",
)

# 5. Group texts into fixed-length blocks (e.g., 1024 tokens)
block_size = 1024


def group_texts(examples):
    # Concatenate all tokens in batch
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = (len(concatenated["input_ids"]) // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated.items()
    }
    return result


print("Grouping into 1024-token blocks...")
lm_datasets = tokenized.map(
    group_texts, batched=True, batch_size=1000, num_proc=16, desc="Grouping"
)

# 6. Save pre-tokenized dataset to disk
print("Saving dataset...")
lm_datasets.save_to_disk("openwebtext-tokenized")

print("✅ Done! Tokenized dataset saved at: openwebtext-tokenized")
