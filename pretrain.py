import os
import time
import torch
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer, AdamW
from datasets import load_from_disk
from torch.utils.tensorboard import SummaryWriter

# -----------------------------
# CONFIG
# -----------------------------
DATASET_PATH = "openwebtext-tokenized"  # path to tokenized dataset
SAVE_DIR = "checkpoints"
BATCH_SIZE = 2  # configurable
BLOCK_SIZE = 1024
LR = 5e-5
EPOCHS = 3
CKPT_INTERVAL_SEC = 3600  # save checkpoint every hour
LOG_DIR = "runs/gpt2_scratch"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(SAVE_DIR, exist_ok=True)
writer = SummaryWriter(LOG_DIR)

# -----------------------------
# LOAD DATA
# -----------------------------
print("Loading dataset...")
lm_datasets = load_from_disk(DATASET_PATH)
split = lm_datasets["train"].train_test_split(test_size=0.1, seed=42)
train_ds = split["train"]
val_ds = split["test"]


# Collate function
def collate_fn(batch):
    input_ids = torch.tensor([x["input_ids"] for x in batch], dtype=torch.long)
    labels = input_ids.clone()
    return {"input_ids": input_ids, "labels": labels}


train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn)

# -----------------------------
# MODEL + OPTIMIZER
# -----------------------------
print("Initializing GPT-2 model from scratch...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

config = GPT2Config(
    vocab_size=len(tokenizer),
    n_layer=12,
    n_head=12,
    n_embd=768,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)
model = GPT2LMHeadModel(config)  # randomly initialized
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=LR)
scaler = torch.cuda.amp.GradScaler()  # FP16 mixed precision

# -----------------------------
# TRAINING LOOP
# -----------------------------
for epoch in range(EPOCHS):
    print(f"\n🚀 Epoch {epoch+1}/{EPOCHS}")
    model.train()
    total_loss = 0
    start_time = time.time()
    last_ckpt_time = start_time

    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        # FP16 autocast context
        with torch.cuda.amp.autocast():
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        writer.add_scalar("train/loss", loss.item(), epoch * len(train_loader) + step)

        if (step + 1) % 50 == 0:
            avg_loss = total_loss / (step + 1)
            print(f"Step {step+1}, avg loss: {avg_loss:.4f}")

        # Save checkpoint every hour
        now = time.time()
        if now - last_ckpt_time >= CKPT_INTERVAL_SEC:
            ckpt_path = os.path.join(SAVE_DIR, f"gpt2_epoch{epoch+1}_step{step+1}")
            model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)
            print(f"💾 Saved intermediate checkpoint: {ckpt_path}")
            last_ckpt_time = now

    # Save checkpoint at end of epoch
    ckpt_path = os.path.join(SAVE_DIR, f"gpt2_epoch{epoch+1}_end")
    model.save_pretrained(ckpt_path)
    tokenizer.save_pretrained(ckpt_path)
    print(f"✅ Epoch {epoch+1} finished. Checkpoint saved: {ckpt_path}")

    # -----------------------------
    # VALIDATION
    # -----------------------------
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            with torch.cuda.amp.autocast():
                outputs = model(input_ids, labels=labels)
                val_loss += outputs.loss.item()
    val_loss /= len(val_loader)
    writer.add_scalar("val/loss", val_loss, epoch)
    print(f"Validation loss after epoch {epoch+1}: {val_loss:.4f}")

writer.close()
