# ============================================================
# HARD SAFETY: disable meta tensors globally
# ============================================================
import os
os.environ["TRANSFORMERS_NO_META"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

assert torch.cuda.is_available(), "CUDA GPU required"

# ============================================================
# CONFIG (H200 OPTIMIZED)
# ============================================================
MODEL_NAME = "facebook/nllb-200-distilled-600M"

MAX_LEN = 512
BATCH_SIZE = 4
GRAD_ACCUM = 8              
LR = 3e-5
MAX_STEPS = 8000
WARMUP_STEPS = 800

SAVE_DIR = "./pretrained-nllb-akk"
DATA_FILE = "generated/pretrain.src.txt"

# ============================================================
# TOKENIZER
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True
)

# ============================================================
# MODEL  
# ============================================================

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    use_safetensors=True,
    device_map="cuda",              
    torch_dtype=torch.bfloat16      
)

model.config.use_cache = False
model.gradient_checkpointing_enable()
model.train()

# ============================================================
# DATASET
# ============================================================
dataset = load_dataset(
    "text",
    data_files={"train": DATA_FILE}
)

def tokenize(batch):
    out = tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN
    )
    out["labels"] = out["input_ids"].copy()
    return out

dataset = dataset.map(
    tokenize,
    batched=True,
    remove_columns=["text"],
    desc="Tokenizing"
)

# ============================================================
# COLLATOR
# ============================================================
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding="max_length"
)

# ============================================================
# TRAINING ARGS
# ============================================================
training_args = TrainingArguments(
    output_dir=SAVE_DIR,
    overwrite_output_dir=True,

    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,

    learning_rate=LR,
    lr_scheduler_type="linear",
    warmup_steps=WARMUP_STEPS,
    max_steps=MAX_STEPS,

    bf16=True,
    fp16=False,

    logging_steps=50,
    save_steps=1000,
    save_total_limit=2,

    report_to="none",
    dataloader_num_workers=4,
    disable_tqdm=False
)

# ============================================================
# TRAINER
# ============================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    data_collator=data_collator,
    processing_class=tokenizer,
)

# ============================================================
# TRAIN
# ============================================================
trainer.train()

# ============================================================
# SAVE
# ============================================================
trainer.save_model(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

print("✅ DOMAIN-ADAPTIVE PRETRAINING COMPLETE")
print(f"📁 Saved to: {SAVE_DIR}")
