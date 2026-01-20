# ============================================================
# HARD SAFETY — must be first
# ============================================================
import os
os.environ["TRANSFORMERS_NO_META"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from huggingface_hub import snapshot_download

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
# DOWNLOAD PRETRAINED CHECKPOINT (SHARDED)
# ============================================================
# IMPORTANT:
# - This directory ALREADY contains:
#   config.json
#   model.safetensors.index.json
#   model-0000x-of-0000y.safetensors
# - DO NOT append subfolders
# ============================================================
LOCAL_MODEL_DIR = snapshot_download(
    repo_id=MODEL_NAME
)

# ============================================================
# TOKENIZER
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True
)

# ============================================================
# MODEL (META-SAFE, CLUSTER-SAFE)
# ============================================================
# 1) Load config from SNAPSHOT ROOT
config = AutoConfig.from_pretrained(LOCAL_MODEL_DIR)
config.use_cache = False

# 2) Build empty architecture (no tensors allocated)
with init_empty_weights():
    model = AutoModelForSeq2SeqLM.from_config(config)

# 3) Tie embeddings (REQUIRED for seq2seq models)
model.tie_weights()

# 4) Load sharded checkpoint directly onto GPU
model = load_checkpoint_and_dispatch(
    model,
    checkpoint=LOCAL_MODEL_DIR,   # SNAPSHOT ROOT
    device_map={"": "cuda"},
    dtype=torch.bfloat16,
)

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
    disable_tqdm=False,
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
