# ------ 7B model training parameters ---------
from __future__ import annotations
import torch

# Global defaults
MAX_SEQ_LEN: int = 4096
PER_DEVICE_BATCH_SIZE: int = 1
GRAD_ACCUM_STEPS: int = 8
SEED: int = 42
MODEL_TEMP: float = 0.5  # Temperature for model generation
PACKING = True                                                  # ---- change this to False when using comprehensive datasets
PAD_TO_MULTIPLE = 128

# QLoRA quantization
LOAD_IN_4BIT: bool = True
BNB_4BIT_QUANT_TYPE: str = "nf4"
BNB_4BIT_COMPUTE_DTYPE = torch.bfloat16
BNB_4BIT_USE_DOUBLE_QUANT: bool = True

# Batching
PER_DEVICE_TRAIN_BATCH_SIZE = 3     # microbatch = 3
GRADIENT_ACCUMULATION_STEPS = 4     # effective batch = 12
PER_DEVICE_EVAL_BATCH_SIZE = 3

# Precision & Kernels
BF16 = True
FP16 = False
TF32 = True                         # matmul allow tf32
ATTN_IMPLEMENTATION = "flash_attention_2"
USE_GRADIENT_CHECKPOINTING = "mlp_only"
USE_FLASH_ATTENTION_2 = True       # Enable Flash Attention 2 by default

# LoRA
"""
old defaults for heavy, domain-shifting training. too strong for baseline task style training - use with comprehensive, verified dataset
R: int = 96
ALPHA: int = 256
DROPOUT: float = 0.05
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]
"""
# LoRA config for baseline task training
R: int = 16
ALPHA: int = 32
DROPOUT: float = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Trainer/runtime
GRADIENT_CHECKPOINTING: bool = True
OPTIM: str = "paged_adamw_8bit"
BF16: bool = True                   # ok on Ampere/Ada; set False if you hit issues
DEVICE_MAP: str = "auto"
# PACKING: bool = False               # True if your SFT examples are short & homogeneous

# Save/log
LOGGING_STEPS: int = 10
SAVE_STEPS: int = 200
SAVE_STRATEGY: str = "steps"
EVAL_STRATEGY: str = "steps"           # flip to "steps" if you add eval
EVAL_STEPS = 100

# SFT-specific
SFT_MAX_STEPS: int = 1000
SFT_LR: float = 2e-4
SFT_WARMUP_RATIO: float = 0.03
SFT_LR_SCHEDULER: str = "cosine"

# DPO-specific
DPO_MAX_STEPS: int = 1000
DPO_LR: float = 5e-5
DPO_BETA: float = 0.1               # preference sharpness
DPO_LABEL_SMOOTH: float = 0.0
DPO_MAX_PROMPT_LEN: int = min(512, MAX_SEQ_LEN // 2)

# Two-part response parameters
USE_TWO_PART_RESPONSE: bool = True
STRATEGY_MAX_TOKENS: int = 150
RESPONSE_MAX_TOKENS: int = 256
STRATEGY_TEMPERATURE: float = 0.3  # Lower temp for strategy generation