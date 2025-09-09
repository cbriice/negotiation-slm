# ------ 7B model training parameters ---------
from __future__ import annotations
import torch

# Global defaults
MAX_SEQ_LEN: int = 1536
PER_DEVICE_BATCH_SIZE: int = 1
GRAD_ACCUM_STEPS: int = 8
SEED: int = 42
MODEL_TEMP: float = 0.5  # Temperature for model generation

# QLoRA quantization
LOAD_IN_4BIT: bool = True
BNB_4BIT_QUANT_TYPE: str = "nf4"
BNB_4BIT_COMPUTE_DTYPE = torch.bfloat16
BNB_4BIT_USE_DOUBLE_QUANT: bool = True

# LoRA
R: int = 16
ALPHA: int = 32
DROPOUT: float = 0.05
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

# Trainer/runtime
GRADIENT_CHECKPOINTING: bool = True
OPTIM: str = "paged_adamw_8bit"
BF16: bool = True                   # ok on Ampere/Ada; set False if you hit issues
DEVICE_MAP: str = "auto"
PACKING: bool = False               # True if your SFT examples are short & homogeneous

# Save/log
LOGGING_STEPS: int = 10
SAVE_STEPS: int = 200
SAVE_STRATEGY: str = "steps"
EVAL_STRATEGY: str = "no"           # flip to "steps" if you add eval

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

# validation - make sure parameters aren't changed to outrageous values
def validate():
    assert 256 <= MAX_SEQ_LEN <= 4096, "MAX_SEQ_LEN out of sane range"
    assert PER_DEVICE_BATCH_SIZE >= 1
    assert 0 < DROPOUT < 0.5
    assert R in (4, 8, 16, 32), "Pick a reasonable LoRA rank"
    assert 0 < SFT_LR < 5e-3
    assert 0 < DPO_LR < 5e-3
    assert 0.0 <= DPO_BETA <= 1.0
    assert BNB_4BIT_QUANT_TYPE in ("nf4", "fp4")
    # Optional: guard VRAM by reducing seq len when using 13B on 12GB GPUs
    # (hook in your own device check here if desired)

validate()