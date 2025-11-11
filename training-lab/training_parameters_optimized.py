# ------ Optimized 7B model training parameters for RTX 4070 Ti Super 16GB ---------
from __future__ import annotations
import torch

# === SEQUENCE LENGTH OPTIMIZATION ===
# Reduced from 2048 to 1536 (covers your 1200 max with buffer)
# This allows 25% more samples in memory
MAX_SEQ_LEN: int = 1536

# === BATCH SIZE OPTIMIZATION ===
# VRAM Budget (12.5GB target with 3.5GB safety from 16GB total):
# - 4-bit model: ~3.5GB
# - LoRA adapters: ~0.2GB
# - 8-bit optimizer (LoRA params only): ~0.4GB
# - Activations/gradients (batch=4): ~7.5GB
# - Attention matrices (Flash Attention): ~0.8GB
# - Working memory buffer: ~0.5GB
# Total: ~12.5GB, leaving 3.5GB safety buffer
PER_DEVICE_TRAIN_BATCH_SIZE = 4     # Slightly reduced from 6
GRADIENT_ACCUMULATION_STEPS = 6     # Increased for effective batch = 24
PER_DEVICE_EVAL_BATCH_SIZE = 8      # Keep eval batch reasonable

# === 4-BIT QUANTIZATION (CRITICAL FOR SPEED) ===
LOAD_IN_4BIT: bool = True           # Enable 4-bit quantization
BNB_4BIT_QUANT_TYPE: str = "nf4"    # Normal Float 4 - best quality
BNB_4BIT_COMPUTE_DTYPE = torch.bfloat16
BNB_4BIT_USE_DOUBLE_QUANT: bool = True  # Further compression

# === PRECISION & KERNELS ===
BF16 = True                          # Use bfloat16 for training
FP16 = False                         # Don't mix FP16 with BF16
TF32 = True                          # Enable TF32 for matmul operations
ATTN_IMPLEMENTATION = "flash_attention_2"  # Critical for speed
USE_FLASH_ATTENTION_2 = True

# === LORA CONFIGURATION FOR QUALITY ===
# Keep higher rank for quality as you requested
R: int = 64                          # Higher rank for better quality
ALPHA: int = 128                     # 2x rank as recommended
DROPOUT: float = 0.1                 # Keep dropout for regularization
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

# === MEMORY OPTIMIZATION ===
GRADIENT_CHECKPOINTING: bool = True  # Trade compute for memory
OPTIM: str = "paged_adamw_8bit"     # 8-bit optimizer saves memory
DEVICE_MAP: str = "auto"
DATALOADER_NUM_WORKERS: int = 2     # Parallel data loading (not 0!)
DATALOADER_PIN_MEMORY: bool = True  # Faster GPU transfers

# === TRAINING SCHEDULE ===
SFT_MAX_STEPS: int = 8000
SFT_LR: float = 3e-5                # Slightly lower for stability with larger batch
SFT_WARMUP_RATIO: float = 0.05      # 400 steps warmup for 8000 total
SFT_LR_SCHEDULER: str = "cosine"

# === LOGGING & CHECKPOINTING ===
LOGGING_STEPS: int = 10
SAVE_STEPS: int = 500               # Save every 500 steps
SAVE_STRATEGY: str = "steps"
EVAL_STRATEGY: str = "no"           # Curriculum handles validation
EVAL_STEPS = 500

# === DATA PROCESSING ===
PACKING = False                     # Don't pack sequences
PAD_TO_MULTIPLE = 64               # Optimize padding for tensor cores
SEED: int = 42
MODEL_TEMP: float = 0.5

# === CURRICULUM-SPECIFIC ===
# Phase 1 (0-2000): Contextual learning
# Phase 2 (2000-4000): Guided practice
# Phase 3 (4000-8000): Specialized training
USE_TWO_PART_RESPONSE: bool = True
STRATEGY_MAX_TOKENS: int = 150
RESPONSE_MAX_TOKENS: int = 256
STRATEGY_TEMPERATURE: float = 0.3

# === RESPONSE MASKING ===
RESPONSE_TEMPLATES = [
    "### Response:\n",
    "### Response:",
    "Assistant:",
    "You:",
    "Agent:",
]
INSTRUCTION_TEMPLATES = [
    "### Instruction:\n",
    "### Context:\n",
    "User:",
    "Partner:",
    "State:",
]
MASKING_IGNORE_INDEX: int = -100