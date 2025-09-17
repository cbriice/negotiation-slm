# Training Guide for Mistral-7B Negotiation Agent

This guide explains how to use the centralized training script `scripts/train.py` and details the underlying modules it routes to.

## Overview

The `train.py` script provides a thin wrapper around three main training modules:
- **SFT (Supervised Fine-Tuning)**: Initial training on dialogue data
- **DPO (Direct Preference Optimization)**: Preference-based training using chosen/rejected pairs
- **QLoRA Wrapper**: Unified interface that can run both SFT and DPO

## Architecture

```
scripts/train.py
    ├── sft command  →  training/sft_trainer.py::main()
    ├── dpo command  →  training/dpo_trainer.py::main()
    ├── qlora command → training/qlora_trainer.py::QLoRATrainer
    └── info command →  training_parameters.py (display only)
```

## Commands

### 1. SFT Training (`sft` command)

Routes directly to `training/sft_trainer.py::main()`

**Basic Usage:**
```bash
python scripts/train.py sft data/sft_data.jsonl
```

**Full Options:**
```bash
python scripts/train.py sft data/sft_data.jsonl \
    --eval-data data/sft_eval.jsonl \
    --model mistralai/Mistral-7B-v0.1 \
    --output-dir models/sft \
    --checkpoint-dir checkpoints/sft \
    --resume-from checkpoints/sft/checkpoint-500 \
    --num-epochs 3 \
    --max-steps 1000 \
    --suffix _v2
```

**Module Details:**
- **Class**: `SFTTrainingPipeline` (training/sft_trainer.py:52-601)
- **Key Methods**:
  - `load_ndjson_dataset()`: Loads JSONL training data (line 148)
  - `setup_model()`: Initializes model with QLoRA and Flash Attention 2 (line 193)
  - `train()`: Main training loop with TRL's SFTTrainer (line 241)
- **Error Handling**: Comprehensive error recovery including:
  - OOM recovery with batch size reduction (line 515)
  - Gradient health monitoring (line 438)
  - Loss validation (line 410)
  - Training state persistence (line 446)

### 2. DPO Training (`dpo` command)

Routes directly to `training/dpo_trainer.py::main()`

**Basic Usage with DPO pairs:**
```bash
python scripts/train.py dpo --train-data data/dpo_pairs.jsonl
```

**From episodes (auto-generates pairs):**
```bash
python scripts/train.py dpo \
    --from-episodes data/episodes.jsonl \
    --min-score-gap 0.5
```

**Full Options:**
```bash
python scripts/train.py dpo \
    --train-data data/dpo_pairs.jsonl \
    --eval-data data/dpo_eval.jsonl \
    --model mistralai/Mistral-7B-v0.1 \
    --reference-model models/sft/final_model \
    --output-dir checkpoints/dpo \
    --epochs 2 \
    --resume checkpoints/dpo/checkpoint-100
```

**Module Details:**
- **Class**: `DPOTrainingModule` (training/dpo_trainer.py:246-836)
- **Key Methods**:
  - `prepare_dataset()`: Processes DPO pairs using UnifiedDatasetLoader (line 391)
  - `train()`: Main DPO training with TRL's DPOTrainer (line 530)
  - `train_from_episodes()`: Auto-generates DPO pairs from episodes (line 753)
  - `merge_and_save()`: Merges LoRA weights with base model (line 786)
- **DPO-Specific Features**:
  - Reward margin tracking (line 459)
  - Early stopping on reward collapse (line 508)
  - Preference pair validation (line 362)
  - Custom error callback for DPO metrics (line 47)

### 3. QLoRA Wrapper (`qlora` command)

Uses `training/qlora_trainer.py::QLoRATrainer` directly

**SFT mode:**
```bash
python scripts/train.py qlora \
    --mode sft \
    --train-data data/sft.jsonl \
    --output-dir models/negotiator
```

**DPO mode:**
```bash
python scripts/train.py qlora \
    --mode dpo \
    --train-data data/dpo.jsonl \
    --reference-model models/sft/final_model
```

**Test mode:**
```bash
python scripts/train.py qlora \
    --mode test \
    --checkpoint models/final_model \
    --prompt "You are negotiating for a car. Respond:" \
    --temperature 0.7
```

**Module Details:**
- **Class**: `QLoRATrainer` (training/qlora_trainer.py:31-461)
- **Key Methods**:
  - `setup_model()`: Central model loading with QLoRA (line 106)
  - `train_sft()`: Delegates to SFTTrainingPipeline (line 138)
  - `train_dpo()`: Delegates to DPOTrainingModule (line 220)
  - `generate()`: Test generation from model (line 343)
  - `load_checkpoint()`: Load saved checkpoints (line 395)
- **Shared Components**:
  - Uses ModelLoader for consistent 4-bit quantization
  - Uses LoRAConfigurator for LoRA adapter setup
  - Manages checkpoint saving/loading centrally

### 4. Show Parameters (`info` command)

Displays current training parameters from `training_parameters.py`

```bash
python scripts/train.py info
```

**Displays:**
- Model configuration (Flash Attention 2 status)
- QLoRA settings (4-bit, NF4, LoRA rank/alpha/dropout)
- Training hyperparameters (batch size, gradient accumulation, etc.)
- SFT-specific parameters (learning rate, warmup, scheduler)
- DPO-specific parameters (beta, label smoothing)
- Logging configuration

## Module Dependencies

### Core Training Modules

1. **training/model_loader.py**
   - Loads models with 4-bit quantization
   - Configures BitsAndBytes for QLoRA
   - Handles Flash Attention 2 setup

2. **training/lora_config.py**
   - Creates LoRA configuration
   - Applies LoRA adapters to model
   - Target modules: q_proj, k_proj, v_proj, o_proj, etc.

3. **training/checkpoint_manager.py**
   - Saves and loads checkpoints
   - Tracks training versions and stages

### Data Processing

1. **utils/dataset_loader.py (UnifiedDatasetLoader)**
   - Loads JSONL files for SFT and DPO
   - Formats dialogues with chat templates
   - Preprocesses datasets for training

2. **training/data_converter.py**
   - Converts between data formats
   - Auto-generates DPO pairs from episodes
   - Calculates protocol scores for pair selection

### Error Handling Modules

1. **utils/error_handler.py**
   - Central error recovery system
   - Handles OOM, gradient explosions, etc.
   - Automatic retry with adjusted parameters

2. **utils/batch_size_optimizer.py**
   - Dynamically adjusts batch size on OOM
   - Maintains effective batch size with gradient accumulation

3. **utils/gradient_monitor.py**
   - Monitors gradient health
   - Detects explosions/vanishing gradients
   - Applies adaptive gradient clipping

4. **utils/loss_validator.py**
   - Validates loss values (NaN/Inf detection)
   - Detects training plateaus
   - Triggers recovery actions

5. **utils/precision_handler.py**
   - Manages mixed precision training (bf16/fp16)
   - Handles automatic mixed precision (AMP)
   - Provides fallback for overflow

## Training Parameters Configuration

All training parameters are centralized in `training_parameters.py`:

### Key Parameters

```python
# Model
LOAD_IN_4BIT = True           # 4-bit quantization for QLoRA
NF4 = True                    # Use NF4 quantization
USE_FLASH_ATTENTION_2 = True # Flash Attention 2 for efficiency

# LoRA
R = 32                        # LoRA rank
ALPHA = 64                    # LoRA alpha (typically 2*R)
DROPOUT = 0.1                 # LoRA dropout

# Training
PER_DEVICE_BATCH_SIZE = 4    # Batch size per device
GRAD_ACCUM_STEPS = 4          # Gradient accumulation
MAX_SEQ_LEN = 2048           # Maximum sequence length

# SFT
SFT_LR = 2e-4                # SFT learning rate
SFT_MAX_STEPS = 1000         # SFT max training steps
SFT_WARMUP_RATIO = 0.1       # Warmup ratio

# DPO
DPO_LR = 5e-7                # DPO learning rate (much lower)
DPO_BETA = 0.1               # DPO beta parameter
DPO_MAX_STEPS = 500          # DPO max training steps
```

## Training Workflow

### Typical Training Pipeline

1. **Baseline SFT Training**
   ```bash
   # Train on negotiation dialogues
   python scripts/train.py sft data/baseline_dialogues.jsonl \
       --output-dir models/baseline \
       --num-epochs 1
   ```

2. **Generate Episodes** (using ParlAI - separate process)
   ```bash
   # Run self-play or dialog partner world to generate episodes
   python parlai_integration/run_world.py self-chat \
       --model models/baseline/final_model \
       --output data/episodes.jsonl
   ```

3. **DPO Training from Episodes**
   ```bash
   # Auto-generate pairs and train
   python scripts/train.py dpo \
       --from-episodes data/episodes.jsonl \
       --reference-model models/baseline/final_model \
       --output-dir models/dpo_v1 \
       --min-score-gap 0.3
   ```

4. **Test the Model**
   ```bash
   python scripts/train.py qlora \
       --mode test \
       --checkpoint models/dpo_v1/final_model \
       --prompt "I want to buy your car for $5000"
   ```

## Error Recovery

Both SFT and DPO trainers include comprehensive error handling:

- **OOM Recovery**: Automatically reduces batch size and retries
- **Gradient Issues**: Monitors and clips exploding gradients
- **Loss Validation**: Detects NaN/Inf and training divergence
- **State Persistence**: Saves training state every 100 steps for recovery
- **Precision Fallback**: Falls back from bf16 to fp32 if needed

Error logs are saved to:
- SFT: `models/sft/error_logs/`
- DPO: `checkpoints/dpo/error_logs/`

## TensorBoard Monitoring

Training metrics are logged to TensorBoard:

```bash
# View training metrics
tensorboard --logdir models/sft/runs  # For SFT
tensorboard --logdir checkpoints/runs  # For DPO
```

Tracked metrics include:
- Loss curves (train/eval)
- Learning rate schedule
- Gradient norms
- DPO reward margins (DPO only)
- Error events and recovery actions

## Notes

- All trainers use Flash Attention 2 when `USE_FLASH_ATTENTION_2=True` in training_parameters.py
- Models are automatically loaded in 4-bit with QLoRA configuration
- The wrapper script doesn't add logic - it just routes to existing implementations
- Each trainer can also be run directly via their own CLI interfaces