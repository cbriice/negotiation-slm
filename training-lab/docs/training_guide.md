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

## Checkpoint Management

The training script includes comprehensive checkpoint management features via the `checkpoint` and `save` commands.

### Manual Checkpoint Saving

You can manually save a checkpoint at any point:

```bash
# Save a checkpoint from an existing model
python scripts/train.py save \
    --load-from models/sft/final_model \
    --stage sft \
    --description "Good performance on negotiation tasks"
```

### Checkpoint Operations

**List all checkpoints:**
```bash
python scripts/train.py checkpoint list
python scripts/train.py checkpoint list --stage sft  # Filter by stage
```

**Find best checkpoint by metric:**
```bash
python scripts/train.py checkpoint best --stage sft --metric loss
python scripts/train.py checkpoint best --stage dpo --metric reward_margin
```

**View checkpoint history:**
```bash
python scripts/train.py checkpoint history
python scripts/train.py checkpoint history --stage dpo
```

**Rollback to a previous checkpoint:**
```bash
# Rollback sets a checkpoint as "current" for future operations
python scripts/train.py checkpoint rollback \
    --checkpoint checkpoints/negotiation_sft_v3_20240315_143022
```

**Compare two checkpoints:**
```bash
python scripts/train.py checkpoint compare \
    --checkpoint1 checkpoints/negotiation_sft_v2_20240315_140000 \
    --checkpoint2 checkpoints/negotiation_sft_v3_20240315_143022
```

**Clean up old checkpoints:**
```bash
# Keep only the 3 most recent checkpoints
python scripts/train.py checkpoint cleanup --keep 3

# Clean up only SFT checkpoints
python scripts/train.py checkpoint cleanup --keep 5 --stage sft
```

**Export checkpoint summary:**
```bash
python scripts/train.py checkpoint export --output checkpoint_report.json
```

### Using Checkpoints in Training

**Resume training from checkpoint:**
```bash
# SFT training
python scripts/train.py sft data/sft_data.jsonl \
    --resume-from checkpoints/negotiation_sft_v2_20240315_140000

# DPO training
python scripts/train.py dpo --train-data data/dpo.jsonl \
    --resume checkpoints/negotiation_dpo_v1_20240315_150000
```

**Use checkpoint as reference model for DPO:**
```bash
python scripts/train.py dpo \
    --train-data data/dpo_pairs.jsonl \
    --reference-model checkpoints/negotiation_sft_v3_20240315_143022
```

### Checkpoint Storage Structure

Checkpoints are stored with the following naming convention:
```
checkpoints/
├── negotiation_sft_v1_20240315_140000/
│   ├── adapter_model.bin          # LoRA weights
│   ├── adapter_config.json        # LoRA configuration
│   ├── tokenizer_config.json      # Tokenizer config
│   ├── special_tokens_map.json    # Special tokens
│   └── checkpoint_info.json       # Metadata and metrics
├── negotiation_sft_v2_20240315_141500/
├── negotiation_dpo_v1_20240315_150000/
└── checkpoints_summary.json       # Summary report (if exported)
```

### Automatic Checkpointing

During training, checkpoints are automatically saved based on `SAVE_STEPS` and `SAVE_STRATEGY` in `training_parameters.py`:

- **By steps**: Saves every N steps (default: 100)
- **By epoch**: Saves at the end of each epoch
- **Best model**: Automatically tracks and saves the best model based on evaluation metric

The automatic checkpoints are saved to:
- SFT: `models/sft/checkpoint-{step}/`
- DPO: `checkpoints/dpo/checkpoint-{step}/`

### Reverting Model State

To revert a model to an earlier checkpoint:

1. **Find the checkpoint you want:**
   ```bash
   python scripts/train.py checkpoint history --stage sft
   ```

2. **Rollback to that checkpoint:**
   ```bash
   python scripts/train.py checkpoint rollback \
       --checkpoint checkpoints/negotiation_sft_v2_20240315_140000
   ```

3. **Use the checkpoint in training or testing:**
   ```bash
   # Test the rolled-back model
   python scripts/train.py qlora --mode test \
       --checkpoint checkpoints/negotiation_sft_v2_20240315_140000 \
       --prompt "Test negotiation prompt"

   # Continue training from that checkpoint
   python scripts/train.py sft data/new_data.jsonl \
       --resume-from checkpoints/negotiation_sft_v2_20240315_140000
   ```

### Checkpoint Metadata

Each checkpoint includes metadata in `checkpoint_info.json`:
```json
{
  "stage": "sft",
  "version": 2,
  "timestamp": "20240315_140000",
  "metrics": {
    "train_loss": 0.8234,
    "eval_loss": 0.9123,
    "train_samples_per_second": 45.2
  },
  "training_args": {
    "learning_rate": 0.0002,
    "batch_size": 4,
    "gradient_accumulation_steps": 4
  },
  "dataset_stats": {
    "num_examples": 1500,
    "avg_length": 512
  },
  "model_type": "LoRA adapters",
  "base_model": "mistralai/Mistral-7B-v0.1",
  "training_duration_seconds": 3600,
  "checkpoint_size_mb": 134.5
}
```

## Notes

- All trainers use Flash Attention 2 when `USE_FLASH_ATTENTION_2=True` in training_parameters.py
- Models are automatically loaded in 4-bit with QLoRA configuration
- The wrapper script doesn't add logic - it just routes to existing implementations
- Each trainer can also be run directly via their own CLI interfaces
- Checkpoints save only LoRA adapters (not the full model) to save space
- The CheckpointManager tracks rollback history and current checkpoint state