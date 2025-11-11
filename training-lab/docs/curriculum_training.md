# Curriculum Training for Preventing Catastrophic Forgetting

## Quick Start

```bash
# Full training (8000 steps)
python train_curriculum.py

# Test mode (10 steps)
python train_curriculum.py --test-mode

# Custom parameters
python train_curriculum.py \
    --base-lr 5e-6 \
    --max-steps 10000 \
    --output-dir models/my_curriculum
```

## Architecture

The curriculum wrapper implements three-phase training:

1. **Contextual Learning (0-2000 steps)**
   - 50/50 data mix
   - Minimal masking (system only)
   - Reduced learning rate (50%)

2. **Guided Practice (2000-4000 steps)**
   - 30/70 conversational/negotiation
   - Alternating masking
   - Full learning rate

3. **Specialized Training (4000-8000 steps)**
   - 10/90 mix
   - Full SFT masking
   - Reduced LR (80%)

## Automatic Recovery

If token collapse is detected:
1. LR drops to 10%
2. Data mix becomes 70% conversational
3. All masking disabled
4. 200 recovery steps
5. Emergency checkpoint saved

## Monitoring

Health checks run every 100-500 steps (phase-dependent) and monitor:
- Token repetition ratios
- Response length
- Perplexity changes
- Specific collapse patterns ("eller", "0000")

## Components

### Phase Configuration (`training/curriculum/phase_config.py`)
- Defines 3 training phases with transitions at steps 0, 2000, 4000, 8000
- Configures data ratios, learning rates, and masking strategies per phase
- Validation frequency increases from 100 to 500 steps as training progresses

### Dynamic Masking Collator (`training/curriculum/masking_collator.py`)
- Implements 4 masking strategies: none, system_only, alternating, assistant_only
- Phase-aware strategy switching
- Handles tokenization, padding, and label masking

### Data Sampler (`training/curriculum/data_sampler.py`)
- Loads and mixes conversational (10,000 examples) and negotiation (10,770 examples) datasets
- Dynamic ratio-based sampling
- Emergency override for recovery mode

### Health Monitor (`training/curriculum/monitor.py`)
- Validates model health using conversation/negotiation prompts
- Detects token collapse patterns and perplexity degradation
- Saves emergency checkpoints with full diagnostics

### Curriculum Wrapper (`training/curriculum/wrapper.py`)
- Main orchestrator coordinating all components
- Manages phase transitions and recovery protocols
- Integrates with Transformers' `TrainerCallback` system

## Running Tests

```bash
# Run all curriculum tests
pytest tests/test_phase_config.py tests/test_curriculum_masking.py tests/test_curriculum_sampler.py tests/test_curriculum_monitor.py tests/test_curriculum_wrapper.py -v

# Run integration test
python test_curriculum_integration.py
```

## Command Line Options

```
--conversational-data    Path to conversational dataset (default: data/training_data/conversational-dataset_formatted.jsonl)
--negotiation-data       Path to negotiation dataset (default: data/training_data/negotiation-dataset_formatted.jsonl)
--model-name            HuggingFace model name (default: mistralai/Mistral-7B-v0.1)
--output-dir            Output directory (default: models/curriculum)
--max-steps             Maximum training steps (default: 8000)
--batch-size            Per-device batch size (default: 4)
--gradient-accumulation Gradient accumulation steps (default: 4)
--base-lr               Base learning rate (default: 1e-5)
--disable-recovery      Disable automatic recovery protocol
--test-mode             Run with only 10 steps for testing
```

## Implementation Details

### LoRA Configuration
- Rank: 64 (increased capacity)
- Alpha: 128
- Dropout: 0.1
- Target modules: All attention and MLP projections

### Training Configuration
- Optimizer: adamw_torch (from training_parameters)
- LR Scheduler: cosine with 10% warmup
- Precision: BF16 with TF32
- Gradient checkpointing enabled

## Troubleshooting

### Token Collapse Detected
If the monitor detects token collapse:
1. Check emergency checkpoint in `models/curriculum/emergency_checkpoint_step_*`
2. Review diagnostics.json for metrics comparison
3. Recovery protocol auto-activates unless `--disable-recovery` used

### Phase Transition Issues
Phase transitions occur automatically at configured steps. If transitions don't happen:
1. Check `CurriculumCallback` is registered with trainer
2. Verify `global_step` is incrementing
3. Review logs for phase configuration messages

### Memory Issues
If OOM errors occur:
1. Reduce `--batch-size`
2. Increase `--gradient-accumulation`
3. Reduce LoRA rank in train_curriculum.py
4. Check VRAM usage: `nvidia-smi`
