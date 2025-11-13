# Curriculum Checkpoint and Resume Design

**Date:** 2025-01-13

## Overview

Add checkpoint saving at curriculum phase boundaries and create a wrapper script to resume training from any phase with modified parameters.

## Requirements

1. Save checkpoints at the end of each curriculum phase
2. Create wrapper script to resume training from a specific phase
3. Allow modification of curriculum parameters for subsequent phases
4. Avoid redundant retraining of already-completed phases

## Design Decisions

### 1. Checkpoint Content
- **Save LoRA adapter weights only** (not full trainer state)
- Rationale: Resuming between phases with different parameters anyway, so reinitializing optimizer/scheduler is acceptable
- Checkpoint size: ~500MB vs ~14GB+ for full checkpoint

### 2. Phase Selection Method
- **Use phase number** (e.g., `--start-from-phase 2`)
- Simple numeric index: phase 1, 2, 3
- Assumes user knows phase order from curriculum_parameters.py

### 3. Parameter Modification Method
- **Edit curriculum_parameters.py directly**
- Simple, no additional config file management
- User modifies PHASES list before running resume script

### 4. Step Counter Behavior
- **Reset to 0 for resumed training**
- No step offset or fake checkpoint hacks
- Phase ranges adjusted to start from 0
- Clear banner printed showing what was loaded and new step ranges

## Architecture

### Checkpoint Structure

```
models/curriculum/
├── phase_1_step_800/
│   ├── adapter_model.safetensors    # LoRA weights
│   ├── adapter_config.json          # LoRA config
│   └── checkpoint_metadata.json     # Custom metadata
├── phase_2_step_2100/
│   └── ...
└── phase_3_step_3000/
    └── ...
```

### Metadata Format

```json
{
  "phase_number": 2,
  "phase_name": "guided_practice",
  "step": 2100,
  "total_phases": 3,
  "timestamp": "2025-01-13T10:30:00"
}
```

### Phase Range Adjustment

When resuming from phase N:
1. Filter to keep only phases >= N
2. Calculate offset = starting_phase.step_start
3. Subtract offset from all phase step ranges
4. Result: phases start from step 0

Example:
- Original: Phase 1 (0-800), Phase 2 (800-2100), Phase 3 (2100-3000)
- Resume from phase 2: Phase 2 (0-1300), Phase 3 (1300-2200)

## Implementation Plan

### Files to Modify

1. **training/curriculum/wrapper.py**
   - Update `_save_phase_checkpoint()` to save LoRA weights + metadata
   - Accept model reference for saving
   - Called from `_transition_phase()`

2. **train_curriculum.py**
   - Pass model to wrapper for checkpoint saving
   - Save final checkpoint as phase checkpoint (not just "final/")

### New Files

3. **resume_curriculum.py** (new script)
   - CLI args: `--start-from-phase`, `--checkpoint-dir`, plus inherited args
   - Load checkpoint and validate metadata
   - Adjust phase ranges
   - Print resume banner
   - Initialize model with loaded adapters
   - Run training from step 0

4. **training/curriculum/checkpoint_utils.py** (new module)
   - `save_phase_checkpoint()` - save adapters + metadata
   - `load_phase_checkpoint()` - load and validate
   - `adjust_phase_ranges()` - shift phase step ranges
   - `print_resume_banner()` - clear status message

## Error Handling

- Validate checkpoint directory exists and contains required files
- Validate phase number is in valid range
- Warn if curriculum_parameters.py differs from checkpoint metadata
- Check adjusted phase ranges are sensible

## Example Usage

### Initial training:
```bash
python train_curriculum.py
# Saves: phase_1_step_800/, phase_2_step_2100/, phase_3_step_3000/
```

### Resume from phase 2 with modified parameters:
1. Edit `curriculum_parameters.py` - change phase 2 and 3 parameters
2. Run:
```bash
python resume_curriculum.py \
  --start-from-phase 2 \
  --checkpoint-dir models/curriculum/phase_1_step_800
```

Output:
```
=================================================================
RESUMING CURRICULUM TRAINING
=================================================================
Loaded adapters from: models/curriculum/phase_1_step_800
Previous training completed through step 800 (end of phase 1)
Resuming with phases 2-3, step counter reset to 0
New step ranges: Phase 2 (0-1300), Phase 3 (1300-2200)
=================================================================
```

## Benefits

- **No redundant training**: Skip already-completed phases
- **Full flexibility**: Modify any parameter for remaining phases
- **Simple and explicit**: No hidden state, clear about what's happening
- **Disk efficient**: Only save LoRA adapters (~500MB per checkpoint)
- **Clean integration**: Minimal changes to existing code
