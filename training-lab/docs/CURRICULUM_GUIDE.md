# Weighted Masking Curriculum Training Guide

## Quick Start

### 1. Configure Your Training

Edit `curriculum_parameters.py` to adjust:
- **MAX_STEPS**: Total training duration
- **Phase boundaries**: When each phase starts/ends
- **User weights**: Loss weights for user turns (0.0 = fully masked, 1.0 = full weight)
- **Data ratios**: Mix of conversational vs negotiation data

### 2. Use Presets for Quick Testing

Uncomment a preset in `curriculum_parameters.py`:

```python
# Ultra-quick 500-step diagnostic (~10-15 min)
MAX_STEPS, PHASES = DIAGNOSTIC_500["MAX_STEPS"], DIAGNOSTIC_500["PHASES"]

# OR 2000-step validation (~1-2 hours)
MAX_STEPS, PHASES = VALIDATION_2000["MAX_STEPS"], VALIDATION_2000["PHASES"]
```

### 3. Test Your Setup

```bash
python test_curriculum_quick.py
```

This validates that weighted masking is working correctly.

### 4. Run Training

```bash
# Use MAX_STEPS from curriculum_parameters.py
python train_curriculum.py --output-dir models/my_run

# Override with command line
python train_curriculum.py --max-steps 500 --output-dir models/diagnostic
```

## Understanding Weighted Masking

### The Problem (Binary Masking)
Your original training used **binary masking** with `ignore_index=-100`:
- User turns: **Completely removed** from loss (ignore_index=-100)
- Assistant turns: **Fully trained** on

Result: Model learned to generate full dialogs because it saw complete conversations in the input but only predicted assistant turns. It learned the pattern of "conversations look like this: Buyer:... Seller:..."

### The Solution (Weighted Masking)
New training uses **weighted loss calculation**:
- User turns: **Tiny weight** (0.05 → 0.02 → 0.0)
- Assistant turns: **Full weight** (1.0)

Result: Model learns from context without being rewarded for generating both sides.

## Key Parameters to Tune

### User Weights (Most Important)
```python
"user_weight": 0.05   # Phase 1: Learn from context
"user_weight": 0.02   # Phase 2: Reduce influence
"user_weight": 0.0    # Phase 3: Pure assistant responses
```

**Too high** (>0.1): Model learns to generate both sides
**Too low** (<0.01): Model loses conversational context
**Sweet spot**: 0.02-0.05 for early phases

### Phase Duration
```python
"step_start": 0,
"step_end": 2000,
```

**Longer phases**: More stable learning, slower specialization
**Shorter phases**: Faster iteration, risk of abrupt transitions

### Data Ratios
```python
"data_ratio": {"conversational": 0.5, "negotiation": 0.5}
```

**More conversational**: Preserves general ability, slower specialization
**More negotiation**: Faster specialization, risk of forgetting

## Checking Model Behavior

After training, test if the model:

**Good signs:**
- Generates single-party responses ("I can offer $11,000")
- Responds contextually to user input
- Uses negotiation strategies

**Bad signs:**
- Generates "Buyer:" or "Seller:" labels
- Produces meta-commentary ("### Summary:")
- Creates full dialogs instead of responses

## Quick Iteration Workflow

1. **Start with 500-step diagnostic**
   ```bash
   # Edit curriculum_parameters.py
   MAX_STEPS, PHASES = DIAGNOSTIC_500["MAX_STEPS"], DIAGNOSTIC_500["PHASES"]

   # Run training
   python train_curriculum.py --output-dir models/test1
   ```

2. **Test checkpoint behavior**
   ```bash
   python custom_eval/run_selfchat.py \
       --adapter-path models/test1/checkpoint-500 \
       --num-seeds 2 \
       --verbose
   ```

3. **Adjust weights if needed**
   - Still generating dialogs? → **Reduce user_weight** (try 0.03 → 0.01)
   - Losing context? → **Increase user_weight** (try 0.05 → 0.08)

4. **Run 2000-step validation** once 500-step looks good

5. **Full 8000-step training** once validated

## Example Configurations

### Conservative (Safe but Slow)
```python
PHASE_1: user_weight=0.03, steps=0-3000
PHASE_2: user_weight=0.02, steps=3000-6000
PHASE_3: user_weight=0.0, steps=6000-8000
```

### Aggressive (Fast but Risky)
```python
PHASE_1: user_weight=0.05, steps=0-1000
PHASE_2: user_weight=0.01, steps=1000-2000
PHASE_3: user_weight=0.0, steps=2000-3000
```

### Recommended (Balanced)
```python
PHASE_1: user_weight=0.05, steps=0-2000    # Default in curriculum_parameters.py
PHASE_2: user_weight=0.02, steps=2000-6000
PHASE_3: user_weight=0.0, steps=6000-8000
```

## Troubleshooting

### "Model still generates full dialogs at step 500"
- **Reduce user_weight** in Phase 1 from 0.05 to 0.02
- Check that weighted masking is active: `python test_curriculum_quick.py`

### "Model responses are incoherent"
- **Increase user_weight** in early phases to 0.08
- **Extend Phase 1** duration (more contextual learning)

### "Training loss is unstable"
- **Lower learning rate**: `--base-lr 5e-6` instead of `1e-5`
- **Longer warmup**: Edit `train_curriculum.py` warmup_ratio from 0.1 to 0.15

### "OOM (Out of Memory) errors"
- **Reduce batch size**: `--batch-size 2` instead of 4
- **Increase gradient accumulation**: `--gradient-accumulation 8` instead of 4

## Files Modified

- `curriculum_parameters.py` - **NEW**: All training configuration
- `training/curriculum/phase_config.py` - Loads from parameters file
- `training/curriculum/masking_collator.py` - Weighted masking implementation
- `train_curriculum.py` - Uses WeightedLossSFTTrainer
- `test_curriculum_quick.py` - **NEW**: Validation script
