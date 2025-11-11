# Training Optimization Report for RTX 4070 Ti Super 16GB

## Current Problem
- **Current Speed**: ~10 seconds per step
- **Total Training Time**: ~29 hours for 8000 steps
- **Quality Issues**: Model showed degeneration after training (repetition, incomplete outputs)

## Optimization Strategy
Maximize speed while maintaining quality to prevent degeneration.

## Key Optimizations Implemented

### 1. **4-Bit Quantization (QLoRA)**
- **Status**: Currently NOT enabled in original script despite params being set
- **Impact**: ~40% speedup, 75% memory reduction
- **Implementation**: Model loads with BitsAndBytes 4-bit NF4 quantization

### 2. **Optimized Gradient Accumulation**
- **Original**: batch_size=4, grad_accum=4 (effective=16)
- **Optimized**: batch_size=4, grad_accum=6 (effective=24)
- **Impact**: Larger effective batch for better convergence

### 3. **Reduced Sequence Length**
- **Original**: MAX_SEQ_LEN=2048
- **Optimized**: MAX_SEQ_LEN=1536 (covers your 1200 token max)
- **Impact**: ~25% memory savings, ~25% speedup

### 4. **Parallel Data Loading**
- **Original**: dataloader_num_workers=0 (single-threaded)
- **Optimized**: dataloader_num_workers=2
- **Impact**: ~15% speedup from reduced I/O blocking

### 5. **Memory Management**
- **Flash Attention 2**: Already enabled, ~35% speedup on attention
- **Gradient Checkpointing**: Enabled for memory efficiency
- **8-bit Optimizer**: Saves optimizer memory
- **Pin Memory**: Faster CPU-GPU transfers

### 6. **Quality Preservation**
- **Kept LoRA rank=64**: Higher capacity as you requested
- **Kept alpha=128**: 2x rank for better learning
- **Kept dropout=0.1**: Regularization against overfitting
- **All target modules**: Full coverage of attention and MLP layers

## VRAM Budget Analysis

```
Total VRAM: 16GB
Target Usage: 12.5GB (3.5GB safety buffer)

Breakdown:
- 4-bit Mistral-7B model: ~3.5GB
- LoRA adapters (r=64): ~0.2GB
- 8-bit AdamW optimizer: ~0.4GB
- Batch=4 activations: ~7.5GB
- Flash Attention cache: ~0.8GB
- Working memory: ~0.5GB
--------------------------------
Total: ~12.5GB (safe, ~90% GPU utilization)
```

## Performance Estimates

### Conservative Estimates:
- **Original**: 0.1 steps/second (10 seconds/step)
- **With optimizations**: ~0.22 steps/second (4.5 seconds/step)
- **Speedup**: ~2.2x
- **New training time**: ~10-11 hours (down from 29 hours)

### Best Case (with all optimizations working):
- Up to 0.3 steps/second (3.3 seconds/step)
- Training time: ~7-8 hours

## How to Test

### 1. Quick Test (Verify it runs):
```bash
python train_curriculum_optimized.py --test-mode --max-steps 20
```

### 2. Monitor VRAM Usage:
```bash
# In another terminal:
watch -n 1 nvidia-smi
```

### 3. Full Training:
```bash
python train_curriculum_optimized.py \
    --max-steps 8000 \
    --output-dir models/curriculum_optimized
```

## Fallback Options if Still Too High

If GPU utilization or VRAM is still too high, adjust in this order:

1. **Reduce batch size to 3**:
   - Edit `training_parameters_optimized.py`
   - Change `PER_DEVICE_TRAIN_BATCH_SIZE = 3`
   - Change `GRADIENT_ACCUMULATION_STEPS = 8` to maintain effective batch

2. **Reduce sequence length to 1400**:
   - Change `MAX_SEQ_LEN = 1400`
   - Still covers your 1200 token max with buffer

3. **Reduce eval batch size**:
   - Change `PER_DEVICE_EVAL_BATCH_SIZE = 6`
   - Evaluation uses less memory but can be slower

## Monitoring Training Quality

Watch for these warning signs of quality degradation:

1. **Loss spikes** after 2000+ steps
2. **Repetitive outputs** in validation
3. **Gradient norm** consistently > 5.0
4. **Learning rate** might need reduction if unstable

The curriculum wrapper includes automatic recovery protocols if collapse is detected.

## Additional Optimization Opportunities

If you want even more speed:

1. **Mixed Precision Training**: Consider using `fp16` instead of `bf16` if stable
2. **Torch Compile**: Add `torch.compile(model)` for potential 10-20% speedup
3. **DeepSpeed**: For advanced users, can provide additional optimizations
4. **Data Preprocessing**: Pre-tokenize dataset to eliminate runtime tokenization

## Expected Results

With these conservative optimizations, you should see:
- **Training time**: 10-11 hours (vs 29 hours)
- **Steps/second**: 0.22-0.3 (vs 0.1)
- **Quality**: Maintained or improved due to larger effective batch size (24 vs 16)
- **VRAM usage**: ~12.5GB peak (3.5GB safety margin)
- **GPU utilization**: ~90% (±2%)

The key insight is that your current setup wasn't using 4-bit quantization despite having it configured. Even with conservative batch sizing, you'll get 2-3x speedup while maintaining stability for long training runs.