# Model Training Recovery Plan

## Immediate Fixes

### 1. Reduce Learning Rate and Extend Training
```python
# Adjust in training_parameters.py
SFT_LR: float = 1e-5  # Reduced from 5e-5
SFT_MAX_STEPS: int = 8000  # Extended from 4000 (~4.7 epochs)
SFT_WARMUP_RATIO: float = 0.10  # Increased from 0.03 for gentler start
```

### 2. Increase LoRA Capacity
```python
# More capacity for complex task adaptation
R: int = 64  # Increased from 32
ALPHA: int = 128  # Increased from 64
DROPOUT: float = 0.1  # Increased from 0.05 for regularization
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"  # Add FFN layers back
]
```

### 3. Adjust Training Strategy

#### Option A: Full Visibility Training (Recommended)
- Remove masking during initial training phase
- Train on complete conversations first
- Add masking only in final refinement phase

#### Option B: Progressive Training
1. **Phase 1 (2000 steps):** Train on 50% conversational + 50% negotiation data with no masking
2. **Phase 2 (3000 steps):** Train on negotiation data with partial masking
3. **Phase 3 (3000 steps):** Full masked training on negotiation data

### 4. Data Preprocessing Fixes

Create a data verification script:
```python
def verify_dataset_formatting(jsonl_path):
    """Verify dataset consistency and masking alignment"""
    issues = []

    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            text = data['text']
            regions = data.get('masking_regions', [])

            # Check for response template alignment
            for region in regions:
                if region['role'] == 'assistant':
                    substr = text[region['start']:region['end']]
                    if not any(template in substr for template in RESPONSE_TEMPLATES):
                        issues.append(f"Line {i}: Assistant region missing response template")

            # Check for token length consistency
            if len(text) > MAX_SEQ_LEN:
                issues.append(f"Line {i}: Text exceeds max sequence length")

    return issues
```

### 5. Generation Parameter Tuning

During inference/evaluation:
```python
# Add to model generation
gen_kwargs = {
    "max_new_tokens": 150,  # Reduced from 256
    "temperature": 0.5,  # Reduced from 0.7
    "do_sample": True,
    "top_p": 0.85,  # Tightened from 0.9
    "repetition_penalty": 1.2,  # Add repetition penalty
    "no_repeat_ngram_size": 3,  # Prevent trigram repetition
}
```

## Monitoring During Retraining

1. **Early Stopping Criteria:**
   - Monitor perplexity on held-out conversational data
   - Stop if conversational perplexity increases by >20%

2. **Validation Checkpoints:**
   - Save every 500 steps
   - Test each checkpoint with simple prompts:
     ```
     "Hello, how are you?"
     "What is 2+2?"
     "Tell me about yourself"
     ```
   - If basic responses degrade, reduce learning rate

3. **Gradient Monitoring:**
   ```python
   # Add to training loop
   if step % 100 == 0:
       grad_norm = compute_gradient_norm(model)
       if grad_norm > 10:
           logger.warning(f"High gradient norm: {grad_norm}")
           # Consider gradient clipping
   ```

## Dataset Mixing Strategy

Instead of 25% conversational data throughout:
1. **Curriculum Learning:**
   - Steps 0-1000: 50% conversational, 50% negotiation
   - Steps 1000-4000: 30% conversational, 70% negotiation
   - Steps 4000+: 10% conversational, 90% negotiation

2. **Ensure Diversity:**
   - Add more varied negotiation scenarios
   - Include failed negotiations (no deal outcomes)
   - Mix different negotiation styles

## Testing Recovery

After retraining, test with:
```bash
python custom_eval/run_interactive.py --checkpoint [new_checkpoint] --test-basic-conversation
```

Add basic conversation tests before negotiation tests to ensure base capabilities are preserved.

## Root Cause Summary

Your model experienced catastrophic forgetting because:
1. **Learning rate too high** → Overwrote base knowledge too quickly
2. **Training too short** → Insufficient time to learn new patterns properly
3. **LoRA capacity too small** → Forced oversimplified pattern compression
4. **Masking complexity** → Model learned fragmented, context-free tokens

The "eller" repetition and zeros indicate the model collapsed to high-frequency token patterns in your dataset without understanding their contextual usage.