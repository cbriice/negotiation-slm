# Real Performance Analysis - Why Training is Slow

## Current Performance
- **11.7 seconds/step** with batch_size=4, grad_accum=4
- For 8000 steps = ~26 hours

## The ACTUAL Bottleneck: Dynamic Dataset

The `CurriculumIterableDataset` is tokenizing on-the-fly for EVERY sample:

```python
def __iter__(self):
    while True:
        batch = self.wrapper.data_sampler.sample_batch(1)  # Sample 1 at a time
        tokenized = self.tokenizer(text, ...)  # Tokenize EVERY sample
        labels = self.wrapper.masking_collator.compute_labels(...)  # Compute labels EVERY time
```

**This means per step:**
- 16 tokenizations (batch_size=4 × grad_accum=4)
- 16 label computations
- All happening sequentially during training!

## Real Optimizations That Would Help

### Option 1: Pre-tokenize the Dataset (Biggest Win)
Create a script to pre-tokenize both datasets once:

```python
# pre_tokenize_data.py
import json
from transformers import AutoTokenizer
from tqdm import tqdm

def pre_tokenize_dataset(input_path, output_path, tokenizer, max_seq_len=1536):
    with open(input_path) as f_in, open(output_path, 'w') as f_out:
        for line in tqdm(f_in):
            data = json.loads(line)
            text = data.get("text", "")

            # Tokenize once
            tokenized = tokenizer(
                text,
                truncation=True,
                max_length=max_seq_len,
                padding=False,
                return_tensors=None
            )

            # Save tokenized version
            data["input_ids"] = tokenized["input_ids"]
            data["attention_mask"] = tokenized["attention_mask"]
            f_out.write(json.dumps(data) + '\n')

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
pre_tokenize_dataset(
    "data/training_data/conversational-dataset_formatted.jsonl",
    "data/training_data/conversational-dataset_tokenized.jsonl",
    tokenizer
)
pre_tokenize_dataset(
    "data/training_data/negotiation-dataset_formatted.jsonl",
    "data/training_data/negotiation-dataset_tokenized.jsonl",
    tokenizer
)
```

### Option 2: Batch Processing (Moderate Win)
Instead of sampling 1 at a time, sample and tokenize in batches:

```python
def __iter__(self):
    while True:
        # Sample full batch at once
        batch = self.wrapper.data_sampler.sample_batch(16)  # Full batch
        texts = [ex.get("text", "") for ex in batch]

        # Batch tokenization (much faster)
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_seq_len,
            padding=True,
            return_tensors="pt"
        )

        # Yield individual examples
        for i in range(len(batch)):
            yield {
                "input_ids": tokenized["input_ids"][i],
                "attention_mask": tokenized["attention_mask"][i],
                ...
            }
```

### Option 3: Reduce Sequence Length More Aggressively
Since your max is 1200 tokens, try MAX_SEQ_LEN=1280 instead of 1536:
- Saves ~20% compute per attention layer
- Still covers all your data

## Expected Speedup with Pre-tokenization

- **Current**: 11.7 sec/step (tokenizing during training)
- **With pre-tokenization**: ~6-8 sec/step
- **Speedup**: ~1.5-2x
- **New total time**: ~13-18 hours (down from 26)

## Why My Original "Optimizations" Failed

1. **4-bit quantization**: Already enabled via ModelLoader
2. **Flash Attention**: Already enabled
3. **Increased grad_accum**: Made it WORSE (more forward passes)
4. **Reduced seq_len by 25%**: Not enough to offset grad_accum increase

## The Hardware Reality

Your RTX 4070 Ti Super with proper optimization should achieve:
- **Inference**: 50-100 tokens/sec
- **Training (batch=4)**: ~0.15-0.2 steps/sec (5-7 sec/step)

Current 11.7 sec/step suggests CPU bottleneck from tokenization, not GPU.

## Recommended Action

1. **Keep using baseline `train_curriculum.py`** (it's already well-optimized)
2. **Pre-tokenize your datasets** (biggest win)
3. **Consider reducing MAX_SEQ_LEN to 1280** in `training_parameters.py`
4. **Monitor with `htop`** during training - if CPU is at 100%, that confirms tokenization bottleneck