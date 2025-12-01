#!/usr/bin/env python3
"""
Pre-tokenize curriculum training datasets to eliminate runtime tokenization overhead.
Stores tokenized data in RAM for instant access during training.
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer
import training_parameters as params


def pre_tokenize_dataset(input_path: str, output_path: str, tokenizer, max_seq_len: int):
    """Pre-tokenize a dataset and save with input_ids/attention_mask."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing: {input_path.name}")
    print(f"Output: {output_path}")

    processed = 0
    total_tokens = 0

    with open(input_path) as f_in, open(output_path, 'w') as f_out:
        # Count lines first
        lines = f_in.readlines()
        f_in.seek(0)

        for line in tqdm(lines, desc="Tokenizing"):
            data = json.loads(line)
            text = data.get("text", "")

            if not text:
                continue

            # Tokenize once
            tokenized = tokenizer(
                text,
                truncation=True,
                max_length=max_seq_len,
                padding=False,
                return_tensors=None,
                add_special_tokens=True
            )

            # Store tokenized data
            data["input_ids"] = tokenized["input_ids"]
            data["attention_mask"] = tokenized["attention_mask"]

            # Keep original text for label computation
            # (labels are computed dynamically based on phase)

            f_out.write(json.dumps(data) + '\n')

            processed += 1
            total_tokens += len(tokenized["input_ids"])

    avg_tokens = total_tokens / processed if processed > 0 else 0
    print(f"✓ Processed {processed} examples")
    print(f"  Average tokens: {avg_tokens:.1f}")
    print(f"  Max sequence length: {max_seq_len}")

    return processed, avg_tokens


def main():
    parser = argparse.ArgumentParser(description="Pre-tokenize curriculum datasets")
    parser.add_argument("--conversational-data", type=str,
                      default="data/training_data/conversational-dataset_formatted.jsonl")
    parser.add_argument("--negotiation-data", type=str,
                      default="data/training_data/negotiation-dataset_formatted.jsonl")
    parser.add_argument("--model-name", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--max-seq-len", type=int, default=params.MAX_SEQ_LEN)
    parser.add_argument("--output-suffix", type=str, default="_tokenized",
                      help="Suffix to add to output filenames")

    args = parser.parse_args()

    print("=" * 60)
    print("PRE-TOKENIZATION FOR CURRICULUM TRAINING")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Max sequence length: {args.max_seq_len}")
    print("=" * 60)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Process conversational dataset
    conv_input = Path(args.conversational_data)
    conv_output = conv_input.parent / f"{conv_input.stem}{args.output_suffix}.jsonl"

    conv_count, conv_avg = pre_tokenize_dataset(
        str(conv_input),
        str(conv_output),
        tokenizer,
        args.max_seq_len
    )

    # Process negotiation dataset
    neg_input = Path(args.negotiation_data)
    neg_output = neg_input.parent / f"{neg_input.stem}{args.output_suffix}.jsonl"

    neg_count, neg_avg = pre_tokenize_dataset(
        str(neg_input),
        str(neg_output),
        tokenizer,
        args.max_seq_len
    )

    # Summary
    print("\n" + "=" * 60)
    print("TOKENIZATION COMPLETE")
    print("=" * 60)
    print(f"Total examples: {conv_count + neg_count}")
    print(f"Conversational: {conv_count} examples (avg {conv_avg:.1f} tokens)")
    print(f"Negotiation: {neg_count} examples (avg {neg_avg:.1f} tokens)")
    print(f"\nTokenized files:")
    print(f"  - {conv_output}")
    print(f"  - {neg_output}")
    print("\nTo use pre-tokenized data:")
    print(f"  python train_curriculum.py \\")
    print(f"    --conversational-data {conv_output} \\")
    print(f"    --negotiation-data {neg_output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
