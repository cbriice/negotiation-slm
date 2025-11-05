#!/usr/bin/env python3
"""
Test the import script with a small sample (10 packets).
"""

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from tqdm import tqdm

# Test configuration - very small sample
TARGET_PACKETS = 10  # Very small for testing
MIN_TOKENS = 300
MAX_TOKENS = 1000
CHARS_PER_TOKEN = 4

# System prompt variations
SYSTEM_PROMPTS = [
    "",  # 60% chance
    "",
    "",
    "You are a helpful assistant.",
    "You are a helpful conversational assistant.",
    "You are an AI assistant engaged in conversation.",
    "You are a friendly and knowledgeable assistant.",
    "You are an assistant here to help with various topics.",
    "You are a conversational AI assistant.",
    "You are here to assist and engage in dialogue.",
]

def estimate_tokens(text: str) -> int:
    """Estimate token count from character count."""
    return len(text) // CHARS_PER_TOKEN

def format_conversation_to_sft(conversation: List[dict]) -> Optional[Tuple[Dict, int]]:
    """Convert a conversation to SFT packet format."""
    # Sort turns by turn_index
    conversation.sort(key=lambda x: x['turn_index'])

    # Calculate total tokens
    total_text = " ".join([turn['text'] for turn in conversation])
    token_count = estimate_tokens(total_text)

    # Filter by token count
    if token_count < MIN_TOKENS or token_count > MAX_TOKENS:
        return None

    # Build messages array
    messages = []

    # Add system prompt (random selection)
    system_prompt = random.choice(SYSTEM_PROMPTS)
    if system_prompt:  # Only add if not empty
        messages.append({"role": "system", "content": system_prompt})

    # Add conversation turns (LLM1 = assistant, LLM2 = user)
    for turn in conversation:
        role = "assistant" if turn['speaker'] == "LLM1" else "user"
        messages.append({"role": role, "content": turn['text']})

    # Create SFT packet with empty negotiation fields
    packet = {
        "messages": messages,
        "system": {
            "assistant_style": "",
            "difficulty": "",
            "key_strategy": ""
        },
        "meta": {
            "opponent_style": "",
            "favorable_outcome": "",
            "deal_closed": "",
            "rounds": len(conversation)
        }
    }

    return packet, token_count

def main():
    """Test import with very small sample."""
    print("TEST: Loading small sample from Parquet file...")

    # Use existing downloaded file
    parquet_path = Path(__file__).parent.parent / "temp_data" / "train.parquet"

    if not parquet_path.exists():
        print(f"Error: Parquet file not found at {parquet_path}")
        print("Please run test_parquet.py first to download the file.")
        return

    # Load the Parquet file
    print("Loading Parquet file...")
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} turns from {df['conversation_id'].nunique()} conversations")

    # Group by conversation_id and deduplicate by turn_index/speaker
    conversations = defaultdict(list)
    seen_turns = set()  # Track (conv_id, turn_index, speaker) to deduplicate

    for _, row in df.iterrows():
        key = (row['conversation_id'], row['turn_index'], row['speaker'])
        if key not in seen_turns:
            seen_turns.add(key)
            conversations[row['conversation_id']].append(row.to_dict())

    print(f"Found {len(conversations)} unique conversations (after deduplication)")

    # Process conversations to find valid packets
    packets = []
    processed = 0
    filtered_short = 0
    filtered_long = 0

    for conv_id, turns in conversations.items():
        if len(packets) >= TARGET_PACKETS:
            break

        processed += 1
        result = format_conversation_to_sft(turns)

        if result is None:
            avg_tokens = estimate_tokens(" ".join([t['text'] for t in turns]))
            if avg_tokens < MIN_TOKENS:
                filtered_short += 1
            else:
                filtered_long += 1
            continue

        packet, token_count = result
        packets.append(packet)
        print(f"  Added conversation {conv_id}: {token_count} tokens, {len(turns)} turns")

    # Save test output
    output_path = Path(__file__).parent.parent / "data" / "test_small_sample.jsonl"
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w') as f:
        for packet in packets:
            f.write(json.dumps(packet) + '\n')

    # Print results
    print("\n" + "="*50)
    print("TEST COMPLETE")
    print("="*50)
    print(f"Conversations processed: {processed}")
    print(f"Valid packets created: {len(packets)}")
    print(f"Filtered too short: {filtered_short}")
    print(f"Filtered too long: {filtered_long}")
    print(f"\nTest output: {output_path}")

    # Show sample packet
    if packets:
        print("\nSample packet structure:")
        sample = packets[0]
        print(f"  Messages: {len(sample['messages'])} turns")
        print(f"  Has system prompt: {bool(sample['messages'] and sample['messages'][0]['role'] == 'system')}")
        print(f"  Rounds in meta: {sample['meta']['rounds']}")
        print("\nFirst message preview:")
        if sample['messages']:
            msg = sample['messages'][0]
            preview = msg['content'][:200] + "..." if len(msg['content']) > 200 else msg['content']
            print(f"  Role: {msg['role']}")
            print(f"  Content: {preview}")

if __name__ == "__main__":
    main()