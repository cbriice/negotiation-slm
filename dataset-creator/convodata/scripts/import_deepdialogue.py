#!/usr/bin/env python3
"""
Import DeepDialogue-orpheus dataset and convert to SFT format.
Loads from the Parquet file to access the conversation data.
"""

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from tqdm import tqdm
import requests
import os


# Configuration
TARGET_PACKETS = 3000
MIN_TOKENS = 100  # Lowered from 300 to capture more conversations
MAX_TOKENS = 1000
TARGET_AVG_TOKENS = 475  # Middle of 450-500 range
CHARS_PER_TOKEN = 4  # Rough estimate

# System prompt variations (60% blank, 40% variations)
SYSTEM_PROMPTS = [
    "",  # 60% chance
    "",
    "",
    "You are a helpful assistant.",  # 40% chance split among these
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


class ReservoirSampler:
    """Reservoir sampling for random selection of conversations."""

    def __init__(self, target_size: int):
        self.target_size = target_size
        self.reservoir = []
        self.seen = 0
        self.token_counts = []
        self.domain_counts = defaultdict(int)

    def add(self, packet: Dict, token_count: int, domain: str):
        """Add a packet to the reservoir."""
        self.seen += 1

        if len(self.reservoir) < self.target_size:
            self.reservoir.append(packet)
            self.token_counts.append(token_count)
            self.domain_counts[domain] += 1
        else:
            # Randomly replace with decreasing probability
            j = random.randint(0, self.seen - 1)
            if j < self.target_size:
                # Track domain change
                old_packet = self.reservoir[j]
                # Note: we don't have domain in packet, just track separately
                self.reservoir[j] = packet
                self.token_counts[j] = token_count
                self.domain_counts[domain] += 1

    def get_results(self) -> Tuple[List[Dict], Dict]:
        """Get final sampled packets and statistics."""
        stats = {
            "total_seen": self.seen,
            "total_selected": len(self.reservoir),
            "avg_tokens": sum(self.token_counts) / len(self.token_counts) if self.token_counts else 0,
            "min_tokens": min(self.token_counts) if self.token_counts else 0,
            "max_tokens": max(self.token_counts) if self.token_counts else 0,
            "domain_distribution": dict(self.domain_counts)
        }
        return self.reservoir, stats


def main():
    """Main import function."""
    print("Loading DeepDialogue-orpheus dataset from Parquet file...")

    # Download or locate the Parquet file
    # The dataset is at https://huggingface.co/datasets/SALT-Research/DeepDialogue-orpheus
    parquet_url = "https://huggingface.co/datasets/SALT-Research/DeepDialogue-orpheus/resolve/main/data/train-00000-of-00001.parquet"

    # Create a temp directory for data
    data_dir = Path(__file__).parent.parent / "temp_data"
    data_dir.mkdir(exist_ok=True)
    parquet_path = data_dir / "train.parquet"

    # Download the parquet file if it doesn't exist
    if not parquet_path.exists():
        print(f"Downloading Parquet file from {parquet_url}...")
        response = requests.get(parquet_url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        with open(parquet_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
    else:
        print(f"Using existing Parquet file at {parquet_path}")

    # Load the Parquet file
    print("Loading Parquet file into memory...")
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} turns from {df['conversation_id'].nunique()} conversations")

    # Initialize components
    sampler = ReservoirSampler(TARGET_PACKETS)

    # Process statistics
    processed_convs = 0
    filtered_too_short = 0
    filtered_too_long = 0
    domain_seen = defaultdict(int)

    print("Processing conversations...")

    # Group by conversation_id and deduplicate by turn_index/speaker
    conversations = defaultdict(list)
    seen_turns = set()  # Track (conv_id, turn_index, speaker) to deduplicate

    for _, row in df.iterrows():
        key = (row['conversation_id'], row['turn_index'], row['speaker'])
        if key not in seen_turns:
            seen_turns.add(key)
            conversations[row['conversation_id']].append(row.to_dict())

    print(f"Found {len(conversations)} unique conversations (after deduplication)")

    # Process each conversation
    for conv_id, turns in tqdm(conversations.items(), desc="Processing conversations"):
        processed_convs += 1

        # Track domain if available
        if turns:
            domain = turns[0].get('domain', 'unknown')
            domain_seen[domain] += 1

        # Format to SFT
        result = format_conversation_to_sft(turns)
        if result is None:
            # Filtered out by token count
            avg_tokens = estimate_tokens(" ".join([t.get('text', '') for t in turns]))
            if avg_tokens < MIN_TOKENS:
                filtered_too_short += 1
            else:
                filtered_too_long += 1
            continue

        packet, token_count = result

        # Add to reservoir sampler
        sampler.add(packet, token_count, domain)

        # Early exit if we've seen enough good conversations
        if sampler.seen >= TARGET_PACKETS * 10:
            print(f"\nProcessed sufficient conversations ({sampler.seen} valid)")
            break

    # Get final results
    packets, stats = sampler.get_results()

    # Save to JSONL file
    output_path = Path(__file__).parent.parent / "data" / "deepdialogue_sft.jsonl"
    output_path.parent.mkdir(exist_ok=True)

    print(f"\nSaving {len(packets)} packets to {output_path}...")
    with open(output_path, 'w') as f:
        for packet in packets:
            f.write(json.dumps(packet) + '\n')

    # Enhance statistics
    stats.update({
        "total_conversations_processed": processed_convs,
        "filtered_too_short": filtered_too_short,
        "filtered_too_long": filtered_too_long,
        "domains_seen": dict(domain_seen),
        "system_prompt_distribution": {
            "blank": sum(1 for p in packets if not p["messages"] or
                        p["messages"][0]["role"] != "system" or
                        not p["messages"][0]["content"]),
            "with_prompt": sum(1 for p in packets if p["messages"] and
                             p["messages"][0]["role"] == "system" and
                             p["messages"][0]["content"])
        }
    })

    # Save statistics
    stats_path = Path(__file__).parent.parent / "logs" / "import_stats.json"
    stats_path.parent.mkdir(exist_ok=True)

    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    # Print summary
    print("\n" + "="*50)
    print("IMPORT COMPLETE")
    print("="*50)
    print(f"Total conversations processed: {stats['total_conversations_processed']:,}")
    print(f"Valid conversations found: {stats['total_seen']:,}")
    print(f"Conversations selected: {stats['total_selected']:,}")
    print(f"Filtered too short (<{MIN_TOKENS} tokens): {stats['filtered_too_short']:,}")
    print(f"Filtered too long (>{MAX_TOKENS} tokens): {stats['filtered_too_long']:,}")
    print(f"\nToken Statistics:")
    print(f"  Average: {stats['avg_tokens']:.1f} tokens")
    print(f"  Min: {stats['min_tokens']} tokens")
    print(f"  Max: {stats['max_tokens']} tokens")
    print(f"\nSystem Prompts:")
    print(f"  Blank: {stats['system_prompt_distribution']['blank']}")
    print(f"  With prompt: {stats['system_prompt_distribution']['with_prompt']}")
    print(f"\nDomains seen: {len(stats['domains_seen'])}")
    print(f"\nOutput: {output_path}")
    print(f"Stats: {stats_path}")


if __name__ == "__main__":
    main()