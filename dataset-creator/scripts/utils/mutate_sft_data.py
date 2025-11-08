#!/usr/bin/env python3
"""
SFT Data Mutation Script

Applies mutations to SFT training data to prevent model overfitting.

Mutation types:
- remove_first_turn: Remove first conversation turn
- remove_system_prompt: Remove system prompt
- mutate_system_prompt: Change system prompt to variations
- remove_assistant_style: Remove assistant_style field from system
- remove_opponent_style: Remove opponent_style field from meta
- remove_difficulty: Remove difficulty field from system
- remove_difficulty_bin: Remove difficulty_bin field from system
- remove_key_strategy: Remove key_strategy field from system

Usage:
    python mutate_sft_data.py --input data.jsonl --num-mutations 1000 --mutation-type remove_difficulty
"""

import argparse
import json
import random
import os
import sys
from pathlib import Path
from enum import Enum
from typing import Dict, Optional
from copy import deepcopy
from datetime import datetime
from collections import Counter


class MutationType(Enum):
    """Available mutation types."""
    REMOVE_FIRST_TURN = "remove_first_turn"
    REMOVE_SYSTEM_PROMPT = "remove_system_prompt"
    MUTATE_SYSTEM_PROMPT = "mutate_system_prompt"
    REMOVE_ASSISTANT_STYLE = "remove_assistant_style"
    REMOVE_OPPONENT_STYLE = "remove_opponent_style"
    REMOVE_DIFFICULTY = "remove_difficulty"
    REMOVE_DIFFICULTY_BIN = "remove_difficulty_bin"
    REMOVE_KEY_STRATEGY = "remove_key_strategy"


class SFTMutator:
    """Handles mutations for SFT training data."""

    # System prompt variations
    SYSTEM_PROMPT_VARIATIONS = [
        "You are proficient at negotiation and deal-making.",
        "You are negotiating.",
        "You're an expert negotiator.",
        "You serve as a skilled negotiator.",
        "Assume the role of a good negotiator.",
        "You are acting as a negotiation expert.",
        "You are using negotiation skills to make a deal.",
        "Act as a negotiator to make a deal.",
        "You are negotiating for something."
    ]

    def __init__(self, seed: Optional[int] = None):
        """Initialize the mutator with optional random seed."""
        if seed is not None:
            random.seed(seed)
            self.seed = seed
        else:
            self.seed = random.randint(0, 2**32 - 1)
            random.seed(self.seed)

    def validate_packet(self, packet: Dict) -> bool:
        """Check if packet has valid structure."""
        if not isinstance(packet, dict):
            return False

        # Check required fields
        if "messages" not in packet or "system" not in packet or "meta" not in packet:
            return False

        # Check messages is a non-empty list
        if not isinstance(packet["messages"], list) or len(packet["messages"]) == 0:
            return False

        return True

    def count_turns(self, packet: Dict) -> int:
        """Count conversation turns (excluding system prompt)."""
        return sum(1 for msg in packet["messages"] if msg.get("role") != "system")

    def has_system_prompt(self, packet: Dict) -> bool:
        """Check if packet has a system prompt."""
        return (len(packet["messages"]) > 0 and
                packet["messages"][0].get("role") == "system")

    def remove_first_turn(self, packet: Dict) -> Optional[Dict]:
        """Remove first conversation turn after system prompt."""
        # Need at least 3 turns total to remove 1 and keep 2
        if self.count_turns(packet) < 3:
            return None

        mutated = deepcopy(packet)

        # Find where to start removal (after system prompt if present)
        start_idx = 1 if self.has_system_prompt(packet) else 0

        # Remove 1 turn
        if start_idx < len(mutated["messages"]):
            del mutated["messages"][start_idx]
            return mutated

        return None

    def remove_system_prompt(self, packet: Dict) -> Optional[Dict]:
        """Remove the system prompt."""
        if not self.has_system_prompt(packet):
            return None

        mutated = deepcopy(packet)
        del mutated["messages"][0]
        return mutated

    def mutate_system_prompt(self, packet: Dict) -> Optional[Dict]:
        """Change system prompt to a variation."""
        if not self.has_system_prompt(packet):
            return None

        mutated = deepcopy(packet)
        current = mutated["messages"][0]["content"]

        # Pick a different variation
        variations = [v for v in self.SYSTEM_PROMPT_VARIATIONS if v != current]
        if variations:
            mutated["messages"][0]["content"] = random.choice(variations)
            return mutated

        # If current prompt isn't in our list, replace with random variation
        mutated["messages"][0]["content"] = random.choice(self.SYSTEM_PROMPT_VARIATIONS)
        return mutated

    def remove_assistant_style(self, packet: Dict) -> Optional[Dict]:
        """Remove assistant_style from system field."""
        if "assistant_style" not in packet.get("system", {}):
            return None

        mutated = deepcopy(packet)
        del mutated["system"]["assistant_style"]
        return mutated

    def remove_opponent_style(self, packet: Dict) -> Optional[Dict]:
        """Remove opponent_style from meta field."""
        if "opponent_style" not in packet.get("meta", {}):
            return None

        mutated = deepcopy(packet)
        del mutated["meta"]["opponent_style"]
        return mutated

    def remove_difficulty(self, packet: Dict) -> Optional[Dict]:
        """Remove difficulty from system field.

        If removing difficulty would result in both difficulty and
        difficulty_bin being absent, only proceeds 50% of the time."""
        if "difficulty" not in packet.get("system", {}):
            return None

        # Check if removing difficulty would leave packet without both difficulty fields
        # If difficulty_bin is also missing (or would be missing), apply 50% chance
        if "difficulty_bin" not in packet.get("system", {}):
            # Both would be missing, 50% chance to proceed
            if random.random() > 0.5:
                return None

        mutated = deepcopy(packet)
        del mutated["system"]["difficulty"]
        return mutated

    def remove_difficulty_bin(self, packet: Dict) -> Optional[Dict]:
        """Remove difficulty_bin from system field.

        If removing difficulty_bin would result in both difficulty and
        difficulty_bin being absent, only proceeds 50% of the time."""
        if "difficulty_bin" not in packet.get("system", {}):
            return None

        # Check if removing difficulty_bin would leave packet without both difficulty fields
        # If difficulty is also missing (or would be missing), apply 50% chance
        if "difficulty" not in packet.get("system", {}):
            # Both would be missing, 50% chance to proceed
            if random.random() > 0.5:
                return None

        mutated = deepcopy(packet)
        del mutated["system"]["difficulty_bin"]
        return mutated

    def remove_key_strategy(self, packet: Dict) -> Optional[Dict]:
        """Remove key_strategy from system field."""
        if "key_strategy" not in packet.get("system", {}):
            return None

        mutated = deepcopy(packet)
        del mutated["system"]["key_strategy"]
        return mutated

    def apply_mutation(self, packet: Dict, mutation_type: MutationType) -> Optional[Dict]:
        """Apply specified mutation to packet."""
        if mutation_type == MutationType.REMOVE_FIRST_TURN:
            return self.remove_first_turn(packet)
        elif mutation_type == MutationType.REMOVE_SYSTEM_PROMPT:
            return self.remove_system_prompt(packet)
        elif mutation_type == MutationType.MUTATE_SYSTEM_PROMPT:
            return self.mutate_system_prompt(packet)
        elif mutation_type == MutationType.REMOVE_ASSISTANT_STYLE:
            return self.remove_assistant_style(packet)
        elif mutation_type == MutationType.REMOVE_OPPONENT_STYLE:
            return self.remove_opponent_style(packet)
        elif mutation_type == MutationType.REMOVE_DIFFICULTY:
            return self.remove_difficulty(packet)
        elif mutation_type == MutationType.REMOVE_DIFFICULTY_BIN:
            return self.remove_difficulty_bin(packet)
        elif mutation_type == MutationType.REMOVE_KEY_STRATEGY:
            return self.remove_key_strategy(packet)
        else:
            raise ValueError(f"Unknown mutation type: {mutation_type}")

    def mutate_dataset(self,
                      input_file: str,
                      num_mutations: int,
                      mutation_type: Optional[MutationType] = None,
                      output_file: Optional[str] = None,
                      dry_run: bool = False) -> Dict:
        """
        Apply mutations to a dataset.

        Args:
            input_file: Input JSONL file
            num_mutations: Number of mutations to apply
            mutation_type: Specific mutation to apply (random if None)
            output_file: Output file path (auto-generated if None)
            dry_run: If True, don't write output

        Returns:
            Statistics dictionary
        """
        # Load packets
        print(f"Loading packets from {input_file}...")
        packets = []
        with open(input_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    packet = json.loads(line)
                    if self.validate_packet(packet):
                        packets.append(packet)
                    else:
                        print(f"Warning: Invalid packet at line {line_num}")
                except json.JSONDecodeError as e:
                    print(f"Warning: JSON error at line {line_num}: {e}")

        if not packets:
            raise ValueError("No valid packets found in input file")

        print(f"Loaded {len(packets)} valid packets")

        # Generate output filename
        if output_file is None:
            base = Path(input_file).stem
            parent = Path(input_file).parent
            output_file = str(parent / f"{base}-mutated.jsonl")

            # Add timestamp if exists
            if Path(output_file).exists() and not dry_run:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = str(parent / f"{base}-mutated-{timestamp}.jsonl")

        # Apply mutations to randomly selected packets
        if mutation_type:
            print(f"\nApplying {num_mutations} '{mutation_type.value}' mutations...")
        else:
            print(f"\nApplying {num_mutations} random mutations...")

        mutation_counts = Counter()
        failed = 0
        mutated_indices = set()
        mutation_types = list(MutationType)

        # Create a copy of all packets to modify in place
        output_packets = deepcopy(packets)

        for i in range(num_mutations):
            # Progress
            if (i + 1) % 100 == 0:
                print(f"Progress: {i + 1}/{num_mutations}")

            # Select random packet index
            packet_idx = random.randint(0, len(packets) - 1)
            packet = packets[packet_idx]

            # Apply mutation
            mutation_applied = False

            if mutation_type:
                # Use specified mutation
                mutated = self.apply_mutation(packet, mutation_type)
                if mutated is not None and self.validate_packet(mutated):
                    output_packets[packet_idx] = mutated
                    mutated_indices.add(packet_idx)
                    mutation_counts[mutation_type.value] += 1
                    mutation_applied = True
            else:
                # Try random mutations until one works
                attempts = 0
                while not mutation_applied and attempts < 10:
                    selected_mutation = random.choice(mutation_types)
                    mutated = self.apply_mutation(packet, selected_mutation)

                    if mutated is not None and self.validate_packet(mutated):
                        output_packets[packet_idx] = mutated
                        mutated_indices.add(packet_idx)
                        mutation_counts[selected_mutation.value] += 1
                        mutation_applied = True

                    attempts += 1

            if not mutation_applied:
                failed += 1

        # Count packets with both difficulty fields removed
        both_removed_count = 0
        for packet in output_packets:
            system = packet.get("system", {})
            if "difficulty" not in system and "difficulty_bin" not in system:
                both_removed_count += 1

        # Save output
        if not dry_run:
            print(f"\nSaving to {output_file}...")
            with open(output_file, 'w') as f:
                for packet in output_packets:
                    json.dump(packet, f, ensure_ascii=False, separators=(',', ':'))
                    f.write('\n')

        # Print summary
        print("\n" + "="*60)
        print("MUTATION SUMMARY")
        print("="*60)
        print(f"Input:              {input_file}")
        print(f"Output:             {output_file if not dry_run else '(dry run)'}")
        print(f"Total packets:      {len(packets)}")
        print(f"Packets mutated:    {len(mutated_indices)}")
        print(f"Mutations applied:  {sum(mutation_counts.values())}")
        print(f"Mutations failed:   {failed}")
        print(f"Both difficulty fields removed: {both_removed_count} packets")
        print(f"Random seed:        {self.seed}")

        if mutation_counts:
            print("\nMutation distribution:")
            total = sum(mutation_counts.values())
            for mut_type, count in sorted(mutation_counts.items()):
                pct = count / total * 100
                print(f"  {mut_type:25s}: {count:4d} ({pct:5.1f}%)")

        print("="*60)

        return {
            "input_file": input_file,
            "output_file": str(output_file) if not dry_run else None,
            "total_packets": len(packets),
            "packets_mutated": len(mutated_indices),
            "mutations_applied": sum(mutation_counts.values()),
            "mutations_failed": failed,
            "both_difficulty_removed": both_removed_count,
            "seed": self.seed,
            "distribution": dict(mutation_counts)
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Mutate SFT training data to prevent overfitting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Mutation types:
  remove_first_turn       Remove first conversation turn
  remove_system_prompt    Remove system prompt
  mutate_system_prompt    Change system prompt to variations
  remove_assistant_style  Remove assistant_style field from system
  remove_opponent_style   Remove opponent_style field from meta
  remove_difficulty       Remove difficulty field from system
  remove_difficulty_bin   Remove difficulty_bin field from system
  remove_key_strategy     Remove key_strategy field from system

Examples:
  # Apply specific mutation to 1000 packets
  python mutate_sft_data.py --input data.jsonl --num-mutations 1000 --mutation-type remove_difficulty

  # Apply random mutations (including the new ones)
  python mutate_sft_data.py --input data.jsonl --num-mutations 500

  # Use specific seed for reproducibility
  python mutate_sft_data.py --input data.jsonl --num-mutations 500 --mutation-type remove_key_strategy --seed 42

  # Dry run to preview
  python mutate_sft_data.py --input data.jsonl --num-mutations 100 --mutation-type remove_difficulty_bin --dry-run
        """
    )

    parser.add_argument('--input', '-i', required=True,
                       help='Input JSONL file')
    parser.add_argument('--num-mutations', '-n', type=int, default=100,
                       help='Number of mutations to apply (default: 100)')
    parser.add_argument('--mutation-type', '-m',
                       choices=['remove_first_turn', 'remove_system_prompt',
                               'mutate_system_prompt', 'remove_assistant_style',
                               'remove_opponent_style', 'remove_difficulty',
                               'remove_difficulty_bin', 'remove_key_strategy'],
                       help='Specific mutation type to apply (random if not specified)')
    parser.add_argument('--output', '-o',
                       help='Output file (default: auto-generated)')
    parser.add_argument('--seed', '-s', type=int,
                       help='Random seed for reproducibility')
    parser.add_argument('--dry-run', action='store_true',
                       help='Preview without writing output')

    args = parser.parse_args()

    # Check input exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found")
        sys.exit(1)

    # Parse mutation type
    mutation_type = None
    if args.mutation_type:
        mutation_type = MutationType(args.mutation_type)

    # Run mutations
    try:
        mutator = SFTMutator(seed=args.seed)
        mutator.mutate_dataset(
            input_file=args.input,
            num_mutations=args.num_mutations,
            mutation_type=mutation_type,
            output_file=args.output,
            dry_run=args.dry_run
        )
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()