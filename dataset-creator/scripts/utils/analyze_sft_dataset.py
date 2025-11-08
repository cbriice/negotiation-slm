#!/usr/bin/env python3
"""
Analyze SFT dataset for mutation statistics and restore over-mutated packets.

Functions:
- analyze_mutations: Get statistics on mutations applied by mutate_sft_data.py
- restore_packets: Restore the most mutated packets from reference file
"""

import json
import argparse
import sys
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional

def detect_mutations(mutated_packet: Dict, reference_packet: Dict) -> List[str]:
    """Detect which mutations were applied to a packet."""
    mutations = []

    # Check system prompt mutations
    mutated_msgs = mutated_packet.get("messages", [])
    ref_msgs = reference_packet.get("messages", [])

    # Check if first turn was removed
    if len(mutated_msgs) < len(ref_msgs):
        mutations.append("remove_first_turn")

    # Check system prompt changes
    if mutated_msgs and ref_msgs:
        mutated_has_sys = mutated_msgs[0].get("role") == "system"
        ref_has_sys = ref_msgs[0].get("role") == "system"

        if ref_has_sys and not mutated_has_sys:
            mutations.append("remove_system_prompt")
        elif mutated_has_sys and ref_has_sys:
            if mutated_msgs[0].get("content") != ref_msgs[0].get("content"):
                mutations.append("mutate_system_prompt")

    # Check field removals in system
    mutated_sys = mutated_packet.get("system", {})
    ref_sys = reference_packet.get("system", {})

    if "assistant_style" in ref_sys and "assistant_style" not in mutated_sys:
        mutations.append("remove_assistant_style")

    if "difficulty" in ref_sys and "difficulty" not in mutated_sys:
        mutations.append("remove_difficulty")

    if "difficulty_bin" in ref_sys and "difficulty_bin" not in mutated_sys:
        mutations.append("remove_difficulty_bin")

    if "key_strategy" in ref_sys and "key_strategy" not in mutated_sys:
        mutations.append("remove_key_strategy")

    # Check field removals in meta
    mutated_meta = mutated_packet.get("meta", {})
    ref_meta = reference_packet.get("meta", {})

    if "opponent_style" in ref_meta and "opponent_style" not in mutated_meta:
        mutations.append("remove_opponent_style")

    return mutations


def analyze_mutations(mutated_file: str, reference_file: str, add_mutation_field: bool = False, output_file: Optional[str] = None) -> Dict:
    """
    Analyze mutations in a dataset by comparing to reference.

    Args:
        mutated_file: Path to mutated dataset
        reference_file: Path to reference dataset
        add_mutation_field: If True, add "mutations_applied" to meta field and save
        output_file: Output file path (required if add_mutation_field is True)

    Returns statistics on:
    - Number of mutations per packet
    - Number of untouched packets
    - Personality distributions for packets with both difficulty fields removed
    """
    # Load files line by line
    print(f"Analyzing mutations in: {mutated_file}")
    print(f"Using reference: {reference_file}")

    with open(mutated_file, 'r') as mf, open(reference_file, 'r') as rf:
        mutated_lines = mf.readlines()
        reference_lines = rf.readlines()

    if len(mutated_lines) != len(reference_lines):
        print(f"Warning: File lengths differ - mutated: {len(mutated_lines)}, reference: {len(reference_lines)}")
        print("Analyzing up to the shorter length...")

    # Statistics tracking
    mutations_per_packet = Counter()  # Count of packets with 0, 1, 2, 3+ mutations
    mutation_types = Counter()  # Count of each mutation type
    untouched = 0
    both_difficulty_removed_personalities = Counter()
    packets_analyzed = 0

    # Store modified packets if needed
    modified_packets = [] if add_mutation_field else None

    # Analyze each packet
    min_len = min(len(mutated_lines), len(reference_lines))
    for i in range(min_len):
        if not mutated_lines[i].strip() or not reference_lines[i].strip():
            continue

        try:
            mutated_packet = json.loads(mutated_lines[i])
            reference_packet = json.loads(reference_lines[i])
            packets_analyzed += 1

            # Detect mutations
            mutations = detect_mutations(mutated_packet, reference_packet)

            # Add mutations_applied field to meta if requested
            if add_mutation_field:
                if "meta" not in mutated_packet:
                    mutated_packet["meta"] = {}
                mutated_packet["meta"]["mutations_applied"] = mutations
                modified_packets.append(json.dumps(mutated_packet, ensure_ascii=False, separators=(',', ':')) + '\n')

            # Track statistics
            num_mutations = len(mutations)
            if num_mutations == 0:
                untouched += 1
                mutations_per_packet[0] += 1
            elif num_mutations >= 3:
                mutations_per_packet["3+"] += 1
            else:
                mutations_per_packet[num_mutations] += 1

            # Track mutation types
            for mut in mutations:
                mutation_types[mut] += 1

            # Check for both difficulty fields removed
            mutated_sys = mutated_packet.get("system", {})
            if ("difficulty" not in mutated_sys and
                "difficulty_bin" not in mutated_sys):
                # Track personality for this packet
                if "assistant_style" in mutated_sys:
                    both_difficulty_removed_personalities[mutated_sys["assistant_style"]] += 1

        except json.JSONDecodeError:
            print(f"Skipping line {i+1} due to JSON error")
            continue

    # Save modified packets if requested
    if add_mutation_field:
        if output_file is None:
            base = Path(mutated_file).stem
            parent = Path(mutated_file).parent
            output_file = str(parent / f"{base}-with-mutations.jsonl")

        print(f"\nSaving dataset with mutation tracking to: {output_file}")
        with open(output_file, 'w') as f:
            f.writelines(modified_packets)

    # Calculate percentages
    total_with_both_removed = sum(both_difficulty_removed_personalities.values())
    personality_percentages = {}
    if total_with_both_removed > 0:
        for personality, count in both_difficulty_removed_personalities.items():
            personality_percentages[personality] = (count / total_with_both_removed) * 100

    # Return statistics
    result = {
        "packets_analyzed": packets_analyzed,
        "untouched_packets": untouched,
        "mutations_per_packet": dict(mutations_per_packet),
        "mutation_type_counts": dict(mutation_types),
        "packets_with_both_difficulty_removed": total_with_both_removed,
        "personality_distribution": personality_percentages
    }

    if add_mutation_field:
        result["mutations_tracked_file"] = output_file

    return result


def restore_packets(mutated_file: str,
                    reference_file: str,
                    num_restore: int,
                    output_file: Optional[str] = None) -> Dict:
    """
    Restore the most mutated packets from reference file.

    Args:
        mutated_file: Path to mutated dataset
        reference_file: Path to original reference dataset
        num_restore: Number of packets to restore
        output_file: Output file path (auto-generated if None)

    Returns:
        Statistics about the restoration
    """
    print(f"Preparing to restore {num_restore} packets...")

    # Load files
    with open(mutated_file, 'r') as mf, open(reference_file, 'r') as rf:
        mutated_lines = mf.readlines()
        reference_lines = rf.readlines()

    if len(mutated_lines) != len(reference_lines):
        raise ValueError(f"File lengths must match - mutated: {len(mutated_lines)}, reference: {len(reference_lines)}")

    # Calculate mutations per packet with line indices
    packet_mutations = []  # List of (line_index, num_mutations, mutations_list)

    for i in range(len(mutated_lines)):
        if not mutated_lines[i].strip() or not reference_lines[i].strip():
            continue

        try:
            mutated_packet = json.loads(mutated_lines[i])
            reference_packet = json.loads(reference_lines[i])

            mutations = detect_mutations(mutated_packet, reference_packet)
            packet_mutations.append((i, len(mutations), mutations))

        except json.JSONDecodeError:
            continue

    # Sort by number of mutations (descending)
    packet_mutations.sort(key=lambda x: x[1], reverse=True)

    # Select packets to restore
    packets_to_restore = packet_mutations[:num_restore]
    restore_indices = {idx for idx, _, _ in packets_to_restore}

    # Create output
    output_lines = []
    restored_count = 0

    for i in range(len(mutated_lines)):
        if i in restore_indices:
            output_lines.append(reference_lines[i])
            restored_count += 1
        else:
            output_lines.append(mutated_lines[i])

    # Generate output filename if needed
    if output_file is None:
        base = Path(mutated_file).stem
        parent = Path(mutated_file).parent
        output_file = str(parent / f"{base}-restored.jsonl")

    # Write output
    print(f"Writing restored dataset to: {output_file}")
    with open(output_file, 'w') as f:
        f.writelines(output_lines)

    # Prepare statistics
    mutations_restored = Counter()
    for _, num_muts, muts_list in packets_to_restore:
        for mut in muts_list:
            mutations_restored[mut] += 1

    return {
        "output_file": output_file,
        "packets_restored": restored_count,
        "total_packets": len(mutated_lines),
        "mutations_restored": dict(mutations_restored),
        "max_mutations_in_restored_packet": packets_to_restore[0][1] if packets_to_restore else 0
    }


def print_analysis_results(stats: Dict):
    """Pretty print analysis statistics."""
    print("\n" + "="*70)
    print("MUTATION ANALYSIS RESULTS")
    print("="*70)

    print(f"\nTotal packets analyzed: {stats['packets_analyzed']}")
    print(f"Untouched packets: {stats['untouched_packets']}")

    if "mutations_tracked_file" in stats:
        print(f"\nMutations tracked and saved to: {stats['mutations_tracked_file']}")

    print("\nPackets by mutation count:")
    for num_muts in [0, 1, 2, "3+"]:
        key = str(num_muts) if isinstance(num_muts, str) else num_muts
        if key in stats['mutations_per_packet']:
            count = stats['mutations_per_packet'][key]
            pct = (count / stats['packets_analyzed']) * 100 if stats['packets_analyzed'] > 0 else 0
            print(f"  {num_muts} mutations: {count} packets ({pct:.1f}%)")

    print("\nMutation types applied:")
    for mut_type, count in sorted(stats['mutation_type_counts'].items()):
        print(f"  {mut_type}: {count}")

    print(f"\nPackets with both difficulty fields removed: {stats['packets_with_both_difficulty_removed']}")

    if stats['personality_distribution']:
        print("\nPersonality distribution (for packets with both difficulty removed):")
        for personality, pct in sorted(stats['personality_distribution'].items(),
                                       key=lambda x: x[1], reverse=True):
            print(f"  {personality}: {pct:.1f}%")


def print_restore_results(stats: Dict):
    """Pretty print restoration statistics."""
    print("\n" + "="*70)
    print("RESTORATION RESULTS")
    print("="*70)

    print(f"\nOutput file: {stats['output_file']}")
    print(f"Packets restored: {stats['packets_restored']} / {stats['total_packets']}")
    print(f"Max mutations in a restored packet: {stats['max_mutations_in_restored_packet']}")

    if stats['mutations_restored']:
        print("\nMutations restored:")
        for mut_type, count in sorted(stats['mutations_restored'].items()):
            print(f"  {mut_type}: {count}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze SFT dataset mutations and restore over-mutated packets",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze',
                                          help='Analyze mutations in dataset')
    analyze_parser.add_argument('--mutated', '-m', required=True,
                               help='Path to mutated dataset')
    analyze_parser.add_argument('--reference', '-r',
                               default='data/runs/bulk_data/real_training_data/all_negotiations_cleaned.jsonl',
                               help='Path to reference dataset (default: all_negotiations_cleaned.jsonl)')
    analyze_parser.add_argument('--add-mutations-field', action='store_true',
                               help='Add "mutations_applied" field to meta and save modified dataset')
    analyze_parser.add_argument('--output', '-o',
                               help='Output file for dataset with mutations tracked (auto-generated if not provided)')

    # Restore command
    restore_parser = subparsers.add_parser('restore',
                                          help='Restore most mutated packets')
    restore_parser.add_argument('--mutated', '-m', required=True,
                               help='Path to mutated dataset')
    restore_parser.add_argument('--reference', '-r',
                               default='data/runs/bulk_data/real_training_data/all_negotiations_cleaned.jsonl',
                               help='Path to reference dataset (default: all_negotiations_cleaned.jsonl)')
    restore_parser.add_argument('--num-restore', '-n', type=int, required=True,
                               help='Number of packets to restore')
    restore_parser.add_argument('--output', '-o',
                               help='Output file path (auto-generated if not provided)')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Check files exist
    if not Path(args.mutated).exists():
        print(f"Error: Mutated file not found - {args.mutated}")
        sys.exit(1)

    if not Path(args.reference).exists():
        print(f"Error: Reference file not found - {args.reference}")
        sys.exit(1)

    try:
        if args.command == 'analyze':
            stats = analyze_mutations(
                args.mutated,
                args.reference,
                add_mutation_field=args.add_mutations_field,
                output_file=args.output if hasattr(args, 'output') else None
            )
            print_analysis_results(stats)

        elif args.command == 'restore':
            stats = restore_packets(args.mutated, args.reference,
                                   args.num_restore, args.output)
            print_restore_results(stats)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()