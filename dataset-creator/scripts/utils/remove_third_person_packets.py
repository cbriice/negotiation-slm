#!/usr/bin/env python3
"""
Remove Third-Person Packets Script

Removes negotiation packets containing problematic third-person pronoun usage
where the model speaks about the other party in meta-commentary rather than
directly addressing them.

This script detects and removes packets containing patterns like:
- "They clearly don't understand..."
- "They must be out of their minds..."
- "They seem unwilling..."
- "they're being so difficult"
- Third-person references with "Goodbye/Bye"

Usage:
    python remove_third_person_packets.py --input data.jsonl --output cleaned.jsonl
    python remove_third_person_packets.py --dry-run --verbose
    python remove_third_person_packets.py --log-removed removed_packets.jsonl
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import Dict, List, Tuple, Optional


class PacketCleaner:
    """Removes packets with problematic third-person pronoun usage."""

    # Problematic patterns to detect
    PROBLEMATIC_PATTERNS = [
        "They clearly don't",
        "They clearly do not",
        "They must be out of their minds",
        "They must be out of your minds",
        "They seem",
        "they're being so difficult",
        "they're being difficult",
        "They're being so difficult",
        "They're being difficult",
        "They don't see",
        "They do not see",
        "They don't understand",
        "They do not understand",
        "they don't understand",
        "they do not understand",
        "They can't see",
        "They cannot see",
        "they can't see",
        "they cannot see",
    ]

    def __init__(self, verbose: bool = False):
        """Initialize the cleaner."""
        self.verbose = verbose
        self.stats = {
            'total_packets': 0,
            'removed_packets': 0,
            'kept_packets': 0,
            'pattern_counts': Counter(),
            'removed_lines': []
        }

    def has_problematic_pattern(self, packet: Dict, line_num: int) -> Optional[Tuple[str, str]]:
        """
        Check if packet contains problematic third-person patterns.

        Returns:
            Tuple of (pattern found, message excerpt) if problematic, None otherwise
        """
        if not self.validate_packet(packet):
            return None

        messages = packet.get('messages', [])

        for msg in messages:
            content = msg.get('content', '')

            # Check for direct problematic patterns
            for pattern in self.PROBLEMATIC_PATTERNS:
                if pattern in content:
                    excerpt = self._get_excerpt(content, pattern)
                    return (pattern, excerpt)

            # Special rule: "They " with "Goodbye" or "Bye"
            if ("They " in content and
                ("Goodbye" in content or "goodbye" in content or
                 "Bye" in content or "bye" in content)):
                excerpt = self._get_excerpt(content, "They ")
                return ("They + Goodbye/Bye", excerpt)

        return None

    def _get_excerpt(self, content: str, pattern: str, context_chars: int = 50) -> str:
        """Get an excerpt of the content around the pattern."""
        idx = content.find(pattern)
        if idx == -1:
            return content[:100] + "..." if len(content) > 100 else content

        start = max(0, idx - context_chars)
        end = min(len(content), idx + len(pattern) + context_chars)

        excerpt = content[start:end]
        if start > 0:
            excerpt = "..." + excerpt
        if end < len(content):
            excerpt = excerpt + "..."

        return excerpt.replace('\n', ' ')

    def validate_packet(self, packet: Dict) -> bool:
        """Check if packet has valid structure."""
        if not isinstance(packet, dict):
            return False

        # Check required fields
        if "messages" not in packet:
            return False

        # Check messages is a list
        if not isinstance(packet["messages"], list):
            return False

        return True

    def process_file(self, input_path: str, output_path: Optional[str] = None,
                    dry_run: bool = False, log_removed_path: Optional[str] = None) -> None:
        """Process the input file and remove problematic packets."""

        # Generate output filename if not provided
        if output_path is None and not dry_run:
            input_file = Path(input_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = input_file.parent / f"{input_file.stem}_cleaned_{timestamp}.jsonl"

        print("=" * 60)
        print("THIRD-PERSON PACKET REMOVAL")
        print("=" * 60)
        print(f"Input file: {input_path}")
        if not dry_run:
            print(f"Output file: {output_path}")
            if log_removed_path:
                print(f"Removed packets log: {log_removed_path}")
        else:
            print("Mode: DRY RUN (no files will be written)")
        print("=" * 60)

        # Check input file exists
        if not os.path.exists(input_path):
            print(f"Error: Input file not found: {input_path}")
            sys.exit(1)

        # Process packets
        removed_packets = []
        kept_packets = []

        with open(input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue

                try:
                    packet = json.loads(line)
                    self.stats['total_packets'] += 1

                    # Check for problematic patterns
                    problem = self.has_problematic_pattern(packet, line_num)

                    if problem:
                        pattern, excerpt = problem
                        self.stats['removed_packets'] += 1
                        self.stats['pattern_counts'][pattern] += 1
                        self.stats['removed_lines'].append(line_num)
                        removed_packets.append(packet)

                        if self.verbose:
                            print(f"\nLine {line_num}: REMOVING")
                            print(f"  Pattern: {pattern}")
                            print(f"  Excerpt: {excerpt}")
                    else:
                        self.stats['kept_packets'] += 1
                        kept_packets.append(packet)

                        if self.verbose and line_num % 1000 == 0:
                            print(f"  Line {line_num}: keeping")

                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping line {line_num} - Invalid JSON: {e}")
                    continue
                except Exception as e:
                    print(f"Warning: Error processing line {line_num}: {e}")
                    continue

        # Write output files if not dry run
        if not dry_run:
            # Write cleaned data
            with open(output_path, 'w', encoding='utf-8') as f:
                for packet in kept_packets:
                    f.write(json.dumps(packet, ensure_ascii=False) + '\n')

            # Write removed packets if requested
            if log_removed_path and removed_packets:
                with open(log_removed_path, 'w', encoding='utf-8') as f:
                    for packet in removed_packets:
                        f.write(json.dumps(packet, ensure_ascii=False) + '\n')

        # Print statistics
        self.print_statistics()

        if not dry_run:
            print(f"\nCleaned data written to: {output_path}")
            if log_removed_path and removed_packets:
                print(f"Removed packets logged to: {log_removed_path}")

    def print_statistics(self) -> None:
        """Print removal statistics."""
        print("\n" + "=" * 60)
        print("REMOVAL STATISTICS")
        print("=" * 60)

        total = self.stats['total_packets']
        removed = self.stats['removed_packets']
        kept = self.stats['kept_packets']

        print(f"Total packets processed: {total:,}")
        print(f"Packets removed: {removed:,} ({removed/total*100:.2f}%)")
        print(f"Packets kept: {kept:,} ({kept/total*100:.2f}%)")

        if self.stats['pattern_counts']:
            print("\nRemoval reasons:")
            for pattern, count in sorted(self.stats['pattern_counts'].items(),
                                        key=lambda x: x[1], reverse=True):
                print(f"  {pattern}: {count}")

        if self.verbose and self.stats['removed_lines']:
            print(f"\nRemoved line numbers (first 20):")
            for i, line_num in enumerate(self.stats['removed_lines'][:20]):
                if i % 10 == 0 and i > 0:
                    print()
                print(f"{line_num:6d}", end="  ")
            print()
            if len(self.stats['removed_lines']) > 20:
                print(f"  ... and {len(self.stats['removed_lines']) - 20} more")

        print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Remove negotiation packets with problematic third-person pronoun usage.',
        epilog="""
Examples:
  # Dry run to preview what would be removed
  python remove_third_person_packets.py --dry-run --verbose

  # Remove packets and save to new file
  python remove_third_person_packets.py --input data.jsonl --output cleaned.jsonl

  # Remove packets and log removed ones for review
  python remove_third_person_packets.py --log-removed removed.jsonl
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--input',
        type=str,
        default='data/runs/bulk_data/real_training_data/all_negotiations.jsonl',
        help='Input JSONL file (default: data/runs/bulk_data/real_training_data/all_negotiations.jsonl)'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Output file path (default: auto-generated with timestamp)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview what would be removed without writing files'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed information about each removed packet'
    )

    parser.add_argument(
        '--log-removed',
        type=str,
        help='Save removed packets to this file for review'
    )

    args = parser.parse_args()

    # Create cleaner and process
    cleaner = PacketCleaner(verbose=args.verbose)
    cleaner.process_file(
        input_path=args.input,
        output_path=args.output,
        dry_run=args.dry_run,
        log_removed_path=args.log_removed
    )


if __name__ == "__main__":
    main()