"""
SFT Dataset Converter Module.

Converts existing JSONL simulation datasets to the SFT training packet format.
Creates two SFT packets per negotiation (one with buyer as assistant, one with seller as assistant).
"""

import json
import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import sys

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompts.personality_deck import (
    PersonalityDeck, StraightToPoint, Desperate, Skeptic, Bluffer,
    Confrontational, Haggler, WeakEmotional, Rationalizer, StoneWall,
    Opportunist, HighHorse, Stubborn
)


@dataclass
class SFTPacket:
    """Represents a single SFT training packet."""
    messages: List[Dict[str, str]]
    system: Dict[str, any]
    meta: Dict[str, any]


class SFTConverter:
    """Converts negotiation datasets to SFT training format."""

    def __init__(self, seed_dir: str = "seeds/"):
        """Initialize converter with personality mappings.

        Args:
            seed_dir: Directory containing seed files (default: seeds/)
        """
        # Map personality class names to their tags from personality_deck.py
        self.personality_tags = {
            "StraightToPoint": StraightToPoint.tag,
            "Desperate": Desperate.tag,
            "Skeptic": Skeptic.tag,
            "Bluffer": Bluffer.tag,
            "Confrontational": Confrontational.tag,
            "Haggler": Haggler.tag,
            "WeakEmotional": WeakEmotional.tag,
            "Rationalizer": Rationalizer.tag,
            "StoneWall": StoneWall.tag,
            "Opportunist": Opportunist.tag,
            "HighHorse": HighHorse.tag,
            "Stubborn": Stubborn.tag
        }

        # Initialize personality deck for trait access
        self.personality_deck = PersonalityDeck()

        # Store seed directory for lookups
        self.seed_dir = seed_dir

        # Map seed ID prefixes to seed files
        self.seed_file_map = {
            "1": "tech_seeds.json",
            "2": "vehicle_seeds.json",
            "3": "service_seeds.json",
            "4": "travel_seeds.json"
        }

        # Cache for loaded seed data
        self.seed_cache = {}

    def get_seed_prices(self, seed_id: int) -> Dict[str, float]:
        """
        Look up seed price data from the appropriate seed file.

        Args:
            seed_id: The seed ID to look up

        Returns:
            Dictionary with buyer_max, seller_min, and seller_list prices
        """
        # Determine which file to look in based on seed ID prefix
        seed_id_str = str(seed_id)
        if not seed_id_str:
            return {}

        prefix = seed_id_str[0]
        seed_file = self.seed_file_map.get(prefix)

        if not seed_file:
            print(f"Warning: Unknown seed ID prefix {prefix} for seed {seed_id}")
            return {}

        # Check cache first
        if seed_file not in self.seed_cache:
            # Load the seed file
            seed_path = os.path.join(self.seed_dir, seed_file)
            if not os.path.exists(seed_path):
                print(f"Warning: Seed file not found: {seed_path}")
                return {}

            try:
                with open(seed_path, 'r', encoding='utf-8') as f:
                    self.seed_cache[seed_file] = json.load(f)
            except Exception as e:
                print(f"Error loading seed file {seed_path}: {e}")
                return {}

        # Find the specific seed
        seeds = self.seed_cache.get(seed_file, [])
        for seed in seeds:
            if seed.get("seed_id") == seed_id:
                return {
                    "buyer_max": seed.get("buyer_max", 0),
                    "seller_min": seed.get("seller_min", 0),
                    "seller_list": seed.get("seller_list", 0)
                }

        print(f"Warning: Seed {seed_id} not found in {seed_file}")
        return {}

    def calculate_difficulty(self, negotiation: Dict) -> float:
        """
        Calculate negotiation difficulty based on price differences.

        High difficulty: buyer max much lower than seller min
        Low difficulty: buyer max over seller min

        Args:
            negotiation: Negotiation data dictionary

        Returns:
            Difficulty score between 0.0 and 1.0
        """
        # Look up price data from seed files using seed_id
        seed_id = negotiation.get("seed_id")
        if not seed_id:
            return 0.5  # Default if no seed_id available

        seed_prices = self.get_seed_prices(seed_id)
        buyer_max = seed_prices.get("buyer_max", 0)
        seller_min = seed_prices.get("seller_min", 0)
        seller_list = seed_prices.get("seller_list", seller_min)

        if buyer_max == 0 or seller_min == 0:
            return 0.5  # Default if prices not available

        # Calculate overlap and difficulty
        if buyer_max >= seller_min:
            # There's price overlap - easier negotiation
            overlap = buyer_max - seller_min
            range_size = seller_list - seller_min if seller_list > seller_min else 1
            # More overlap = lower difficulty
            difficulty = max(0.0, 0.5 - (overlap / range_size) * 0.5)
        else:
            # No overlap - harder negotiation
            gap = seller_min - buyer_max
            range_size = seller_list - seller_min if seller_list > seller_min else 1
            # Larger gap = higher difficulty
            difficulty = min(1.0, 0.5 + (gap / range_size) * 0.5)

        return round(difficulty, 2)

    def convert_dialog_to_messages(self, dialog: List[Tuple[str, str]],
                                  assistant_side: str) -> List[Dict[str, str]]:
        """
        Convert dialog to SFT message format.

        Args:
            dialog: List of (speaker, message) tuples
            assistant_side: "BUYER" or "SELLER" - who is the assistant

        Returns:
            List of message dictionaries with roles
        """
        messages = [
            {"role": "system", "content": "You are an expert negotiator."}
        ]

        for speaker, message in dialog:
            if speaker == assistant_side:
                role = "assistant"
            else:
                role = "user"

            messages.append({
                "role": role,
                "content": message
            })

        return messages

    def create_sft_packets(self, negotiation: Dict) -> Tuple[Dict, Dict]:
        """
        Create two SFT packets from a negotiation (buyer and seller as assistant).

        Args:
            negotiation: Negotiation data dictionary

        Returns:
            Tuple of (buyer_packet, seller_packet)
        """
        # Extract basic information
        dialog = negotiation.get("dialog", [])
        outcome = negotiation.get("outcome", "unknown")
        final_price = negotiation.get("final_price")
        deal_closed = negotiation.get("deal_closed", False)

        # Get personality tags
        buyer_personality = negotiation.get("buyer_personality", "unknown")
        seller_personality = negotiation.get("seller_personality", "unknown")

        buyer_tag = self.personality_tags.get(buyer_personality, "balanced")
        seller_tag = self.personality_tags.get(seller_personality, "balanced")

        # Calculate difficulty
        difficulty = self.calculate_difficulty(negotiation)

        # Determine favorable outcomes
        buyer_favorable = negotiation.get("buyer_favorable", False)
        seller_favorable = negotiation.get("seller_favorable", False)

        favorable_outcome_buyer = "favorable" if buyer_favorable else ("unfavorable" if seller_favorable else "neutral")
        favorable_outcome_seller = "favorable" if seller_favorable else ("unfavorable" if buyer_favorable else "neutral")

        # Get number of rounds
        num_rounds = len(dialog)

        # Create buyer-as-assistant packet
        buyer_packet = {
            "messages": self.convert_dialog_to_messages(dialog, "BUYER"),
            "system": {
                "assistant_style": buyer_tag,
                "difficulty": difficulty,
                "key_strategy": ""  # To be filled by separate module
            },
            "meta": {
                "opponent_style": seller_tag,
                "favorable_outcome": favorable_outcome_buyer,
                "deal_closed": str(deal_closed).lower(),
                "rounds": num_rounds
            }
        }

        # Create seller-as-assistant packet
        seller_packet = {
            "messages": self.convert_dialog_to_messages(dialog, "SELLER"),
            "system": {
                "assistant_style": seller_tag,
                "difficulty": difficulty,
                "key_strategy": ""  # To be filled by separate module
            },
            "meta": {
                "opponent_style": buyer_tag,
                "favorable_outcome": favorable_outcome_seller,
                "deal_closed": str(deal_closed).lower(),
                "rounds": num_rounds
            }
        }

        return buyer_packet, seller_packet

    def convert_file(self, input_file: str, output_file: str,
                    include_no_deals: bool = True) -> int:
        """
        Convert a JSONL file of negotiations to SFT format.

        Args:
            input_file: Path to input JSONL file
            output_file: Path to output JSONL file
            include_no_deals: Whether to include negotiations that didn't close

        Returns:
            Number of packets written
        """
        packets_written = 0

        # Create output directory if needed
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        with open(input_file, 'r', encoding='utf-8') as infile:
            with open(output_file, 'w', encoding='utf-8') as outfile:
                for line in infile:
                    try:
                        negotiation = json.loads(line.strip())

                        # Skip no-deals if requested
                        if not include_no_deals and not negotiation.get("deal_closed", False):
                            continue

                        # Create two packets
                        buyer_packet, seller_packet = self.create_sft_packets(negotiation)

                        # Write packets as JSONL
                        outfile.write(json.dumps(buyer_packet, ensure_ascii=False) + '\n')
                        outfile.write(json.dumps(seller_packet, ensure_ascii=False) + '\n')
                        packets_written += 2

                    except json.JSONDecodeError as e:
                        print(f"Error parsing line: {e}")
                        continue
                    except Exception as e:
                        print(f"Error converting negotiation: {e}")
                        continue

        print(f"Converted {packets_written // 2} negotiations to {packets_written} SFT packets")
        print(f"Output saved to: {output_file}")

        return packets_written

    def convert_run_directory(self, run_dir: str, output_file: Optional[str] = None,
                            include_no_deals: bool = True) -> int:
        """
        Convert all dataset files in a run directory.

        Args:
            run_dir: Path to run directory
            output_file: Optional output file path (default: same dir with _sft suffix)
            include_no_deals: Whether to include negotiations that didn't close

        Returns:
            Number of packets written
        """
        # Find dataset files
        dataset_files = list(Path(run_dir).glob("dataset_*.jsonl"))

        if not dataset_files:
            print(f"No dataset files found in {run_dir}")
            return 0

        total_packets = 0

        for dataset_file in dataset_files:
            # Determine output file name
            if output_file is None:
                output_name = dataset_file.stem + "_sft.jsonl"
                output_path = dataset_file.parent / output_name
            else:
                output_path = Path(output_file)

            # Convert the file
            packets = self.convert_file(
                str(dataset_file),
                str(output_path),
                include_no_deals=include_no_deals
            )
            total_packets += packets

        return total_packets

    def convert_multiple_runs(self, run_dirs: List[str],
                            output_file: str,
                            include_no_deals: bool = True) -> int:
        """
        Combine and convert multiple run directories into single SFT dataset.

        Args:
            run_dirs: List of run directory paths
            output_file: Output file path for combined SFT dataset
            include_no_deals: Whether to include negotiations that didn't close

        Returns:
            Total number of packets written
        """
        all_packets = []

        for run_dir in run_dirs:
            dataset_files = list(Path(run_dir).glob("dataset_*.jsonl"))

            for dataset_file in dataset_files:
                with open(dataset_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            negotiation = json.loads(line.strip())

                            # Skip no-deals if requested
                            if not include_no_deals and not negotiation.get("deal_closed", False):
                                continue

                            # Create two packets
                            buyer_packet, seller_packet = self.create_sft_packets(negotiation)
                            all_packets.extend([buyer_packet, seller_packet])

                        except Exception as e:
                            print(f"Error processing negotiation: {e}")
                            continue

        # Write all packets to output file
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            for packet in all_packets:
                f.write(json.dumps(packet, ensure_ascii=False) + '\n')

        print(f"Converted {len(all_packets) // 2} negotiations to {len(all_packets)} SFT packets")
        print(f"Combined output saved to: {output_file}")

        return len(all_packets)


def main():
    """Command-line interface for SFT converter."""
    import argparse

    parser = argparse.ArgumentParser(description="Convert negotiation datasets to SFT format")
    parser.add_argument("input", help="Input file, directory, or list of directories")
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument("--exclude-no-deals", action="store_true",
                       help="Exclude negotiations that didn't close")
    parser.add_argument("--combine", action="store_true",
                       help="Combine multiple runs into single output file")
    parser.add_argument("--seed-dir", default="seeds/",
                       help="Directory containing seed files (default: seeds/)")

    args = parser.parse_args()

    converter = SFTConverter(seed_dir=args.seed_dir)

    # Determine input type and convert accordingly
    input_path = Path(args.input)

    if args.combine:
        # Expecting comma-separated list of directories
        run_dirs = args.input.split(',')
        output_file = args.output or "combined_sft_dataset.jsonl"
        converter.convert_multiple_runs(
            run_dirs,
            output_file,
            include_no_deals=not args.exclude_no_deals
        )
    elif input_path.is_file():
        # Single file conversion
        output_file = args.output or str(input_path.with_suffix('')) + "_sft.jsonl"
        converter.convert_file(
            str(input_path),
            output_file,
            include_no_deals=not args.exclude_no_deals
        )
    elif input_path.is_dir():
        # Directory conversion
        converter.convert_run_directory(
            str(input_path),
            args.output,
            include_no_deals=not args.exclude_no_deals
        )
    else:
        print(f"Error: {args.input} is not a valid file or directory")
        return 1

    return 0


if __name__ == "__main__":
    main()