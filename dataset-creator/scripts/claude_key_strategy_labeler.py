#!/usr/bin/env python3
"""
Claude API script to label negotiation key strategies in SFT packets.
Processes JSONL files containing negotiation data and adds strategy tags.
"""

import json
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import time
from tqdm import tqdm
import anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Strategy tags as defined in schema
STRATEGY_TAGS = {
    "anchor",
    "compromise",
    "market_logic",
    "connection_appeal",
    "empathy_appeal",
    "aggro",
    "concede",
    "walk_away",
    "flatter"
}

# Claude API prompt template
LABELING_PROMPT = """You are a negotiation labeler labeling negotiations based on the tactic **MOST used** by the ASSISTANT during the negotiation.
You can choose from ONE of these tags:
  - **anchor** - assistant stands firm at one price and doesn't waver for anything, whether or not the deal is closed
  - **compromise** - assistant understands counterparty's position and works towards a compromise (problem-solving)
  - **market_logic** - assistant uses market context to persuade the counterparty
  - **connection_appeal** - assistant offers connections to friends/others as incentive for their offer
  - **empathy_appeal** - assistant appeals to empathy of counterparty, using personal situation as leverage
  - **aggro** - assistant aggressively attacks counterparty to get a favorable outcome
  - **concede** - assistant concedes to counterparty in order to close deal
  - **walk_away** - assistant quickly exits negotiation upon realizing a favorable deal can't be reached
  - **flatter** - assistant flatters the counterparty to get a more favorable deal
You must output ONE unformatted tag and nothing else.
Negotiation:
"""


class ClaudeStrategyLabeler:
    """Handles Claude API calls for strategy labeling."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize Claude client.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Claude model to use (defaults to env var or Haiku for cost efficiency)
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("No API key provided. Set ANTHROPIC_API_KEY environment variable or pass api_key parameter.")

        self.client = anthropic.Anthropic(api_key=self.api_key)
        # Use provided model, env var, or default to Haiku for cost efficiency
        self.model = model or os.getenv("CLAUDE_MODEL") or "claude-3-haiku-20240307"
        print(f"Using model: {self.model}")
        self.stats = {
            "total": 0,
            "successful": 0,
            "failed": 0,
            "skipped": 0
        }

    def format_negotiation(self, messages: List[Dict[str, str]]) -> str:
        """
        Format messages into readable negotiation text.

        Args:
            messages: List of message dictionaries with 'role' and 'content'

        Returns:
            Formatted negotiation string
        """
        formatted = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            # Skip system messages in the negotiation display
            if role == "system":
                continue

            # Format as "ROLE: content"
            role_label = role.upper()
            formatted.append(f"{role_label}: {content}")

        return "\n\n".join(formatted)

    def get_strategy_label(self, messages: List[Dict[str, str]], retry_count: int = 3) -> str:
        """
        Get strategy label from Claude API.

        Args:
            messages: Negotiation messages
            retry_count: Number of retries on failure

        Returns:
            Strategy tag or empty string on failure
        """
        negotiation_text = self.format_negotiation(messages)
        full_prompt = LABELING_PROMPT + "\n" + negotiation_text

        for attempt in range(retry_count):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=10,  # We only need one word
                    temperature=0,  # Deterministic for consistency
                    messages=[
                        {"role": "user", "content": full_prompt}
                    ]
                )

                # Extract and clean the response
                # The response content is a list of content blocks
                # For text responses, we get the text from the first block
                try:
                    # Get the text from the first content block
                    content_block = response.content[0]
                    # Access text attribute (for TextBlock type)
                    label = getattr(content_block, 'text', '').strip().lower()
                    if not label:
                        print(f"Warning: Empty response received")
                        continue
                except (IndexError, AttributeError) as e:
                    print(f"Warning: Unexpected response format: {e}")
                    continue

                # Validate the label
                if label in STRATEGY_TAGS:
                    return label
                else:
                    print(f"Warning: Invalid label '{label}' received. Retrying...")

            except Exception as e:
                print(f"API call failed (attempt {attempt + 1}/{retry_count}): {e}")
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff

        return ""

    def process_packet(self, packet: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single SFT packet.

        Args:
            packet: SFT packet dictionary

        Returns:
            Updated packet with key_strategy filled
        """
        # Check if key_strategy already exists and is non-empty
        if packet.get("system", {}).get("key_strategy"):
            self.stats["skipped"] += 1
            return packet

        # Get messages
        messages = packet.get("messages", [])
        if not messages:
            print("Warning: Packet has no messages")
            self.stats["failed"] += 1
            return packet

        # Get strategy label
        strategy = self.get_strategy_label(messages)

        if strategy:
            # Update the packet
            if "system" not in packet:
                packet["system"] = {}
            packet["system"]["key_strategy"] = strategy
            self.stats["successful"] += 1
        else:
            self.stats["failed"] += 1
            print("Warning: Failed to get valid strategy label")

        return packet

    def process_file(self, input_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None, batch_size: int = 10):
        """
        Process entire JSONL file.

        Args:
            input_path: Path to input JSONL file
            output_path: Path to output file (defaults to input_labeled.jsonl)
            batch_size: Number of packets to process before saving
        """
        input_path = Path(input_path) if isinstance(input_path, str) else input_path
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Default output path
        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_labeled.jsonl"
        else:
            output_path = Path(output_path) if isinstance(output_path, str) else output_path

        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Count total lines for progress bar
        with open(input_path, 'r') as f:
            total_lines = sum(1 for _ in f)

        print(f"Processing {total_lines} packets from {input_path}")
        print(f"Output will be saved to {output_path}")

        # Process packets
        processed_packets = []

        with open(input_path, 'r') as infile:
            with tqdm(total=total_lines, desc="Labeling packets") as pbar:
                for line_num, line in enumerate(infile, 1):
                    try:
                        packet = json.loads(line.strip())
                        self.stats["total"] += 1

                        # Process the packet
                        updated_packet = self.process_packet(packet)
                        processed_packets.append(updated_packet)

                        # Save periodically
                        if len(processed_packets) >= batch_size:
                            self._save_batch(output_path, processed_packets, append=line_num > batch_size)
                            processed_packets = []

                        pbar.update(1)

                        # Rate limiting
                        time.sleep(0.5)  # Adjust based on your API rate limits

                    except json.JSONDecodeError:
                        print(f"Warning: Invalid JSON at line {line_num}")
                        self.stats["failed"] += 1
                    except Exception as e:
                        print(f"Error processing line {line_num}: {e}")
                        self.stats["failed"] += 1

        # Save remaining packets
        if processed_packets:
            self._save_batch(output_path, processed_packets, append=True)

        # Print statistics
        self.print_stats()
        print(f"Output saved to {output_path}")

    def _save_batch(self, output_path: Path, packets: List[Dict], append: bool = False):
        """Save a batch of packets to file."""
        mode = 'a' if append else 'w'
        with open(output_path, mode) as f:
            for packet in packets:
                f.write(json.dumps(packet) + '\n')

    def print_stats(self):
        """Print processing statistics."""
        print("\n=== Processing Statistics ===")
        print(f"Total packets: {self.stats['total']}")
        print(f"Successfully labeled: {self.stats['successful']}")
        print(f"Failed to label: {self.stats['failed']}")
        print(f"Skipped (already labeled): {self.stats['skipped']}")
        if self.stats['total'] > 0:
            success_rate = (self.stats['successful'] / self.stats['total']) * 100
            print(f"Success rate: {success_rate:.1f}%")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Label negotiation strategies in SFT packets using Claude API"
    )
    parser.add_argument(
        "input_file",
        help="Path to input JSONL file containing SFT packets"
    )
    parser.add_argument(
        "-o", "--output",
        help="Path to output file (defaults to input_labeled.jsonl)"
    )
    parser.add_argument(
        "-m", "--model",
        help="Claude model to use (e.g., claude-3-5-sonnet-20241022, claude-3-haiku-20240307)"
    )
    parser.add_argument(
        "-k", "--api-key",
        help="Anthropic API key (defaults to ANTHROPIC_API_KEY env var)"
    )
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=10,
        help="Number of packets to process before saving (default: 10)"
    )

    args = parser.parse_args()

    try:
        # Initialize labeler
        labeler = ClaudeStrategyLabeler(
            api_key=args.api_key,
            model=args.model
        )

        # Process file
        labeler.process_file(
            args.input_file,
            args.output,
            args.batch_size
        )

    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()