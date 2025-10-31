#!/usr/bin/env python3
"""
Optimized Claude API script to label negotiation key strategies in SFT packets.
Batches multiple negotiations per API call for efficiency.
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
import random

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

# Batch labeling prompt template
BATCH_LABELING_PROMPT = """
You are a negotiation strategy expert labeling negotiations based on the strategy used by the ASSISTANT. 
IMPORTANT: These negotiations contain a diverse range of tactics. Look for SPECIFIC markers in negotiation that match the following tags, not just general tone. Pick the tag that could be paired with the dialog as a label for someone learning negotiation tactics.

Strategy identification tags:
- anchor: assistant stands firm at one price and repeats it multiple times, won't budge, "final offer"
- compromise: assistant acknowledges both positions and works with counterparty - "meet halfway", etc.
- market_logic: assistant cites SPECIFIC external prices/data, compares to other items in the market, "market value"
- connection_appeal: assistant offers connections to friends/others as incentive
- empathy_appeal: assistant uses personal hardship story, "I really need", sympathy play
- aggro: assistant aggressively attacks counterparty. Insults, "ridiculous", "crazy", righteousness
- walk_away: assistant QUICKLY and explicitly exits upon realizing no favorable deal. short negotiation
- flatter: assistant compliments/flatters counterparty to try and get a favorable deal

Avoid defaulting to common labels. Find what makes each negotiation unique.

CRITICAL RESPONSE GUIDELINES: For each negotiation below, respond with ONLY the negotiation ID and its tag in format:
ID: tag

Example response format:
1001: compromise
2035: anchor
3197: market_logic

Now label these negotiations:

"""


class OptimizedClaudeStrategyLabeler:
    """Handles batched Claude API calls for strategy labeling."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, batch_size: int = 10, temperature: float = 0.4):
        """
        Initialize Claude client.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Claude model to use (defaults to env var or Haiku)
            batch_size: Number of negotiations to process per API call
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("No API key provided. Set ANTHROPIC_API_KEY environment variable or pass api_key parameter.")

        self.client = anthropic.Anthropic(api_key=self.api_key)
        # Use provided model, env var, or default to Haiku for cost efficiency
        self.model = model or os.getenv("CLAUDE_MODEL") or "claude-3-haiku-20240307"
        print(f"Using model: {self.model}")
        print(f"Batch size: {batch_size} negotiations per API call")

        self.batch_size = batch_size
        self.temp = temperature
        self.stats = {
            "total": 0,
            "successful": 0,
            "failed": 0,
            "skipped": 0,
            "api_calls": 0
        }

    def format_negotiation(self, messages: List[Dict[str, str]], truncate: bool = True) -> str:
        """
        Format messages into readable negotiation text.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            truncate: Whether to truncate long negotiations

        Returns:
            Formatted negotiation string
        """
        formatted = []
        total_chars = 0
        max_chars = 3000  # Limit per negotiation to keep prompt reasonable

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            # Skip system messages
            if role == "system":
                continue

            # Format as "ROLE: content"
            role_label = role.upper()
            line = f"{role_label}: {content}"

            # Truncate if needed
            if truncate and total_chars + len(line) > max_chars:
                if total_chars == 0:
                    # At least include part of first message
                    formatted.append(line[:max_chars] + "... [truncated]")
                else:
                    formatted.append("[remaining messages truncated for length]")
                break

            formatted.append(line)
            total_chars += len(line)

        return "\n".join(formatted)

    def get_batch_labels(self, negotiations_batch: List[Dict[str, Any]], retry_count: int = 3) -> Dict[str, str]:
        """
        Get strategy labels for a batch of negotiations.

        Args:
            negotiations_batch: List of dicts with 'id' and 'messages'
            retry_count: Number of retries on failure

        Returns:
            Dictionary mapping negotiation IDs to strategy tags
        """
        # Build the batch prompt
        prompt_parts = [BATCH_LABELING_PROMPT]

        for neg in negotiations_batch:
            neg_id = neg['id']
            messages = neg['messages']
            formatted = self.format_negotiation(messages)

            prompt_parts.append(f"=== NEGOTIATION {neg_id} ===")
            prompt_parts.append(formatted)
            prompt_parts.append("")

        full_prompt = "\n".join(prompt_parts)

        for attempt in range(retry_count):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=100 * len(negotiations_batch),  # Enough for all labels
                    temperature=self.temp, 
                    messages=[
                        {"role": "user", "content": full_prompt}
                    ]
                )

                self.stats["api_calls"] += 1

                # Parse the response
                try:
                    content_block = response.content[0]
                    response_text = getattr(content_block, 'text', '').strip()

                    # Parse each line for ID: tag format
                    labels = {}
                    for line in response_text.split('\n'):
                        line = line.strip()
                        if ':' in line:
                            parts = line.split(':', 1)
                            if len(parts) == 2:
                                neg_id = parts[0].strip()
                                tag = parts[1].strip().lower()

                                # Validate tag
                                if tag in STRATEGY_TAGS:
                                    labels[neg_id] = tag
                                else:
                                    print(f"Warning: Invalid tag '{tag}' for negotiation {neg_id}")

                    # Check we got all labels
                    expected_ids = {str(neg['id']) for neg in negotiations_batch}
                    got_ids = set(labels.keys())

                    if expected_ids != got_ids:
                        missing = expected_ids - got_ids
                        print(f"Warning: Missing labels for negotiations: {missing}")
                        if len(labels) > 0:
                            # Partial success - return what we have
                            return labels
                    else:
                        return labels

                except (IndexError, AttributeError) as e:
                    print(f"Warning: Failed to parse response: {e}")
                    continue

            except Exception as e:
                print(f"API call failed (attempt {attempt + 1}/{retry_count}): {e}")
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff

        return {}

    def process_file(self, input_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None):
        """
        Process entire JSONL file with batching.

        Args:
            input_path: Path to input JSONL file
            output_path: Path to output file (defaults to input_labeled.jsonl)
        """
        input_path = Path(input_path) if isinstance(input_path, str) else input_path
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Default output path
        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_labeled_optimized.jsonl"
        else:
            output_path = Path(output_path) if isinstance(output_path, str) else output_path

        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load all packets first
        packets = []
        with open(input_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    packet = json.loads(line.strip())
                    packets.append(packet)
                except json.JSONDecodeError:
                    print(f"Warning: Invalid JSON at line {line_num}")

        print(f"Loaded {len(packets)} packets from {input_path}")
        print(f"Output will be saved to {output_path}")

        # Separate packets that need labeling
        to_label = []
        already_labeled = []

        for idx, packet in enumerate(packets):
            self.stats["total"] += 1

            # Check if already labeled
            if packet.get("system", {}).get("key_strategy"):
                already_labeled.append(packet)
                self.stats["skipped"] += 1
            else:
                # Add to batch with ID for tracking
                messages = packet.get("messages", [])
                if messages:
                    to_label.append({
                        'id': idx,
                        'messages': messages,
                        'packet': packet
                    })
                else:
                    print(f"Warning: Packet {idx} has no messages")
                    self.stats["failed"] += 1

        print(f"Need to label: {len(to_label)} packets")
        print(f"Already labeled: {len(already_labeled)} packets")

        # Process in batches
        labeled_packets = already_labeled.copy()

        with tqdm(total=len(to_label), desc="Labeling packets") as pbar:
            for i in range(0, len(to_label), self.batch_size):
                batch = to_label[i:i + self.batch_size]
                # shuffle batch
                random.shuffle(batch)

                # Get labels for this batch
                labels = self.get_batch_labels(batch)

                # Update packets with labels
                for item in batch:
                    idx = str(item['id'])
                    packet = item['packet']

                    if idx in labels:
                        # Update the packet
                        if "system" not in packet:
                            packet["system"] = {}
                        packet["system"]["key_strategy"] = labels[idx]
                        self.stats["successful"] += 1
                    else:
                        self.stats["failed"] += 1
                        print(f"Warning: Failed to get label for packet {idx}")

                    labeled_packets.append(packet)
                    pbar.update(1)

                # Rate limiting between batches
                if i + self.batch_size < len(to_label):
                    time.sleep(1.0)  # Adjust based on your API rate limits

        # Sort packets by original order (if indices were preserved)
        # This ensures output order matches input order

        # Save all packets
        with open(output_path, 'w') as f:
            for packet in packets:
                # Find the updated version if it was labeled
                for labeled in labeled_packets:
                    if labeled.get("messages") == packet.get("messages"):
                        packet = labeled
                        break
                f.write(json.dumps(packet) + '\n')

        # Print statistics
        self.print_stats()
        print(f"Output saved to {output_path}")

    def print_stats(self):
        """Print processing statistics."""
        print("\n=== Processing Statistics ===")
        print(f"Total packets: {self.stats['total']}")
        print(f"Successfully labeled: {self.stats['successful']}")
        print(f"Failed to label: {self.stats['failed']}")
        print(f"Skipped (already labeled): {self.stats['skipped']}")
        print(f"Total API calls made: {self.stats['api_calls']}")

        if self.stats['total'] > 0:
            success_rate = (self.stats['successful'] / self.stats['total']) * 100
            print(f"Success rate: {success_rate:.1f}%")

        if self.stats['api_calls'] > 0:
            avg_per_call = self.stats['successful'] / self.stats['api_calls']
            print(f"Average negotiations per API call: {avg_per_call:.1f}")

            # Calculate efficiency improvement
            old_api_calls = self.stats['successful']  # Old version: 1 call per negotiation
            efficiency_gain = ((old_api_calls - self.stats['api_calls']) / old_api_calls) * 100
            print(f"API call reduction: {efficiency_gain:.1f}% fewer calls")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Optimized batch labeling of negotiation strategies using Claude API"
    )
    parser.add_argument(
        "input_file",
        help="Path to input JSONL file containing SFT packets"
    )
    parser.add_argument(
        "-o", "--output",
        help="Path to output file (defaults to input_labeled_optimized.jsonl)"
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
        help="Number of negotiations to process per API call (default: 10)"
    )

    args = parser.parse_args()

    try:
        # Initialize labeler
        labeler = OptimizedClaudeStrategyLabeler(
            api_key=args.api_key,
            model=args.model,
            batch_size=args.batch_size
        )

        # Process file
        labeler.process_file(
            args.input_file,
            args.output
        )

    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()