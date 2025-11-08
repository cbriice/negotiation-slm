#!/usr/bin/env python3
"""Script to remove emojis from text in JSONL files."""

import json
import re
import argparse
from pathlib import Path


def remove_emojis(text):
    """Remove all emojis and emoticons from text."""
    if not isinstance(text, str):
        return text

    # Comprehensive emoji removal using multiple patterns
    # Pattern 1: Unicode emoji blocks
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons (including 😊)
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002600-\U000027BF"  # Miscellaneous symbols
        "\U0001F000-\U0001F02F"  # Mahjong/Domino tiles
        "\U0001F0A0-\U0001F0FF"  # Playing cards
        "\U0001F100-\U0001F1FF"  # Enclosed chars
        "\U00002700-\U000027BF"  # Dingbats
        "\U0001F650-\U0001F67F"  # Ornamental Dingbats
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U00002000-\U0000206F"  # General Punctuation
        "\U00002300-\U000023FF"  # Miscellaneous Technical
        "\U00002460-\U000024FF"  # Enclosed Alphanumerics
        "\U000025A0-\U000025FF"  # Geometric Shapes
        "\U00002B00-\U00002BFF"  # Miscellaneous Symbols and Arrows
        "\U0000FE00-\U0000FE0F"  # Variation Selectors
        "\U0001F200-\U0001F2FF"  # Enclosed Ideographic Supplement
        "\u2639-\u263B"  # Various smileys (☹️☺️☻)
        "]+",
        flags=re.UNICODE
    )

    # Remove emojis
    text = emoji_pattern.sub('', text)

    # Additional pattern for emoji modifiers and zero-width joiners
    text = re.sub(r'[\u200d\ufe0f\u203c\u2049\u20e3]+', '', text)

    # Remove any remaining non-ASCII characters that might be emojis
    # This is more aggressive but ensures all emojis are removed
    text = ''.join(char for char in text if ord(char) < 0x1F000 or ord(char) > 0x1FFFF)

    return text.strip()


def clean_json_data(data):
    """Recursively clean emojis from JSON data of any structure."""
    if isinstance(data, str):
        return remove_emojis(data)
    elif isinstance(data, list):
        return [clean_json_data(item) for item in data]
    elif isinstance(data, dict):
        return {key: clean_json_data(value) for key, value in data.items()}
    else:
        return data


def process_dialog_entry(dialog_entry):
    """Process a single dialog entry to remove emojis."""
    if isinstance(dialog_entry, list) and len(dialog_entry) >= 2:
        # Dialog entries are typically ["BUYER"/"SELLER", "message"]
        return [dialog_entry[0], remove_emojis(dialog_entry[1])]
    return dialog_entry


def clean_jsonl_file(input_path, output_path=None):
    """Remove emojis from all text fields in a JSONL file."""

    input_path = Path(input_path)
    if not input_path.exists():
        print(f"Error: File {input_path} does not exist")
        return False

    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_cleaned{input_path.suffix}"
    else:
        output_path = Path(output_path)

    lines_processed = 0
    lines_with_emojis = 0

    print(f"Processing: {input_path}")
    print(f"Output to: {output_path}")

    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:

        for line_num, line in enumerate(infile, 1):
            try:
                data = json.loads(line.strip())
                original_str = json.dumps(data, ensure_ascii=False)

                # Clean all data recursively - handles any JSON structure
                cleaned_data = clean_json_data(data)

                cleaned_str = json.dumps(cleaned_data, ensure_ascii=False)

                if original_str != cleaned_str:
                    lines_with_emojis += 1
                    if lines_with_emojis <= 3:  # Show first 3 examples
                        print(f"  Line {line_num}: Found and removed emojis")

                outfile.write(json.dumps(cleaned_data, ensure_ascii=False) + '\n')
                lines_processed += 1

            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue

    print(f"\nProcessed {lines_processed} lines")
    print(f"Found and cleaned {lines_with_emojis} lines with emojis")
    print(f"Output saved to: {output_path}")

    return True


def main():
    parser = argparse.ArgumentParser(description='Remove emojis from JSONL files')
    parser.add_argument('input', help='Input JSONL file path')
    parser.add_argument('-o', '--output', help='Output file path (default: input_cleaned.jsonl)')

    args = parser.parse_args()

    clean_jsonl_file(args.input, args.output)


if __name__ == "__main__":
    main()