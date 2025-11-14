#!/usr/bin/env python3
"""
Reformats training data from full conversation blobs to next-utterance prediction format.
Splits each conversation into 2-3 sub-packets focusing on early, mid, and final stages.
"""

import json
import sys
from pathlib import Path


def parse_conversation(text):
    """Parse text into system section and turns."""
    lines = text.split('\n')

    # Extract system section (between --- markers)
    system_end = 0
    if lines[0].strip() == '---':
        for i in range(1, len(lines)):
            if lines[i].strip() == '---':
                system_end = i + 1
                break

    system_section = '\n'.join(lines[:system_end])
    conversation_text = '\n'.join(lines[system_end:])

    # Parse turns
    turns = []
    current_turn = None
    current_text = []

    for line in conversation_text.split('\n'):
        if line.startswith('Assistant: ') or line.startswith('User: ') or line.startswith('System: '):
            if current_turn:
                turns.append((current_turn, '\n'.join(current_text)))
            current_turn = line.split(':', 1)[0]
            current_text = [line]
        elif current_text:
            current_text.append(line)

    if current_turn:
        turns.append((current_turn, '\n'.join(current_text)))

    return system_section, turns


def create_masking_regions(text):
    """Create masking regions for the text."""
    regions = []
    current_pos = 0

    for line in text.split('\n'):
        line_with_newline = line + '\n'
        line_len = len(line_with_newline)

        # Determine role and mask value
        if line.startswith('---') or line.startswith('system:') or \
           line.startswith('assistant_style:') or line.startswith('key_strategy:') or \
           line.startswith('difficulty'):
            role = 'system'
            mask = True
        elif line.startswith('Assistant:'):
            role = 'assistant'
            mask = False
        elif line.startswith('User:'):
            role = 'user'
            mask = True
        elif line.startswith('System:'):
            role = 'system'
            mask = True
        else:
            # Continue previous region if exists
            if regions:
                regions[-1]['end'] = current_pos + line_len
                current_pos += line_len
                continue
            else:
                role = 'system'
                mask = True

        regions.append({
            'start': current_pos,
            'end': current_pos + line_len,
            'mask': mask,
            'role': role
        })
        current_pos += line_len

    # Remove trailing newline from last region
    if regions:
        regions[-1]['end'] -= 1

    return regions


def split_conversation(record):
    """Split a conversation record into 2-3 sub-packets."""
    text = record['text']
    meta = record['meta']
    assistant_turns_count = meta.get('assistant_turns', 0)

    if assistant_turns_count == 0:
        return [record]

    system_section, turns = parse_conversation(text)

    # Find assistant turn indices
    assistant_indices = [i for i, (role, _) in enumerate(turns) if role == 'Assistant']

    if not assistant_indices:
        return [record]

    # Determine split points based on assistant turn count
    if assistant_turns_count <= 4:
        # Create 2 packets: early and final
        split_points = [
            assistant_indices[len(assistant_indices) // 2],
            assistant_indices[-1]
        ]
    else:
        # Create 3 packets: early, mid, final
        third = len(assistant_indices) // 3
        split_points = [
            assistant_indices[third],
            assistant_indices[third * 2],
            assistant_indices[-1]
        ]

    # Create sub-packets
    sub_packets = []
    for split_idx in split_points:
        # Build text up to and including this split point
        sub_turns = turns[:split_idx + 1]
        sub_text_parts = [system_section]
        for role, turn_text in sub_turns:
            sub_text_parts.append(turn_text)

        sub_text = '\n'.join(sub_text_parts)

        # Create masking regions
        masking_regions = create_masking_regions(sub_text)

        # Update metadata
        sub_meta = meta.copy()
        sub_meta['num_turns'] = len(sub_turns)
        sub_meta['assistant_turns'] = sum(1 for r, _ in sub_turns if r == 'Assistant')
        sub_meta['user_turns'] = sum(1 for r, _ in sub_turns if r == 'User')

        sub_packet = {
            'text': sub_text,
            'meta': sub_meta,
            'masking_regions': masking_regions
        }
        sub_packets.append(sub_packet)

    return sub_packets


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <input_file.jsonl>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)

    output_path = input_path.parent / f"{input_path.stem}_next_utterance.jsonl"

    # Process input file
    total_original = 0
    total_output = 0

    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            total_original += 1

            sub_packets = split_conversation(record)
            for sub_packet in sub_packets:
                outfile.write(json.dumps(sub_packet) + '\n')
                total_output += 1

    print(f"Processed {total_original} conversations into {total_output} sub-packets")
    print(f"Output written to: {output_path}")


if __name__ == '__main__':
    main()
