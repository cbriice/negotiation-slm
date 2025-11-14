#!/usr/bin/env python3
"""
Simple script to analyze turn statistics in training data JSONL files.
Usage: python analyze_turns.py <jsonl_file>
"""
import json
import sys

def analyze_turns(filepath):
    assistant_turns = 0
    user_turns = 0
    total_conversations = 0
    turns_per_conversation = []

    with open(filepath, 'r') as f:
        for line in f:
            if not line.strip():
                continue

            data = json.loads(line)
            text = data['text']

            # Count turns in this conversation
            conv_assistant = text.count('Assistant:')
            conv_user = text.count('User:')

            assistant_turns += conv_assistant
            user_turns += conv_user
            total_conversations += 1
            turns_per_conversation.append(conv_assistant + conv_user)

    # Calculate statistics
    total_turns = assistant_turns + user_turns
    avg_turns = sum(turns_per_conversation) / len(turns_per_conversation) if turns_per_conversation else 0
    min_turns = min(turns_per_conversation) if turns_per_conversation else 0
    max_turns = max(turns_per_conversation) if turns_per_conversation else 0

    # Print results
    print(f"Total conversations: {total_conversations}")
    print(f"Total turns: {total_turns}")
    print(f"  Assistant turns: {assistant_turns}")
    print(f"  User turns: {user_turns}")
    print(f"Average turns per conversation: {avg_turns:.2f}")
    print(f"Min turns per conversation: {min_turns}")
    print(f"Max turns per conversation: {max_turns}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python analyze_turns.py <jsonl_file>")
        sys.exit(1)

    analyze_turns(sys.argv[1])
