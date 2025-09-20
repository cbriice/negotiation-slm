#!/usr/bin/env python3
"""
Audit training data for contamination and quality issues.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import re

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def check_contamination_markers(text: str) -> List[str]:
    """Check text for contamination markers."""
    issues = []

    # Check for prompt/instruction markers
    prompt_markers = [
        ('##', 'Markdown header'),
        ('You are a', 'Prompt instruction'),
        ('Your target', 'Metadata leak'),
        ('Make a negotiation', 'Task instruction'),
        ('[PROPOSE]', 'Action tag in text'),
        ('[COMMIT]', 'Action tag in text'),
        ('What is the', 'Question prompt'),
        ('Subject:', 'Email header'),
        ('Dear Seller', 'Formal letter'),
        ('As a helpful', 'Assistant prompt'),
        ('Negotiator A', 'Role label'),
        ('Negotiator B', 'Role label'),
    ]

    for marker, issue_type in prompt_markers:
        if marker in text:
            issues.append(f"{issue_type}: '{marker}'")

    # Check for excessive repetition (sign of generation issues)
    lines = text.split('\n')
    if len(lines) > 3:
        # Check if lines repeat
        unique_lines = set(lines)
        if len(unique_lines) < len(lines) / 2:
            issues.append("Excessive repetition")

    return issues


def audit_sft_file(filepath: str) -> Dict:
    """Audit an SFT training file for quality issues."""
    if not Path(filepath).exists():
        return {'error': f'File not found: {filepath}'}

    stats = {
        'total_packets': 0,
        'contaminated_packets': 0,
        'empty_responses': 0,
        'short_responses': 0,
        'long_responses': 0,
        'contamination_types': {},
        'samples': []
    }

    with open(filepath) as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            stats['total_packets'] += 1
            packet = json.loads(line)

            # Check response field
            response = packet.get('response', '')

            if not response:
                stats['empty_responses'] += 1
                continue

            # Length checks
            if len(response) < 10:
                stats['short_responses'] += 1
            elif len(response) > 1000:
                stats['long_responses'] += 1

            # Contamination checks
            issues = check_contamination_markers(response)

            # Also check input if present
            if 'input' in packet:
                input_text = str(packet['input'])
                input_issues = check_contamination_markers(input_text)
                for issue in input_issues:
                    issues.append(f"INPUT: {issue}")

            if issues:
                stats['contaminated_packets'] += 1

                # Track contamination types
                for issue in issues:
                    issue_type = issue.split(':')[0]
                    stats['contamination_types'][issue_type] = stats['contamination_types'].get(issue_type, 0) + 1

                # Store sample for inspection
                if len(stats['samples']) < 5:
                    stats['samples'].append({
                        'line': line_num,
                        'response_preview': response[:100],
                        'issues': issues
                    })

    return stats


def audit_episode_file(filepath: str) -> Dict:
    """Audit an episode file for quality issues."""
    if not Path(filepath).exists():
        return {'error': f'File not found: {filepath}'}

    stats = {
        'total_episodes': 0,
        'total_turns': 0,
        'contaminated_turns': 0,
        'empty_dialogues': 0,
        'contamination_types': {},
        'samples': []
    }

    with open(filepath) as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            stats['total_episodes'] += 1
            episode = json.loads(line)

            dialogue = episode.get('dialogue', [])

            if not dialogue:
                stats['empty_dialogues'] += 1
                continue

            for turn in dialogue:
                stats['total_turns'] += 1

                # Check both text and response fields
                text = turn.get('text', '')
                response = turn.get('response', '')

                all_issues = []

                if text:
                    text_issues = check_contamination_markers(text)
                    for issue in text_issues:
                        all_issues.append(f"TEXT: {issue}")

                if response:
                    response_issues = check_contamination_markers(response)
                    for issue in response_issues:
                        all_issues.append(f"RESPONSE: {issue}")

                if all_issues:
                    stats['contaminated_turns'] += 1

                    # Track contamination types
                    for issue in all_issues:
                        issue_type = issue.split(':')[0]
                        stats['contamination_types'][issue_type] = stats['contamination_types'].get(issue_type, 0) + 1

                    # Store sample
                    if len(stats['samples']) < 5:
                        stats['samples'].append({
                            'episode_line': line_num,
                            'turn_text': text[:50] if text else 'N/A',
                            'turn_response': response[:50] if response else 'N/A',
                            'issues': all_issues
                        })

    return stats


def main():
    """Main audit function."""
    print("=" * 80)
    print("TRAINING DATA AUDIT")
    print("=" * 80)

    # Files to audit
    files_to_audit = [
        # SFT files
        ('data/training/sft/baseline_negotiation_sft.jsonl', 'sft'),
        ('data/training/sft/dealnodeal_sft.jsonl', 'sft'),

        # Episode files
        ('data/training/builtin_tasks/dealnodeal_episodes.jsonl', 'episodes'),
        ('data/negotiations/selfchat/episodes/selfchat_episodes.jsonl', 'episodes'),
    ]

    for filepath, file_type in files_to_audit:
        print(f"\n{'=' * 80}")
        print(f"Auditing: {filepath}")
        print(f"Type: {file_type}")
        print('=' * 80)

        if not Path(filepath).exists():
            print(f"  ⚠️  File not found")
            continue

        if file_type == 'sft':
            stats = audit_sft_file(filepath)
        else:
            stats = audit_episode_file(filepath)

        if 'error' in stats:
            print(f"  ⚠️  {stats['error']}")
            continue

        # Print statistics
        if file_type == 'sft':
            print(f"\n📊 Statistics:")
            print(f"  Total packets: {stats['total_packets']}")
            print(f"  Contaminated: {stats['contaminated_packets']} ({stats['contaminated_packets']/max(1, stats['total_packets'])*100:.1f}%)")
            print(f"  Empty responses: {stats['empty_responses']}")
            print(f"  Short responses (<10 chars): {stats['short_responses']}")
            print(f"  Long responses (>1000 chars): {stats['long_responses']}")
        else:
            print(f"\n📊 Statistics:")
            print(f"  Total episodes: {stats['total_episodes']}")
            print(f"  Total turns: {stats['total_turns']}")
            print(f"  Contaminated turns: {stats['contaminated_turns']} ({stats['contaminated_turns']/max(1, stats['total_turns'])*100:.1f}%)")
            print(f"  Empty dialogues: {stats['empty_dialogues']}")

        if stats.get('contamination_types'):
            print(f"\n🚨 Contamination Types:")
            for cont_type, count in sorted(stats['contamination_types'].items(), key=lambda x: x[1], reverse=True):
                print(f"  - {cont_type}: {count} occurrences")

        if stats.get('samples'):
            print(f"\n📝 Sample Issues (showing first {len(stats['samples'])}):")
            for i, sample in enumerate(stats['samples'], 1):
                print(f"\n  Sample {i}:")
                if 'line' in sample:
                    print(f"    Line: {sample['line']}")
                    print(f"    Response: {sample['response_preview']}...")
                else:
                    print(f"    Episode Line: {sample['episode_line']}")
                    print(f"    Text: {sample['turn_text']}...")
                    print(f"    Response: {sample['turn_response']}...")
                print(f"    Issues: {', '.join(sample['issues'][:3])}")

    print("\n" + "=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)
    print("\nRecommendations:")
    print("1. If contamination found in SFT files, re-export with sanitization")
    print("2. Consider filtering out short/long responses")
    print("3. Review samples to understand contamination patterns")
    print("4. Re-train model with clean data")


if __name__ == "__main__":
    main()