#!/usr/bin/env python3
"""
Test script to do a dry run of baseline training and intercept/analyze the training data.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from training.converters.builtin_export import BuiltinTaskExporter
from training.data_converter import DataConverter


def analyze_training_packets(sft_file: str, sample_size: int = 10) -> Dict:
    """
    Analyze the SFT training packets that would be sent to the model.

    Args:
        sft_file: Path to the SFT JSONL file
        sample_size: Number of samples to display

    Returns:
        Dictionary with analysis results
    """
    if not Path(sft_file).exists():
        return {'error': f'File not found: {sft_file}'}

    analysis = {
        'total_packets': 0,
        'clean_packets': 0,
        'contaminated_packets': 0,
        'packet_lengths': [],
        'contamination_markers': {},
        'samples': [],
        'issues': []
    }

    # Contamination markers to check for
    contamination_markers = [
        ('##', 'Markdown header'),
        ('###', 'Markdown subheader'),
        ('You are a', 'Role prompt'),
        ('Your target', 'Metadata leak'),
        ('Make a negotiation', 'Instruction'),
        ('[PROPOSE]', 'Action tag'),
        ('[COMMIT]', 'Action tag'),
        ('What is the', 'Question prompt'),
        ('Subject:', 'Email header'),
        ('Negotiator A', 'Role label'),
        ('Negotiator B', 'Role label'),
        ('\n\n\n', 'Excessive newlines'),
        ('Buyer\n\nYou are', 'Full prompt pattern'),
    ]

    with open(sft_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            analysis['total_packets'] += 1
            packet = json.loads(line)

            # Extract response (the actual training target)
            response = packet.get('response', '')

            # Also check input if present
            input_data = packet.get('input', {})
            input_text = ''
            if isinstance(input_data, dict):
                # Check partner_msg field which might contain prompts
                partner_msg = input_data.get('partner_msg', '')
                if partner_msg:
                    input_text = partner_msg

            # Track packet length
            analysis['packet_lengths'].append(len(response))

            # Check for contamination in response
            is_contaminated = False
            packet_issues = []

            for marker, issue_type in contamination_markers:
                if marker in response:
                    is_contaminated = True
                    packet_issues.append(f"Response contains {issue_type}: '{marker}'")
                    analysis['contamination_markers'][issue_type] = \
                        analysis['contamination_markers'].get(issue_type, 0) + 1

                if marker in input_text:
                    is_contaminated = True
                    packet_issues.append(f"Input contains {issue_type}: '{marker}'")
                    analysis['contamination_markers'][f"Input_{issue_type}"] = \
                        analysis['contamination_markers'].get(f"Input_{issue_type}", 0) + 1

            if is_contaminated:
                analysis['contaminated_packets'] += 1
                analysis['issues'].extend(packet_issues)

                # Store contaminated sample
                if len([s for s in analysis['samples'] if s.get('contaminated')]) < sample_size // 2:
                    analysis['samples'].append({
                        'line': line_num,
                        'contaminated': True,
                        'input': input_text[:100] if input_text else 'N/A',
                        'response': response[:200],
                        'issues': packet_issues[:3]  # First 3 issues
                    })
            else:
                analysis['clean_packets'] += 1

                # Store clean sample
                if len([s for s in analysis['samples'] if not s.get('contaminated')]) < sample_size // 2:
                    analysis['samples'].append({
                        'line': line_num,
                        'contaminated': False,
                        'input': input_text[:100] if input_text else 'N/A',
                        'response': response[:200],
                        'issues': []
                    })

    return analysis


def run_dry_export(episodes_per_task: int = 10):
    """
    Run a dry export of Deal or No Deal data with the new sanitization.

    Args:
        episodes_per_task: Number of episodes to export for testing

    Returns:
        Paths to generated files
    """
    print("\n" + "=" * 80)
    print("STEP 1: DRY RUN EXPORT WITH SANITIZATION")
    print("=" * 80)

    output_dir = "data/test_dry_run"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Export with new sanitization
    exporter = BuiltinTaskExporter(output_dir=f"{output_dir}/builtin_tasks")

    print(f"Exporting {episodes_per_task} Deal or No Deal episodes...")
    task_files = exporter.export_all_tasks(
        task_names=['dealnodeal'],
        max_episodes_per_task=episodes_per_task
    )

    print(f"Exported to: {task_files}")

    # Convert to SFT format
    print("\n" + "=" * 80)
    print("STEP 2: CONVERTING TO SFT FORMAT")
    print("=" * 80)

    converter = DataConverter(output_dir=output_dir)
    sft_file = converter.convert_episodes_to_sft(
        episodes_file=task_files['dealnodeal'],
        output_file=f"{output_dir}/sft/dealnodeal_test.jsonl"
    )

    print(f"Generated SFT file: {sft_file}")

    return task_files['dealnodeal'], sft_file


def display_analysis(analysis: Dict):
    """Display analysis results in a readable format."""

    if 'error' in analysis:
        print(f"❌ Error: {analysis['error']}")
        return

    print("\n" + "=" * 80)
    print("TRAINING DATA ANALYSIS")
    print("=" * 80)

    # Overall statistics
    print("\n📊 Overall Statistics:")
    print(f"  Total training packets: {analysis['total_packets']}")
    print(f"  Clean packets: {analysis['clean_packets']} ({analysis['clean_packets']/max(1, analysis['total_packets'])*100:.1f}%)")
    print(f"  Contaminated packets: {analysis['contaminated_packets']} ({analysis['contaminated_packets']/max(1, analysis['total_packets'])*100:.1f}%)")

    if analysis['packet_lengths']:
        avg_length = sum(analysis['packet_lengths']) / len(analysis['packet_lengths'])
        print(f"  Average response length: {avg_length:.1f} characters")
        print(f"  Min/Max length: {min(analysis['packet_lengths'])}/{max(analysis['packet_lengths'])}")

    # Contamination breakdown
    if analysis['contamination_markers']:
        print("\n🚨 Contamination Types Found:")
        for marker_type, count in sorted(analysis['contamination_markers'].items(),
                                        key=lambda x: x[1], reverse=True):
            print(f"  - {marker_type}: {count} occurrences")
    else:
        print("\n✅ No contamination markers found!")

    # Sample packets
    print("\n📝 Sample Training Packets:")

    # Separate clean and contaminated samples
    clean_samples = [s for s in analysis['samples'] if not s['contaminated']]
    contaminated_samples = [s for s in analysis['samples'] if s['contaminated']]

    if clean_samples:
        print("\n  ✅ CLEAN SAMPLES:")
        for i, sample in enumerate(clean_samples[:3], 1):
            print(f"\n  Sample {i} (Line {sample['line']}):")
            print(f"    Input: {sample['input']}")
            print(f"    Response: {sample['response']}")

    if contaminated_samples:
        print("\n  ❌ CONTAMINATED SAMPLES:")
        for i, sample in enumerate(contaminated_samples[:3], 1):
            print(f"\n  Sample {i} (Line {sample['line']}):")
            print(f"    Input: {sample['input']}")
            print(f"    Response: {sample['response']}")
            print(f"    Issues: {', '.join(sample['issues'])}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if analysis['contaminated_packets'] == 0:
        print("✅ Training data is CLEAN - no contamination detected!")
        print("   The sanitization is working correctly.")
    elif analysis['contaminated_packets'] < analysis['total_packets'] * 0.05:
        print("⚠️  Minor contamination detected (<5% of packets)")
        print("   Consider additional filtering but data is mostly clean.")
    else:
        print("❌ Significant contamination detected!")
        print("   Further sanitization needed before training.")


def main():
    """Main function to run the dry run test."""
    parser = argparse.ArgumentParser(
        description="Test baseline training data generation with sanitization"
    )
    parser.add_argument('--episodes', type=int, default=50,
                       help='Number of episodes to test (default: 50)')
    parser.add_argument('--analyze-existing', type=str,
                       help='Analyze an existing SFT file instead of generating new')

    args = parser.parse_args()

    print("=" * 80)
    print("BASELINE TRAINING DRY RUN - DATA INSPECTION")
    print("=" * 80)
    print("This will generate training data and analyze it for contamination")

    if args.analyze_existing:
        # Analyze existing file
        print(f"\nAnalyzing existing file: {args.analyze_existing}")
        analysis = analyze_training_packets(args.analyze_existing)
    else:
        # Generate new data
        episodes_file, sft_file = run_dry_export(episodes_per_task=args.episodes)

        # Analyze the generated SFT file
        print("\n" + "=" * 80)
        print("STEP 3: ANALYZING TRAINING DATA")
        print("=" * 80)

        analysis = analyze_training_packets(sft_file)

    # Display results
    display_analysis(analysis)

    print("\n🔍 Files generated in: data/test_dry_run/")
    print("   - Episode file: builtin_tasks/dealnodeal_episodes.jsonl")
    print("   - SFT file: sft/dealnodeal_test.jsonl")
    print("\n📋 Next steps:")
    print("   1. Review the samples above")
    print("   2. If clean, run full baseline training")
    print("   3. If contaminated, adjust sanitization rules")


if __name__ == "__main__":
    main()