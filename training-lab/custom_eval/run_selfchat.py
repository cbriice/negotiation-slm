#!/usr/bin/env python3
"""
Run self-chat evaluation with the custom evaluation framework.
Simple, clean evaluation without ParlAI complexity.
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

from custom_eval.model_wrapper import SimpleModelWrapper
from custom_eval.simple_negotiation import NegotiationSession
from custom_eval.eval_seeds import EvalSeedLoader
from custom_eval.prompt_builder import PromptBuilder

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_single_negotiation(session: NegotiationSession,
                          seed: Dict[str, Any],
                          verbose: bool = False) -> Dict[str, Any]:
    """Run a single negotiation and return results."""

    if verbose:
        logger.info(f"Running negotiation for seed {seed.get('seed_id', 'unknown')}")
        logger.info(f"Item: {seed['item']['name']} at ${seed['item']['list_price']}")
        logger.info(f"Buyer max: ${seed['buyer_state']['max_price']}, "
                   f"Seller min: ${seed['seller_state']['min_price']}")

    # Run negotiation
    result = session.run_negotiation(seed)

    # Add seed ID to result
    result['seed_id'] = seed.get('seed_id')

    if verbose:
        # Print conversation
        print("\n" + "="*60)
        print(f"NEGOTIATION: {seed['item']['name']}")
        print("="*60)
        for turn in result['dialogue']:
            print(f"\n[{turn['speaker'].upper()}]: {turn['text']}")
        print("\n" + "="*60)

        # Print outcome
        if result['metrics']['deal_made']:
            print(f"✓ DEAL at ${result['metrics']['final_price']}")
        else:
            print("✗ NO DEAL")
        print(f"Turns: {result['metrics']['turns']}")
        print(f"Quality: {result['metrics']['conversation_quality']:.2f}")
        print("="*60 + "\n")

    # Reset session for next negotiation
    session.reset()

    return result


def main():
    parser = argparse.ArgumentParser(description='Run self-chat evaluation')

    # Model arguments
    parser.add_argument('--model-name', type=str, default='mistralai/Mistral-7B-v0.1',
                       help='HuggingFace model to use')
    parser.add_argument('--adapter-path', type=str, default='models/sft/sft_run',
                       help='Path to LoRA adapter checkpoint directory')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Specific checkpoint to use (e.g., checkpoint-3000)')

    # Evaluation arguments
    parser.add_argument('--num-seeds', type=int, default=5,
                       help='Number of seeds to evaluate')
    parser.add_argument('--seed-file', type=str, default=None,
                       help='Path to seed file')
    parser.add_argument('--category', type=str, default=None,
                       help='Filter seeds by category')
    parser.add_argument('--max-turns', type=int, default=20,
                       help='Maximum turns per negotiation')

    # Generation arguments
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Generation temperature')
    parser.add_argument('--max-tokens', type=int, default=256,
                       help='Maximum tokens to generate per turn')

    # Output arguments
    parser.add_argument('--output-dir', type=str, default='custom_eval/results',
                       help='Directory to save results')
    parser.add_argument('--verbose', action='store_true',
                       help='Print conversations as they run')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Construct adapter path
    adapter_path = args.adapter_path
    if args.checkpoint:
        adapter_path = str(Path(adapter_path) / args.checkpoint)

    logger.info(f"Loading model from {args.model_name}")
    logger.info(f"Loading adapter from {adapter_path}")

    # Initialize model wrapper
    try:
        model = SimpleModelWrapper(
            model_name=args.model_name,
            adapter_path=adapter_path,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return 1

    # Print model info
    model_info = model.get_model_info()
    logger.info(f"Model loaded: {model_info}")

    # Initialize negotiation session
    session = NegotiationSession(
        model_wrapper=model,
        max_turns=args.max_turns,
        temperature=args.temperature
    )

    # Load evaluation seeds
    logger.info("Loading evaluation seeds...")
    seed_loader = EvalSeedLoader(args.seed_file)

    # Get seeds
    seeds = seed_loader.get_seeds(
        n=args.num_seeds,
        category=args.category,
        shuffle=True
    )

    logger.info(f"Loaded {len(seeds)} seeds for evaluation")

    # Run evaluations
    results = []
    successful = 0
    deals_made = 0
    total_quality = 0.0

    for i, seed in enumerate(seeds, 1):
        logger.info(f"\n[{i}/{len(seeds)}] Running negotiation...")

        try:
            result = run_single_negotiation(session, seed, verbose=args.verbose)
            results.append(result)

            successful += 1
            if result['metrics']['deal_made']:
                deals_made += 1
            total_quality += result['metrics']['conversation_quality']

        except Exception as e:
            logger.error(f"Error in negotiation {i}: {e}")
            continue

    # Calculate summary statistics
    summary = {
        'total_negotiations': len(seeds),
        'successful_negotiations': successful,
        'deals_made': deals_made,
        'deal_rate': deals_made / successful if successful > 0 else 0,
        'average_quality': total_quality / successful if successful > 0 else 0,
        'average_turns': sum(r['metrics']['turns'] for r in results) / len(results) if results else 0,
        'model_info': model_info,
        'evaluation_params': {
            'num_seeds': args.num_seeds,
            'max_turns': args.max_turns,
            'temperature': args.temperature,
            'category': args.category,
        }
    }

    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Negotiations run: {summary['successful_negotiations']}/{summary['total_negotiations']}")
    print(f"Deals made: {summary['deals_made']} ({summary['deal_rate']:.1%})")
    print(f"Average quality: {summary['average_quality']:.2f}")
    print(f"Average turns: {summary['average_turns']:.1f}")
    print("="*60)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = output_dir / f"selfchat_results_{timestamp}.json"

    with open(result_file, 'w') as f:
        json.dump({
            'summary': summary,
            'negotiations': results
        }, f, indent=2)

    logger.info(f"Results saved to {result_file}")

    # Save just the conversations in a readable format
    conv_file = output_dir / f"conversations_{timestamp}.txt"
    prompt_builder = PromptBuilder()

    with open(conv_file, 'w') as f:
        for result in results:
            # Format conversation
            history = [(d['speaker'], d['text']) for d in result['dialogue']]
            outcome = {
                'deal_made': result['metrics']['deal_made'],
                'final_price': result['metrics']['final_price']
            }

            formatted = prompt_builder.format_for_logging(history, outcome)
            f.write(f"Seed ID: {result.get('seed_id', 'unknown')}\n")
            f.write(f"Item: {result['seed_data']['item']['name']}\n")
            f.write(formatted)
            f.write("\n\n")

    logger.info(f"Conversations saved to {conv_file}")

    return 0


if __name__ == '__main__':
    exit(main())