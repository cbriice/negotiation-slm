#!/usr/bin/env python3
"""
Interactive negotiation testing - play against the model.
Allows manual testing to observe model behavior in real-time.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

from custom_eval.model_wrapper import SimpleModelWrapper
from custom_eval.eval_seeds import EvalSeedLoader
from custom_eval.prompt_builder import PromptBuilder

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InteractiveNegotiation:
    """Manage an interactive negotiation session."""

    def __init__(self,
                 model: SimpleModelWrapper,
                 user_role: str = 'buyer',
                 temperature: float = 0.7):
        """
        Initialize interactive negotiation.

        Args:
            model: Model wrapper
            user_role: Role for human player ('buyer' or 'seller')
            temperature: Generation temperature
        """
        self.model = model
        self.user_role = user_role
        self.model_role = 'seller' if user_role == 'buyer' else 'buyer'
        self.temperature = temperature
        self.prompt_builder = PromptBuilder(include_strategy_hints=True)

        self.conversation_history: List[Tuple[str, str]] = []
        self.turns = 0
        self.max_turns = 30

    def run(self, seed: Dict[str, Any]):
        """Run interactive negotiation session."""

        # Display scenario
        self._display_scenario(seed)

        # Get role parameters
        item = seed['item']
        if self.user_role == 'buyer':
            user_params = seed['buyer_state']
            model_params = seed['seller_state']
        else:
            user_params = seed['seller_state']
            model_params = seed['buyer_state']

        # Build initial model prompt
        model_context = self.prompt_builder.build_initial_prompt(
            self.model_role,
            item,
            model_params
        )

        print("\n" + "="*60)
        print("NEGOTIATION START")
        print("="*60)
        print(f"You are the {self.user_role.upper()}")
        print(f"Type your messages and press Enter. Type 'quit' to end.")
        print(f"Type 'deal' to accept the current offer.")
        print(f"Type 'no deal' to walk away.")
        print("="*60 + "\n")

        # Determine who starts
        if self.user_role == 'buyer':
            print("You start the negotiation. What's your opening?")
        else:
            print("The buyer will start. Generating their opening...")
            self._model_turn(model_context, item, model_params)

        # Main conversation loop
        while self.turns < self.max_turns:
            # User turn
            user_input = input(f"\n{self.user_role.upper()} > ").strip()

            if user_input.lower() == 'quit':
                print("\nEnding negotiation...")
                break

            if not user_input:
                print("Please enter a message.")
                continue

            # Add to history
            self.conversation_history.append((self.user_role, user_input))
            self.turns += 1

            # Check for deal/no deal
            if 'deal' in user_input.lower():
                if 'no deal' in user_input.lower():
                    print("\n✗ Negotiation ended - No deal reached")
                else:
                    print("\n✓ Deal accepted!")
                break

            # Model turn
            print(f"\n{self.model_role.upper()} is thinking...")

            # Build prompt with history
            prompt = self.prompt_builder.build_initial_prompt(
                self.model_role,
                item,
                model_params,
                self.conversation_history
            )

            # Generate response
            try:
                response = self.model.generate(
                    prompt,
                    temperature=self.temperature,
                    max_tokens=150,
                    stop_sequences=["\n\n", f"{self.user_role.capitalize()}:", f"{self.model_role.capitalize()}:"]
                )

                # Clean response
                response = self.prompt_builder.extract_response(response, self.model_role)

                # Display response
                print(f"\n{self.model_role.upper()} > {response}")

                # Add to history
                self.conversation_history.append((self.model_role, response))
                self.turns += 1

                # Check if model ended negotiation
                if 'deal' in response.lower():
                    if 'no deal' in response.lower():
                        print("\n✗ Model walked away - No deal reached")
                    else:
                        print("\n✓ Model accepted the deal!")
                    break

            except Exception as e:
                logger.error(f"Error generating response: {e}")
                print(f"\n[Error generating response: {e}]")

        if self.turns >= self.max_turns:
            print(f"\n⏱ Maximum turns ({self.max_turns}) reached")

        # Display summary
        self._display_summary()

    def _model_turn(self,
                    context: str,
                    item: Dict[str, Any],
                    params: Dict[str, Any]):
        """Handle model's opening turn."""
        try:
            # Generate opening
            prompt = context + f"\n\n### Start the negotiation:\n{self.model_role.capitalize()}: "

            response = self.model.generate(
                prompt,
                temperature=self.temperature,
                max_tokens=150
            )

            # Clean and display
            response = self.prompt_builder.extract_response(response, self.model_role)
            print(f"\n{self.model_role.upper()} > {response}")

            # Add to history
            self.conversation_history.append((self.model_role, response))
            self.turns += 1

        except Exception as e:
            logger.error(f"Error in model opening: {e}")
            print(f"\n[Error generating opening: {e}]")

    def _display_scenario(self, seed: Dict[str, Any]):
        """Display the negotiation scenario to the user."""
        item = seed['item']

        print("\n" + "="*60)
        print("NEGOTIATION SCENARIO")
        print("="*60)

        print(f"\nItem: {item['name']}")
        if 'brand' in item:
            print(f"Brand: {item['brand']}")
        if 'model' in item:
            print(f"Model: {item['model']}")
        print(f"Condition: {item.get('condition', 'used')}")
        print(f"Listed Price: ${item['list_price']}")

        if self.user_role == 'buyer':
            params = seed['buyer_state']
            print(f"\n--- YOUR ROLE: BUYER ---")
            print(f"Your maximum budget: ${params['max_price']}")
            print(f"Your target price: ${params.get('target_price', params['max_price'] * 0.85):.0f}")
            print(f"Remember: Don't exceed your maximum!")
        else:
            params = seed['seller_state']
            print(f"\n--- YOUR ROLE: SELLER ---")
            print(f"Your minimum acceptable: ${params['min_price']}")
            print(f"Your target price: ${params.get('target_price', item['list_price'])}")
            if params.get('item_issues'):
                print(f"Known issues: {', '.join(params['item_issues'])}")

        print("="*60)

    def _display_summary(self):
        """Display conversation summary."""
        print("\n" + "="*60)
        print("CONVERSATION SUMMARY")
        print("="*60)
        print(f"Total turns: {self.turns}")
        print(f"Your role: {self.user_role}")

        # Format conversation
        formatted = self.prompt_builder.format_for_logging(self.conversation_history)
        print("\n" + formatted)


def main():
    parser = argparse.ArgumentParser(description='Interactive negotiation testing')

    # Model arguments
    parser.add_argument('--model-name', type=str, default='mistralai/Mistral-7B-v0.1',
                       help='HuggingFace model to use')
    parser.add_argument('--adapter-path', type=str, default='models/sft/sft_run',
                       help='Path to LoRA adapter checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Specific checkpoint to use')

    # Game arguments
    parser.add_argument('--role', type=str, default='buyer',
                       choices=['buyer', 'seller'],
                       help='Your role in the negotiation')
    parser.add_argument('--seed-id', type=int, default=None,
                       help='Specific seed ID to use')
    parser.add_argument('--category', type=str, default=None,
                       help='Category to select from')

    # Generation arguments
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Generation temperature')

    args = parser.parse_args()

    # Construct adapter path
    adapter_path = args.adapter_path
    if args.checkpoint:
        adapter_path = str(Path(adapter_path) / args.checkpoint)

    print(f"Loading model from {args.model_name}")
    print(f"Loading adapter from {adapter_path}")

    # Initialize model
    try:
        model = SimpleModelWrapper(
            model_name=args.model_name,
            adapter_path=adapter_path,
            temperature=args.temperature,
            max_tokens=256
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return 1

    print("Model loaded successfully!\n")

    # Load seeds
    seed_loader = EvalSeedLoader()

    while True:
        # Get seed
        if args.seed_id:
            seed = seed_loader.get_seed(args.seed_id)
        else:
            # Get random seed from category if specified
            seeds = seed_loader.get_seeds(n=1, category=args.category)
            if not seeds:
                print(f"No seeds found for category '{args.category}'")
                categories = seed_loader.get_categories()
                print(f"Available categories: {', '.join(categories)}")
                return 1
            seed = seeds[0]

        # Run interactive negotiation
        session = InteractiveNegotiation(
            model=model,
            user_role=args.role,
            temperature=args.temperature
        )

        session.run(seed)

        # Ask if user wants to continue
        again = input("\nPlay another round? (y/n): ").strip().lower()
        if again != 'y':
            break

        print("\n" + "="*60)
        print("Starting new negotiation...")
        print("="*60)

    print("\nThanks for playing!")
    return 0


if __name__ == '__main__':
    exit(main())