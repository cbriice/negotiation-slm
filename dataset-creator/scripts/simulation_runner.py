"""
Main Simulation Runner for Negotiation Dataset Creation.

This module orchestrates the entire negotiation simulation process, managing
turn-by-turn dialog generation between buyer and seller personalities using
the control function and prompt selection system.
"""

import json
import random
import numpy as np
from typing import Dict, List, Optional, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.llm_interface import LLMInterface, LLMConfig
from utils.control_function import ControlFunction, NegotiationState, NegotiationParams
from utils.seed_loader import MultiFileSeedLoader
from utils.prompt_manager import PromptManager
from utils.price_dynamics import PriceDynamics, PriceState
from prompts.personality_deck import PersonalityDeck
from prompts.prompts_unified import get_unified_prompt
from schema.seed_initializer import NegotiationSeed


class SimulationRunner:
    """Main orchestrator for negotiation simulations."""

    def __init__(self, seed_directory: str = "seeds/",
                 model_name: str = "llama3.3:70b",
                 max_turns: int = 20):
        """
        Initialize simulation runner with all required components.

        Args:
            seed_directory: Directory containing seed JSON files
            model_name: Model name for LLM
            max_turns: Maximum number of turns before ending negotiation
        """
        # Initialize components
        self.seed_loader = MultiFileSeedLoader(seed_directory)
        self.seed_loader.load_all_seeds()

        # Initialize control function with default params
        self.control_params = NegotiationParams(
            beta_b=0.1,  # Will be overridden by personality
            beta_s=0.1   # Will be overridden by personality
        )
        self.control_function = ControlFunction(self.control_params)

        # Initialize LLM interface
        llm_config = LLMConfig(model=model_name)
        self.llm_interface = LLMInterface(llm_config)

        self.personality_deck = PersonalityDeck()
        self.prompt_manager = PromptManager()
        self.price_dynamics = PriceDynamics()
        self.max_turns = max_turns

        # Track simulation results
        self.completed_negotiations = []

    def create_negotiation_packet(self, seed: NegotiationSeed,
                                 buyer_personality, seller_personality) -> Dict:
        """
        Create negotiation packet from components.

        Args:
            seed: NegotiationSeed object
            buyer_personality: Buyer personality object
            seller_personality: Seller personality object

        Returns:
            Negotiation packet dictionary
        """
        return {
            "seed": seed,
            "buyer_personality": buyer_personality,
            "seller_personality": seller_personality,
            "dialog": [],  # Will store tuples of (speaker, message)
            "deal_closed": False,
            "buyer_favorable": False,
            "seller_favorable": False,
            "outcome": None,
            "final_price": None
        }

    def run_simulation(self, num_negotiations: int = 1) -> List[Dict]:
        """
        Run multiple negotiation simulations.

        Args:
            num_negotiations: Number of negotiations to simulate

        Returns:
            List of completed negotiation packets
        """
        results = []

        for i in range(num_negotiations):
            print(f"\n=== Starting Negotiation {i+1}/{num_negotiations} ===")

            # Load seed and personalities
            seed = self.seed_loader.get_random_seed()
            if not seed:
                print("No seeds available!")
                break

            buyer_personality, seller_personality = self.personality_deck.select_two_personalities()

            # Create negotiation packet
            packet = self.create_negotiation_packet(seed, buyer_personality, seller_personality)

            # Update control function params with personality betas
            self.control_params.beta_b = buyer_personality.traits.beta
            self.control_params.beta_s = seller_personality.traits.beta

            # Initialize negotiation state
            state = NegotiationState(
                r=0,  # Turn index
                Ps=seed.seller_min,
                Pb=seed.buyer_max,
                P_k=seed.seller_list,  # Current price starts at list price
                n_b_k=0,  # Extra items for buyer (to be determined from seed)
                n_s_k=len(seed.extras_available) if seed.extras_available else 0,
                l_k=0  # Number of extras included
            )

            # Initialize price state
            price_state = PriceState(
                buyer_price=0,  # Will be set after initial offer
                seller_price=seed.seller_list,
                initial_buyer_price=0,  # Will be set after initial offer
                initial_seller_price=seed.seller_list,
                current_price=seed.seller_list,
                turn_number=0
            )

            # Reset prompt manager for new negotiation
            self.prompt_manager.reset_negotiation()
            self.price_dynamics.reset()

            # Run the negotiation
            result = self._run_single_negotiation(packet, state, price_state)
            results.append(result)

            # Store completed negotiation
            self.completed_negotiations.append(result)

        return results

    def _run_single_negotiation(self, packet: Dict, state: NegotiationState,
                                price_state: PriceState) -> Dict:
        """
        Run a single negotiation simulation.

        Args:
            packet: Negotiation packet with seed and personalities
            state: Initial negotiation state
            price_state: Initial price state

        Returns:
            Completed negotiation packet with dialog
        """
        seed = packet["seed"]
        buyer_personality = packet["buyer_personality"]
        seller_personality = packet["seller_personality"]

        # Track forced response flags
        forced_response = None
        action_blocked = False

        # Track control features for state updates
        control_features = {'cg': 0, 'cz': 0, 's': 0, 'A_k': 0}

        for turn in range(self.max_turns):
            state.r = turn
            price_state.turn_number = turn
            is_buyer_turn = (turn % 2 == 0)

            print(f"\nTurn {turn}: {'Buyer' if is_buyer_turn else 'Seller'}")

            # Determine action and prompt based on turn number and state
            if turn == 0:
                # Buyer opener (forced)
                action = "opener"
                prompt = self.prompt_manager.get_forced_response_prompt("opener", True)

            elif turn == 1:
                # Seller opener response (forced)
                action = "opener_response"
                prompt = self.prompt_manager.get_forced_response_prompt("opener", False)

            elif turn == 2:
                # Buyer initial price offer (forced)
                action = "price_adjust"
                # Calculate initial buyer offer
                initial_offer = self.price_dynamics.calculate_initial_buyer_offer(
                    seed.seller_min, buyer_personality.traits.alpha
                )
                price_state.buyer_price = initial_offer
                price_state.initial_buyer_price = initial_offer
                price_state.current_price = initial_offer
                state.P_k = initial_offer

                # Get price adjust prompt for initial offer
                prompt = self.prompt_manager.get_price_adjust_prompt(
                    buyer_personality,
                    eta=buyer_personality.traits.eta,
                    cg=control_features['cg'],
                    cz=control_features['cz'],
                    is_buyer=True,
                    current_price=initial_offer
                )

            else:
                # Turn 3+ - Control function driven

                # Check for forced responses
                if forced_response == "accept":
                    action = "accept"
                    prompt = self.prompt_manager.get_action_prompt(
                        "accept",
                        buyer_personality if is_buyer_turn else seller_personality,
                        is_buyer_turn,
                        price=state.P_k
                    )
                    forced_response = None

                elif forced_response == "reject":
                    action = "reject"
                    prompt = self.prompt_manager.get_action_prompt(
                        "reject",
                        buyer_personality if is_buyer_turn else seller_personality,
                        is_buyer_turn,
                        price=state.P_k
                    )
                    forced_response = None
                    action_blocked = True  # Block actions after rejection

                elif forced_response == "clarify_respond":
                    action = "clarify_respond"
                    prompt = self.prompt_manager.get_forced_response_prompt(
                        "clarify_respond", is_buyer_turn
                    )
                    forced_response = None

                elif action_blocked:
                    # After rejection, negotiation ends
                    print("Negotiation ended after rejection.")
                    packet["outcome"] = "no_deal"
                    packet["deal_closed"] = False
                    break

                else:
                    # Normal control function action selection
                    action, prompt, control_features = self._select_action_and_prompt(
                        state, price_state, is_buyer_turn, packet
                    )

                    # Check if action triggers forced response
                    if action == "close":
                        # Next turn must be accept or reject
                        # Use personality p parameter for decision
                        personality = seller_personality if is_buyer_turn else buyer_personality
                        if random.random() < personality.traits.p:
                            forced_response = "accept"
                        else:
                            forced_response = "reject"

                    elif action == "clarify":
                        # Next turn must be clarify_respond
                        forced_response = "clarify_respond"

            # Generate response using LLM
            response = self._generate_response(packet, state, prompt)

            # Add to dialog
            speaker = "BUYER" if is_buyer_turn else "SELLER"
            packet["dialog"].append((speaker, response))

            # Update state based on action
            if action == "price_adjust" and turn > 2:
                # Update price state if price was adjusted
                new_price, was_conceded = self.price_dynamics.calculate_price_adjustment(
                    price_state,
                    buyer_personality if is_buyer_turn else seller_personality,
                    is_buyer_turn
                )
                if was_conceded:
                    price_state = self.price_dynamics.update_state_with_new_price(
                        price_state, new_price, is_buyer_turn
                    )
                    state.P_k = new_price

            elif action == "accept":
                # Deal accepted
                print(f"Deal accepted at price: ${state.P_k}")
                packet["outcome"] = "deal"
                packet["deal_closed"] = True
                packet["final_price"] = state.P_k

                # Determine who got favorable deal
                midpoint = (seed.buyer_max + seed.seller_min) / 2
                packet["buyer_favorable"] = state.P_k < midpoint
                packet["seller_favorable"] = state.P_k > midpoint
                break

            # Check for convergence
            if turn > 5:  # After some initial turns
                price_gap = self.price_dynamics.calculate_price_gap(price_state)
                if price_gap < 0.01 * seed.seller_list:  # Less than 1% of list price
                    print(f"Prices converged. Deal at ${state.P_k}")
                    packet["outcome"] = "deal"
                    packet["deal_closed"] = True
                    packet["final_price"] = state.P_k

                    midpoint = (seed.buyer_max + seed.seller_min) / 2
                    packet["buyer_favorable"] = state.P_k < midpoint
                    packet["seller_favorable"] = state.P_k > midpoint
                    break

            # Update state for next turn
            state.prev_s = control_features.get('s', 0)

        # If max turns reached without deal
        if packet["outcome"] is None:
            packet["outcome"] = "no_deal"
            packet["deal_closed"] = False
            print("Maximum turns reached without agreement.")

        return packet

    def _select_action_and_prompt(self, state: NegotiationState, price_state: PriceState,
                                  is_buyer: bool, packet: Dict) -> Tuple[str, str, Dict]:
        """
        Select action using control function and corresponding prompt.

        Args:
            state: Current negotiation state
            price_state: Current price state
            is_buyer: Whether this is buyer's turn
            packet: Negotiation packet

        Returns:
            Tuple of (action, prompt, control_features)
        """
        personality = packet["buyer_personality"] if is_buyer else packet["seller_personality"]

        # Evaluate control function
        evaluation = self.control_function.evaluate(state, is_buyer)

        # Get control features
        control_features = evaluation['features']

        # Get action probabilities
        action_probs = evaluation['action_probs']

        # Check for ultimatum conditions
        if self.prompt_manager.should_add_ultimatum_to_pool(
            personality.traits.eta, control_features['A_k']
        ):
            # Add ultimatum to possible actions with high probability
            action_probs["ultimatum"] = 0.3  # Give it reasonable probability

        # Normalize probabilities
        total = sum(action_probs.values())
        action_probs = {k: v/total for k, v in action_probs.items()}

        # Sample action
        actions = list(action_probs.keys())
        probabilities = list(action_probs.values())
        action = np.random.choice(actions, p=probabilities)

        # Get appropriate prompt for action
        if action == "ultimatum":
            prompt = self.prompt_manager.get_ultimatum_prompt(
                personality, is_buyer, state.P_k
            )
            action = "close"  # Ultimatum is treated as close

        elif action == "price_adjust":
            # Handle price adjustment with concession
            new_price, was_conceded = self.price_dynamics.calculate_price_adjustment(
                price_state, personality, is_buyer
            )

            prompt = self.prompt_manager.get_price_adjust_prompt(
                personality,
                eta=personality.traits.eta,
                cg=control_features['cg'],
                cz=control_features['cz'],
                is_buyer=is_buyer,
                current_price=new_price if was_conceded else state.P_k
            )

        elif action == "extra":
            # Map "extra" to "extras" for prompt manager
            prompt = self.prompt_manager.get_action_prompt(
                "extras", personality, is_buyer, price=state.P_k
            )

        else:
            # Get prompt for other actions
            prompt = self.prompt_manager.get_action_prompt(
                action, personality, is_buyer, price=state.P_k
            )

        return action, prompt, control_features

    def _generate_response(self, packet: Dict, state: NegotiationState,
                          action_prompt: str) -> str:
        """
        Generate response using LLM.

        Args:
            packet: Negotiation packet
            state: Current negotiation state
            action_prompt: Prompt for the action

        Returns:
            Generated response string
        """
        # Build unified prompt
        unified_prompt = get_unified_prompt(packet, state, action_prompt)

        # Generate response using appropriate temperature for personality
        turn_idx = state.r
        is_buyer_turn = (turn_idx % 2 == 0)
        personality = packet["buyer_personality"] if is_buyer_turn else packet["seller_personality"]

        response = self.llm_interface.generate(
            unified_prompt,
            temperature=personality.traits.temp,
            sim_id=f"sim_{len(self.completed_negotiations)}"
        )

        return response.strip()

    def save_results(self, output_file: str = "negotiation_dataset.jsonl"):
        """
        Save completed negotiations to JSONL file.

        Args:
            output_file: Path to output file
        """
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(output_file, 'w') as f:
            for negotiation in self.completed_negotiations:
                # Format for dataset - convert objects to serializable format
                dataset_entry = {
                    "seed_id": negotiation["seed"].seed_id,
                    "category": negotiation["seed"].category,
                    "item": negotiation["seed"].item,
                    "buyer_personality": negotiation["buyer_personality"].__class__.__name__,
                    "seller_personality": negotiation["seller_personality"].__class__.__name__,
                    "dialog": negotiation["dialog"],
                    "outcome": negotiation.get("outcome", "unknown"),
                    "final_price": negotiation.get("final_price", None),
                    "deal_closed": negotiation.get("deal_closed", False),
                    "buyer_favorable": negotiation.get("buyer_favorable", False),
                    "seller_favorable": negotiation.get("seller_favorable", False)
                }
                f.write(json.dumps(dataset_entry) + '\n')

        print(f"\nSaved {len(self.completed_negotiations)} negotiations to {output_file}")


def main():
    """Main entry point for simulation runner."""
    # Create simulation runner
    runner = SimulationRunner(
        seed_directory="../seeds/",
        model_name="llama3.3:70b",
        max_turns=20
    )

    # Run simulations
    print("Starting negotiation simulations...")
    results = runner.run_simulation(num_negotiations=2)

    # Save results
    runner.save_results("../data/negotiation_dataset.jsonl")

    print("\nSimulation complete!")
    print(f"Generated {len(results)} negotiations")

    # Print summary statistics
    deals_made = sum(1 for r in results if r.get("deal_closed"))
    print(f"Deals completed: {deals_made}/{len(results)}")

    if deals_made > 0:
        avg_price = sum(r.get("final_price", 0) for r in results if r.get("deal_closed")) / deals_made
        print(f"Average final price: ${avg_price:.2f}")


if __name__ == "__main__":
    main()