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
from datetime import datetime
import time
from pathlib import Path

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
from utils.logger_config import get_logger, setup_logger


class SimulationRunner:
    """Main orchestrator for negotiation simulations."""

    def __init__(self, seed_directory: str = "seeds/",
                 model_name: str = "Qwen/Qwen2.5-14B-Instruct",
                 max_turns: int = 25,
                 seed_category: Optional[str] = None,
                 seed_selection_method: str = "random",
                 specific_seed_file: Optional[str] = None):
        """
        Initialize simulation runner with all required components.

        Args:
            seed_directory: Directory containing seed JSON files
            model_name: Model name for LLM
            max_turns: Maximum number of turns before ending negotiation
            seed_category: Optional category filter for seeds
            seed_selection_method: How to select seeds ("random", "sequential", "all")
            specific_seed_file: Optional specific seed file to load instead of directory
        """
        # Initialize logger
        self.logger = get_logger()
        self.logger.sim_logger.info("="*60)
        self.logger.sim_logger.info("Initializing SimulationRunner")
        self.logger.sim_logger.info(f"Configuration:")
        self.logger.sim_logger.info(f"  - Model: {model_name}")
        self.logger.sim_logger.info(f"  - Max turns: {max_turns}")
        self.logger.sim_logger.info(f"  - Seed directory: {seed_directory}")
        self.logger.sim_logger.info(f"  - Seed category: {seed_category}")
        self.logger.sim_logger.info(f"  - Seed selection: {seed_selection_method}")
        if specific_seed_file:
            self.logger.sim_logger.info(f"  - Specific seed file: {specific_seed_file}")

        # Initialize components
        self.logger.sim_logger.info("Loading seeds...")

        # Check if we're loading a specific file or a directory
        self.using_single_file = False
        if specific_seed_file:
            # Load only the specific file
            print(f"\n[SEED LOADER] Loading specific seed file: {specific_seed_file}")

            from utils.seed_loader import SeedLoader
            single_loader = SeedLoader(specific_seed_file)

            # Create a MultiFileSeedLoader-compatible structure
            self.seed_loader = MultiFileSeedLoader(seed_directory)
            # Manually set up the loader with just this single file
            category_name = Path(specific_seed_file).stem.replace("_seeds", "")
            self.seed_loader.loaders[category_name] = single_loader
            self.seed_loader.all_seeds = single_loader.get_all_seeds()
            self.using_single_file = True

            # print(f"[SEED LOADER] Loaded {len(self.seed_loader.all_seeds)} seeds from {specific_seed_file}")
            print(f"[SEED LOADER] Category: {category_name}\n")
            self.logger.sim_logger.info(f"Loaded {len(self.seed_loader.all_seeds)} seeds from specific file: {specific_seed_file}")
        else:
            # Load all seeds from directory
            print(f"\n[SEED LOADER] Loading ALL seed files from directory: {seed_directory}")
            self.seed_loader = MultiFileSeedLoader(seed_directory)
            self.seed_loader.load_all_seeds()
            self.logger.sim_logger.info(f"Loaded {len(self.seed_loader.all_seeds)} total seeds from directory")

        # Store configuration
        self.seed_category = seed_category
        self.seed_selection_method = seed_selection_method
        self.seed_index = 0  # For sequential selection

        # Initialize control function with default params
        self.logger.sim_logger.info("Initializing control function...")
        self.control_params = NegotiationParams(
            beta_b=0.1,  # Will be overridden by personality
            beta_s=0.1   # Will be overridden by personality
        )
        self.control_function = ControlFunction(self.control_params)

        # Initialize LLM interface
        self.logger.sim_logger.info(f"Initializing LLM interface with {model_name}...")
        llm_config = LLMConfig(model=model_name)
        self.llm_interface = LLMInterface(llm_config)

        self.logger.sim_logger.info("Initializing personality deck and managers...")
        self.personality_deck = PersonalityDeck()
        self.prompt_manager = PromptManager()
        self.price_dynamics = PriceDynamics()
        self.max_turns = max_turns

        # Track simulation results
        self.completed_negotiations = []

        self.logger.sim_logger.info("SimulationRunner initialization complete!")
        self.logger.sim_logger.info("="*60)

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

    def get_next_seed(self) -> Optional[NegotiationSeed]:
        """
        Get next seed based on configured selection method and category filter.

        Returns:
            NegotiationSeed or None if no seeds available
        """
        self.logger.sim_logger.debug(f"Getting next seed with method: {self.seed_selection_method}")

        if self.seed_selection_method == "random":
            # Random selection, respecting category filter
            seed = self.seed_loader.get_random_seed(self.seed_category)
            if seed:
                self.logger.sim_logger.debug(f"Selected random seed: {seed.seed_id} - {seed.item}")
            return seed

        elif self.seed_selection_method == "sequential":
            # Sequential selection
            if self.seed_category:
                # Get seeds for specific category
                category_loader = self.seed_loader.get_loader_for_category(self.seed_category)
                if category_loader:
                    seed = category_loader.get_next_seed()
                    if seed:
                        self.logger.sim_logger.debug(f"Selected sequential seed from category {self.seed_category}: {seed.seed_id}")
                    return seed
            else:
                # Get from all seeds sequentially
                if self.seed_index < len(self.seed_loader.all_seeds):
                    seed = self.seed_loader.all_seeds[self.seed_index]
                    self.seed_index += 1
                    self.logger.sim_logger.debug(f"Selected sequential seed {self.seed_index}/{len(self.seed_loader.all_seeds)}: {seed.seed_id}")
                    return seed

        elif self.seed_selection_method == "all":
            # Return seeds in order (handled by iteration in run_simulation)
            if self.seed_index < len(self.seed_loader.all_seeds):
                seed = self.seed_loader.all_seeds[self.seed_index]
                self.seed_index += 1
                self.logger.sim_logger.debug(f"Selected seed {self.seed_index}/{len(self.seed_loader.all_seeds)}: {seed.seed_id}")
                return seed

        self.logger.sim_logger.warning("No more seeds available")
        return None

    def run_simulation(self, num_negotiations: int = 1,
                      specific_seed: Optional[NegotiationSeed] = None) -> List[Dict]:
        """
        Run multiple negotiation simulations.

        Args:
            num_negotiations: Number of negotiations to simulate
            specific_seed: Optional specific seed to use for all negotiations

        Returns:
            List of completed negotiation packets
        """
        self.logger.sim_logger.info("="*60)
        self.logger.sim_logger.info(f"Starting simulation run with {num_negotiations} negotiations")

        results = []
        start_time = time.time()

        # Determine number of iterations based on selection method
        if self.seed_selection_method == "all":
            # Run once for each seed
            if self.seed_category:
                category_loader = self.seed_loader.get_loader_for_category(self.seed_category)
                total_seeds = len(category_loader.get_all_seeds()) if category_loader else 0
            else:
                total_seeds = len(self.seed_loader.all_seeds)
            num_negotiations = total_seeds
            self.logger.sim_logger.info(f"Running negotiations for all {num_negotiations} seeds")
            print(f"Running negotiations for all {num_negotiations} seeds")

        for i in range(num_negotiations):
            self.logger.sim_logger.info(f"Starting negotiation {i+1}/{num_negotiations}")
            print(f"\n=== Starting Negotiation {i+1}/{num_negotiations} ===")

            # Generate unique simulation ID
            sim_id = f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i+1}"
            self.logger.sim_logger.info(f"Simulation ID: {sim_id}")

            # Load seed and personalities
            if specific_seed:
                seed = specific_seed
                self.logger.sim_logger.info(f"Using specific seed: {seed.seed_id} - {seed.item}")
            else:
                seed = self.get_next_seed()

            if not seed:
                self.logger.sim_logger.warning("No more seeds available!")
                print("No more seeds available!")
                break
            
            print(f"\nSeed ID: {seed.seed_id}")

            buyer_personality, seller_personality = self.personality_deck.select_two_personalities()
            self.logger.log_simulation_start(
                sim_id, seed.seed_id,
                buyer_personality.__class__.__name__,
                seller_personality.__class__.__name__
            )

            # Create negotiation packet
            packet = self.create_negotiation_packet(seed, buyer_personality, seller_personality)
            packet["sim_id"] = sim_id  # Add sim_id to packet

            # Update control function params with personality betas
            self.control_params.beta_b = buyer_personality.traits.beta
            self.control_params.beta_s = seller_personality.traits.beta
            self.logger.sim_logger.debug(f"[{sim_id}] Control params - beta_b: {self.control_params.beta_b}, beta_s: {self.control_params.beta_s}")

            # Initialize negotiation state with separate buyer/seller prices
            state = NegotiationState(
                r=0,  # Turn index
                Ps=seed.seller_min,
                Pb=seed.buyer_max,
                buyer_offer=0.0,  # Buyer hasn't made an offer yet
                seller_offer=seed.seller_list,  # Seller starts at list price
                n_b_k=0,  # Extra items for buyer (to be determined from seed)
                n_s_k=len(seed.extras_available) if seed.extras_available else 0,
                l_k=0  # Number of extras included
            )
            self.logger.sim_logger.debug(f"[{sim_id}] Initial state - Ps: ${seed.seller_min}, Pb: ${seed.buyer_max}, P_k: ${seed.seller_list}")

            # Initialize price state
            price_state = PriceState(
                buyer_price=0,  # Will be set after initial offer
                seller_price=seed.seller_list,
                initial_buyer_price=0,  # Will be set after initial offer
                initial_seller_price=seed.seller_list,
                current_price=seed.seller_list,
                turn_number=0
            )
            self.logger.sim_logger.debug(f"[{sim_id}] Initial price state - seller_price: ${seed.seller_list}")

            # Reset prompt manager for new negotiation
            self.prompt_manager.reset_negotiation()
            self.price_dynamics.reset()
            self.logger.sim_logger.debug(f"[{sim_id}] Reset managers for new negotiation")

            # Run the negotiation
            negotiation_start = time.time()
            self.logger.sim_logger.info(f"[{sim_id}] Starting negotiation turns...")
            result = self._run_single_negotiation(packet, state, price_state)
            negotiation_duration = time.time() - negotiation_start

            # Log outcome
            self.logger.log_simulation_end(sim_id, result, negotiation_duration)
            results.append(result)

            # Store completed negotiation
            self.completed_negotiations.append(result)

        total_duration = time.time() - start_time
        self.logger.sim_logger.info(f"Simulation run complete. Total time: {total_duration:.2f}s, Average: {total_duration/len(results):.2f}s per negotiation")

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
        sim_id = packet.get("sim_id", "unknown")

        # Track forced response flags
        forced_response = None

        # Track rejection attempts for persistence mechanic
        has_had_rejection = False

        # Track control features for state updates
        control_features = {'cg': 0, 'cz': 0, 's': 0, 'A_k': 0}

        for turn in range(self.max_turns):
            state.r = turn
            price_state.turn_number = turn
            is_buyer_turn = (turn % 2 == 0)
            speaker = "BUYER" if is_buyer_turn else "SELLER"

            self.logger.sim_logger.info(f"[{sim_id}] Turn {turn}: {speaker}")
            print(f"\nTurn {turn}: {'Buyer' if is_buyer_turn else 'Seller'}")

            # Determine action and prompt based on turn number and state
            if turn == 0:
                # Buyer opener (forced)
                action = "opener"
                prompt = self.prompt_manager.get_forced_response_prompt("opener", True)
                self.logger.sim_logger.debug(f"[{sim_id}] Turn {turn}: Forced action - {action}")

            elif turn == 1:
                # Seller opener response (forced)
                action = "opener_response"
                prompt = self.prompt_manager.get_forced_response_prompt("opener", False)
                self.logger.sim_logger.debug(f"[{sim_id}] Turn {turn}: Forced action - {action}")

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
                state.buyer_offer = initial_offer  # Update buyer's offer
                self.logger.sim_logger.info(f"[{sim_id}] Turn {turn}: Initial buyer offer - ${initial_offer}")

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
                self.logger.sim_logger.debug(f"[{sim_id}] Turn {turn}: Control function driven turn")

                # Check for forced responses
                if forced_response == "accept":
                    action = "accept"
                    # Use convergence price if available, otherwise accept other party's price
                    if "convergence_price" in packet:
                        accept_price = packet["convergence_price"]
                        self.logger.sim_logger.debug(f"Using convergence price: ${accept_price:.2f}")
                    else:
                        accept_price = state.get_active_price_for(not is_buyer_turn)  # Accept other party's price
                    prompt = self.prompt_manager.get_action_prompt(
                        "accept",
                        buyer_personality if is_buyer_turn else seller_personality,
                        is_buyer_turn,
                        accept_price=accept_price,
                        buyer_max=seed.buyer_max,
                        seller_min=seed.seller_min
                    )
                    self.logger.sim_logger.info(f"[{sim_id}] Turn {turn}: Forced response - ACCEPT at ${accept_price:.2f}")
                    forced_response = None

                elif forced_response == "reject":
                    action = "reject"
                    prompt = self.prompt_manager.get_action_prompt(
                        "reject",
                        buyer_personality if is_buyer_turn else seller_personality,
                        is_buyer_turn,
                        price=state.get_active_price_for(not is_buyer_turn)  # Reject other party's price
                    )
                    self.logger.sim_logger.info(f"[{sim_id}] Turn {turn}: Forced response - REJECT")
                    forced_response = None

                elif forced_response == "clarify_respond":
                    action = "clarify_respond"
                    prompt = self.prompt_manager.get_forced_response_prompt(
                        "clarify_respond", is_buyer_turn
                    )
                    self.logger.sim_logger.info(f"[{sim_id}] Turn {turn}: Forced response - CLARIFY_RESPOND")
                    forced_response = None

                else:
                    # Normal control function action selection
                    action, prompt, control_features = self._select_action_and_prompt(
                        state, price_state, is_buyer_turn, packet
                    )
                    self.logger.log_turn(sim_id, turn, speaker, action)

                    # Check if action triggers forced response
                    if action == "close":
                        # Next turn must be accept or reject
                        # Use personality p parameter for decision
                        personality = seller_personality if is_buyer_turn else buyer_personality
                        if random.random() < personality.traits.p:
                            forced_response = "accept"
                            self.logger.sim_logger.debug(f"[{sim_id}] Close action - forcing ACCEPT next turn (p={personality.traits.p})")
                        else:
                            forced_response = "reject"
                            self.logger.sim_logger.debug(f"[{sim_id}] Close action - forcing REJECT next turn (p={personality.traits.p})")

                    elif action == "clarify":
                        # Next turn must be clarify_respond
                        forced_response = "clarify_respond"
                        self.logger.sim_logger.debug(f"[{sim_id}] Clarify action - forcing CLARIFY_RESPOND next turn")

            # Generate response using LLM
            llm_start = time.time()
            response = self._generate_response(packet, state, prompt)
            llm_duration = time.time() - llm_start
            self.logger.log_llm_call(sim_id, prompt[:200] + "...", response[:200] + "...", llm_duration, success=True)

            # Add to dialog
            speaker = "BUYER" if is_buyer_turn else "SELLER"
            packet["dialog"].append((speaker, response))
            self.logger.log_turn(sim_id, turn, speaker, action, response)

            # Check if negotiation should end after rejection
            if action == "reject":
                # Determine if the other party will persist after rejection
                # The rejecting party is the current speaker
                rejecting_party = "BUYER" if is_buyer_turn else "SELLER"
                other_party_personality = seller_personality if is_buyer_turn else buyer_personality

                if not has_had_rejection:
                    # First rejection - check if other party persists
                    persistence_roll = random.random()
                    persistence_threshold = other_party_personality.traits.persistence

                    if persistence_roll < persistence_threshold:
                        # Other party persists - negotiation continues
                        self.logger.sim_logger.info(
                            f"[{sim_id}] {rejecting_party} REJECTED at ${state.P_k}, but other party persists "
                            f"(roll: {persistence_roll:.3f} < persistence: {persistence_threshold:.3f})"
                        )
                        print(f"{rejecting_party} rejected at ${state.P_k}, but negotiation continues (persistence check passed)")
                        has_had_rejection = True
                        # Clear any forced response to allow negotiation to continue naturally
                        forced_response = None
                        # Continue negotiation - don't break
                    else:
                        # Other party doesn't persist - negotiation ends
                        self.logger.sim_logger.info(
                            f"[{sim_id}] DEAL REJECTED at price: ${state.P_k} "
                            f"(persistence check failed: {persistence_roll:.3f} >= {persistence_threshold:.3f})"
                        )
                        print(f"Deal rejected at price: ${state.P_k} (other party walks away)")
                        packet["outcome"] = "no_deal"
                        packet["deal_closed"] = False
                        break
                else:
                    # Second rejection - definitively ends negotiation
                    self.logger.sim_logger.info(f"[{sim_id}] SECOND REJECTION - negotiation definitively ended at price: ${state.P_k}")
                    print(f"Second rejection - negotiation definitively ended at price: ${state.P_k}")
                    packet["outcome"] = "no_deal"
                    packet["deal_closed"] = False
                    break

            # Update state based on action
            if action == "price_adjust" and turn > 2:
                # Update price state if price was adjusted
                new_price, was_conceded = self.price_dynamics.calculate_price_adjustment(
                    price_state,
                    buyer_personality if is_buyer_turn else seller_personality,
                    is_buyer_turn
                )
                if was_conceded:
                    # Get current price for this party
                    old_price = state.get_active_price_for(is_buyer_turn)
                    self.logger.sim_logger.info(f"[{sim_id}] Price adjustment - {speaker} moved from ${old_price:.2f} to ${new_price:.2f}")
                    price_state = self.price_dynamics.update_state_with_new_price(
                        price_state, new_price, is_buyer_turn
                    )
                    # Update the appropriate party's offer
                    if is_buyer_turn:
                        state.buyer_offer = new_price
                    else:
                        state.seller_offer = new_price
                else:
                    current_price = state.get_active_price_for(is_buyer_turn)
                    self.logger.sim_logger.debug(f"[{sim_id}] Price adjustment - {speaker} held at ${current_price:.2f}")

            elif action == "accept":
                # Deal accepted - use convergence price if available, otherwise accept other party's offer
                if "convergence_price" in packet:
                    accepted_price = packet["convergence_price"]
                    self.logger.sim_logger.info(f"[{sim_id}] DEAL ACCEPTED (convergence) - {speaker} accepts ${accepted_price:.2f}")
                else:
                    accepted_price = state.get_active_price_for(not is_buyer_turn)
                    self.logger.sim_logger.info(f"[{sim_id}] DEAL ACCEPTED - {speaker} accepts ${accepted_price:.2f}")
                print(f"Deal accepted at price: ${accepted_price:.2f}")
                packet["outcome"] = "deal"
                packet["deal_closed"] = True
                packet["final_price"] = accepted_price

                # Determine who got favorable deal
                midpoint = (seed.buyer_max + seed.seller_min) / 2
                packet["buyer_favorable"] = accepted_price < midpoint
                packet["seller_favorable"] = accepted_price > midpoint

                self.logger.sim_logger.info(f"[{sim_id}] Deal analysis - midpoint: ${midpoint:.2f}, buyer_favorable: {packet['buyer_favorable']}, seller_favorable: {packet['seller_favorable']}")
                break

            # Check for convergence (only if deal not already closed and no pending forced response)
            # Skip convergence check if there's a pending forced response to ensure proper dialog flow
            if turn > 5 and not packet.get("deal_closed", False) and forced_response is None:  # After some initial turns
                # Check for price overlap (automatic deal convergence)
                if state.buyer_offer >= state.seller_offer:
                    # Prices have crossed - deal zone reached
                    # Use seller's offer (lower price) as that's what buyer would accept
                    final_price = state.seller_offer
                    self.logger.sim_logger.info(f"[{sim_id}] Prices OVERLAPPED - Buyer: ${state.buyer_offer:.2f} >= Seller: ${state.seller_offer:.2f}. Deal at seller's price: ${final_price:.2f}")
                    print(f"Prices overlapped. Deal at ${final_price:.2f}")
                    packet["outcome"] = "deal"
                    packet["deal_closed"] = True
                    packet["final_price"] = final_price

                    midpoint = (seed.buyer_max + seed.seller_min) / 2
                    packet["buyer_favorable"] = final_price < midpoint
                    packet["seller_favorable"] = final_price > midpoint
                    break
                elif state.price_gap < 0.01 * seed.seller_list:  # Gap less than 1% of list price
                    # Price convergence detected - force an accept response on next turn
                    final_price = state.P_k  # Use midpoint
                    self.logger.sim_logger.info(f"[{sim_id}] Prices CONVERGED (gap: ${state.price_gap:.2f} < 1% of ${seed.seller_list}). Forcing accept on next turn at: ${final_price:.2f}")
                    print(f"Prices converged. Forcing accept on next turn at ${final_price:.2f}")

                    # Force accept on the next turn with favorable conditions
                    forced_response = "accept"
                    accept_price = final_price

                    # Store convergence price in packet for the accept handler to use
                    packet["convergence_price"] = final_price
                    self.logger.sim_logger.debug(f"Next turn will be forced ACCEPT at ${accept_price:.2f}")
                else:
                    self.logger.sim_logger.debug(f"[{sim_id}] Price gap: ${state.price_gap:.2f} (Buyer: ${state.buyer_offer:.2f} - Seller: ${state.seller_offer:.2f})")

            # Update state for next turn
            state.prev_s = control_features.get('s', 0)

        # If max turns reached without deal
        if packet["outcome"] is None:
            # Check if there's a pending forced accept response
            if forced_response == "accept":
                # Generate the final accept response before closing
                self.logger.sim_logger.info(f"[{sim_id}] Max turns reached with pending ACCEPT - generating final accept response")

                # Determine who's turn it would be
                is_next_buyer_turn = (len(packet["dialog"]) % 2 == 0)
                accept_price = state.get_active_price_for(not is_next_buyer_turn)  # Accept other party's price

                # Get the accept prompt
                accept_prompt = self.prompt_manager.get_action_prompt(
                    "accept",
                    buyer_personality if is_next_buyer_turn else seller_personality,
                    is_next_buyer_turn,
                    accept_price=accept_price,
                    buyer_max=seed.buyer_max,
                    seller_min=seed.seller_min
                )

                # Generate the accept response
                accept_response = self._generate_response(packet, state, accept_prompt)
                speaker = "BUYER" if is_next_buyer_turn else "SELLER"
                packet["dialog"].append((speaker, accept_response))

                # Mark deal as closed
                packet["outcome"] = "deal"
                packet["deal_closed"] = True
                packet["final_price"] = accept_price

                # Determine who got favorable deal
                midpoint = (seed.buyer_max + seed.seller_min) / 2
                packet["buyer_favorable"] = accept_price < midpoint
                packet["seller_favorable"] = accept_price > midpoint

                self.logger.sim_logger.info(f"[{sim_id}] Deal accepted at price: ${accept_price:.2f} (forced accept at max turns)")
                print(f"Deal accepted at price: ${accept_price:.2f} (pending accept completed)")
            else:
                packet["outcome"] = "no_deal"
                packet["deal_closed"] = False
                self.logger.sim_logger.info(f"[{sim_id}] Maximum turns ({self.max_turns}) reached without agreement")
                print("Maximum turns reached without agreement.")

        # Log final outcome
        self.logger.sim_logger.info(f"[{sim_id}] Negotiation complete - outcome: {packet['outcome']}, turns: {len(packet['dialog'])}, final_price: {packet.get('final_price', 'N/A')}")

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
        sim_id = packet.get("sim_id", "unknown")
        personality = packet["buyer_personality"] if is_buyer else packet["seller_personality"]

        # Evaluate control function
        evaluation = self.control_function.evaluate(state, is_buyer)

        # Get control features
        control_features = evaluation['features']

        # Get action probabilities
        action_probs = evaluation['action_probs']

        # Log control decision
        self.logger.log_control_decision(sim_id, state.r, {
            'features': control_features,
            'action_probs': action_probs,
            'recommended_action': evaluation.get('recommended_action')
        })

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

        self.logger.sim_logger.info(f"[{sim_id}] Selected action: {action} (prob: {action_probs.get(action, 0):.3f})")

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
                current_price=new_price if was_conceded else state.get_active_price_for(is_buyer)
            )

        elif action == "extra":
            # Map "extra" to "extras" for prompt manager
            prompt = self.prompt_manager.get_action_prompt(
                "extras", personality, is_buyer, price=state.get_active_price_for(is_buyer)
            )

        elif action == "accept":
            # Special handling for accept to determine favorability
            seed = packet["seed"]
            accept_price = state.get_active_price_for(not is_buyer)  # Accepting other party's price
            prompt = self.prompt_manager.get_action_prompt(
                "accept", personality, is_buyer,
                accept_price=accept_price,
                buyer_max=seed.buyer_max,
                seller_min=seed.seller_min
            )

        else:
            # Get prompt for other actions
            prompt = self.prompt_manager.get_action_prompt(
                action, personality, is_buyer, price=state.get_active_price_for(is_buyer)
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

    def save_results(self, output_base: Optional[str] = None, save_structured: bool = True):
        """
        Save completed negotiations with datetime labeling and organized structure.

        Args:
            output_base: Base path for output directory (default: data/runs/)
            save_structured: Whether to also save structured (non-flattened) version
        """
        self.logger.sim_logger.info("="*60)
        self.logger.sim_logger.info("Saving results...")

        # Generate timestamp for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"run_{timestamp}"
        self.logger.sim_logger.info(f"Run ID: {run_id}")

        # Determine output directory
        if output_base is None:
            # Use data/runs/ as default, relative to script location
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            output_base = os.path.join(project_root, "data", "runs")

        # Create run-specific directory
        run_dir = os.path.join(output_base, run_id)
        os.makedirs(run_dir, exist_ok=True)
        self.logger.sim_logger.info(f"Output directory: {run_dir}")

        # Save metadata about the run
        metadata = {
            "timestamp": timestamp,
            "run_id": run_id,
            "total_negotiations": len(self.completed_negotiations),
            "max_turns": self.max_turns,
            "model": self.llm_interface.config.model,
            "seed_category": self.seed_category,
            "seed_selection_method": self.seed_selection_method,
            "deals_completed": sum(1 for n in self.completed_negotiations if n.get("deal_closed")),
            "deals_failed": sum(1 for n in self.completed_negotiations if not n.get("deal_closed"))
        }

        self.logger.sim_logger.info(f"Run statistics - Total: {metadata['total_negotiations']}, Deals: {metadata['deals_completed']}, No-deals: {metadata['deals_failed']}")

        metadata_file = os.path.join(run_dir, "metadata.json")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        self.logger.sim_logger.debug(f"Saved metadata to {metadata_file}")

        # Save flattened dataset for training (JSONL format)
        flattened_file = os.path.join(run_dir, f"dataset_{timestamp}.jsonl")
        with open(flattened_file, 'w', encoding='utf-8') as f:
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
                f.write(json.dumps(dataset_entry, ensure_ascii=False) + '\n')

        # Save structured version if requested
        if save_structured:
            structured_file = os.path.join(run_dir, f"negotiations_structured_{timestamp}.json")
            structured_data = []

            for i, negotiation in enumerate(self.completed_negotiations):
                # Keep more detailed structure with full objects
                structured_entry = {
                    "negotiation_id": i + 1,
                    "seed": {
                        "id": negotiation["seed"].seed_id,
                        "category": negotiation["seed"].category,
                        "item": negotiation["seed"].item,
                        "seller_list": negotiation["seed"].seller_list,
                        "seller_min": negotiation["seed"].seller_min,
                        "buyer_max": negotiation["seed"].buyer_max,
                        "extras_available": negotiation["seed"].extras_available
                    },
                    "personalities": {
                        "buyer": {
                            "type": negotiation["buyer_personality"].__class__.__name__,
                            "traits": {
                                "alpha": negotiation["buyer_personality"].traits.alpha,
                                "beta": negotiation["buyer_personality"].traits.beta,
                                "eta": negotiation["buyer_personality"].traits.eta,
                                "p": negotiation["buyer_personality"].traits.p,
                                "temp": negotiation["buyer_personality"].traits.temp
                            }
                        },
                        "seller": {
                            "type": negotiation["seller_personality"].__class__.__name__,
                            "traits": {
                                "alpha": negotiation["seller_personality"].traits.alpha,
                                "beta": negotiation["seller_personality"].traits.beta,
                                "eta": negotiation["seller_personality"].traits.eta,
                                "p": negotiation["seller_personality"].traits.p,
                                "temp": negotiation["seller_personality"].traits.temp
                            }
                        }
                    },
                    "dialog": negotiation["dialog"],
                    "outcome": {
                        "status": negotiation.get("outcome", "unknown"),
                        "final_price": negotiation.get("final_price", None),
                        "deal_closed": negotiation.get("deal_closed", False),
                        "buyer_favorable": negotiation.get("buyer_favorable", False),
                        "seller_favorable": negotiation.get("seller_favorable", False),
                        "turn_count": len(negotiation["dialog"])
                    }
                }
                structured_data.append(structured_entry)

            with open(structured_file, 'w', encoding='utf-8') as f:
                json.dump(structured_data, f, indent=2, ensure_ascii=False)

        # Create a summary file
        summary_file = os.path.join(run_dir, f"summary_{timestamp}.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"=== Negotiation Run Summary ===\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Run ID: {run_id}\n")
            f.write(f"Total Negotiations: {metadata['total_negotiations']}\n")
            f.write(f"Deals Completed: {metadata['deals_completed']}\n")
            f.write(f"Deals Failed: {metadata['deals_failed']}\n")
            success_rate = (metadata['deals_completed']/metadata['total_negotiations']*100) if metadata['total_negotiations'] > 0 else 0
            f.write(f"Success Rate: {success_rate:.1f}%\n\n")

            if metadata['deals_completed'] > 0:
                prices = [n.get("final_price", 0) for n in self.completed_negotiations if n.get("deal_closed")]
                if prices:
                    avg_price = sum(prices) / len(prices)
                    f.write(f"Average Final Price: ${avg_price:.2f}\n")
                    f.write(f"Min Price: ${min(prices):.2f}\n")
                    f.write(f"Max Price: ${max(prices):.2f}\n")

        # Log saved files
        self.logger.sim_logger.info(f"Saved flattened dataset to {flattened_file}")
        if save_structured:
            self.logger.sim_logger.info(f"Saved structured data to negotiations_structured_{timestamp}.json")
        self.logger.sim_logger.info(f"Saved summary to summary_{timestamp}.txt")

        print(f"\n=== Save Results ===")
        print(f"Saved {len(self.completed_negotiations)} negotiations to: {run_dir}")
        print(f"  - Metadata: metadata.json")
        print(f"  - Training dataset: dataset_{timestamp}.jsonl")
        if save_structured:
            print(f"  - Structured data: negotiations_structured_{timestamp}.json")
        print(f"  - Summary: summary_{timestamp}.txt")

        self.logger.sim_logger.info("Results saved successfully!")
        self.logger.sim_logger.info("="*60)

        return run_dir  # Return the directory path for reference


def combine_runs(run_directories: List[str], output_file: Optional[str] = None):
    """
    Combine multiple simulation runs into a single dataset file.

    Args:
        run_directories: List of run directory paths to combine
        output_file: Output file path (default: data/combined_dataset_<timestamp>.jsonl)
    """
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        output_file = os.path.join(project_root, "data", f"combined_dataset_{timestamp}.jsonl")

    combined_entries = []
    total_negotiations = 0

    for run_dir in run_directories:
        # Find the dataset JSONL file in the run directory
        jsonl_files = [f for f in os.listdir(run_dir) if f.startswith("dataset_") and f.endswith(".jsonl")]

        if not jsonl_files:
            print(f"Warning: No dataset file found in {run_dir}")
            continue

        dataset_file = os.path.join(run_dir, jsonl_files[0])

        # Read and combine the data
        with open(dataset_file, 'r') as f:
            for line in f:
                combined_entries.append(line.strip())
                total_negotiations += 1

    # Save combined dataset
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, 'w') as f:
        for entry in combined_entries:
            f.write(entry + '\n')

    print(f"Combined {total_negotiations} negotiations from {len(run_directories)} runs")
    print(f"Saved to: {output_file}")

    return output_file


def main():
    """Main entry point for simulation runner."""
    import argparse

    # Initialize logger first
    logger = setup_logger()
    logger.sim_logger.info("="*60)
    logger.sim_logger.info("NEGOTIATION SIMULATION RUNNER STARTED")
    logger.sim_logger.info("="*60)

    parser = argparse.ArgumentParser(description="Run negotiation simulations")

    # Seed configuration options
    parser.add_argument("--seed-dir", type=str, default="seeds/",
                       help="Directory containing seed JSON files (default: seeds/)")
    parser.add_argument("--seed-file", type=str,
                       help="Use a specific seed file instead of directory")
    parser.add_argument("--seed-category", type=str,
                       help="Only use seeds from specific category")
    parser.add_argument("--seed-id", type=int,
                       help="Use a specific seed by ID (requires --seed-file)")
    parser.add_argument("--seed-selection", type=str, choices=["random", "sequential", "all"],
                       default="random",
                       help="How to select seeds: random (default), sequential, or all")

    # Simulation configuration
    parser.add_argument("--num-negotiations", type=int, default=2,
                       help="Number of negotiations to simulate (default: 2)")
    parser.add_argument("--max-turns", type=int, default=20,
                       help="Maximum turns per negotiation (default: 20)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-14B-Instruct",
                       help="Model name for LLM (default: Qwen/Qwen2.5-14B-Instruct)")

    # Output configuration
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Base output directory for runs (default: data/runs/)")
    parser.add_argument("--no-structured", action="store_true",
                       help="Skip saving structured version of data")

    args = parser.parse_args()

    # Handle seed-specific configuration
    seed_directory = args.seed_dir
    specific_seed_file = None
    if args.seed_file:
        # Resolve the seed file path
        if not os.path.dirname(args.seed_file):
            # If just a filename, look in the seed directory
            specific_seed_file = os.path.join(args.seed_dir, args.seed_file)
        else:
            # Full path provided
            specific_seed_file = args.seed_file

        # Check if file exists
        if not os.path.exists(specific_seed_file):
            print(f"Error: Seed file not found: {specific_seed_file}")
            return

    # Create simulation runner
    runner = SimulationRunner(
        seed_directory=seed_directory,
        model_name=args.model,
        max_turns=args.max_turns,
        seed_category=args.seed_category,
        seed_selection_method=args.seed_selection,
        specific_seed_file=specific_seed_file
    )

    # Handle specific seed selection
    specific_seed = None
    if args.seed_file and args.seed_id:
        # Load specific seed by ID from file
        from utils.seed_loader import SeedLoader
        single_loader = SeedLoader(args.seed_file)
        specific_seed = single_loader.get_seed_by_id(args.seed_id)
        if specific_seed:
            print(f"Using seed ID {args.seed_id}: {specific_seed.item}")
        else:
            print(f"Seed ID {args.seed_id} not found in {args.seed_file}")
            return

    # Run simulations
    start = time.time()
    print("Starting negotiation simulations...")
    print(f"Configuration:")
    if specific_seed_file:
        print(f"  - Seed file: {specific_seed_file} (only this file will be used)")
    else:
        print(f"  - Seed directory: {seed_directory} (loading ALL seed files)")
    print(f"  - Seed selection: {args.seed_selection}")
    if args.seed_category:
        print(f"  - Category filter: {args.seed_category}")
    print(f"  - Number of negotiations: {args.num_negotiations}")
    print(f"  - Max turns: {args.max_turns}")
    print(f"  - Model: {args.model}")

    results = runner.run_simulation(
        num_negotiations=args.num_negotiations,
        specific_seed=specific_seed
    )

    # Save results
    runner.save_results(
        output_base=args.output_dir,
        save_structured=not args.no_structured
    )
    
    end = time.time()
    runtime = round((end - start) / 60, 1)        # in minutes
    if runtime > 120:
        runtime = round(runtime / 60, 1)          # in hours
        runtime_string = f"{runtime} hrs"
    else:
        runtime_string = f"{runtime} min"

    logger.sim_logger.info("="*60)
    logger.sim_logger.info("SIMULATION COMPLETE")
    logger.sim_logger.info(f"Generated {len(results)} negotiations")

    print("\nSimulation complete!")
    print(f"Generated {len(results)} negotiations in {runtime_string}")

    # Print summary statistics
    deals_made = sum(1 for r in results if r.get("deal_closed"))
    print(f"Deals completed: {deals_made}/{len(results)}")
    logger.sim_logger.info(f"Final statistics - Deals: {deals_made}/{len(results)}")
    print(f"Average time per negotiation: {runtime / len(results)} min")

    if deals_made > 0:
        avg_price = sum(r.get("final_price", 0) for r in results if r.get("deal_closed")) / deals_made
        # print(f"Average final price: ${avg_price:.2f}")
        logger.sim_logger.info(f"Average final price: ${avg_price:.2f}")

    logger.sim_logger.info("="*60)


if __name__ == "__main__":
    main()