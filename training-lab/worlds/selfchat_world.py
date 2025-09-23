"""
SelfChatWorld Runner for negotiation training.
Manages dual OllamaNegotiator agents simulating negotiations between buyer and seller.
"""

import json
import time
import random
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from copy import deepcopy

import pandas as pd
import numpy as np
from parlai.core.worlds import World
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt

from training_parameters import MODEL_TEMP

from agents.ollama_wrapper import OllamaNegotiator
from agents.negotiator_base import NegotiationState, NegotiationAction
from tasks.selfplay_teacher import SelfPlayTeacher

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SelfChatWorld(World):
    """
    World for self-play negotiation simulations.
    
    Manages two OllamaNegotiator agents playing different sides (buyer/seller)
    with comprehensive metrics tracking and structured logging.
    """
    
    @classmethod
    def add_cmdline_args(cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None):
        """Add command-line arguments for SelfChatWorld."""
        group = parser.add_argument_group('SelfChat World Arguments')
        group.add_argument(
            '--num-episodes',
            type=int,
            default=1500,
            help='Target number of episodes to generate'
        )
        group.add_argument(
            '--batch-size',
            type=int,
            default=100,
            help='Episodes per output file'
        )
        group.add_argument(
            '--seed-mutation-rate',
            type=float,
            default=0.1,
            help='Probability of seed mutation'
        )
        group.add_argument(
            '--regression-ratio',
            type=float,
            default=0.1,
            help='Portion of seeds for regression testing'
        )
        group.add_argument(
            '--max-turns',
            type=int,
            default=20,
            help='Maximum turns per episode'
        )
        group.add_argument(
            '--output-dir',
            type=str,
            default='data/negotiations/selfchat/',
            help='Base directory for outputs'
        )
        group.add_argument(
            '--model-checkpoint',
            type=str,
            default=None,
            help='Path to LoRA checkpoint for fine-tuned model'
        )
        group.add_argument(
            '--temperature',
            type=float,
            default=MODEL_TEMP,
            help='Temperature for generation'
        )
        group.add_argument(
            '--save-metrics-every',
            type=int,
            default=50,
            help='Save metrics after this many episodes'
        )
        group.add_argument(
            '--parallel-episodes',
            type=int,
            default=4,
            help='Number of episodes to process in parallel (for GPU efficiency)'
        )
        group.add_argument(
            '--use-batch-inference',
            type=str,
            default='True',
            choices=['True', 'False', 'true', 'false'],
            help='Use batch inference for parallel episodes'
        )
        return parser
    
    def __init__(self, opt: Opt, agents=None, shared=None):
        """Initialize the SelfChatWorld."""
        super().__init__(opt)

        # Configuration
        self.target_episodes = opt.get('num_episodes', 1500)
        self.batch_size = opt.get('batch_size', 100)
        self.seed_mutation_rate = opt.get('seed_mutation_rate', 0.1)
        self.regression_ratio = opt.get('regression_ratio', 0.1)
        self.max_turns = opt.get('max_turns', 20)
        self.output_dir = Path(opt.get('output_dir', 'data/negotiations/selfchat/'))
        self.save_metrics_every = opt.get('save_metrics_every', 50)

        # Parallel processing configuration
        self.parallel_episodes = opt.get('parallel_episodes', 4)  # Process 4 episodes in parallel
        # Parse string boolean for batch inference
        batch_inference_str = opt.get('use_batch_inference', 'True')
        self.use_batch_inference = batch_inference_str.lower() == 'true'
        
        # Create output directories with timestamped transcripts folder
        self.episodes_dir = self.output_dir / 'episodes'
        self.metrics_dir = self.output_dir / 'metrics'
        self.seeds_dir = self.output_dir / 'seeds'

        # Create timestamped folder for this session's transcripts
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.transcripts_dir = self.output_dir / 'transcripts' / f'run_{timestamp}'

        for dir_path in [self.episodes_dir, self.metrics_dir, self.seeds_dir, self.transcripts_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize teacher for state seeds
        teacher_opt = deepcopy(opt)
        teacher_opt['negotiation_data_dir'] = str(self.output_dir)
        teacher_opt['selfplay_seed_file'] = 'state_seeds.jsonl'
        self.teacher = SelfPlayTeacher(teacher_opt, shared)
        
        # Load state seeds
        self.state_seeds = self.teacher.state_seeds
        self.regression_seeds = self.teacher.regression_seeds
        self.active_seed_idx = 0
        
        # Initialize agents if not provided
        if agents:
            self.agents = agents
        else:
            self.agents = self._create_agents(opt)
        
        # Episode tracking
        self.current_episode = None
        self.episode_count = 0
        self.batch_episodes = []
        self.current_batch = 0
        
        # Session metrics using DataFrames
        self.session_metrics = {
            'episode_df': pd.DataFrame(),
            'aggregate_stats': {},
            'session_id': f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'start_time': time.time()
        }
        
        # Protocol violations DataFrame
        self.violations_df = pd.DataFrame()
        
        logger.info(f"Initialized SelfChatWorld with {len(self.state_seeds)} seeds")
        logger.info(f"Regression seeds: {len(self.regression_seeds)}")
    
    def _create_agents(self, opt: Opt) -> List[OllamaNegotiator]:
        """Create buyer and seller agents."""
        agents = []
        shared = {}

        # Share model between agents for better memory efficiency
        # HF models can't truly parallelize on single GPU anyway

        for i, side in enumerate(['buyer', 'seller']):
            agent_opt = deepcopy(opt)
            agent_opt['id'] = f"negotiator_{side}"

            if opt.get('model_checkpoint'):
                agent_opt['lora_checkpoint'] = opt['model_checkpoint']

            # Share model between agents
            agent = OllamaNegotiator(agent_opt, shared)
            agent.side = side
            agents.append(agent)

            # After first agent, share its model
            if i == 0:
                shared = agent.share()

        return agents
    
    def load_state_seeds(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Load state seeds from NDJSON files.
        Returns (all_seeds, regression_seeds).
        """
        # Save regression seeds if not already saved
        regression_path = self.seeds_dir / 'regression_seeds.jsonl'
        if not regression_path.exists() and self.regression_seeds:
            with open(regression_path, 'w') as f:
                for seed in self.regression_seeds:
                    f.write(json.dumps(seed) + '\n')
            logger.info(f"Saved {len(self.regression_seeds)} regression seeds")
        
        # Save active seeds
        active_path = self.seeds_dir / 'active_seeds.jsonl'
        if not active_path.exists() and self.state_seeds:
            with open(active_path, 'w') as f:
                for seed in self.state_seeds:
                    f.write(json.dumps(seed) + '\n')
            logger.info(f"Saved {len(self.state_seeds)} active seeds")
        
        return self.state_seeds, self.regression_seeds
    
    def _select_seed(self) -> Tuple[Dict[str, Any], bool, bool]:
        """
        Select next seed for episode.
        Returns (seed, is_regression, was_mutated).
        """
        # 10% chance to use regression seed
        is_regression = bool(random.random() < 0.1 and len(self.regression_seeds) > 0)
        
        if is_regression:
            seed = random.choice(self.regression_seeds)
        else:
            # Cycle through seeds
            seed = self.state_seeds[self.active_seed_idx % len(self.state_seeds)]
            self.active_seed_idx += 1
        
        # Apply mutation if configured
        was_mutated = False
        if not is_regression and random.random() < self.seed_mutation_rate:
            seed = self.teacher._mutate_seed(seed)
            was_mutated = True
        
        return seed, is_regression, was_mutated
    
    def run_episode(self) -> Dict[str, Any]:
        """
        Execute a single negotiation episode.
        Returns complete episode data with metrics.
        """
        logger.info(f"[run_episode] Starting episode {self.episode_count}")
        # Select seed
        seed, is_regression, was_mutated = self._select_seed()
        logger.debug(f"[run_episode] Seed selected: regression={is_regression}, mutated={was_mutated}")
        
        # Initialize episode structure
        episode = {
            'episode_id': f"selfchat_ep_{self.episode_count:04d}",
            'dialogue_id': f"selfchat_d_{self.episode_count:04d}",
            'seed_id': f"seed_{self.active_seed_idx:03d}",
            'state': seed,
            'dialogue': [],
            'metrics': {},
            'meta': {
                'model_version': 'mistral-7b-v0.1',
                'lora_checkpoint': self.opt.get('model_checkpoint'),
                'temperature': self.opt.get('temperature', MODEL_TEMP),
                'is_regression_seed': is_regression,
                'was_mutated': was_mutated,
                'batch_id': f"batch_{self.current_batch:03d}",
                'session_id': self.session_metrics['session_id']
            }
        }
        
        # Initialize agents with seed state
        buyer, seller = self.agents
        buyer.initialize_from_state(seed, 'buyer')
        seller.initialize_from_state(seed, 'seller')
        
        # Reset agent states
        for agent in self.agents:
            agent.negotiation_history = []
            agent.negotiation_state = NegotiationState.PROPOSING
            agent.turn_count = 0
            agent.terms_buffer = None
            agent.metrics = {
                'turns': 0,
                'tokens_sent': 0,
                'tokens_received': 0,
                'protocol_violations': [],
                'action_history': []
            }
        
        # Start timer
        episode_start = time.time()
        
        # Generate opening from buyer
        opening = self.teacher._generate_opening_prompt(seed, 'buyer')
        if opening:
            episode['dialogue'].append({
                'turn': 0,
                'speaker': 'buyer',
                'text': opening,
                'action': 'PROPOSE' if '[PROPOSE]' in opening else None,
                'tokens': len(opening.split()),
                'timestamp': datetime.now().isoformat()
            })
            
            # Seller observes buyer's opening
            seller.observe({'text': opening, 'episode_done': False})
        
        # Run negotiation turns
        turn = 1
        max_turns = self.max_turns
        episode_done = False
        
        while turn < max_turns and not episode_done:
            # Determine current speaker (alternate turns)
            current_agent = seller if turn % 2 == 1 else buyer
            other_agent = buyer if turn % 2 == 1 else seller
            speaker = current_agent.side
            logger.debug(f"[run_episode] Turn {turn}: {speaker}'s turn")

            # Generate response
            logger.debug(f"[run_episode] Calling {speaker}.act()")
            response = current_agent.act()
            logger.debug(f"[run_episode] {speaker} response received")
            
            # Extract text and check if episode is done
            text = response.get('text', '')
            episode_done = response.get('episode_done', False)
            strategy = response.get('strategy', None)
            token_metrics = response.get('token_metrics', None)
            
            # REMOVED: Action parsing from text
            # Actions are now in backend metadata only, NOT in messages!
            # Legacy 'action' field kept for backwards compatibility
            action = response.get('backend_tag', response.get('action_tag', None))
            
            # Build turn log entry
            turn_entry = {
                'turn': turn,
                'speaker': speaker,
                'text': text,
                'action': action,  # Keep for backwards compat
                'tokens': len(text.split()),  # Keep for backwards compat
                'timestamp': datetime.now().isoformat(),
                # CRITICAL: Store backend tag from response metadata
                'backend_tag': response.get('backend_tag', response.get('action_tag', '')),
                'action_tag': response.get('action_tag', response.get('backend_tag', ''))
            }
            
            # Add two-part response data if available
            if strategy and token_metrics:
                # Strategy generation phase
                turn_entry['strategy_generation'] = {
                    'prompt': current_agent._build_strategy_prompt() if hasattr(current_agent, '_build_strategy_prompt') else '',
                    'output': strategy,
                    'prompt_tokens': token_metrics['strategy']['prompt_tokens'],
                    'output_tokens': token_metrics['strategy']['output_tokens'],
                    'duration_ms': token_metrics['strategy']['generation_time_ms']
                }
                
                # Response generation phase
                turn_entry['response_generation'] = {
                    'prompt': current_agent._build_response_prompt(strategy) if hasattr(current_agent, '_build_response_prompt') else '',
                    'output': text,
                    'prompt_tokens': token_metrics['response']['prompt_tokens'],
                    'output_tokens': token_metrics['response']['output_tokens'],
                    'duration_ms': token_metrics['response']['generation_time_ms']
                }
            
            # Log turn
            episode['dialogue'].append(turn_entry)
            
            # Other agent observes
            if not episode_done:
                other_agent.observe({'text': text, 'episode_done': episode_done})
            
            # Check for conclusion
            if episode_done or current_agent.negotiation_state in [
                NegotiationState.CONCLUDED, 
                NegotiationState.FAILED
            ]:
                break
            
            turn += 1
        
        # Calculate episode metrics
        episode_duration = time.time() - episode_start
        
        # Determine deal outcome
        deal_closed = False
        final_price = None
        terms_met_percentage = 0.0
        
        for agent in self.agents:
            if agent.negotiation_state == NegotiationState.CONCLUDED:
                deal_closed = True
                if agent.terms_buffer and 'total_price' in agent.terms_buffer:
                    final_price = agent.terms_buffer['total_price']
                break
        
        # Calculate terms met percentage if deal closed
        if deal_closed and final_price:
            buyer_target = seed['buyer_state'].get('target_price', 0)
            seller_min = seed['seller_state'].get('min_price', 0)
            
            if buyer_target and seller_min:
                # Calculate how well the price meets both parties' goals
                buyer_satisfaction = max(0, (seed['buyer_state']['max_price'] - final_price) / 
                                        (seed['buyer_state']['max_price'] - buyer_target))
                seller_satisfaction = max(0, (final_price - seller_min) / 
                                         (seed['item']['list_price'] - seller_min))
                terms_met_percentage = (buyer_satisfaction + seller_satisfaction) / 2 * 100
        
        # Aggregate protocol violations
        all_violations = []
        for agent in self.agents:
            all_violations.extend(agent.metrics.get('protocol_violations', []))
        
        # Calculate token metrics for two-part system
        strategy_tokens = 0
        response_tokens = 0
        total_tokens = buyer.metrics['tokens_sent'] + seller.metrics['tokens_sent']
        
        # If using two-part response, calculate separate metrics
        for turn_data in episode['dialogue']:
            if 'strategy_generation' in turn_data:
                strategy_tokens += turn_data['strategy_generation']['output_tokens']
            if 'response_generation' in turn_data:
                response_tokens += turn_data['response_generation']['output_tokens']
        
        # Fill episode metrics
        episode['metrics'] = {
            'turns': turn,
            'deal_closed': deal_closed,
            'final_price': final_price,
            'terms_met_percentage': terms_met_percentage,
            'buyer_tokens': buyer.metrics['tokens_sent'],
            'seller_tokens': seller.metrics['tokens_sent'],
            'total_tokens': total_tokens,
            # New breakdown for two-part system
            'strategy_tokens': strategy_tokens,
            'response_tokens': response_tokens,
            'protocol_violations': all_violations,
            'duration_seconds': episode_duration,
            'tokens_per_second': total_tokens / episode_duration if episode_duration > 0 else 0
        }
        
        return episode
    
    def run_parallel_episodes(self, num_episodes: int) -> List[Dict[str, Any]]:
        """
        Run multiple episodes in parallel using batch inference.
        This maximizes GPU utilization by processing multiple conversations simultaneously.
        """
        parallel_episodes = []
        episode_states = []

        # Initialize multiple episodes
        for i in range(num_episodes):
            if self.episode_count >= self.target_episodes:
                break

            # Select seed for each episode
            seed, is_regression, was_mutated = self._select_seed()

            # Initialize episode structure
            episode = {
                'episode_id': f"selfchat_ep_{self.episode_count + i:04d}",
                'dialogue_id': f"selfchat_d_{self.episode_count + i:04d}",
                'seed_id': f"seed_{self.active_seed_idx:03d}",
                'state': seed,
                'dialogue': [],
                'metrics': {},
                'meta': {
                    'model_version': 'mistral-7b-v0.1',
                    'lora_checkpoint': self.opt.get('model_checkpoint'),
                    'temperature': self.opt.get('temperature', MODEL_TEMP),
                    'is_regression_seed': is_regression,
                    'was_mutated': was_mutated,
                    'batch_id': f"batch_{self.current_batch:03d}",
                    'session_id': self.session_metrics['session_id']
                }
            }

            # Track episode state for parallel processing
            episode_state = {
                'episode': episode,
                'seed': seed,
                'turn': 0,
                'done': False,
                'buyer_history': [],
                'seller_history': []
            }

            parallel_episodes.append(episode)
            episode_states.append(episode_state)

        # Process all episodes in parallel for each turn
        for turn in range(self.max_turns):
            active_states = [s for s in episode_states if not s['done']]
            if not active_states:
                break

            # Collect prompts for batch processing
            buyer_prompts = []
            seller_prompts = []

            for state in active_states:
                # Build prompts for buyer and seller
                seed = state['seed']

                # Buyer prompt
                buyer_prompt = self._build_parallel_prompt(
                    seed, 'buyer', state['buyer_history'], turn
                )
                buyer_prompts.append(buyer_prompt)

                # Seller prompt
                seller_prompt = self._build_parallel_prompt(
                    seed, 'seller', state['seller_history'], turn
                )
                seller_prompts.append(seller_prompt)

            # Batch generate responses for all active episodes
            if self.use_batch_inference and hasattr(self.agents[0], '_batch_generate'):
                # Use batch inference for maximum GPU efficiency
                all_prompts = buyer_prompts + seller_prompts
                all_responses = self.agents[0]._batch_generate(
                    all_prompts,
                    temperature=self.opt.get('temperature', MODEL_TEMP),
                    max_tokens=256
                )

                buyer_responses = all_responses[:len(buyer_prompts)]
                seller_responses = all_responses[len(buyer_prompts):]
            else:
                # Fallback to sequential generation
                buyer_responses = []
                seller_responses = []
                for prompt in buyer_prompts:
                    response, _, _ = self.agents[0]._generate_response(prompt)
                    buyer_responses.append(response)
                for prompt in seller_prompts:
                    response, _, _ = self.agents[1]._generate_response(prompt)
                    seller_responses.append(response)

            # Update episode states with responses
            for i, state in enumerate(active_states):
                buyer_text = buyer_responses[i]
                seller_text = seller_responses[i]

                # Add to dialogue
                state['episode']['dialogue'].append({
                    'turn': turn * 2,
                    'speaker': 'buyer',
                    'text': buyer_text,
                    'action': 'PROPOSE',
                    'tokens': len(buyer_text.split()),
                    'timestamp': datetime.now().isoformat()
                })

                state['episode']['dialogue'].append({
                    'turn': turn * 2 + 1,
                    'speaker': 'seller',
                    'text': seller_text,
                    'action': 'PROPOSE',
                    'tokens': len(seller_text.split()),
                    'timestamp': datetime.now().isoformat()
                })

                # Update histories
                state['buyer_history'].append(buyer_text)
                state['seller_history'].append(seller_text)

                # Check if episode is done
                if self._check_parallel_episode_done(buyer_text, seller_text, turn):
                    state['done'] = True

        # Finalize episodes
        completed_episodes = []
        for state in episode_states:
            episode = state['episode']
            self._finalize_parallel_episode(episode)
            completed_episodes.append(episode)

        return completed_episodes

    def _build_parallel_prompt(self, seed: Dict, side: str, history: List[str], turn: int) -> str:
        """Build prompt for parallel episode processing."""
        prompt_parts = []

        # Role and context
        prompt_parts.append(f"You are a {side} in a negotiation.")

        # Item details
        item = seed.get('item', {})
        prompt_parts.append(f"Item: {item.get('name', 'Unknown')}")
        prompt_parts.append(f"List Price: ${item.get('list_price', 0)}")

        # Side-specific constraints
        if side == 'buyer':
            buyer_state = seed.get('buyer_state', {})
            prompt_parts.append(f"Your maximum price: ${buyer_state.get('max_price', 0)}")
            prompt_parts.append(f"Your target price: ${buyer_state.get('target_price', 0)}")
        else:
            seller_state = seed.get('seller_state', {})
            prompt_parts.append(f"Your minimum price: ${seller_state.get('min_price', 0)}")

        # Add history if available
        if history:
            prompt_parts.append("\nPrevious messages:")
            for msg in history[-3:]:  # Last 3 messages
                prompt_parts.append(f"- {msg}")

        prompt_parts.append(f"\nGenerate your response for turn {turn + 1}:")

        return "\n".join(prompt_parts)

    def _check_parallel_episode_done(self, buyer_text: str, seller_text: str, turn: int) -> bool:
        """Check if a parallel episode should end."""
        # Check for deal keywords
        combined_text = (buyer_text + " " + seller_text).lower()
        if any(word in combined_text for word in ['deal accepted', 'agreed', 'no deal']):
            return True

        # Check turn limit
        if turn >= self.max_turns - 1:
            return True

        return False

    def _finalize_parallel_episode(self, episode: Dict):
        """Finalize a parallel episode with metrics."""
        dialogue = episode['dialogue']
        total_turns = len(dialogue) // 2 if dialogue else 0

        # Calculate simple metrics
        episode['metrics'] = {
            'turns': total_turns,
            'deal_closed': False,  # Simplified for now
            'final_price': None,
            'terms_met_percentage': 0.0,
            'buyer_tokens': sum(d['tokens'] for d in dialogue if d['speaker'] == 'buyer'),
            'seller_tokens': sum(d['tokens'] for d in dialogue if d['speaker'] == 'seller'),
            'total_tokens': sum(d['tokens'] for d in dialogue),
            'protocol_violations': [],
            'duration_seconds': 0,  # Would need timing
            'tokens_per_second': 0
        }

    def parley(self):
        """
        Execute episodes of self-chat negotiation.
        Override of ParlAI World method to support parallel processing.
        """
        if self.episode_count >= self.target_episodes:
            return

        if self.use_batch_inference and self.parallel_episodes > 1:
            # Run multiple episodes in parallel
            episodes_to_run = min(self.parallel_episodes, self.target_episodes - self.episode_count)
            episodes = self.run_parallel_episodes(episodes_to_run)

            # Process all completed episodes
            for episode in episodes:
                self.batch_episodes.append(episode)
                self.log_episode(episode)

                # Update session metrics DataFrame for each episode
                episode_metrics = {
                    'episode_id': episode['episode_id'],
                    'dialogue_id': episode['dialogue_id'],
                    'seed_id': episode['seed_id'],
                    'turns': episode['metrics']['turns'],
                    'deal_closed': episode['metrics']['deal_closed'],
                    'final_price': episode['metrics']['final_price'],
                    'terms_met_percentage': episode['metrics']['terms_met_percentage'],
                    'buyer_tokens': episode['metrics']['buyer_tokens'],
                    'seller_tokens': episode['metrics']['seller_tokens'],
                    'total_tokens': episode['metrics']['total_tokens'],
                    'protocol_violations': len(episode['metrics']['protocol_violations']),
                    'duration_seconds': episode['metrics']['duration_seconds'],
                    'tokens_per_second': episode['metrics']['tokens_per_second'],
                    'is_regression_seed': episode['meta']['is_regression_seed'],
                    'was_mutated': episode['meta']['was_mutated'],
                    'batch_id': episode['meta']['batch_id'],
                    'timestamp': datetime.now().isoformat()
                }

                # Append to DataFrame
                self.session_metrics['episode_df'] = pd.concat([
                    self.session_metrics['episode_df'],
                    pd.DataFrame([episode_metrics])
                ], ignore_index=True)

                self.episode_count += 1

                # Save batch if full
                if len(self.batch_episodes) >= self.batch_size:
                    self.save_batch()
                    self.batch_episodes = []
                    self.current_batch += 1

            # Save metrics periodically
            if self.episode_count % self.save_metrics_every == 0:
                self.save_session_metrics()
        else:
            # Original single episode logic
            episode = self.run_episode()
            self.batch_episodes.append(episode)

            # Update session metrics DataFrame for single episode
            episode_metrics = {
                'episode_id': episode['episode_id'],
                'dialogue_id': episode['dialogue_id'],
                'seed_id': episode['seed_id'],
                'turns': episode['metrics']['turns'],
                'deal_closed': episode['metrics']['deal_closed'],
                'final_price': episode['metrics']['final_price'],
                'terms_met_percentage': episode['metrics']['terms_met_percentage'],
                'buyer_tokens': episode['metrics']['buyer_tokens'],
                'seller_tokens': episode['metrics']['seller_tokens'],
                'total_tokens': episode['metrics']['total_tokens'],
                'protocol_violations': len(episode['metrics']['protocol_violations']),
                'duration_seconds': episode['metrics']['duration_seconds'],
                'tokens_per_second': episode['metrics']['tokens_per_second'],
                'is_regression_seed': episode['meta']['is_regression_seed'],
                'was_mutated': episode['meta']['was_mutated'],
                'batch_id': episode['meta']['batch_id'],
                'timestamp': datetime.now().isoformat()
            }

            # Append to DataFrame
            self.session_metrics['episode_df'] = pd.concat([
                self.session_metrics['episode_df'],
                pd.DataFrame([episode_metrics])
            ], ignore_index=True)

            # Track violations separately
            if episode['metrics']['protocol_violations']:
                for violation in episode['metrics']['protocol_violations']:
                    violation_record = {
                        'episode_id': episode['episode_id'],
                        'violation_type': violation,
                        'timestamp': datetime.now().isoformat()
                    }
                    self.violations_df = pd.concat([
                        self.violations_df,
                        pd.DataFrame([violation_record])
                    ], ignore_index=True)

            # Log episode
            self.log_episode(episode)

            # Save batch if full
            if len(self.batch_episodes) >= self.batch_size:
                self.save_batch()
                self.batch_episodes = []
                self.current_batch += 1

            # Save metrics periodically
            if self.episode_count % self.save_metrics_every == 0:
                self.save_session_metrics()

            self.episode_count += 1

            # Log progress
            if self.episode_count % 10 == 0:
                logger.info(f"Completed {self.episode_count}/{self.target_episodes} episodes")
    
    def episode_done(self):
        """Check if we've generated enough episodes."""
        return self.episode_count >= self.target_episodes
    
    def _save_transcript(self, episode: Dict[str, Any]):
        """Save human-readable transcript of the episode."""
        # Ensure transcript directory exists (in case of parallel execution)
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)
        transcript_file = self.transcripts_dir / f"{episode['episode_id']}_transcript.txt"
        
        with open(transcript_file, 'w') as f:
            # Write header
            f.write("=" * 80 + "\n")
            f.write(f"SELFCHAT NEGOTIATION TRANSCRIPT\n")
            f.write(f"Episode ID: {episode['episode_id']}\n")
            f.write(f"Dialogue ID: {episode['dialogue_id']}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")
            
            # Write scenario info
            meta = episode['meta']
            f.write(f"SCENARIO:\n")
            f.write(f"  Seed ID: {meta.get('seed_id', 'Unknown')}\n")
            f.write(f"  Is Regression Seed: {meta.get('is_regression_seed', False)}\n")
            
            # Write item info
            state = episode['state']
            item = state.get('item', {})
            f.write(f"\nITEM:\n")
            f.write(f"  Name: {item.get('name', 'Unknown')}\n")
            f.write(f"  List Price: ${item.get('list_price', 0)}\n")
            
            # Write negotiation parameters
            f.write(f"\nNEGOTIATION PARAMETERS:\n")
            if 'buyer_state' in state:
                buyer = state['buyer_state']
                f.write(f"  Buyer Max Price: ${buyer.get('max_price', 0)}\n")
                f.write(f"  Buyer Target Price: ${buyer.get('target_price', 0)}\n")
            if 'seller_state' in state:
                seller = state['seller_state']
                f.write(f"  Seller Min Price: ${seller.get('min_price', 0)}\n")
                f.write(f"  Seller Target Price: ${seller.get('target_price', 0)}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("DIALOGUE:\n")
            f.write("=" * 80 + "\n\n")
            
            # Write dialogue
            for turn in episode['dialogue']:
                speaker = turn['speaker'].upper()
                action = f"[{turn['action']}]" if turn['action'] else ""
                f.write(f"Turn {turn['turn']} - {speaker} {action}:\n")
                f.write(f"{turn['text']}\n")
                f.write("-" * 40 + "\n\n")
            
            # Write outcome
            metrics = episode['metrics']
            f.write("=" * 80 + "\n")
            f.write("OUTCOME:\n")
            f.write("=" * 80 + "\n")
            f.write(f"  Deal Closed: {'Yes' if metrics['deal_closed'] else 'No'}\n")
            if metrics.get('final_price'):
                f.write(f"  Final Price: ${metrics['final_price']}\n")
            f.write(f"  Total Turns: {metrics['turns']}\n")
            f.write(f"  Duration: {metrics['duration_seconds']:.2f} seconds\n")
            f.write(f"  Total Tokens: {metrics['total_tokens']}\n")
            
            # Write action distribution
            if metrics.get('action_distribution'):
                f.write(f"\nAction Distribution:\n")
                for action, count in metrics['action_distribution'].items():
                    f.write(f"  {action}: {count}\n")
            
            # Write protocol violations if any
            if metrics.get('protocol_violations'):
                f.write(f"\nProtocol Violations: {metrics['protocol_violations']}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF TRANSCRIPT\n")
            f.write("=" * 80 + "\n")
    
    def log_episode(self, episode: Dict[str, Any]):
        """Write episode to JSONL file."""
        # Save transcript first
        self._save_transcript(episode)
        
        # Immediate write to current batch file
        batch_file = self.episodes_dir / f"selfchat_batch_{self.current_batch:03d}.jsonl"
        
        with open(batch_file, 'a') as f:
            f.write(json.dumps(episode) + '\n')
    
    def save_batch(self):
        """Save current batch of episodes."""
        if not self.batch_episodes:
            return
        
        batch_file = self.episodes_dir / f"selfchat_batch_{self.current_batch:03d}_complete.jsonl"
        
        with open(batch_file, 'w') as f:
            for episode in self.batch_episodes:
                f.write(json.dumps(episode) + '\n')
        
        logger.info(f"Saved batch {self.current_batch} with {len(self.batch_episodes)} episodes")
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate aggregate metrics for the session using pandas."""
        df = self.session_metrics['episode_df']
        
        if df.empty:
            return {}
        
        # Basic statistics
        total_episodes = len(df)
        deals_closed = df['deal_closed'].sum()
        
        aggregate = {
            'total_episodes': total_episodes,
            'deals_closed': int(deals_closed),
            'deal_closure_rate': deals_closed / total_episodes,
            'avg_turns': df['turns'].mean(),
            'std_turns': df['turns'].std(),
            'avg_buyer_tokens': df['buyer_tokens'].mean(),
            'avg_seller_tokens': df['seller_tokens'].mean(),
            'avg_total_tokens': df['total_tokens'].mean(),
            'avg_duration': df['duration_seconds'].mean(),
            'avg_tokens_per_second': df['tokens_per_second'].mean(),
            'total_violations': df['protocol_violations'].sum(),
            'episodes_with_violations': (df['protocol_violations'] > 0).sum(),
            'session_duration': time.time() - self.session_metrics['start_time']
        }
        
        # Statistics for closed deals only
        closed_deals = df[df['deal_closed'] == True]
        if not closed_deals.empty:
            aggregate['avg_terms_met'] = closed_deals['terms_met_percentage'].mean()
            aggregate['std_terms_met'] = closed_deals['terms_met_percentage'].std()
            
            # Price statistics
            price_data = closed_deals['final_price'].dropna()
            if not price_data.empty:
                aggregate['avg_final_price'] = price_data.mean()
                aggregate['std_final_price'] = price_data.std()
                aggregate['min_final_price'] = price_data.min()
                aggregate['max_final_price'] = price_data.max()
                aggregate['median_final_price'] = price_data.median()
        
        # Regression seed performance
        if 'is_regression_seed' in df.columns:
            regression_episodes = df[df['is_regression_seed'] == True]
            if not regression_episodes.empty:
                aggregate['regression_deal_rate'] = regression_episodes['deal_closed'].mean()
                aggregate['regression_avg_turns'] = regression_episodes['turns'].mean()
        
        return aggregate
    
    def save_session_metrics(self):
        """Save session metrics to CSV files using pandas."""
        # Save per-episode metrics
        if not self.session_metrics['episode_df'].empty:
            episode_csv = self.metrics_dir / 'episode_metrics.csv'
            self.session_metrics['episode_df'].to_csv(episode_csv, index=False)
            logger.info(f"Saved episode metrics to {episode_csv}")
        
        # Calculate and save aggregate stats
        aggregates = self.calculate_metrics()
        self.session_metrics['aggregate_stats'] = aggregates
        
        if aggregates:
            session_csv = self.metrics_dir / 'session_stats.csv'
            pd.DataFrame([aggregates]).to_csv(session_csv, index=False)
            logger.info(f"Saved session stats to {session_csv}")
        
        # Save protocol violations if any
        if not self.violations_df.empty:
            violations_csv = self.metrics_dir / 'protocol_violations.csv'
            
            # Add violation statistics
            violation_stats = self.violations_df.groupby('violation_type').size().reset_index(name='count')
            violation_stats.to_csv(violations_csv, index=False)
            
            # Also save detailed violations
            detailed_violations_csv = self.metrics_dir / 'protocol_violations_detailed.csv'
            self.violations_df.to_csv(detailed_violations_csv, index=False)
            logger.info(f"Saved violation data to {violations_csv}")
        
        logger.info(f"Metrics saved after {len(self.session_metrics['episode_df'])} episodes")
    
    def aggregate_session_stats(self):
        """Generate final session summary with pandas analysis."""
        aggregates = self.calculate_metrics()
        
        # Print summary
        print("\n" + "="*50)
        print("SELFCHAT SESSION SUMMARY")
        print("="*50)
        print(f"Session ID: {self.session_metrics['session_id']}")
        print(f"Total Episodes: {aggregates.get('total_episodes', 0)}")
        print(f"Deals Closed: {aggregates.get('deals_closed', 0)} ({aggregates.get('deal_closure_rate', 0)*100:.1f}%)")
        print(f"Average Turns: {aggregates.get('avg_turns', 0):.1f} (±{aggregates.get('std_turns', 0):.1f})")
        print(f"Average Total Tokens: {aggregates.get('avg_total_tokens', 0):.1f}")
        print(f"  - Buyer: {aggregates.get('avg_buyer_tokens', 0):.1f}")
        print(f"  - Seller: {aggregates.get('avg_seller_tokens', 0):.1f}")
        print(f"Average Duration: {aggregates.get('avg_duration', 0):.2f}s")
        print(f"Average Throughput: {aggregates.get('avg_tokens_per_second', 0):.1f} tokens/s")
        print(f"Protocol Violations: {aggregates.get('total_violations', 0)} in {aggregates.get('episodes_with_violations', 0)} episodes")
        
        if aggregates.get('avg_final_price'):
            print(f"\nPrice Statistics (closed deals):")
            print(f"  Average: ${aggregates['avg_final_price']:.2f} (±${aggregates.get('std_final_price', 0):.2f})")
            print(f"  Median: ${aggregates.get('median_final_price', 0):.2f}")
            print(f"  Range: ${aggregates.get('min_final_price', 0):.2f} - ${aggregates.get('max_final_price', 0):.2f}")
            print(f"  Terms Met: {aggregates.get('avg_terms_met', 0):.1f}% (±{aggregates.get('std_terms_met', 0):.1f}%)")
        
        if aggregates.get('regression_deal_rate') is not None:
            print(f"\nRegression Seed Performance:")
            print(f"  Deal Rate: {aggregates['regression_deal_rate']*100:.1f}%")
            print(f"  Avg Turns: {aggregates.get('regression_avg_turns', 0):.1f}")
        
        print(f"\nSession Duration: {aggregates.get('session_duration', 0)/60:.1f} minutes")
        print("="*50 + "\n")
        
        # Generate detailed analysis DataFrame
        if not self.session_metrics['episode_df'].empty:
            df = self.session_metrics['episode_df']
            
            # Turn distribution analysis
            turn_dist = df['turns'].describe()
            print("\nTurn Distribution:")
            print(turn_dist)
            
            # Deal closure analysis by batch
            if 'batch_id' in df.columns:
                batch_performance = df.groupby('batch_id').agg({
                    'deal_closed': 'mean',
                    'turns': 'mean',
                    'total_tokens': 'mean'
                }).round(2)
                print("\nPerformance by Batch:")
                print(batch_performance)
        
        return aggregates
    
    def _convert_to_native(self, obj):
        """Convert NumPy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_native(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    def shutdown(self):
        """Clean up and save final data."""
        # Save any remaining episodes
        if self.batch_episodes:
            self.save_batch()
        
        # Save final metrics
        self.save_session_metrics()
        
        # Generate summary
        self.aggregate_session_stats()
        
        # Export final analysis report
        if not self.session_metrics['episode_df'].empty:
            report_path = self.metrics_dir / f"session_report_{self.session_metrics['session_id']}.json"
            report = {
                'session_id': self.session_metrics['session_id'],
                'configuration': {
                    'num_episodes': self.target_episodes,
                    'max_turns': self.max_turns,
                    'seed_mutation_rate': self.seed_mutation_rate,
                    'regression_ratio': self.regression_ratio,
                    'temperature': self.opt.get('temperature', MODEL_TEMP),
                    'model_checkpoint': self.opt.get('model_checkpoint')
                },
                'summary_statistics': self._convert_to_native(self.session_metrics['aggregate_stats']),
                'timestamp': datetime.now().isoformat()
            }
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Saved session report to {report_path}")
        
        logger.info("SelfChatWorld shutdown complete")