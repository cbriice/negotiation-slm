"""
DialogPartnerWorld for negotiation training with RuleAgent or Human partners.
Integrates with rule_opponent_teacher for scenario management.
"""

import json
import time
import random
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from copy import deepcopy
import numpy as np

import pandas as pd
from parlai.core.worlds import World
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
from parlai.core.agents import create_agent

from training_parameters import MODEL_TEMP, USE_TWO_PART_RESPONSE
from agents.ollama_wrapper import OllamaNegotiator
from agents.rule_agent import RuleAgent
from agents.negotiator_base import NegotiationState, NegotiationAction
from tasks.rule_opponent_teacher import RuleOpponentTeacher
from worlds.dialogue_flow import DialogueFlowManager

# Set up logging - ensure logs directory exists
Path('logs').mkdir(exist_ok=True)

# Configure logging to save to file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/agent_logs.log', mode='a')  # Only log to file
    ],
    force=True  # Force reconfiguration if already configured
)
logger = logging.getLogger(__name__)


class DialogPartnerWorld(World):
    """
    World for negotiation training with dialog partners.
    
    Supports two modes:
    1. RuleAgent mode: Train against rule-based agents with personas
    2. Human mode: Interactive negotiation with human via console
    
    Uses two-part dialogue flow (strategy → response) with proper action management.
    """
    
    @classmethod
    def add_cmdline_args(cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None):
        """Add command-line arguments for DialogPartnerWorld."""
        group = parser.add_argument_group('DialogPartner World Arguments')
        group.add_argument(
            '--partner-mode',
            type=str,
            default='rule',
            choices=['rule', 'human'],
            help='Partner type: rule-based agent or human console'
        )
        group.add_argument(
            '--num-episodes',
            type=int,
            default=2000,
            help='Target number of episodes (rule mode only)'
        )
        group.add_argument(
            '--side-split',
            type=float,
            default=0.5,
            help='Fraction of episodes where agent plays buyer (0.5 = 50/50 split)'
        )
        group.add_argument(
            '--batch-size',
            type=int,
            default=100,
            help='Episodes per output file'
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
            default='data/negotiations/dialog_partner/',
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
            '--use-two-part-response',
            type=lambda x: x.lower() in ['true', '1', 'yes'],
            default=USE_TWO_PART_RESPONSE,
            help='Use two-part strategy/response generation. Accepts: true/false, 1/0, yes/no'
        )
        group.add_argument(
            '--enforce-strict-protocol',
            type=bool,
            default=True,
            help='Enforce strict protocol rules (rule mode)'
        )
        group.add_argument(
            '--save-metrics-every',
            type=int,
            default=50,
            help='Save metrics after this many episodes'
        )
        
        # Add RuleAgent arguments
        RuleAgent.add_cmdline_args(parser, partial_opt)
        
        # Add RuleOpponentTeacher arguments
        RuleOpponentTeacher.add_cmdline_args(parser, partial_opt)
        
        return parser
    
    def __init__(self, opt: Opt, agents=None, shared=None):
        """Initialize the DialogPartnerWorld."""
        super().__init__(opt)
        
        # Configuration
        self.partner_mode = opt.get('partner_mode', 'rule')
        self.target_episodes = opt.get('num_episodes', 2000)
        self.side_split = opt.get('side_split', 0.5)
        self.batch_size = opt.get('batch_size', 100)
        self.max_turns = opt.get('max_turns', 20)
        self.output_dir = Path(opt.get('output_dir', 'data/negotiations/dialog_partner/'))
        self.use_two_part = opt.get('use_two_part_response', USE_TWO_PART_RESPONSE)
        self.enforce_strict = opt.get('enforce_strict_protocol', True)
        self.save_metrics_every = opt.get('save_metrics_every', 50)
        
        # Create output directories with timestamped transcripts folder
        self.episodes_dir = self.output_dir / 'episodes'
        self.metrics_dir = self.output_dir / 'metrics'

        # Create timestamped folder for this session's transcripts
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.transcripts_dir = self.output_dir / 'transcripts' / f'run_{timestamp}'

        for dir_path in [self.episodes_dir, self.metrics_dir, self.transcripts_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize dialogue flow manager
        self.dialogue_flow = DialogueFlowManager()
        
        # Initialize teacher for scenarios and personas
        teacher_opt = deepcopy(opt)
        teacher_opt['negotiation_data_dir'] = str(self.output_dir)
        self.teacher = RuleOpponentTeacher(teacher_opt, shared)
        
        # Initialize agents
        self._initialize_agents(opt, agents)
        
        # Episode tracking
        self.current_episode = None
        self.episode_count = 0
        self.batch_episodes = []
        self.current_batch = 0
        
        # Metrics tracking
        self.session_metrics = {
            'episode_df': pd.DataFrame(),
            'aggregate_stats': {},
            'session_id': f"dialog_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'persona_stats': {}
        }
        
        logger.info(f"DialogPartnerWorld initialized in {self.partner_mode} mode")
        logger.info(f"Target episodes: {self.target_episodes}, Batch size: {self.batch_size}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Two-part response: {self.use_two_part}, Strict protocol: {self.enforce_strict}")
    
    def _initialize_agents(self, opt: Opt, agents=None):
        """Initialize negotiation and partner agents."""
        logger.info("Initializing agents...")
        if agents:
            self.agents = agents
            logger.info(f"Using provided agents: {[type(a).__name__ for a in agents]}")
        else:
            # Create negotiation agent (model being trained)
            negotiator_opt = deepcopy(opt)
            negotiator_opt['model'] = 'agents.ollama_wrapper:OllamaNegotiator'
            negotiator_opt['temperature'] = opt.get('temperature', MODEL_TEMP)
            negotiator_opt['use_two_part_response'] = self.use_two_part
            negotiator_opt['model_checkpoint'] = opt.get('model_checkpoint')
            logger.info(f"Creating OllamaNegotiator with temperature={negotiator_opt.get('temperature')}")
            self.negotiator = OllamaNegotiator(negotiator_opt)
            logger.info(f"OllamaNegotiator initialized successfully")
            
            # Create partner agent based on mode
            logger.info(f"Creating partner agent in {self.partner_mode} mode")
            if self.partner_mode == 'rule':
                partner_opt = deepcopy(opt)
                partner_opt['strict_rules'] = self.enforce_strict
                partner_opt['randomize_each_episode'] = True
                self.partner = RuleAgent(partner_opt)
            else:  # human mode
                # Create console agent for human interaction
                from parlai.core.agents import create_agent
                partner_opt = deepcopy(opt)
                partner_opt['model'] = 'local_human'
                self.partner = create_agent(partner_opt, requireModelExists=False)
            
            self.agents = [self.negotiator, self.partner]
    
    def _select_scenario(self) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
        """
        Select next scenario from teacher.
        Returns (state, agent_side, persona).
        """
        logger.debug("Selecting scenario for new episode")
        # Determine which side the agent plays
        agent_plays_buyer = random.random() < self.side_split
        agent_side = 'buyer' if agent_plays_buyer else 'seller'
        logger.debug(f"Agent will play: {agent_side}")
        
        # Get persona for rule agent
        persona = self.teacher._select_next_persona()
        logger.debug(f"Selected persona: {persona.get('name', 'Unknown') if persona else 'None'}")
        
        # Get state seed from teacher
        obs, _ = next(self.teacher.setup_data())
        scenario = obs.get('scenario', {})
        state = scenario.get('state', {})
        logger.debug(f"Got scenario with state keys: {list(state.keys()) if state else 'Empty'}")
        
        if not state:
            # Fallback to default state
            state = {
                'category': 'general',
                'item': {'name': 'Product', 'list_price': 1000},
                'buyer_state': {
                    'role': 'buyer',
                    'max_price': 1100,
                    'target_price': 900
                },
                'seller_state': {
                    'role': 'seller',
                    'min_price': 800,
                    'target_price': 1000
                }
            }
        
        return state, agent_side, persona
    
    def _generate_strategy_context(self, agent: OllamaNegotiator, state: Dict[str, Any], 
                                  history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate context for strategy generation phase."""
        return {
            'side': agent.side,
            'state': state,
            'history': history,
            'dialogue_id': self.current_episode['dialogue_id'] if self.current_episode else 'unknown',
            'turn': agent.turn_count
        }
    
    def _process_agent_strategy_response(self, agent: Any) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Process response flow for any agent type.
        Returns (strategy, response_text) where strategy is None for non-OllamaNegotiator agents.
        """
        # Check if this is an OllamaNegotiator with two-part flow capability
        if hasattr(agent, 'generate_strategy') and hasattr(agent, 'generate_response'):
            # OllamaNegotiator with two-part flow
            strategy = agent.generate_strategy()
            text = agent.generate_response(strategy)
            return strategy, text
        else:
            # RuleAgent or other agent type - use standard act() method
            response = agent.act()
            text = response.get('text', '')
            return None, text
    
    def run_episode(self) -> Dict[str, Any]:
        """
        Execute a single negotiation episode.
        Returns complete episode data with metrics.
        """
        logger.info(f"Starting episode {self.episode_count + 1}")
        # Select scenario
        state, agent_side, persona = self._select_scenario()
        partner_side = 'seller' if agent_side == 'buyer' else 'buyer'
        
        # Initialize episode structure
        episode = {
            'episode_id': f"dialog_ep_{self.episode_count:04d}",
            'dialogue_id': f"dialog_d_{self.episode_count:04d}",
            'state': state,
            'dialogue': [],
            'strategies': [],  # Track strategies separately
            'metrics': {},
            'meta': {
                'model_version': 'mistral-7b-v0.1',
                'lora_checkpoint': self.opt.get('model_checkpoint'),
                'temperature': self.opt.get('temperature', MODEL_TEMP),
                'partner_mode': self.partner_mode,
                'agent_side': agent_side,
                'partner_side': partner_side,
                'batch_id': f"batch_{self.current_batch:03d}",
                'session_id': self.session_metrics['session_id']
            }
        }
        
        # Add persona info for rule mode
        if self.partner_mode == 'rule':
            episode['meta']['partner_persona'] = str(persona.get('type', 'unknown'))
        
        self.current_episode = episode
        
        # Initialize agents with state
        logger.debug(f"Initializing negotiator as {agent_side}")
        self.negotiator.initialize_from_state(state, agent_side)
        
        if self.partner_mode == 'rule':
            logger.debug(f"Initializing partner as {partner_side}")
            self.partner.reset_episode()
            self.partner.initialize_from_state(state, partner_side)
        
        # Start timer
        episode_start = time.time()
        
        # Generate opening (buyer always starts)
        opening_speaker = self.negotiator if agent_side == 'buyer' else self.partner
        other_agent = self.partner if agent_side == 'buyer' else self.negotiator
        logger.debug(f"Opening speaker: {type(opening_speaker).__name__} as buyer")
        
        # Generate opening - act() method handles two-part logic internally
        logger.debug(f"Generating opening with two_part={self.use_two_part}")
        response = opening_speaker.act()
        opening_text = response.get('text', '')
        logger.debug(f"Opening generated: {opening_text[:100]}..." if opening_text else "No text generated")

        # If two-part response is enabled and strategy was generated, save it
        if self.use_two_part and response.get('strategy'):
            episode['strategies'].append({
                'turn': 0,
                'speaker': 'buyer',  # Opening speaker is always buyer
                'strategy': response.get('strategy')
            })
        
        episode['dialogue'].append({
            'turn': 0,
            'speaker': 'buyer',
            'text': opening_text,
            'action': self._extract_action(opening_text),
            'tokens': len(opening_text.split()),
            'timestamp': datetime.now().isoformat()
        })
        
        # Other agent observes opening
        other_agent.observe({'text': opening_text, 'episode_done': False})
        
        # Run negotiation turns
        turn = 1
        episode_done = False
        logger.debug("Starting negotiation turns")
        
        while turn < self.max_turns and not episode_done:
            logger.debug(f"Turn {turn}/{self.max_turns}")
            # Determine current speaker
            current_is_negotiator = (turn % 2 == 0 and agent_side == 'buyer') or \
                                  (turn % 2 == 1 and agent_side == 'seller')
            
            current_agent = self.negotiator if current_is_negotiator else self.partner
            other_agent = self.partner if current_is_negotiator else self.negotiator
            speaker = agent_side if current_is_negotiator else partner_side
            
            # Generate response - act() method handles two-part logic internally
            response = current_agent.act()
            text = response.get('text', '')
            episode_done = response.get('episode_done', False)

            # If two-part response is enabled and strategy was generated, save it
            if self.use_two_part and current_is_negotiator and response.get('strategy'):
                episode['strategies'].append({
                    'turn': turn,
                    'speaker': speaker,
                    'strategy': response.get('strategy')
                })
            
            # Log turn
            turn_entry = {
                'turn': turn,
                'speaker': speaker,
                'text': text,
                'action': self._extract_action(text),
                'tokens': len(text.split()),
                'timestamp': datetime.now().isoformat()
            }
            episode['dialogue'].append(turn_entry)
            
            # Other agent observes
            if not episode_done:
                other_agent.observe({'text': text, 'episode_done': False})
            
            turn += 1
        
        # Calculate episode metrics
        episode_duration = time.time() - episode_start
        episode['metrics'] = self._calculate_episode_metrics(
            episode, episode_duration
        )
        
        # Log episode
        self._log_episode(episode)
        self.episode_count += 1
        
        return episode
    
    def _extract_action(self, text: str) -> Optional[str]:
        """Extract action tag from text."""
        for action in NegotiationAction:
            if action.value in text:
                return action.value.strip('[]')
        return None
    
    def _check_episode_done(self, text: str, agent) -> bool:
        """Check if episode should end."""
        action = self._extract_action(text)
        
        # Check for terminal actions
        if action in ['NO_DEAL']:
            return True
        
        # Check for successful conclusion (ACK after COMMIT)
        if action == 'ACK' and hasattr(agent, 'negotiation_state'):
            if agent.negotiation_state == NegotiationState.AWAITING_ACK:
                return True
        
        # Check agent's internal state
        if hasattr(agent, 'negotiation_state'):
            if agent.negotiation_state in [NegotiationState.CONCLUDED, NegotiationState.FAILED]:
                return True
        
        return False
    
    def _calculate_episode_metrics(self, episode: Dict[str, Any], 
                                  duration: float) -> Dict[str, Any]:
        """Calculate metrics for completed episode."""
        dialogue = episode['dialogue']
        
        # Basic metrics
        metrics = {
            'turns': len(dialogue),
            'duration_seconds': duration,
            'tokens_total': sum(turn['tokens'] for turn in dialogue),
            'deal_closed': False,
            'final_price': None,
            'protocol_violations': []
        }
        
        # Check if deal was closed
        last_actions = [turn['action'] for turn in dialogue[-3:] if turn['action']]
        if 'ACK' in last_actions:
            metrics['deal_closed'] = True
            
            # Extract final price from COMMIT message
            for turn in reversed(dialogue):
                if turn['action'] == 'COMMIT':
                    # Try to extract price from text
                    import re
                    price_match = re.search(r'"total_price"\s*:\s*(\d+(?:\.\d+)?)', turn['text'])
                    if price_match:
                        metrics['final_price'] = float(price_match.group(1))
                    break
        
        # Calculate side-specific metrics
        agent_turns = [t for t in dialogue if t['speaker'] == episode['meta']['agent_side']]
        partner_turns = [t for t in dialogue if t['speaker'] == episode['meta']['partner_side']]
        
        metrics['agent_tokens'] = sum(t['tokens'] for t in agent_turns)
        metrics['partner_tokens'] = sum(t['tokens'] for t in partner_turns)
        metrics['agent_turns'] = len(agent_turns)
        metrics['partner_turns'] = len(partner_turns)
        
        # Track action distribution
        metrics['action_distribution'] = {}
        for turn in dialogue:
            action = turn['action']
            if action:
                metrics['action_distribution'][action] = \
                    metrics['action_distribution'].get(action, 0) + 1
        
        # Get agent summaries if available
        if hasattr(self.negotiator, 'get_negotiation_summary'):
            metrics['agent_summary'] = self.negotiator.get_negotiation_summary()
        
        if self.partner_mode == 'rule' and hasattr(self.partner, 'get_agent_summary'):
            metrics['partner_summary'] = self.partner.get_agent_summary()
        
        return metrics
    
    def _log_episode(self, episode: Dict[str, Any]):
        """Log episode to appropriate storage."""
        logger.debug(f"Logging episode {episode['episode_id']}")
        self.batch_episodes.append(episode)
        logger.debug(f"Added to batch, current batch size: {len(self.batch_episodes)}")
        
        # Save transcript immediately
        self._save_transcript(episode)
        logger.debug(f"Transcript saved for episode {episode['episode_id']}")
        
        # Update session metrics
        self._update_session_metrics(episode)
        logger.debug("Session metrics updated")
        
        # Save batch if full
        if len(self.batch_episodes) >= self.batch_size:
            self._save_batch()
            self.batch_episodes = []
            self.current_batch += 1
        
        # Save metrics periodically
        if self.episode_count % self.save_metrics_every == 0:
            self._save_metrics()
        
        # Log to teacher for rule mode
        if self.partner_mode == 'rule':
            scenario = {
                'state': episode['state'],
                'persona': {'type': episode['meta'].get('partner_persona')},
                'agent_side': episode['meta']['agent_side'],
                'opponent_side': episode['meta']['partner_side']
            }
            self.teacher.log_rule_episode(
                episode['dialogue'],
                scenario,
                episode['metrics']
            )
    
    def _update_session_metrics(self, episode: Dict[str, Any]):
        """Update session-level metrics."""
        metrics = episode['metrics']
        meta = episode['meta']
        
        # Create episode record
        record = {
            'episode_id': episode['episode_id'],
            'agent_side': meta['agent_side'],
            'partner_mode': meta['partner_mode'],
            'turns': metrics['turns'],
            'deal_closed': metrics['deal_closed'],
            'final_price': metrics['final_price'],
            'duration_seconds': metrics['duration_seconds'],
            'tokens_total': metrics['tokens_total']
        }
        
        if self.partner_mode == 'rule':
            record['partner_persona'] = meta.get('partner_persona')
        
        # Append to DataFrame
        self.session_metrics['episode_df'] = pd.concat([
            self.session_metrics['episode_df'],
            pd.DataFrame([record])
        ], ignore_index=True)
        
        # Update aggregate stats
        self.session_metrics['aggregate_stats'] = {
            'total_episodes': len(self.session_metrics['episode_df']),
            'deals_closed': self.session_metrics['episode_df']['deal_closed'].sum(),
            'avg_turns': self.session_metrics['episode_df']['turns'].mean(),
            'avg_duration': self.session_metrics['episode_df']['duration_seconds'].mean(),
            'close_rate': self.session_metrics['episode_df']['deal_closed'].mean()
        }
        
        # Update persona stats for rule mode
        if self.partner_mode == 'rule' and 'partner_persona' in record:
            persona = record['partner_persona']
            if persona not in self.session_metrics['persona_stats']:
                self.session_metrics['persona_stats'][persona] = {
                    'episodes': 0,
                    'deals_closed': 0,
                    'avg_turns': 0
                }
            
            stats = self.session_metrics['persona_stats'][persona]
            stats['episodes'] += 1
            if metrics['deal_closed']:
                stats['deals_closed'] += 1
            stats['avg_turns'] = (stats['avg_turns'] * (stats['episodes'] - 1) + 
                                 metrics['turns']) / stats['episodes']
    
    def _save_batch(self):
        """Save current batch of episodes."""
        if not self.batch_episodes:
            return
        
        batch_file = self.episodes_dir / f"batch_{self.current_batch:04d}.jsonl"
        
        with open(batch_file, 'w') as f:
            for episode in self.batch_episodes:
                f.write(json.dumps(episode) + '\n')
        
        logger.info(f"Saved batch {self.current_batch} with {len(self.batch_episodes)} episodes")
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        return obj
    
    def _save_metrics(self):
        """Save session metrics."""
        metrics_file = self.metrics_dir / f"session_{self.session_metrics['session_id']}.json"
        
        # Convert DataFrame to dict for JSON serialization
        # Convert numpy types to native Python types
        episode_records = self.session_metrics['episode_df'].to_dict('records')
        episode_records = self._convert_numpy_types(episode_records)
        
        metrics_data = {
            'session_id': self.session_metrics['session_id'],
            'aggregate_stats': self._convert_numpy_types(self.session_metrics['aggregate_stats']),
            'persona_stats': self._convert_numpy_types(self.session_metrics['persona_stats']),
            'episode_records': episode_records
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        logger.info(f"Saved metrics for {self.episode_count} episodes")
    
    def _save_transcript(self, episode: Dict[str, Any]):
        """Save human-readable transcript of the episode."""
        # Ensure transcript directory exists (in case of parallel execution)
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)
        transcript_file = self.transcripts_dir / f"{episode['episode_id']}_transcript.txt"
        
        with open(transcript_file, 'w') as f:
            # Write header
            f.write("=" * 80 + "\n")
            f.write(f"NEGOTIATION TRANSCRIPT\n")
            f.write(f"Episode ID: {episode['episode_id']}\n")
            f.write(f"Dialogue ID: {episode['dialogue_id']}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")
            
            # Write scenario info
            meta = episode['meta']
            f.write(f"SCENARIO:\n")
            f.write(f"  Agent Side: {meta['agent_side']}\n")
            f.write(f"  Partner Side: {meta['partner_side']}\n")
            if self.partner_mode == 'rule':
                f.write(f"  Partner Persona: {meta.get('partner_persona', 'unknown')}\n")
            
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
            if metrics['final_price']:
                f.write(f"  Final Price: ${metrics['final_price']}\n")
            f.write(f"  Total Turns: {metrics['turns']}\n")
            f.write(f"  Duration: {metrics['duration_seconds']:.2f} seconds\n")
            f.write(f"  Total Tokens: {metrics['tokens_total']}\n")
            
            # Write action distribution
            if metrics.get('action_distribution'):
                f.write(f"\nAction Distribution:\n")
                for action, count in metrics['action_distribution'].items():
                    f.write(f"  {action}: {count}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF TRANSCRIPT\n")
            f.write("=" * 80 + "\n")
    
    def parley(self):
        """Execute one episode (ParlAI interface)."""
        logger.info(f"Parley called: episode {self.episode_count + 1}/{self.target_episodes}")
        if self.episode_count >= self.target_episodes:
            logger.info(f"Target episodes reached ({self.target_episodes}), skipping parley")
            return
        
        try:
            logger.info("Calling run_episode()")
            episode = self.run_episode()
            if episode:
                logger.info(f"Episode completed successfully with {len(episode.get('dialogue', []))} turns")
            else:
                logger.warning("run_episode() returned None or empty episode")
        except Exception as e:
            logger.error(f"Error in parley: {e}", exc_info=True)
    
    def shutdown(self):
        """Clean up and save final data."""
        logger.info("Shutting down DialogPartnerWorld")
        # Save remaining batch
        if self.batch_episodes:
            logger.info(f"Saving final batch with {len(self.batch_episodes)} episodes")
            self._save_batch()
        
        # Save final metrics
        self._save_metrics()
        
        # Save teacher data
        self.teacher.save_episodes()
        
        logger.info(f"DialogPartnerWorld shutdown complete. Processed {self.episode_count} episodes")
    
    def share(self):
        """Share data between world instances."""
        shared = {}
        shared['teacher'] = self.teacher.share()
        shared['episode_count'] = self.episode_count
        shared['session_metrics'] = self.session_metrics
        return shared