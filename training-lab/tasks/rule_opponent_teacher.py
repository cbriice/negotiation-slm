"""
Teacher for rule-based opponent scenarios.
Provides scenarios for training against RuleAgent with various personas.
"""

import json
import random
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from tasks.negotiation_teacher import NegotiationJSONTeacher
from schema.rule_agent.personas import PersonaType, PersonaManager
from parlai.core.opt import Opt


class RuleOpponentTeacher(NegotiationJSONTeacher):
    """
    Specialized teacher for rule-based opponent scenarios.
    
    Focuses on:
    - Diverse RuleAgent personas with different negotiation styles
    - Rich conversation structures (embellishment, cheap talk, etc.)
    - Tighter behavior restrictions for precise training
    - Balanced difficulty progression
    """
    
    @classmethod
    def add_cmdline_args(cls, parser, partial_opt: Optional[Opt] = None):
        """Add command-line arguments for rule opponent teacher."""
        # Ensure partial_opt is not None for parent class
        if partial_opt is None:
            partial_opt = Opt()
        super().add_cmdline_args(parser, partial_opt)
        agent = parser.add_argument_group('Rule Opponent Teacher Arguments')
        agent.add_argument(
            '--persona-selection',
            type=str,
            default='random',
            choices=['random', 'sequential', 'difficulty', 'blended'],
            help='How to select personas during training'
        )
        agent.add_argument(
            '--persona-intensity',
            type=float,
            default=0.5,
            help='Intensity of persona traits (0.0=mild, 1.0=extreme)'
        )
        agent.add_argument(
            '--blend-personas',
            type=bool,
            default=False,
            help='Whether to blend personas for nuanced behavior'
        )
        agent.add_argument(
            '--include-hard-constraints',
            type=bool,
            default=True,
            help='Include scenarios with hard behavioral constraints'
        )
        agent.add_argument(
            '--conversation-complexity',
            type=str,
            default='medium',
            choices=['simple', 'medium', 'complex'],
            help='Level of conversation structure complexity'
        )
        return parser
    
    def __init__(self, opt, shared=None):
        """Initialize the rule opponent teacher."""
        # Set default data directory
        if 'negotiation_data_dir' not in opt:
            opt['negotiation_data_dir'] = 'data/rule_opponent'
        if 'negotiation_data_file' not in opt:
            opt['negotiation_data_file'] = 'rule_opponent_episodes.jsonl'
        
        # Initialize parent
        super().__init__(opt, shared)
        
        # Create transcripts directory
        self.transcripts_dir = self.data_dir / 'transcripts'
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)
        
        # Rule opponent specific configuration
        self.persona_selection = opt.get('persona_selection', 'random')
        self.persona_intensity = opt.get('persona_intensity', 0.5)
        self.blend_personas = opt.get('blend_personas', False)
        self.hard_constraints = opt.get('include_hard_constraints', True)
        self.complexity = opt.get('conversation_complexity', 'medium')
        
        # Initialize persona manager
        self.persona_manager = PersonaManager()
        
        # Track persona usage
        if shared:
            self.persona_history = shared.get('persona_history', [])
            self.persona_idx = shared.get('persona_idx', 0)
        else:
            self.persona_history = []
            self.persona_idx = 0
    
    def share(self):
        """Share data between teacher instances."""
        shared = super().share()
        shared['persona_history'] = self.persona_history
        shared['persona_idx'] = self.persona_idx
        return shared
    
    def _select_next_persona(self) -> Dict[str, Any]:
        """Select next persona based on selection strategy."""
        persona_types = list(PersonaType)
        
        if self.persona_selection == 'random':
            if self.blend_personas and random.random() < 0.3:
                # 30% chance to blend personas
                primary = random.choice(persona_types)
                secondary = random.choice([p for p in persona_types if p != primary])
                persona = self.persona_manager.blend_personas(primary, secondary)
            else:
                persona = self.persona_manager.get_random_persona()
        
        elif self.persona_selection == 'sequential':
            # Cycle through all persona types
            persona_type = persona_types[self.persona_idx % len(persona_types)]
            persona = self.persona_manager.get_persona(persona_type)
            self.persona_idx += 1
        
        elif self.persona_selection == 'difficulty':
            # Progress through personas by difficulty
            # Start with cooperative/sympathetic, move to aggressive/deceptive
            difficulty_order = [
                PersonaType.COOPERATIVE,
                PersonaType.SYMPATHETIC,
                PersonaType.DIPLOMATIC,
                PersonaType.ANALYTICAL,
                PersonaType.IMPATIENT,
                PersonaType.DECEPTIVE,
                PersonaType.AGGRESSIVE
            ]
            stage = min(self.persona_idx // 10, len(difficulty_order) - 1)
            persona_type = difficulty_order[stage]
            persona = self.persona_manager.get_persona(persona_type)
            self.persona_idx += 1
        
        elif self.persona_selection == 'blended':
            # Always blend two random personas
            primary = random.choice(persona_types)
            secondary = random.choice([p for p in persona_types if p != primary])
            blend_ratio = random.uniform(0.5, 0.8)
            persona = self.persona_manager.blend_personas(primary, secondary, blend_ratio)
        
        else:
            persona = self.persona_manager.get_random_persona()
        
        # Adjust intensity
        if self.persona_intensity != 0.5:
            persona = self.persona_manager.adjust_persona_intensity(persona, self.persona_intensity)
        
        # Track persona usage
        self.persona_history.append(persona['type'])
        
        return persona
    
    def _generate_conversation_structure(self, persona: Dict[str, Any]) -> Dict[str, Any]:
        """Generate conversation structure elements based on persona and complexity."""
        template = persona['template']
        structure = {
            'tactics': [],
            'language_patterns': [],
            'response_style': {}
        }
        
        if self.complexity == 'simple':
            # Basic negotiation structure
            structure['tactics'] = random.sample(template['preferred_tactics'], 
                                                min(2, len(template['preferred_tactics'])))
            structure['language_patterns'] = random.sample(template['language_patterns'],
                                                          min(2, len(template['language_patterns'])))
        
        elif self.complexity == 'medium':
            # Moderate complexity
            structure['tactics'] = random.sample(template['preferred_tactics'],
                                                min(3, len(template['preferred_tactics'])))
            structure['language_patterns'] = random.sample(template['language_patterns'],
                                                          min(3, len(template['language_patterns'])))
            structure['response_style'] = {
                'use_patterns': True,
                'pattern_frequency': 0.5
            }
        
        elif self.complexity == 'complex':
            # Full complexity
            structure['tactics'] = template['preferred_tactics']
            structure['language_patterns'] = template['language_patterns']
            structure['response_style'] = {
                'use_patterns': True,
                'pattern_frequency': 0.7,
                'adapt_to_opponent': True,
                'escalation_possible': True
            }
            
            # Add complex conversation elements
            structure['advanced_tactics'] = []
            if 'false_competition' in template['preferred_tactics']:
                structure['advanced_tactics'].append('multiple_false_competitors')
            if 'sympathy_appeal' in template['preferred_tactics']:
                structure['advanced_tactics'].append('escalating_emotional_appeal')
            if 'data_reference' in template['preferred_tactics']:
                structure['advanced_tactics'].append('detailed_market_analysis')
        
        return structure
    
    def _format_rule_scenario(self, state: Dict[str, Any], 
                            persona: Dict[str, Any], 
                            side: str) -> Dict[str, Any]:
        """Format a scenario for rule-based opponent training."""
        template = persona['template']
        
        scenario = {
            'state': state,
            'persona': {
                'type': persona['type'],
                'name': template['name'],
                'traits': template['traits'],
                'response_modifiers': template['response_modifiers']
            },
            'agent_side': side,
            'opponent_side': 'seller' if side == 'buyer' else 'buyer',
            'conversation_structure': self._generate_conversation_structure(persona),
            'behavioral_constraints': {
                'concession_rate': template['traits']['concession_rate'],
                'patience_level': template['traits']['patience'],
                'walk_away_threshold': template['response_modifiers']['walk_away_threshold']
            }
        }
        
        # Add hard constraints if enabled
        if self.hard_constraints:
            scenario['hard_constraints'] = {
                'protocol_violations_allowed': 0,
                'max_turns': 15,
                'must_justify_positions': template['traits'].get('firmness', 0.5) > 0.6,
                'require_terms_validation': True,
                'enforce_action_tags': True
            }
        
        return scenario
    
    def _format_persona_context(self, persona: Dict[str, Any]) -> str:
        """Format persona information as context for the agent."""
        template = persona['template']
        
        context_parts = [
            f"\nYour opponent is a {template['name']}:",
            f"- {template['description']}"
        ]
        
        # Add trait-based descriptions
        traits = template['traits']
        if traits.get('patience', 0.5) < 0.3:
            context_parts.append("- Very impatient, wants quick resolution")
        elif traits.get('patience', 0.5) > 0.7:
            context_parts.append("- Patient and willing to negotiate at length")
        
        if traits.get('empathy', 0.5) < 0.3:
            context_parts.append("- Unlikely to be swayed by emotional appeals")
        elif traits.get('empathy', 0.5) > 0.7:
            context_parts.append("- Responsive to personal situations and emotional appeals")
        
        if traits.get('firmness', 0.5) > 0.7:
            context_parts.append("- Very firm on positions, hard to move")
        
        # Add tactical hints
        if self.complexity != 'simple':
            context_parts.append("\nExpected tactics:")
            for tactic in template['preferred_tactics'][:3]:
                tactic_desc = tactic.replace('_', ' ').title()
                context_parts.append(f"- {tactic_desc}")
        
        # Add constraint information
        if self.hard_constraints:
            context_parts.append("\nStrict protocol enforcement is active:")
            context_parts.append("- Use proper action tags ([PROPOSE], [COMMIT], etc.)")
            context_parts.append("- Protocol violations will end the negotiation")
            context_parts.append("- Maximum 15 turns allowed")
        
        return "\n".join(context_parts)
    
    def setup_data(self, datafile=None):
        """Set up rule opponent training data."""
        # If we have existing episodes, use them
        if self.episodes:
            yield from super().setup_data(datafile)
            return
        
        # Generate scenarios with rule opponents
        # Load state seeds (reuse from self-play teacher)
        seed_path = self.data_dir.parent / 'selfplay' / 'state_seeds.jsonl'
        seeds = []
        
        if seed_path.exists():
            with open(seed_path, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            seed = json.loads(line)
                            seeds.append(seed)
                        except json.JSONDecodeError:
                            continue
        
        if not seeds:
            # Generate minimal seeds if none exist
            seeds = [{
                'category': 'general',
                'item': {'name': 'Item', 'list_price': 100},
                'buyer_state': {'role': 'buyer', 'max_price': 90, 'target_price': 80},
                'seller_state': {'role': 'seller', 'min_price': 85}
            }]
        
        # Generate scenarios
        for seed_idx, seed in enumerate(seeds):
            persona = self._select_next_persona()
            
            # Generate for both sides
            for side in ['buyer', 'seller']:
                scenario = self._format_rule_scenario(seed, persona, side)
                
                # Create observation
                context = self._format_state_context(seed, side)
                persona_context = self._format_persona_context(persona)
                
                obs = {
                    'text': context + persona_context,
                    'labels': None,  # Will be filled during actual training
                    'episode_id': f"rule_{seed_idx}_{side}_{persona['type']}",
                    'dialogue_id': f"rule_d{seed_idx}_{side}",
                    'side': side,
                    'persona_type': str(persona['type']),
                    'scenario': scenario,
                    'episode_done': True
                }
                
                yield obs, True
    
    def log_rule_episode(self, dialogue: List[Dict[str, Any]], 
                        scenario: Dict[str, Any],
                        metrics: Dict[str, Any]):
        """Log a completed rule opponent episode."""
        episode = {
            'state': scenario['state'],
            'persona': scenario['persona'],
            'dialogue': dialogue,
            'meta': {
                'episode_id': f"rule_{len(self.episodes)}",
                'dialogue_id': f"rule_d{len(self.episodes)}",
                'agent_side': scenario['agent_side'],
                'opponent_side': scenario['opponent_side'],
                'persona_type': scenario['persona']['type'],
                'conversation_structure': scenario['conversation_structure'],
                'timestamp': metrics.get('timestamp'),
                'model_version': metrics.get('model_version')
            },
            'metrics': metrics
        }
        
        # Add constraint violation tracking
        if 'hard_constraints' in scenario:
            episode['constraint_violations'] = metrics.get('protocol_violations', [])
            episode['constraints_satisfied'] = len(episode['constraint_violations']) == 0
        
        self.episodes.append(episode)
        self.update_episode_metrics(episode['meta']['episode_id'], metrics)
        
        # Save periodically
        if len(self.episodes) % 100 == 0:
            self.save_episodes()
    
    def get_persona_statistics(self) -> Dict[str, Any]:
        """Get statistics about persona performance."""
        stats = {}
        
        for episode in self.episodes:
            persona_type = episode.get('meta', {}).get('persona_type', 'unknown')
            if persona_type not in stats:
                stats[persona_type] = {
                    'episodes': 0,
                    'deals_closed': 0,
                    'avg_turns': 0,
                    'avg_final_price': 0,
                    'constraint_violations': 0
                }
            
            stats[persona_type]['episodes'] += 1
            if episode.get('metrics', {}).get('deal_closed'):
                stats[persona_type]['deals_closed'] += 1
            stats[persona_type]['avg_turns'] += episode.get('metrics', {}).get('turns', 0)
            
            # Track final price if available
            final_terms = episode.get('metrics', {}).get('final_terms', {})
            if final_terms and 'total_price' in final_terms:
                stats[persona_type]['avg_final_price'] += final_terms['total_price']
            
            stats[persona_type]['constraint_violations'] += len(
                episode.get('constraint_violations', [])
            )
        
        # Calculate averages
        for persona_type in stats:
            if stats[persona_type]['episodes'] > 0:
                n = stats[persona_type]['episodes']
                stats[persona_type]['deal_close_rate'] = stats[persona_type]['deals_closed'] / n
                stats[persona_type]['avg_turns'] /= n
                stats[persona_type]['avg_final_price'] /= n
                stats[persona_type]['avg_violations'] = stats[persona_type]['constraint_violations'] / n
        
        return stats