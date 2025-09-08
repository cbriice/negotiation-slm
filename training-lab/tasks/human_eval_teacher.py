"""
Teacher for human evaluation scenarios.
Special handling for human-in-the-loop evaluation and edge case testing.
"""

import json
import random
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

from tasks.negotiation_teacher import NegotiationJSONTeacher


class HumanEvalTeacher(NegotiationJSONTeacher):
    """
    Specialized teacher for human evaluation scenarios.
    
    Focuses on:
    - Edge case scenarios for human testing
    - Controlled test conditions
    - Separate logging from training data to avoid contamination
    - Interactive session management
    """
    
    @classmethod
    def add_cmdline_args(cls, parser, partial_opt=None):
        """Add command-line arguments for human eval teacher."""
        super().add_cmdline_args(parser, partial_opt)
        agent = parser.add_argument_group('Human Eval Teacher Arguments')
        agent.add_argument(
            '--eval-scenario-type',
            type=str,
            default='mixed',
            choices=['edge_cases', 'standard', 'adversarial', 'mixed'],
            help='Type of scenarios for human evaluation'
        )
        agent.add_argument(
            '--eval-session-id',
            type=str,
            default=None,
            help='Unique session ID for this evaluation'
        )
        agent.add_argument(
            '--separate-eval-logs',
            type=bool,
            default=True,
            help='Keep evaluation logs separate from training data'
        )
        agent.add_argument(
            '--include-edge-cases',
            type=bool,
            default=True,
            help='Include extreme edge cases in evaluation'
        )
        agent.add_argument(
            '--prompt-human-feedback',
            type=bool,
            default=True,
            help='Prompt human for feedback after each episode'
        )
        return parser
    
    def __init__(self, opt, shared=None):
        """Initialize the human eval teacher."""
        # Set default data directory for evaluation
        if 'negotiation_data_dir' not in opt:
            opt['negotiation_data_dir'] = 'data/human_eval'
        if 'negotiation_data_file' not in opt:
            opt['negotiation_data_file'] = 'human_eval_episodes.jsonl'
        
        # Initialize parent
        super().__init__(opt, shared)
        
        # Human eval specific configuration
        self.scenario_type = opt.get('eval_scenario_type', 'mixed')
        self.session_id = opt.get('eval_session_id') or f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.separate_logs = opt.get('separate_eval_logs', True)
        self.include_edge_cases = opt.get('include_edge_cases', True)
        self.prompt_feedback = opt.get('prompt_human_feedback', True)
        
        # Evaluation-specific tracking
        if shared:
            self.eval_scenarios = shared.get('eval_scenarios', [])
            self.human_feedback = shared.get('human_feedback', [])
        else:
            self._load_eval_scenarios()
            self.human_feedback = []
        
        # Set up separate logging if requested
        if self.separate_logs:
            self.eval_log_dir = self.data_dir / 'sessions' / self.session_id
            self.eval_log_dir.mkdir(parents=True, exist_ok=True)
    
    def share(self):
        """Share data between teacher instances."""
        shared = super().share()
        shared['eval_scenarios'] = self.eval_scenarios
        shared['human_feedback'] = self.human_feedback
        return shared
    
    def _load_eval_scenarios(self):
        """Load or generate evaluation scenarios."""
        scenario_path = self.data_dir / 'eval_scenarios.jsonl'
        self.eval_scenarios = []
        
        if scenario_path.exists():
            with open(scenario_path, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            scenario = json.loads(line)
                            self.eval_scenarios.append(scenario)
                        except json.JSONDecodeError:
                            continue
            print(f"Loaded {len(self.eval_scenarios)} evaluation scenarios")
        else:
            # Generate default evaluation scenarios
            self.eval_scenarios = self._generate_eval_scenarios()
            self._save_eval_scenarios(scenario_path)
    
    def _generate_eval_scenarios(self) -> List[Dict[str, Any]]:
        """Generate evaluation scenarios including edge cases."""
        scenarios = []
        
        # Standard scenarios
        standard_scenarios = [
            {
                'name': 'standard_electronics',
                'type': 'standard',
                'category': 'electronics',
                'item': {'name': 'Smartphone', 'brand': 'Samsung', 'model': 'Galaxy S23',
                        'condition': 'excellent', 'list_price': 800},
                'buyer_state': {'role': 'buyer', 'max_price': 700, 'target_price': 650},
                'seller_state': {'role': 'seller', 'min_price': 680},
                'test_focus': 'normal_negotiation'
            },
            {
                'name': 'standard_vehicle',
                'type': 'standard',
                'category': 'vehicles',
                'item': {'name': 'Electric Scooter', 'brand': 'Xiaomi', 'model': 'Pro 2',
                        'condition': 'good', 'list_price': 400},
                'buyer_state': {'role': 'buyer', 'max_price': 350, 'target_price': 320},
                'seller_state': {'role': 'seller', 'min_price': 330},
                'test_focus': 'price_negotiation'
            }
        ]
        
        # Edge case scenarios
        edge_cases = []
        if self.include_edge_cases:
            edge_cases = [
                {
                    'name': 'extreme_price_gap',
                    'type': 'edge_case',
                    'category': 'test',
                    'item': {'name': 'Luxury Watch', 'list_price': 10000},
                    'buyer_state': {'role': 'buyer', 'max_price': 3000, 'target_price': 2500},
                    'seller_state': {'role': 'seller', 'min_price': 9000},
                    'test_focus': 'impossible_negotiation',
                    'expected_outcome': 'no_deal'
                },
                {
                    'name': 'no_overlap_zone',
                    'type': 'edge_case',
                    'category': 'test',
                    'item': {'name': 'Antique Vase', 'list_price': 5000},
                    'buyer_state': {'role': 'buyer', 'max_price': 2000, 'target_price': 1800},
                    'seller_state': {'role': 'seller', 'min_price': 3000},
                    'test_focus': 'no_zone_of_agreement',
                    'expected_outcome': 'no_deal'
                },
                {
                    'name': 'identical_values',
                    'type': 'edge_case',
                    'category': 'test',
                    'item': {'name': 'Book Collection', 'list_price': 100},
                    'buyer_state': {'role': 'buyer', 'max_price': 100, 'target_price': 100},
                    'seller_state': {'role': 'seller', 'min_price': 100},
                    'test_focus': 'exact_match',
                    'expected_outcome': 'quick_deal'
                },
                {
                    'name': 'missing_information',
                    'type': 'edge_case',
                    'category': 'test',
                    'item': {'name': 'Mystery Box'},  # No price information
                    'buyer_state': {'role': 'buyer', 'max_price': 500, 'target_price': 400},
                    'seller_state': {'role': 'seller', 'min_price': 450},
                    'test_focus': 'incomplete_information'
                },
                {
                    'name': 'complex_bundle',
                    'type': 'edge_case',
                    'category': 'test',
                    'item': {'name': 'Computer Setup', 'components': ['PC', 'Monitor', 'Keyboard', 'Mouse'],
                            'list_price': 2000},
                    'buyer_state': {'role': 'buyer', 'max_price': 1800, 'target_price': 1600,
                                  'preferences': {'wants_partial': True}},
                    'seller_state': {'role': 'seller', 'min_price': 1700,
                                   'item_issues': ['monitor has dead pixel']},
                    'test_focus': 'complex_negotiation'
                }
            ]
        
        # Adversarial scenarios
        adversarial = []
        if self.scenario_type in ['adversarial', 'mixed']:
            adversarial = [
                {
                    'name': 'aggressive_buyer',
                    'type': 'adversarial',
                    'category': 'test',
                    'item': {'name': 'Gaming Console', 'list_price': 500},
                    'buyer_state': {'role': 'buyer', 'max_price': 400, 'target_price': 300,
                                  'strategy': 'lowball_aggressive'},
                    'seller_state': {'role': 'seller', 'min_price': 450},
                    'test_focus': 'handle_aggression'
                },
                {
                    'name': 'deceptive_seller',
                    'type': 'adversarial',
                    'category': 'test',
                    'item': {'name': 'Used Car', 'list_price': 8000},
                    'buyer_state': {'role': 'buyer', 'max_price': 7500, 'target_price': 7000},
                    'seller_state': {'role': 'seller', 'min_price': 7200,
                                   'hidden_issues': ['engine problems'],
                                   'strategy': 'hide_defects'},
                    'test_focus': 'detect_deception'
                },
                {
                    'name': 'time_pressure',
                    'type': 'adversarial',
                    'category': 'test',
                    'item': {'name': 'Concert Tickets', 'list_price': 200},
                    'buyer_state': {'role': 'buyer', 'max_price': 180, 'target_price': 160,
                                  'constraint': 'need_within_5_turns'},
                    'seller_state': {'role': 'seller', 'min_price': 170,
                                   'strategy': 'stall_tactics'},
                    'test_focus': 'time_pressure_handling'
                }
            ]
        
        # Combine scenarios based on type
        if self.scenario_type == 'standard':
            scenarios = standard_scenarios
        elif self.scenario_type == 'edge_cases':
            scenarios = edge_cases
        elif self.scenario_type == 'adversarial':
            scenarios = adversarial
        elif self.scenario_type == 'mixed':
            scenarios = standard_scenarios + edge_cases + adversarial
            random.shuffle(scenarios)
        
        # Add metadata to all scenarios
        for i, scenario in enumerate(scenarios):
            scenario['scenario_id'] = f"eval_{i}"
            scenario['session_id'] = self.session_id
            if 'seed_metadata' not in scenario:
                scenario['seed_metadata'] = {
                    'evaluation_type': scenario['type'],
                    'test_focus': scenario.get('test_focus', 'general')
                }
        
        return scenarios
    
    def _save_eval_scenarios(self, filepath: Path):
        """Save evaluation scenarios to file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            for scenario in self.eval_scenarios:
                f.write(json.dumps(scenario) + '\n')
        print(f"Saved {len(self.eval_scenarios)} evaluation scenarios to {filepath}")
    
    def _format_eval_context(self, scenario: Dict[str, Any], side: str) -> str:
        """Format evaluation scenario context with special instructions."""
        # Get base context
        context = self._format_state_context(scenario, side)
        
        # Add evaluation-specific information
        eval_context = [context, "\n--- EVALUATION SCENARIO ---"]
        
        # Add scenario type information
        scenario_type = scenario.get('type', 'standard')
        if scenario_type == 'edge_case':
            eval_context.append("This is an EDGE CASE scenario testing extreme conditions.")
            if 'expected_outcome' in scenario:
                eval_context.append(f"Expected outcome: {scenario['expected_outcome']}")
        elif scenario_type == 'adversarial':
            eval_context.append("This is an ADVERSARIAL scenario with challenging opponent behavior.")
        
        # Add test focus
        test_focus = scenario.get('test_focus')
        if test_focus:
            focus_descriptions = {
                'normal_negotiation': "Standard negotiation flow",
                'price_negotiation': "Focus on price discovery and agreement",
                'impossible_negotiation': "Test handling of impossible deals",
                'no_zone_of_agreement': "No overlapping acceptable price range",
                'exact_match': "Prices align exactly",
                'incomplete_information': "Missing critical information",
                'complex_negotiation': "Multiple items or complex terms",
                'handle_aggression': "Dealing with aggressive tactics",
                'detect_deception': "Identifying misleading information",
                'time_pressure_handling': "Negotiating under time constraints"
            }
            eval_context.append(f"Test focus: {focus_descriptions.get(test_focus, test_focus)}")
        
        # Add any special constraints
        role_state = scenario.get(f'{side}_state', {})
        if 'strategy' in role_state:
            eval_context.append(f"Your strategy hint: {role_state['strategy']}")
        if 'constraint' in role_state:
            eval_context.append(f"Special constraint: {role_state['constraint']}")
        
        eval_context.append("--- END EVALUATION SCENARIO ---\n")
        
        return "\n".join(eval_context)
    
    def setup_data(self, datafile=None):
        """Set up human evaluation data."""
        # If we have existing episodes, use them
        if self.episodes and not self.separate_logs:
            yield from super().setup_data(datafile)
            return
        
        # Generate evaluation episodes from scenarios
        for scenario in self.eval_scenarios:
            # Generate for the human's side (usually buyer)
            for side in ['buyer', 'seller']:
                if side == 'buyer' or scenario.get('test_both_sides', False):
                    # Format context
                    context = self._format_eval_context(scenario, side)
                    
                    obs = {
                        'text': context,
                        'labels': None,  # Human will provide responses
                        'episode_id': f"{scenario['scenario_id']}_{side}",
                        'dialogue_id': f"{self.session_id}_{scenario['scenario_id']}_{side}",
                        'side': side,
                        'scenario_type': scenario['type'],
                        'test_focus': scenario.get('test_focus'),
                        'scenario': scenario,
                        'episode_done': True,
                        'is_eval': True  # Mark as evaluation
                    }
                    
                    yield obs, True
    
    def log_human_eval_episode(self, dialogue: List[Dict[str, Any]], 
                              scenario: Dict[str, Any],
                              metrics: Dict[str, Any],
                              human_feedback: Optional[Dict[str, Any]] = None):
        """Log a completed human evaluation episode."""
        episode = {
            'scenario': scenario,
            'dialogue': dialogue,
            'meta': {
                'episode_id': f"human_eval_{len(self.episodes)}",
                'dialogue_id': f"human_{self.session_id}_{len(self.episodes)}",
                'session_id': self.session_id,
                'scenario_id': scenario.get('scenario_id'),
                'scenario_type': scenario.get('type'),
                'test_focus': scenario.get('test_focus'),
                'timestamp': datetime.now().isoformat(),
                'is_human_eval': True
            },
            'metrics': metrics,
            'human_feedback': human_feedback
        }
        
        # Save to separate log if configured
        if self.separate_logs:
            log_path = self.eval_log_dir / f"episode_{episode['meta']['episode_id']}.json"
            with open(log_path, 'w') as f:
                json.dump(episode, f, indent=2)
            
            # Also save to cumulative log
            cumulative_log = self.eval_log_dir / 'all_episodes.jsonl'
            with open(cumulative_log, 'a') as f:
                f.write(json.dumps(episode) + '\n')
        else:
            # Add to regular episodes
            self.episodes.append(episode)
            self.update_episode_metrics(episode['meta']['episode_id'], metrics)
        
        # Track human feedback
        if human_feedback:
            self.human_feedback.append({
                'episode_id': episode['meta']['episode_id'],
                'scenario_id': scenario.get('scenario_id'),
                'feedback': human_feedback,
                'timestamp': datetime.now().isoformat()
            })
    
    def get_human_feedback_prompt(self) -> List[str]:
        """Get prompts for collecting human feedback."""
        return [
            "Rate the agent's negotiation performance (1-10):",
            "Did the agent follow proper protocol? (yes/no):",
            "Was the agent's behavior realistic? (yes/no):",
            "Any specific issues or impressive behaviors? (text):",
            "Would you negotiate with this agent again? (yes/no):",
            "Additional comments (optional):"
        ]
    
    def save_session_summary(self):
        """Save a summary of the evaluation session."""
        if not self.separate_logs:
            return
        
        summary = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'num_scenarios': len(self.eval_scenarios),
            'num_episodes': len(self.human_feedback),
            'scenario_types': {
                'standard': sum(1 for s in self.eval_scenarios if s.get('type') == 'standard'),
                'edge_case': sum(1 for s in self.eval_scenarios if s.get('type') == 'edge_case'),
                'adversarial': sum(1 for s in self.eval_scenarios if s.get('type') == 'adversarial')
            },
            'feedback_summary': self._summarize_feedback()
        }
        
        summary_path = self.eval_log_dir / 'session_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Session summary saved to {summary_path}")
    
    def _summarize_feedback(self) -> Dict[str, Any]:
        """Summarize human feedback collected during the session."""
        if not self.human_feedback:
            return {}
        
        ratings = []
        protocol_follows = []
        realistic_behaviors = []
        would_negotiate_again = []
        
        for feedback_entry in self.human_feedback:
            feedback = feedback_entry.get('feedback', {})
            if 'rating' in feedback:
                ratings.append(feedback['rating'])
            if 'protocol_followed' in feedback:
                protocol_follows.append(feedback['protocol_followed'])
            if 'realistic' in feedback:
                realistic_behaviors.append(feedback['realistic'])
            if 'negotiate_again' in feedback:
                would_negotiate_again.append(feedback['negotiate_again'])
        
        summary = {}
        if ratings:
            summary['avg_rating'] = sum(ratings) / len(ratings)
            summary['min_rating'] = min(ratings)
            summary['max_rating'] = max(ratings)
        if protocol_follows:
            summary['protocol_compliance_rate'] = sum(protocol_follows) / len(protocol_follows)
        if realistic_behaviors:
            summary['realism_rate'] = sum(realistic_behaviors) / len(realistic_behaviors)
        if would_negotiate_again:
            summary['would_negotiate_again_rate'] = sum(would_negotiate_again) / len(would_negotiate_again)
        
        return summary