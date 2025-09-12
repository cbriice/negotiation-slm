"""Rule-based negotiation agent with randomized personas and tactics."""

import json
import random
import re
import time
from typing import Dict, Any, List, Optional

from agents.negotiator_base import NegotiatorBase, NegotiationAction, NegotiationState
from schema.rule_agent.personas import PersonaManager, PersonaType
from schema.rule_agent.tactics import TacticLibrary, TacticType
from schema.rule_agent.rules import RuleEngine, ViolationType

class RuleAgent(NegotiatorBase):
    """Scripted agent with randomized personas, tactics, and rule enforcement."""
    
    @classmethod
    def add_cmdline_args(cls, argparser, partial_opt=None):
        """Add command-line arguments specific to RuleAgent."""
        agent = argparser.add_argument_group('Rule Agent Arguments')
        agent.add_argument(
            '--persona-type',
            type=str,
            default='random',
            choices=['random', 'aggressive', 'cooperative', 'deceptive', 
                    'sympathetic', 'analytical', 'impatient', 'diplomatic'],
            help='Persona type for the agent'
        )
        agent.add_argument(
            '--persona-blend',
            type=str,
            default=None,
            help='Secondary persona to blend with primary (e.g., "aggressive,cooperative")'
        )
        agent.add_argument(
            '--persona-intensity',
            type=float,
            default=0.7,
            help='Intensity of persona traits (0.0-1.0)'
        )
        agent.add_argument(
            '--strict-rules',
            type=bool,
            default=True,
            help='Enforce strict protocol rules'
        )
        agent.add_argument(
            '--terminate-on-violation',
            type=bool,
            default=True,
            help='Terminate negotiation on critical violations'
        )
        agent.add_argument(
            '--response-delay',
            type=float,
            default=0,
            help='Artificial delay before responding (seconds)'
        )
        agent.add_argument(
            '--randomize-each-episode',
            type=bool,
            default=True,
            help='Randomize persona for each new episode'
        )
        return argparser
    
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        
        # Configuration
        self.strict_rules = opt.get('strict_rules', True)
        self.terminate_on_violation = opt.get('terminate_on_violation', True)
        self.response_delay = opt.get('response_delay', 0)
        self.randomize_each_episode = opt.get('randomize_each_episode', True)
        
        # Initialize components
        self.persona_manager = PersonaManager()
        self.tactic_library = TacticLibrary()
        self.rule_engine = RuleEngine(strict_mode=self.strict_rules)
        
        # Set up persona
        self._initialize_persona(opt)
        
        # Response generation state
        self.partner_last_action = None
        self.partner_last_price = None
        self.my_last_price = None
        self.negotiation_round = 0
        self.used_tactics: List[TacticType] = []
        
        # Violation tracking
        self.partner_violations: Dict[ViolationType, int] = {}
        self.terminated = False
        self.termination_reason = None
    
    def _initialize_persona(self, opt):
        """Initialize agent persona based on options."""
        persona_type = opt.get('persona_type', 'random')
        
        if persona_type == 'random':
            self.persona = self.persona_manager.get_random_persona()
        else:
            # Convert string to PersonaType enum
            persona_enum = PersonaType(persona_type.lower())
            self.persona = self.persona_manager.get_persona(persona_enum)
        
        # Handle persona blending
        blend_spec = opt.get('persona_blend')
        if blend_spec and ',' in blend_spec:
            primary, secondary = blend_spec.split(',')
            primary_enum = PersonaType(primary.strip().lower())
            secondary_enum = PersonaType(secondary.strip().lower())
            self.persona = self.persona_manager.blend_personas(
                primary_enum, secondary_enum, blend_ratio=0.7
            )
        
        # Adjust intensity
        intensity = opt.get('persona_intensity', 0.7)
        self.persona = self.persona_manager.adjust_persona_intensity(
            self.persona, intensity
        )
        
        # Extract traits for easy access
        self.traits = self.persona['template']['traits']
        self.language_patterns = self.persona['template']['language_patterns']
        self.preferred_tactics = self.persona['template']['preferred_tactics']
        self.response_modifiers = self.persona['template']['response_modifiers']
    
    def reset_episode(self):
        """Reset for a new episode."""
        # Reset base class state
        self.negotiation_history = []
        self.current_state = {}
        self.side = None
        self.terms_buffer = None
        self.negotiation_state = NegotiationState.INITIALIZING
        self.turn_count = 0
        self.role_state = {}  # Reset role state
        
        # Reset metrics
        self.metrics = {
            'turns': 0,
            'tokens_sent': 0,
            'tokens_received': 0,
            'protocol_violations': [],
            'action_history': [],
        }
        
        # Reset agent-specific state
        self.partner_last_action = None
        self.partner_last_price = None
        self.my_last_price = None
        self.negotiation_round = 0
        self.used_tactics = []
        self.partner_violations = {}
        self.terminated = False
        self.termination_reason = None
        
        # Reset rule engine
        self.rule_engine.reset()
        
        # Optionally randomize persona
        if self.randomize_each_episode:
            self._initialize_persona({'persona_type': 'random'})
    
    def observe(self, observation):
        """Process incoming message and check for violations."""
        super().observe(observation)
        
        if 'text' in observation and observation['text']:
            text = observation['text']
            
            # Parse action
            action, cleaned_text = self.parse_negotiation_action(text)
            self.partner_last_action = action.value if action else None
            
            # Extract price if present
            price_match = re.search(r'\$(\d+(?:\.\d{2})?)', text)
            if price_match:
                self.partner_last_price = float(price_match.group(1))
            
            # Check for violations
            violations, details = self.rule_engine.check_message_violations(
                text, 
                self.partner_last_action,
                self.negotiation_state.value,
                self.turn_count
            )
            
            # Track violations
            for violation in violations:
                self.partner_violations[violation] = self.partner_violations.get(violation, 0) + 1
            
            # Check if we should terminate
            if self.terminate_on_violation:
                should_terminate, reason = self.rule_engine.should_terminate(
                    self.partner_violations,
                    self.rule_engine.warning_count
                )
                if should_terminate:
                    self.terminated = True
                    self.termination_reason = reason
        
        return observation
    
    def _select_action(self) -> NegotiationAction:
        """Select appropriate action based on state and persona."""
        # If terminated, always NO_DEAL
        if self.terminated:
            return NegotiationAction.NO_DEAL
        
        # Handle ACK/NACK for COMMIT
        if self.partner_last_action == "[COMMIT]":
            # Check if we should accept based on persona
            accept_probability = 1.0 - self.response_modifiers.get('reject_low_offers', 0.5)
            
            # Adjust based on price if available
            if self.partner_last_price and self.current_state:
                if self.side == 'buyer':
                    target = self.role_state.get('target_price', float('inf'))
                    if self.partner_last_price <= target:
                        accept_probability += 0.3
                else:
                    target = self.role_state.get('target_price', 0)
                    if self.partner_last_price >= target:
                        accept_probability += 0.3
            
            if random.random() < accept_probability:
                return NegotiationAction.ACK
            else:
                return NegotiationAction.NACK
        
        # Decide between PROPOSE, COMMIT, or NO_DEAL
        if self.negotiation_round > 3:  # After some back and forth
            # Check if we should walk away
            walk_away_prob = self.response_modifiers.get('walk_away_threshold', 0.3)
            if random.random() < walk_away_prob and self.negotiation_round > 5:
                return NegotiationAction.NO_DEAL
            
            # Check if we should commit
            if self.my_last_price:
                commit_prob = self.traits.get('firmness', 0.5)
                if random.random() < commit_prob and self.negotiation_round > 4:
                    return NegotiationAction.COMMIT
        
        return NegotiationAction.PROPOSE
    
    def _generate_price(self) -> float:
        """Generate a price based on persona and negotiation state."""
        # Handle case where side is not set
        if not self.side:
            return 750.0  # Default middle price
            
        if not self.current_state or not self.role_state:
            # Default prices if state not properly initialized
            if self.side == 'buyer':
                return 500.0
            else:
                return 1000.0
        
        if self.side == 'buyer':
            max_price = self.role_state.get('max_price', 1000)
            target_price = self.role_state.get('target_price', 800)
            
            if self.negotiation_round == 0:
                # Opening offer
                opening_style = self.traits.get('opening_style', 'moderate')
                if opening_style == 'extreme':
                    return target_price * 0.5
                elif opening_style == 'reasonable':
                    return target_price * 0.85
                else:
                    return target_price * 0.7
            else:
                # Subsequent offers
                if self.my_last_price:
                    concession_rate = self.traits.get('concession_rate', 0.3)
                    gap = target_price - self.my_last_price
                    concession = gap * concession_rate * random.uniform(0.5, 1.0)
                    new_price = self.my_last_price + concession
                    return min(new_price, max_price)
                else:
                    return target_price * 0.8
        
        else:  # seller
            min_price = self.role_state.get('min_price', 500)
            target_price = self.role_state.get('target_price', 700)
            list_price = self.current_state.get('item', {}).get('list_price', 1000)
            
            if self.negotiation_round == 0:
                # Opening ask
                opening_style = self.traits.get('opening_style', 'moderate')
                if opening_style == 'extreme':
                    return list_price * 1.2
                elif opening_style == 'reasonable':
                    return list_price
                else:
                    return list_price * 1.1
            else:
                # Subsequent asks
                if self.my_last_price:
                    concession_rate = self.traits.get('concession_rate', 0.3)
                    gap = self.my_last_price - target_price
                    concession = gap * concession_rate * random.uniform(0.5, 1.0)
                    new_price = self.my_last_price - concession
                    return max(new_price, min_price)
                else:
                    return list_price * 0.9
    
    def _generate_response_text(self, action: NegotiationAction, price: Optional[float]) -> str:
        """Generate response text based on persona and tactics.
        
        CRITICAL: Do NOT include action tags in the response text!
        Tags are backend-only metadata, never part of agent messages.
        """
        parts = []
        
        # REMOVED: Tag injection into response text
        # Tags should NEVER appear in agent messages - they're backend only!
        
        # Handle different actions
        if action == NegotiationAction.NO_DEAL:
            if self.terminated:
                parts.append(f"I'm terminating this negotiation. {self.termination_reason}")
            else:
                parts.append(random.choice([
                    "We're too far apart. No deal.",
                    "This isn't going to work out.",
                    "I'm walking away.",
                    "We can't reach an agreement."
                ]))
        
        elif action == NegotiationAction.ACK:
            parts.append("Deal! I accept those terms.")
            if self.terms_buffer:
                parts.append(f"\n{json.dumps(self.terms_buffer, indent=2)}")
        
        elif action == NegotiationAction.NACK:
            parts.append(random.choice([
                "I can't accept those terms.",
                "That doesn't work for me.",
                "No, let's keep negotiating."
            ]))
            # Add reason based on persona
            if self.traits.get('empathy', 0) > 0.5:
                parts.append("I understand your position, but I need something different.")
            elif self.traits.get('firmness', 0) > 0.7:
                parts.append("You'll need to do better than that.")
        
        elif action in [NegotiationAction.PROPOSE, NegotiationAction.COMMIT]:
            # Select and apply tactics
            stage = self.tactic_library.get_stage(self.turn_count, self.max_turns)
            tactic = self.tactic_library.select_tactic(
                self.preferred_tactics,
                stage,
                self.used_tactics
            )
            
            if tactic:
                self.used_tactics.append(tactic)
                context = {
                    'price': price,
                    'side': self.side if self.side else 'negotiator',
                    'item_name': self.current_state.get('item', {}).get('name', 'item'),
                }
                tactic_text = self.tactic_library.apply_tactic(tactic, self.side if self.side else 'negotiator', context)
                if tactic_text:
                    parts.append(tactic_text)
            
            # Add price proposal
            if price:
                if self.side == 'buyer':
                    parts.append(f"I'll offer ${price:.0f}")
                elif self.side == 'seller':
                    parts.append(f"I'm asking ${price:.0f}")
                else:
                    parts.append(f"My price is ${price:.0f}")
            
            # Add language pattern
            if random.random() < 0.5 and self.language_patterns:
                pattern = random.choice(self.language_patterns)
                # Simple variable substitution
                pattern = pattern.replace('${price}', f'${price:.0f}' if price else '[price]')
                parts.append(pattern)
            
            # If COMMIT, add terms
            if action == NegotiationAction.COMMIT:
                # Use price if available, otherwise use last price or default
                commit_price = price if price is not None else (self.my_last_price if self.my_last_price else 750.0)
                terms = self._generate_terms(commit_price)
                parts.append(f"\n{json.dumps(terms, indent=2)}")
        
        return " ".join(parts)
    
    def _generate_terms(self, price: float) -> Dict[str, Any]:
        """Generate TERMS_JSON for COMMIT."""
        # Get persona type as string
        persona_type = self.persona.get('type')
        if isinstance(persona_type, PersonaType):
            persona_type_str = persona_type.value
        else:
            persona_type_str = str(persona_type)
        
        return {
            "dialogue_id": f"rule_agent_{self.turn_count}",
            "side_signed": self.side if self.side else "unknown",
            "status": "agreed",
            "total_price": price,
            "items": [self.current_state.get('item', {}).get('name', 'Unknown Item')],
            "conditions": [],
            "metadata": {
                "persona": persona_type_str,
                "rounds": self.negotiation_round,
                "tactics_used": [t.value for t in self.used_tactics[-3:]]
            }
        }
    
    def act(self):
        """Generate a response based on persona, tactics, and rules."""
        # Import protocol validator
        from utils.protocol_validator import check_for_leaked_tags
        
        # Handle response delay if configured
        if self.response_delay > 0:
            time.sleep(self.response_delay)
        
        # Select action
        action = self._select_action()
        
        # Generate price if needed
        price = None
        if action in [NegotiationAction.PROPOSE, NegotiationAction.COMMIT]:
            price = self._generate_price()
            self.my_last_price = price
        
        # Generate response text
        response_text = self._generate_response_text(action, price)
        
        # CRITICAL: Verify no tags leaked into message text
        has_leaked, found_tags = check_for_leaked_tags(response_text)
        if has_leaked:
            # Log critical error and clean the text
            import logging
            logging.error(f"CRITICAL: Action tags leaked into message: {found_tags}")
            # Remove the leaked tags
            for tag in found_tags:
                response_text = response_text.replace(tag, '').strip()
            # Track this violation
            self.metrics['protocol_violations'].append('tag_in_message')
        
        # Update state
        self.update_negotiation_state(action, is_self=True)
        self.negotiation_round += 1
        
        # Track metrics
        self.track_negotiation_metrics(response_text, is_outgoing=True)
        self.metrics['action_history'].append({
            'turn': self.turn_count,
            'side': 'self',
            'action': action.value,
            'price': price,
            'persona': str(self.persona.get('type', 'unknown')),
        })
        
        # Add to history
        self.negotiation_history.append({
            'turn': self.turn_count,
            'is_self': True,
            'text': response_text,
            'action': action.value,
            'price': price,
        })
        
        # Return ParlAI message
        return {
            'id': self.id,
            'text': response_text,
            'episode_done': self.negotiation_state.value in ['concluded', 'failed'] or self.terminated,
            # CRITICAL: Store action tag as backend metadata, NOT in text!
            'backend_tag': action.name,  # e.g., 'PROPOSE', 'COMMIT', etc.
            'action_tag': action.name
        }
    
    def get_agent_summary(self) -> Dict[str, Any]:
        """Get summary including persona and rule enforcement."""
        summary = super().get_negotiation_summary()
        
        # Get persona type as string
        persona_type = self.persona.get('type')
        if isinstance(persona_type, PersonaType):
            persona_type_str = persona_type.value
        else:
            persona_type_str = str(persona_type)
        
        # Add RuleAgent specific info
        summary.update({
            'persona': persona_type_str,
            'persona_traits': self.traits,
            'tactics_used': [t.value for t in self.used_tactics],
            'partner_violations': {k.value: v for k, v in self.partner_violations.items()},
            'enforcement_summary': self.rule_engine.get_enforcement_summary(),
            'terminated': self.terminated,
            'termination_reason': self.termination_reason,
        })
        
        return summary