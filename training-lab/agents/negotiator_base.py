import json
import re
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from parlai.core.agents import Agent

class NegotiationState(Enum):
    """States of a negotiation conversation."""
    INITIALIZING = "initializing"
    PROPOSING = "proposing"
    COMMITTING = "committing"
    AWAITING_ACK = "awaiting_ack"
    CONCLUDED = "concluded"
    FAILED = "failed"

class NegotiationAction(Enum):
    """Valid negotiation actions/tags."""
    PROPOSE = "[PROPOSE]"
    COMMIT = "[COMMIT]"
    ACK = "[ACK]"
    NACK = "[NACK]"
    NO_DEAL = "[NO_DEAL]"

class NegotiatorBase(Agent):
    """Base class for negotiation agents in ParlAI."""
    
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        
        # Negotiation-specific state
        self.negotiation_history: List[Dict[str, Any]] = []
        self.current_state: Dict[str, Any] = {}  # From state_template.jsonl
        self.side: Optional[str] = None  # 'buyer' or 'seller'
        self.terms_buffer: Optional[Dict[str, Any]] = None  # Proposed terms
        self.negotiation_state = NegotiationState.INITIALIZING
        self.turn_count = 0
        self.max_turns = opt.get('max_turns', 20)
        
        # Strategy tracking for two-part response
        self.strategy_history: List[Dict[str, Any]] = []
        self.current_strategy: Optional[Dict[str, Any]] = None
        
        # Role-specific state (populated by initialize_from_state)
        self.role_state: Dict[str, Any] = {}
        
        # Store observation for subclasses
        self.observation = None
        
        # Metrics tracking
        self.metrics = {
            'turns': 0,
            'tokens_sent': 0,
            'tokens_received': 0,
            'protocol_violations': [],
            'action_history': [],
        }
    
    def parse_negotiation_action(self, text: str) -> Tuple[Optional[NegotiationAction], str]:
        """
        Extract negotiation action tag from text.
        Returns (action, cleaned_text).
        """
        for action in NegotiationAction:
            if action.value in text:
                # Extract the action and the rest of the message
                cleaned = text.replace(action.value, '').strip()
                return action, cleaned
        return None, text
    
    def extract_terms_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract and validate TERMS_JSON from message."""
        # Look for JSON block - find first { and last }
        first_brace = text.find('{')
        last_brace = text.rfind('}')
        
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            try:
                json_str = text[first_brace:last_brace + 1]
                terms = json.loads(json_str)
                # Basic validation - check for required fields
                if 'dialogue_id' in terms and 'status' in terms:
                    return terms
            except json.JSONDecodeError:
                pass
        return None
    
    def validate_strategy(self, strategy: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate strategy JSON structure."""
        errors = []
        
        # Check required fields
        required_fields = ['action_indicator', 'target_price', 'tone', 'tactics', 'reasoning']
        for field in required_fields:
            if field not in strategy:
                errors.append(f"Missing required field: {field}")
        
        # Validate action_indicator
        if 'action_indicator' in strategy:
            valid_actions = ['PROPOSE', 'COMMIT', 'ACK', 'NACK', 'NO_DEAL']
            if strategy['action_indicator'] not in valid_actions:
                errors.append(f"Invalid action_indicator: {strategy['action_indicator']}")
        
        # Validate tone
        if 'tone' in strategy:
            valid_tones = ['aggressive', 'reasonable', 'sympathetic', 'firm']
            if strategy['tone'] not in valid_tones:
                errors.append(f"Invalid tone: {strategy['tone']}")
        
        # Validate target_price
        if 'target_price' in strategy:
            if not isinstance(strategy['target_price'], (int, float)) or strategy['target_price'] < 0:
                errors.append("target_price must be a non-negative number")
        
        # Validate tactics is a list
        if 'tactics' in strategy:
            if not isinstance(strategy['tactics'], list):
                errors.append("tactics must be a list")
        
        return len(errors) == 0, errors
    
    def validate_terms_json(self, terms: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate terms against TermsJSON schema.
        Returns (is_valid, list_of_errors).
        """
        errors = []
        
        # Check required fields
        required_fields = ['dialogue_id', 'side_signed', 'status']
        for field in required_fields:
            if field not in terms:
                errors.append(f"Missing required field: {field}")
        
        # Validate status value
        if 'status' in terms:
            valid_statuses = ['agreed', 'no_deal', 'timeout', 'rejected']
            if terms['status'] not in valid_statuses:
                errors.append(f"Invalid status: {terms['status']}")
        
        # Validate side_signed
        if 'side_signed' in terms:
            if terms['side_signed'] not in ['buyer', 'seller']:
                errors.append(f"Invalid side_signed: {terms['side_signed']}")
        
        # Check price is non-negative if present
        if 'total_price' in terms and terms['total_price'] is not None:
            if terms['total_price'] < 0:
                errors.append("total_price must be >= 0")
        
        return len(errors) == 0, errors
    
    def format_response(self, text: str, action: NegotiationAction, 
                       terms: Optional[Dict[str, Any]] = None) -> str:
        """Format response with appropriate action tag and optional terms."""
        response = f"{action.value} {text}"
        
        if terms:
            # Add terms JSON to response
            terms_str = json.dumps(terms, indent=2)
            response = f"{response}\n\nTERMS:\n{terms_str}"
        
        return response
    
    def update_negotiation_state(self, action: NegotiationAction, 
                                is_self: bool = True) -> None:
        """Update internal state machine based on action."""
        current = self.negotiation_state
        
        # State transitions
        if action == NegotiationAction.PROPOSE:
            if current in [NegotiationState.INITIALIZING, NegotiationState.PROPOSING, 
                          NegotiationState.AWAITING_ACK]:
                self.negotiation_state = NegotiationState.PROPOSING
        
        elif action == NegotiationAction.COMMIT:
            if current == NegotiationState.PROPOSING:
                self.negotiation_state = NegotiationState.AWAITING_ACK
        
        elif action == NegotiationAction.ACK:
            if current == NegotiationState.AWAITING_ACK and not is_self:
                self.negotiation_state = NegotiationState.CONCLUDED
        
        elif action == NegotiationAction.NACK:
            if current == NegotiationState.AWAITING_ACK:
                self.negotiation_state = NegotiationState.PROPOSING
        
        elif action == NegotiationAction.NO_DEAL:
            self.negotiation_state = NegotiationState.FAILED
    
    def track_negotiation_metrics(self, message: str, is_outgoing: bool = True) -> None:
        """Track metrics for the current turn."""
        # Simple token estimation (words * 1.3)
        token_estimate = int(len(message.split()) * 1.3)
        
        if is_outgoing:
            self.metrics['tokens_sent'] += token_estimate
        else:
            self.metrics['tokens_received'] += token_estimate
        
        self.metrics['turns'] = self.turn_count
    
    def check_protocol_violations(self, action: Optional[NegotiationAction], 
                                 message: str) -> List[str]:
        """Check for protocol violations in the current turn."""
        violations = []
        
        # Check for missing action tag
        if action is None and self.negotiation_state != NegotiationState.INITIALIZING:
            violations.append("missing_action_tag")
        
        # Check for invalid state transitions
        if action == NegotiationAction.COMMIT:
            if self.negotiation_state not in [NegotiationState.PROPOSING]:
                violations.append("invalid_commit_timing")
        
        # Check for TERMS_JSON in COMMIT messages
        if action == NegotiationAction.COMMIT:
            terms = self.extract_terms_json(message)
            if terms is None:
                violations.append("commit_missing_terms")
            else:
                is_valid, errors = self.validate_terms_json(terms)
                if not is_valid:
                    violations.append(f"invalid_terms_json: {errors[0] if errors else 'unknown'}")
        
        # Check for ACK without matching terms
        if action == NegotiationAction.ACK:
            terms = self.extract_terms_json(message)
            if terms is None:
                violations.append("ack_missing_terms")
            elif self.terms_buffer and terms != self.terms_buffer:
                violations.append("ack_terms_mismatch")
        
        return violations
    
    def initialize_from_state(self, state: Dict[str, Any], side: str) -> None:
        """Initialize agent from a state seed."""
        self.current_state = state
        self.side = side
        self.negotiation_state = NegotiationState.PROPOSING
        
        # Extract role-specific information
        if self.side == 'buyer':
            self.role_state = state.get('buyer_state', {})
        else:
            self.role_state = state.get('seller_state', {})
    
    def get_negotiation_summary(self) -> Dict[str, Any]:
        """Get summary of the negotiation for logging."""
        return {
            'side': self.side if self.side else None,
            'final_state': self.negotiation_state.value,
            'turns': self.turn_count,
            'tokens_sent': self.metrics['tokens_sent'],
            'tokens_received': self.metrics['tokens_received'],
            'protocol_violations': self.metrics['protocol_violations'],
            'action_history': self.metrics['action_history'],
            'deal_closed': self.negotiation_state == NegotiationState.CONCLUDED,
            'final_terms': self.terms_buffer,
        }
    
    def act(self):
        """Main ParlAI interface - must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement act()")
    
    def observe(self, observation):
        """Process incoming message and update state."""
        # Update observation in memory
        self.observation = observation
        
        # Parse incoming message
        if 'text' in observation and observation['text']:
            text = observation['text']
            action, cleaned_text = self.parse_negotiation_action(text)
            
            # Track metrics
            self.track_negotiation_metrics(text, is_outgoing=False)
            
            # Check for violations
            violations = self.check_protocol_violations(action, text)
            if violations:
                self.metrics['protocol_violations'].extend(violations)
            
            # Update state based on partner's action
            if action:
                self.update_negotiation_state(action, is_self=False)
                self.metrics['action_history'].append({
                    'turn': self.turn_count,
                    'side': 'partner',
                    'action': action.value
                })
            
            # Store terms if present
            if action in [NegotiationAction.COMMIT, NegotiationAction.ACK]:
                terms = self.extract_terms_json(text)
                if terms:
                    self.terms_buffer = terms
        
        # Increment turn count
        self.turn_count += 1
        
        # Check for max turns
        if self.turn_count >= self.max_turns:
            self.negotiation_state = NegotiationState.FAILED
        
        return observation