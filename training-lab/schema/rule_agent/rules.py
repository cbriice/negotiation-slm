"""Rule engine for constraint enforcement in negotiations."""

from typing import Dict, Any, List, Tuple, Optional
from enum import Enum
import re
import json

class ViolationType(Enum):
    """Types of protocol violations."""
    MISSING_ACTION = "missing_action"
    INVALID_ACTION = "invalid_action"
    INVALID_STATE_TRANSITION = "invalid_state_transition"
    MISSING_TERMS = "missing_terms"
    INVALID_TERMS = "invalid_terms"
    TERMS_MISMATCH = "terms_mismatch"
    PRICE_OUT_OF_BOUNDS = "price_out_of_bounds"
    EXCESSIVE_LENGTH = "excessive_length"
    OFF_TOPIC = "off_topic"
    PROFANITY = "profanity"
    PERSONAL_ATTACK = "personal_attack"
    CIRCULAR_NEGOTIATION = "circular_negotiation"
    STALLING = "stalling"

class RuleEngine:
    """Enforces negotiation protocol rules and constraints."""
    
    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
        self.violation_history: List[Dict[str, Any]] = []
        self.warning_count = 0
        self.max_warnings = 3 if strict_mode else 5
        
        # Profanity/inappropriate content patterns
        self.inappropriate_patterns = [
            r'\b(stupid|idiot|moron|dumb)\b',
            r'\b(scam|fraud|cheat|steal)\b',
        ]
        
        # Price history for circular negotiation detection
        self.price_history: List[float] = []
        self.max_price_history = 6
    
    def check_message_violations(self, 
                                message: str,
                                action: Optional[str] = None,
                                state: str = "proposing",
                                turn_count: int = 0) -> Tuple[List[ViolationType], List[str]]:
        """Check a message for protocol violations."""
        violations = []
        details = []
        
        # Check for missing action tag
        if self.strict_mode and turn_count > 0:
            if not action:
                violations.append(ViolationType.MISSING_ACTION)
                details.append("Message must include an action tag")
        
        # Check message length
        if len(message) > 500:
            violations.append(ViolationType.EXCESSIVE_LENGTH)
            details.append(f"Message too long ({len(message)} chars, max 500)")
        
        # Check for inappropriate content
        for pattern in self.inappropriate_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                violations.append(ViolationType.PERSONAL_ATTACK)
                details.append("Message contains inappropriate language")
                break
        
        # Check for off-topic content (very basic)
        negotiation_keywords = ['price', 'offer', 'deal', 'sell', 'buy', 'item', 
                              'condition', 'payment', 'delivery', 'discount']
        if turn_count > 2:  # Allow some rapport building initially
            has_relevant_content = any(keyword in message.lower() 
                                      for keyword in negotiation_keywords)
            if not has_relevant_content and len(message) > 50:
                violations.append(ViolationType.OFF_TOPIC)
                details.append("Message appears off-topic")
        
        return violations, details
    
    def check_action_validity(self,
                             action: str,
                             current_state: str,
                             is_response_to: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """Check if an action is valid given the current state."""
        valid_transitions = {
            "initializing": ["[PROPOSE]"],
            "proposing": ["[PROPOSE]", "[COMMIT]", "[NO_DEAL]"],
            "awaiting_ack": ["[ACK]", "[NACK]", "[NO_DEAL]"],
            "committing": ["[ACK]", "[NACK]", "[NO_DEAL]"],
        }
        
        valid_actions = valid_transitions.get(current_state, [])
        
        # Special case: ACK can only follow COMMIT
        if action == "[ACK]" and is_response_to != "[COMMIT]":
            return False, "ACK can only be sent in response to COMMIT"
        
        # Special case: NACK should follow COMMIT or ACK attempt
        if action == "[NACK]" and is_response_to not in ["[COMMIT]", "[ACK]"]:
            return False, "NACK should follow COMMIT or ACK"
        
        if action not in valid_actions:
            return False, f"Action {action} not valid in state {current_state}"
        
        return True, None
    
    def check_terms_validity(self,
                            terms: Dict[str, Any],
                            side: str,
                            constraints: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate terms JSON against constraints."""
        errors = []
        
        # Check required fields
        required_fields = ['dialogue_id', 'side_signed', 'status']
        for field in required_fields:
            if field not in terms:
                errors.append(f"Missing required field: {field}")
        
        # Check side matches
        if terms.get('side_signed') != side:
            errors.append(f"side_signed must be '{side}'")
        
        # Check price constraints
        if 'total_price' in terms and terms['total_price'] is not None:
            price = terms['total_price']
            
            if price < 0:
                errors.append("Price cannot be negative")
            
            # Check against role constraints
            if side == 'buyer' and 'max_price' in constraints:
                if price > constraints['max_price']:
                    errors.append(f"Price ${price} exceeds maximum ${constraints['max_price']}")
            
            elif side == 'seller' and 'min_price' in constraints:
                if price < constraints['min_price']:
                    errors.append(f"Price ${price} below minimum ${constraints['min_price']}")
        
        # Check status validity
        valid_statuses = ['agreed', 'no_deal', 'timeout', 'rejected']
        if terms.get('status') not in valid_statuses:
            errors.append(f"Invalid status: {terms.get('status')}")
        
        return len(errors) == 0, errors
    
    def check_price_progression(self,
                               new_price: float,
                               side: str,
                               last_price: Optional[float] = None) -> Tuple[bool, Optional[str]]:
        """Check if price movement is logical."""
        if last_price is None:
            return True, None
        
        # Buyers should generally increase offers
        if side == 'buyer' and new_price < last_price * 0.95:  # Allow small decreases
            return False, "Buyer offers should generally increase"
        
        # Sellers should generally decrease asks  
        if side == 'seller' and new_price > last_price * 1.05:  # Allow small increases
            return False, "Seller asks should generally decrease"
        
        return True, None
    
    def detect_circular_negotiation(self, price: float) -> bool:
        """Detect if negotiation is going in circles."""
        self.price_history.append(price)
        if len(self.price_history) > self.max_price_history:
            self.price_history.pop(0)
        
        # Check if we've seen this price before recently
        if len(self.price_history) >= 4:
            recent_prices = self.price_history[-4:]
            if len(set(recent_prices)) <= 2:  # Only 2 unique prices in last 4
                return True
        
        return False
    
    def detect_stalling(self,
                       messages: List[str],
                       min_messages: int = 3) -> bool:
        """Detect if a party is stalling without progress."""
        if len(messages) < min_messages:
            return False
        
        # Check last few messages for repetition or no price movement
        recent = messages[-min_messages:]
        
        # Check for repetitive patterns
        unique_messages = len(set(msg[:50] for msg in recent))  # First 50 chars
        if unique_messages <= 1:
            return True
        
        # Check if no prices mentioned recently
        has_prices = any(re.search(r'\$?\d+', msg) for msg in recent)
        if not has_prices and len(recent) >= 3:
            return True
        
        return False
    
    def should_terminate(self, 
                        violation_counts: Dict[ViolationType, int],
                        total_warnings: int) -> Tuple[bool, Optional[str]]:
        """Determine if negotiation should be terminated."""
        # Immediate termination conditions
        critical_violations = [
            ViolationType.PROFANITY,
            ViolationType.PERSONAL_ATTACK,
        ]
        
        for violation in critical_violations:
            if violation_counts.get(violation, 0) > 0:
                return True, f"Terminated due to {violation.value}"
        
        # Warning-based termination
        if total_warnings >= self.max_warnings:
            return True, f"Exceeded maximum warnings ({self.max_warnings})"
        
        # Pattern-based termination
        if violation_counts.get(ViolationType.CIRCULAR_NEGOTIATION, 0) >= 2:
            return True, "Negotiation going in circles"
        
        if violation_counts.get(ViolationType.STALLING, 0) >= 2:
            return True, "Negotiation stalled without progress"
        
        return False, None
    
    def generate_warning(self, 
                        violations: List[ViolationType],
                        details: List[str]) -> str:
        """Generate a warning message for violations."""
        self.warning_count += 1
        
        warning = f"Warning {self.warning_count}/{self.max_warnings}: "
        
        if details:
            warning += "; ".join(details)
        else:
            warning += f"Protocol violation detected: {violations[0].value}"
        
        if self.warning_count == self.max_warnings - 1:
            warning += " (Next violation will terminate negotiation)"
        
        return warning
    
    def reset(self):
        """Reset the rule engine state."""
        self.violation_history.clear()
        self.warning_count = 0
        self.price_history.clear()
    
    def get_enforcement_summary(self) -> Dict[str, Any]:
        """Get summary of rule enforcement."""
        violation_counts = {}
        for violation in self.violation_history:
            vtype = violation.get('type')
            violation_counts[vtype] = violation_counts.get(vtype, 0) + 1
        
        return {
            'total_violations': len(self.violation_history),
            'warning_count': self.warning_count,
            'violation_breakdown': violation_counts,
            'strict_mode': self.strict_mode,
        }