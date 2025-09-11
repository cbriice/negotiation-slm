"""
Dialogue flow management for two-part strategy/response generation.
Handles the flow: strategy generation → action extraction → response generation.
"""

import json
import re
from typing import Dict, Any, Optional, Tuple
from enum import Enum

from agents.negotiator_base import NegotiationAction


class DialogueFlowManager:
    """
    Manages the two-part dialogue flow for negotiation agents.
    
    Flow:
    1. Generate strategy JSON with action indicator
    2. Extract action from strategy
    3. Generate response using action-specific prompt
    """
    
    def __init__(self):
        self.action_prompts = ActionPromptGenerator()
    
    def parse_strategy_action(self, strategy: Dict[str, Any]) -> Optional[str]:
        """
        Extract action indicator from strategy JSON.
        
        Args:
            strategy: Strategy JSON containing 'action_indicator' field
            
        Returns:
            Action string (PROPOSE, COMMIT, ACK, NACK, NO_DEAL) or None
        """
        if isinstance(strategy, dict):
            return strategy.get('action_indicator')
        elif isinstance(strategy, str):
            # Try to parse as JSON if it's a string
            try:
                strategy_dict = json.loads(strategy)
                return strategy_dict.get('action_indicator')
            except (json.JSONDecodeError, AttributeError):
                # Try regex fallback for action indicators
                pattern = r'"action_indicator"\s*:\s*"([A-Z_]+)"'
                match = re.search(pattern, strategy)
                if match:
                    return match.group(1)
        return None
    
    def validate_action(self, action: str) -> bool:
        """Validate that action is a valid negotiation action."""
        valid_actions = ['PROPOSE', 'COMMIT', 'ACK', 'NACK', 'NO_DEAL']
        return action in valid_actions
    
    def generate_response_prompt(self, 
                                strategy: Dict[str, Any],
                                action: str,
                                context: Dict[str, Any]) -> str:
        """
        Generate action-specific prompt for response generation.
        
        Args:
            strategy: Complete strategy JSON
            action: Extracted action indicator
            context: Negotiation context (state, history, etc.)
            
        Returns:
            Formatted prompt for response generation
        """
        return self.action_prompts.generate_prompt(action, strategy, context)
    
    def extract_strategy_metadata(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract useful metadata from strategy for logging.
        
        Args:
            strategy: Strategy JSON
            
        Returns:
            Metadata dictionary
        """
        metadata = {}
        
        if isinstance(strategy, dict):
            metadata['action'] = strategy.get('action_indicator')
            metadata['target_price'] = strategy.get('target_price')
            metadata['tone'] = strategy.get('tone')
            metadata['tactics'] = strategy.get('tactics', [])
            metadata['reasoning'] = strategy.get('reasoning')
            metadata['confidence'] = strategy.get('confidence')
        
        return metadata

class ActionPromptGenerator:
    """Generates action-specific prompts for response generation."""
    
    def generate_prompt(self, action: str, strategy: Dict[str, Any], 
                       context: Dict[str, Any]) -> str:
        """
        Generate prompt based on action type.
        
        Args:
            action: Action indicator (PROPOSE, COMMIT, etc.)
            strategy: Complete strategy JSON
            context: Negotiation context
            
        Returns:
            Action-specific prompt
        """
        if action == 'PROPOSE':
            return self._propose_prompt(strategy, context)
        elif action == 'COMMIT':
            return self._commit_prompt(strategy, context)
        elif action == 'ACK':
            return self._ack_prompt(strategy, context)
        elif action == 'NACK':
            return self._nack_prompt(strategy, context)
        elif action == 'NO_DEAL':
            return self._no_deal_prompt(strategy, context)
        else:
            # Fallback to generic prompt
            return self._generic_prompt(strategy, context)
    
    def _propose_prompt(self, strategy: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate PROPOSE action prompt."""
        side = context.get('side', 'negotiator')
        target_price = strategy.get('target_price', 0)
        tactics = strategy.get('tactics', [])
        
        prompt = f"""You are a {side}. Target price: ${target_price:.0f}.
        
        Make an offer around ${target_price:.0f}. Negotiation tactics: {tactics}"""
        
        return prompt
    
    def _commit_prompt(self, strategy: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate COMMIT action prompt."""
        side = context.get('side', 'negotiator')
        target_price = strategy.get('target_price', 0)
        state = context.get('state', {})
        
        # Generate terms JSON
        terms = {
            "dialogue_id": context.get('dialogue_id', 'unknown'),
            "side_signed": side,
            "status": "agreed",
            "total_price": target_price,
            "items": [state.get('item', {}).get('name', 'Unknown Item')],
            "conditions": [],
            "metadata": {
                "strategy_confidence": strategy.get('confidence', 0.5)
            }
        }
        
        terms_json = json.dumps(terms, indent=2)
        
        prompt = f"""You are a {side}. Agree to ${target_price:.0f}.
        
        Include this TERMS_JSON:
        {terms_json}"""
        
        return prompt
    
    def _ack_prompt(self, strategy: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate ACK action prompt."""
        side = context.get('side', 'negotiator')
        terms = context.get('partner_terms', {})
        
        terms_json = json.dumps(terms, indent=2) if terms else ""
        
        prompt = f"""You are a {side}. Accept the deal.
        
        {f"Echo these terms: {terms_json}" if terms_json else ""}"""
        
        return prompt
    
    def _nack_prompt(self, strategy: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate NACK action prompt."""
        side = context.get('side', 'negotiator')
        reasoning = strategy.get('reasoning', 'The terms need adjustment')
        
        prompt = f"""You are a {side}. Reject the terms.
        
        Reason: {reasoning}"""
        
        return prompt
    
    def _no_deal_prompt(self, strategy: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate NO_DEAL action prompt."""
        side = context.get('side', 'negotiator')
        reasoning = strategy.get('reasoning', 'Unable to reach agreement')
        
        prompt = f"""You are a {side}. Walk away.
        
        Reason: {reasoning}"""
        
        return prompt
    
    def _generic_prompt(self, strategy: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Fallback generic prompt."""
        side = context.get('side', 'negotiator')
        
        prompt = f"""You are a {side}.
        
        Strategy: {json.dumps(strategy)}
        
        Respond based on strategy."""
        
        return prompt