"""
Calculate metrics for individual negotiation episodes.
"""

import json
import logging
import re
from typing import Dict, Any, List
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


class EpisodeMetrics:
    """
    Calculate metrics for individual negotiation episodes.
    """
    
    def __init__(self):
        self.protocol_keywords = {
            'PROPOSE': r'\[PROPOSE\]',
            'COMMIT': r'\[COMMIT\]',
            'ACK': r'\[ACK\]',
            'NACK': r'\[NACK\]',
            'NO_DEAL': r'\[NO_DEAL\]'
        }
        
    def calculate(self, episode: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate comprehensive metrics for a single episode.
        
        Args:
            episode: Episode data containing dialogue, state, and metadata
            
        Returns:
            Dictionary of calculated metrics
        """
        dialogue = episode.get('dialogue', [])
        state = episode.get('state', {})
        meta = episode.get('meta', {})
        
        # Basic identifiers
        metrics = {
            'episode_id': meta.get('episode_id', str(id(episode))),
            'dialogue_id': meta.get('dialogue_id', meta.get('episode_id', '')),
            'side': meta.get('side', 'unknown'),
            'seed': meta.get('seed', ''),
            'timestamp': datetime.now().isoformat()
        }
        
        # Turn and token counts
        metrics['turns'] = len(dialogue)
        metrics['total_tokens'] = self._count_total_tokens(dialogue)
        metrics['agent_tokens'] = self._count_agent_tokens(dialogue, meta.get('side'))
        
        # Deal outcome analysis
        deal_analysis = self._analyze_deal_outcome(dialogue)
        metrics['deal_closed'] = deal_analysis['closed']
        metrics['final_price'] = deal_analysis.get('final_price', 0.0)
        metrics['terms_match_pct'] = deal_analysis.get('terms_match_pct', 0.0)
        
        # Price analysis relative to target/limits
        price_analysis = self._analyze_price_performance(
            deal_analysis.get('final_price', 0.0),
            state,
            meta.get('side', 'buyer')
        )
        metrics['price_vs_target'] = price_analysis.get('vs_target', 0.0)
        metrics['price_vs_limit'] = price_analysis.get('vs_limit', 0.0)
        
        # Protocol compliance
        violations = self._check_protocol_violations(dialogue)
        metrics['protocol_violations'] = violations
        metrics['violation_count'] = len(violations)
        
        # Negotiation dynamics
        dynamics = self._analyze_negotiation_dynamics(dialogue)
        metrics.update(dynamics)
        
        # Communication quality
        quality = self._analyze_communication_quality(dialogue)
        metrics.update(quality)
        
        return metrics
    
    def _count_total_tokens(self, dialogue: List[Dict]) -> int:
        """Count total tokens across all turns."""
        total = 0
        for turn in dialogue:
            # Simple word-based approximation (can be replaced with tokenizer)
            text = turn.get('text', '') + ' ' + turn.get('response', '')
            total += len(text.split())
        return total
    
    def _count_agent_tokens(self, dialogue: List[Dict], agent_side: str) -> int:
        """Count tokens from the agent's side only."""
        total = 0
        for turn in dialogue:
            if turn.get('side') == agent_side:
                text = turn.get('response', turn.get('text', ''))
                total += len(text.split())
        return total
    
    def _analyze_deal_outcome(self, dialogue: List[Dict]) -> Dict[str, Any]:
        """Analyze whether a deal was closed and extract final terms."""
        result = {
            'closed': False,
            'final_price': 0.0,
            'terms_match_pct': 0.0
        }
        
        # Check for successful deal pattern: COMMIT -> ACK
        commit_found = False
        commit_terms = None
        
        for i, turn in enumerate(dialogue):
            text = turn.get('text', '') + ' ' + turn.get('response', '')
            
            # Look for COMMIT action
            if '[COMMIT]' in text:
                commit_found = True
                # Try to extract terms JSON
                try:
                    json_match = re.search(r'\{[^}]+\}', text)
                    if json_match:
                        commit_terms = json.loads(json_match.group())
                        result['final_price'] = commit_terms.get('price', 0.0)
                except (json.JSONDecodeError, AttributeError):
                    pass
            
            # Look for ACK after COMMIT
            elif commit_found and '[ACK]' in text:
                result['closed'] = True
                # Try to match ACK terms with COMMIT terms
                try:
                    json_match = re.search(r'\{[^}]+\}', text)
                    if json_match and commit_terms:
                        ack_terms = json.loads(json_match.group())
                        # Calculate terms match percentage
                        matching_keys = sum(1 for k in commit_terms 
                                          if k in ack_terms and commit_terms[k] == ack_terms[k])
                        total_keys = len(set(commit_terms.keys()) | set(ack_terms.keys()))
                        result['terms_match_pct'] = (matching_keys / total_keys * 100) if total_keys > 0 else 0
                except (json.JSONDecodeError, AttributeError):
                    pass
                break
            
            # NO_DEAL ends negotiation without agreement
            elif '[NO_DEAL]' in text:
                result['closed'] = False
                break
        
        return result
    
    def _analyze_price_performance(self, final_price: float, state: Dict, side: str) -> Dict[str, float]:
        """Analyze how well the final price performed against targets and limits."""
        analysis = {'vs_target': 0.0, 'vs_limit': 0.0}
        
        if final_price <= 0:
            return analysis
        
        if side == 'buyer' and 'buyer_state' in state:
            buyer = state['buyer_state']
            target = buyer.get('target_price', 0)
            max_price = buyer.get('max_price', 0)
            
            if target > 0:
                # Negative is better for buyer (below target)
                analysis['vs_target'] = ((final_price - target) / target) * 100
            if max_price > 0:
                # Negative is better for buyer (below max)
                analysis['vs_limit'] = ((final_price - max_price) / max_price) * 100
                
        elif side == 'seller' and 'seller_state' in state:
            seller = state['seller_state']
            target = seller.get('target_price', seller.get('list_price', 0))
            min_price = seller.get('min_price', 0)
            
            if target > 0:
                # Positive is better for seller (above target)
                analysis['vs_target'] = ((final_price - target) / target) * 100
            if min_price > 0:
                # Positive is better for seller (above min)
                analysis['vs_limit'] = ((final_price - min_price) / min_price) * 100
        
        return analysis
    
    def _check_protocol_violations(self, dialogue: List[Dict]) -> List[str]:
        """Check for protocol violations in the dialogue."""
        violations = []
        
        # Track negotiation state machine
        state = 'INITIAL'
        commit_made = False
        
        for i, turn in enumerate(dialogue):
            text = turn.get('text', '') + ' ' + turn.get('response', '')
            
            # Check for multiple action tags in one turn
            action_count = sum(1 for action in ['[PROPOSE]', '[COMMIT]', '[ACK]', '[NACK]', '[NO_DEAL]'] 
                             if action in text)
            if action_count > 1:
                violations.append(f"Turn {i}: Multiple action tags in single turn")
            
            # State machine validation
            if '[PROPOSE]' in text:
                if state == 'COMMITTED':
                    violations.append(f"Turn {i}: PROPOSE after COMMIT")
                state = 'NEGOTIATING'
                
            elif '[COMMIT]' in text:
                if commit_made:
                    violations.append(f"Turn {i}: Multiple COMMIT attempts")
                commit_made = True
                state = 'COMMITTED'
                # Check for TERMS_JSON
                if '{' not in text or '}' not in text:
                    violations.append(f"Turn {i}: COMMIT without TERMS_JSON")
                    
            elif '[ACK]' in text:
                if state != 'COMMITTED':
                    violations.append(f"Turn {i}: ACK without prior COMMIT")
                # Check for echoed TERMS_JSON
                if '{' not in text or '}' not in text:
                    violations.append(f"Turn {i}: ACK without echoing TERMS_JSON")
                    
            elif '[NACK]' in text:
                if state != 'COMMITTED':
                    violations.append(f"Turn {i}: NACK without prior COMMIT")
                # Check for explanation
                if len(text.split()) < 5:  # Arbitrary minimum for explanation
                    violations.append(f"Turn {i}: NACK without explanation")
                state = 'NEGOTIATING'
                
            elif '[NO_DEAL]' in text:
                if i < 2:  # Too early to walk away
                    violations.append(f"Turn {i}: Premature NO_DEAL")
        
        # Check for incomplete negotiation
        if len(dialogue) > 0 and state == 'COMMITTED' and not any('[ACK]' in (t.get('text', '') + t.get('response', '')) for t in dialogue):
            violations.append("Incomplete: COMMIT without ACK")
        
        return violations
    
    def _analyze_negotiation_dynamics(self, dialogue: List[Dict]) -> Dict[str, Any]:
        """Analyze negotiation patterns and dynamics."""
        dynamics = {
            'num_proposals': 0,
            'num_counteroffers': 0,
            'price_convergence': False,
            'avg_response_length': 0,
            'negotiation_rounds': 0
        }
        
        proposals = []
        last_side = None
        total_response_length = 0
        response_count = 0
        
        for turn in dialogue:
            text = turn.get('text', '') + ' ' + turn.get('response', '')
            side = turn.get('side')
            
            # Count proposals
            if '[PROPOSE]' in text:
                dynamics['num_proposals'] += 1
                # Extract price if possible
                price_match = re.search(r'\$?(\d+(?:\.\d{2})?)', text)
                if price_match:
                    proposals.append(float(price_match.group(1)))
                
                # Count as counteroffer if switching sides
                if last_side and side != last_side:
                    dynamics['num_counteroffers'] += 1
                    
            # Calculate average response length
            if response := turn.get('response'):
                total_response_length += len(response.split())
                response_count += 1
            
            last_side = side
        
        # Check for price convergence
        if len(proposals) >= 3:
            # Calculate if prices are converging (decreasing variance)
            first_half = proposals[:len(proposals)//2]
            second_half = proposals[len(proposals)//2:]
            if np.var(second_half) < np.var(first_half):
                dynamics['price_convergence'] = True
        
        # Average response length
        if response_count > 0:
            dynamics['avg_response_length'] = total_response_length / response_count
        
        # Count negotiation rounds (propose-respond pairs)
        dynamics['negotiation_rounds'] = dynamics['num_proposals'] // 2
        
        return dynamics
    
    def _analyze_communication_quality(self, dialogue: List[Dict]) -> Dict[str, Any]:
        """Analyze quality aspects of the communication."""
        quality = {
            'verbosity_score': 0.0,
            'uses_reasoning': False,
            'uses_empathy': False,
            'uses_urgency': False,
            'professionalism_score': 0.0
        }
        
        total_words = 0
        reasoning_keywords = ['because', 'since', 'therefore', 'considering', 'given that']
        empathy_keywords = ['understand', 'appreciate', 'sympathize', 'realize', 'respect']
        urgency_keywords = ['urgent', 'quickly', 'asap', 'immediately', 'today', 'now']
        unprofessional_indicators = ['!!!', '???', 'wtf', 'lol', 'lmao']
        
        for turn in dialogue:
            text = (turn.get('text', '') + ' ' + turn.get('response', '')).lower()
            words = text.split()
            total_words += len(words)
            
            # Check for reasoning
            if any(keyword in text for keyword in reasoning_keywords):
                quality['uses_reasoning'] = True
            
            # Check for empathy
            if any(keyword in text for keyword in empathy_keywords):
                quality['uses_empathy'] = True
            
            # Check for urgency tactics
            if any(keyword in text for keyword in urgency_keywords):
                quality['uses_urgency'] = True
            
            # Calculate professionalism (inverse of unprofessional indicators)
            unprofessional_count = sum(1 for indicator in unprofessional_indicators if indicator in text)
            quality['professionalism_score'] -= unprofessional_count * 0.1
        
        # Verbosity score (words per turn)
        if len(dialogue) > 0:
            quality['verbosity_score'] = total_words / len(dialogue)
        
        # Normalize professionalism score
        quality['professionalism_score'] = max(0, min(1, 1 + quality['professionalism_score']))
        
        return quality