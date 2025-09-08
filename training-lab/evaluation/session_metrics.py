"""
Aggregate metrics across multiple episodes in a training session.
"""

import logging
from typing import Dict, Any, List
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


class SessionMetrics:
    """
    Aggregate metrics across multiple episodes in a training session.
    """
    
    def __init__(self):
        self.start_time = datetime.now()
        
    def aggregate(self, episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate aggregate statistics across all episodes.
        
        Args:
            episodes: List of episode metrics dictionaries
            
        Returns:
            Dictionary of aggregated session metrics
        """
        if not episodes:
            return self._empty_metrics()
        
        metrics = {
            'session_id': f"session_{self.start_time.strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': (datetime.now() - self.start_time).total_seconds(),
            'total_episodes': len(episodes)
        }
        
        # Calculate averages and totals
        total_turns = sum(ep.get('turns', 0) for ep in episodes)
        total_tokens = sum(ep.get('total_tokens', 0) for ep in episodes)
        agent_tokens = sum(ep.get('agent_tokens', 0) for ep in episodes)
        deals_closed = sum(1 for ep in episodes if ep.get('deal_closed', False))
        total_violations = sum(ep.get('violation_count', 0) for ep in episodes)
        
        metrics['avg_turns'] = total_turns / len(episodes)
        metrics['avg_tokens'] = total_tokens / len(episodes)
        metrics['avg_agent_tokens'] = agent_tokens / len(episodes)
        metrics['deals_closed_pct'] = (deals_closed / len(episodes)) * 100
        metrics['avg_violations_per_episode'] = total_violations / len(episodes)
        
        # Token throughput (tps, ctps, ltps from CLAUDE.md)
        if metrics['duration_seconds'] > 0:
            metrics['tokens_per_second'] = total_tokens / metrics['duration_seconds']
            metrics['conversations_per_second'] = len(episodes) / metrics['duration_seconds']
            metrics['agent_tokens_per_second'] = agent_tokens / metrics['duration_seconds']
        else:
            metrics['tokens_per_second'] = 0
            metrics['conversations_per_second'] = 0
            metrics['agent_tokens_per_second'] = 0
        
        # Terms matching performance
        terms_matches = [ep.get('terms_match_pct', 0) for ep in episodes if ep.get('deal_closed')]
        if terms_matches:
            metrics['avg_terms_match'] = np.mean(terms_matches)
            metrics['std_terms_match'] = np.std(terms_matches)
        else:
            metrics['avg_terms_match'] = 0
            metrics['std_terms_match'] = 0
        
        # Price performance (separated by side)
        buyer_episodes = [ep for ep in episodes if ep.get('side') == 'buyer']
        seller_episodes = [ep for ep in episodes if ep.get('side') == 'seller']
        
        if buyer_episodes:
            buyer_prices = [ep.get('price_vs_target', 0) for ep in buyer_episodes if ep.get('deal_closed')]
            if buyer_prices:
                metrics['buyer_avg_price_vs_target'] = np.mean(buyer_prices)
                metrics['buyer_deals_closed_pct'] = (sum(1 for ep in buyer_episodes if ep.get('deal_closed')) / len(buyer_episodes)) * 100
            else:
                metrics['buyer_avg_price_vs_target'] = 0
                metrics['buyer_deals_closed_pct'] = 0
        
        if seller_episodes:
            seller_prices = [ep.get('price_vs_target', 0) for ep in seller_episodes if ep.get('deal_closed')]
            if seller_prices:
                metrics['seller_avg_price_vs_target'] = np.mean(seller_prices)
                metrics['seller_deals_closed_pct'] = (sum(1 for ep in seller_episodes if ep.get('deal_closed')) / len(seller_episodes)) * 100
            else:
                metrics['seller_avg_price_vs_target'] = 0
                metrics['seller_deals_closed_pct'] = 0
        
        # Negotiation dynamics
        metrics['avg_proposals'] = np.mean([ep.get('num_proposals', 0) for ep in episodes])
        metrics['avg_rounds'] = np.mean([ep.get('negotiation_rounds', 0) for ep in episodes])
        metrics['price_convergence_rate'] = (sum(1 for ep in episodes if ep.get('price_convergence')) / len(episodes)) * 100
        
        # Communication quality
        metrics['avg_verbosity'] = np.mean([ep.get('verbosity_score', 0) for ep in episodes])
        metrics['reasoning_usage_pct'] = (sum(1 for ep in episodes if ep.get('uses_reasoning')) / len(episodes)) * 100
        metrics['empathy_usage_pct'] = (sum(1 for ep in episodes if ep.get('uses_empathy')) / len(episodes)) * 100
        metrics['urgency_usage_pct'] = (sum(1 for ep in episodes if ep.get('uses_urgency')) / len(episodes)) * 100
        metrics['avg_professionalism'] = np.mean([ep.get('professionalism_score', 0) for ep in episodes])
        
        # Violation breakdown
        all_violations = []
        for ep in episodes:
            all_violations.extend(ep.get('protocol_violations', []))
        
        if all_violations:
            # Count violation types
            violation_types = {}
            for violation in all_violations:
                vtype = violation.split(':')[1].strip() if ':' in violation else violation
                violation_types[vtype] = violation_types.get(vtype, 0) + 1
            
            # Get top 3 most common violations
            sorted_violations = sorted(violation_types.items(), key=lambda x: x[1], reverse=True)[:3]
            metrics['top_violations'] = [f"{v[0]} ({v[1]}x)" for v in sorted_violations]
        else:
            metrics['top_violations'] = []
        
        return metrics
    
    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics structure when no episodes available."""
        return {
            'session_id': f"session_{self.start_time.strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': 0,
            'total_episodes': 0,
            'avg_turns': 0,
            'avg_tokens': 0,
            'avg_agent_tokens': 0,
            'deals_closed_pct': 0,
            'avg_violations_per_episode': 0,
            'tokens_per_second': 0,
            'conversations_per_second': 0,
            'agent_tokens_per_second': 0,
            'avg_terms_match': 0,
            'std_terms_match': 0,
            'avg_proposals': 0,
            'avg_rounds': 0,
            'price_convergence_rate': 0,
            'avg_verbosity': 0,
            'reasoning_usage_pct': 0,
            'empathy_usage_pct': 0,
            'urgency_usage_pct': 0,
            'avg_professionalism': 0,
            'top_violations': []
        }