"""
Episode scoring utilities for quality assessment.
"""

from typing import Dict, Any, List


class EpisodeScorer:
    """Calculate quality scores for negotiation episodes."""
    
    def __init__(self, 
                 deal_weight: float = 10.0,
                 efficiency_weight: float = 0.5,
                 violation_penalty: float = 5.0,
                 terms_weight: float = 10.0):
        """
        Initialize scorer with configurable weights.
        
        Args:
            deal_weight: Points for successfully closing a deal
            efficiency_weight: Penalty per turn over optimal (5 turns)
            violation_penalty: Penalty per protocol violation
            terms_weight: Points multiplier for terms met percentage
        """
        self.deal_weight = deal_weight
        self.efficiency_weight = efficiency_weight
        self.violation_penalty = violation_penalty
        self.terms_weight = terms_weight
        self.optimal_turns = 5
    
    def calculate_score(self, episode: Dict[str, Any]) -> float:
        """
        Calculate a quality score for an episode.
        
        Args:
            episode: Episode dictionary with dialogue, meta, and metrics
            
        Returns:
            Numerical quality score
        """
        score = 0.0
        
        meta = episode.get('meta', {})
        metrics = episode.get('metrics', {})
        
        # Deal closure bonus
        if meta.get('deal_closed', False) or metrics.get('deal_closed', False):
            score += self.deal_weight
        
        # Turn efficiency (fewer is better)
        num_turns = len(episode.get('dialogue', []))
        if num_turns > self.optimal_turns:
            score -= self.efficiency_weight * (num_turns - self.optimal_turns)
        
        # Protocol compliance
        violations = metrics.get('protocol_violations', [])
        score -= self.violation_penalty * len(violations)
        
        # Terms met percentage
        terms_met_pct = metrics.get('terms_met_percentage', 0.0)
        score += self.terms_weight * (terms_met_pct / 100.0)
        
        # Additional quality indicators
        if metrics.get('hallucination_detected', False):
            score -= 3.0
        
        if metrics.get('verbosity_score', 0) > 0.7:
            score -= 2.0
        
        # Bonus for good negotiation tactics
        if metrics.get('used_bundling', False):
            score += 1.0
        
        if metrics.get('made_concessions', False):
            score += 0.5
        
        return score
    
    def rank_episodes(self, episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank episodes by quality score.
        
        Args:
            episodes: List of episode dictionaries
            
        Returns:
            Episodes sorted by score (highest first)
        """
        scored_episodes = []
        for episode in episodes:
            score = self.calculate_score(episode)
            episode_with_score = episode.copy()
            episode_with_score['quality_score'] = score
            scored_episodes.append(episode_with_score)
        
        return sorted(scored_episodes, key=lambda x: x['quality_score'], reverse=True)
    
    def classify_episode_quality(self, episode: Dict[str, Any]) -> str:
        """
        Classify episode into quality tiers.
        
        Args:
            episode: Episode dictionary
            
        Returns:
            Quality tier: 'excellent', 'good', 'fair', 'poor'
        """
        score = self.calculate_score(episode)
        
        if score >= 15:
            return 'excellent'
        elif score >= 8:
            return 'good'
        elif score >= 0:
            return 'fair'
        else:
            return 'poor'
    
    def get_score_breakdown(self, episode: Dict[str, Any]) -> Dict[str, float]:
        """
        Get detailed breakdown of score components.
        
        Args:
            episode: Episode dictionary
            
        Returns:
            Dictionary with score component breakdown
        """
        breakdown = {
            'total': 0.0,
            'deal_bonus': 0.0,
            'efficiency': 0.0,
            'violations': 0.0,
            'terms_met': 0.0,
            'quality_modifiers': 0.0
        }
        
        meta = episode.get('meta', {})
        metrics = episode.get('metrics', {})
        
        # Deal closure
        if meta.get('deal_closed', False) or metrics.get('deal_closed', False):
            breakdown['deal_bonus'] = self.deal_weight
        
        # Turn efficiency
        num_turns = len(episode.get('dialogue', []))
        if num_turns > self.optimal_turns:
            breakdown['efficiency'] = -self.efficiency_weight * (num_turns - self.optimal_turns)
        
        # Protocol violations
        violations = metrics.get('protocol_violations', [])
        breakdown['violations'] = -self.violation_penalty * len(violations)
        
        # Terms met
        terms_met_pct = metrics.get('terms_met_percentage', 0.0)
        breakdown['terms_met'] = self.terms_weight * (terms_met_pct / 100.0)
        
        # Quality modifiers
        quality_mod = 0.0
        if metrics.get('hallucination_detected', False):
            quality_mod -= 3.0
        if metrics.get('verbosity_score', 0) > 0.7:
            quality_mod -= 2.0
        if metrics.get('used_bundling', False):
            quality_mod += 1.0
        if metrics.get('made_concessions', False):
            quality_mod += 0.5
        breakdown['quality_modifiers'] = quality_mod
        
        # Total
        breakdown['total'] = sum(v for k, v in breakdown.items() if k != 'total')
        
        return breakdown