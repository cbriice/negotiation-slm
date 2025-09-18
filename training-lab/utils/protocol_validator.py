"""Protocol validation utilities for negotiation training.

This module provides helper functions for validating negotiation protocols,
canonicalizing terms, and scoring violations without duplicating existing logic.

CRITICAL: Action tags ([PROPOSE], [COMMIT], etc.) are BACKEND ONLY and should 
NEVER appear in agent message text. They are tracked separately as metadata.
"""

import json
from typing import Dict, Any, List, Tuple, Optional, Union
from enum import Enum

class ViolationSeverity(Enum):
    """Severity levels for protocol violations."""
    CRITICAL = 5  # Immediate termination worthy
    HIGH = 4      # Major protocol break
    MEDIUM = 3    # Significant but recoverable
    LOW = 2       # Minor issue
    WARNING = 1   # Informational

def check_for_leaked_tags(message_text: str) -> Tuple[bool, List[str]]:
    """
    Check if action tags leaked into the message text.
    This is a CRITICAL violation as tags should NEVER appear in messages.
    
    Args:
        message_text: The actual message text sent by agent
        
    Returns:
        (has_leaked_tags, list_of_found_tags)
    """
    action_tags = ['[PROPOSE]', '[COMMIT]', '[ACK]', '[NACK]', '[NO_DEAL]']
    found_tags = []
    
    for tag in action_tags:
        if tag in message_text:
            found_tags.append(tag)
    
    return len(found_tags) > 0, found_tags

def canonicalize_terms_json(terms: Dict[str, Any]) -> str:
    """
    Canonicalize TERMS_JSON for equivalence checking.
    
    Args:
        terms: Terms dictionary to canonicalize
        
    Returns:
        Canonical JSON string for comparison
    """
    # Remove metadata fields that shouldn't affect equivalence
    canonical = {}
    
    # Core fields that must match for ACK
    core_fields = [
        'dialogue_id', 'side_signed', 'status', 
        'total_price', 'currency', 'line_items'
    ]
    
    for field in core_fields:
        if field in terms:
            canonical[field] = terms[field]
    
    # Normalize line_items if present
    if 'line_items' in canonical and canonical['line_items']:
        # Sort by name for consistent ordering
        canonical['line_items'] = sorted(
            canonical['line_items'], 
            key=lambda x: x.get('name', '') if isinstance(x, dict) else str(x)
        )
    
    # Sort keys and use compact JSON
    return json.dumps(canonical, sort_keys=True, separators=(',', ':'))

def check_terms_equivalence(terms1: Dict[str, Any], terms2: Dict[str, Any]) -> bool:
    """
    Check if two TERMS_JSON objects are equivalent for ACK purposes.
    
    Args:
        terms1: First terms dictionary
        terms2: Second terms dictionary
        
    Returns:
        True if terms are equivalent
    """
    return canonicalize_terms_json(terms1) == canonicalize_terms_json(terms2)

def validate_action_sequence(action_history: List[Union[Dict[str, Any], str]], 
                            max_lookback: int = 10) -> Tuple[bool, List[str]]:
    """
    Validate action sequences follow proper protocol patterns.
    
    NOTE: Actions are backend metadata, NOT part of message text!
    
    Args:
        action_history: List of dicts with 'action' (backend tag), 'side', 'turn' keys
                       OR list of strings (for backward compatibility)
        max_lookback: Maximum number of recent actions to validate
        
    Returns:
        (is_valid, list_of_violations)
    """
    violations = []
    
    if not action_history:
        return True, []
    
    # Convert string format to dict format for validation
    if action_history and isinstance(action_history[0], str):
        action_history = [{'action': action, 'turn': i} for i, action in enumerate(action_history)]
    
    # Look at recent history
    recent = action_history[-max_lookback:] if len(action_history) > max_lookback else action_history
    
    # Check for invalid patterns (using backend action metadata)
    for i in range(len(recent) - 1):
        curr = recent[i].get('action', '')  # Backend tag, not in message
        next_action = recent[i + 1].get('action', '')
        curr_side = recent[i].get('side', 'unknown')
        next_side = recent[i + 1].get('side', 'unknown')
        
        # ACK must follow COMMIT from other party
        if next_action == 'ACK' or next_action == '[ACK]':
            if curr not in ['COMMIT', '[COMMIT]']:
                violations.append(f"ACK without preceding COMMIT at turn {recent[i+1].get('turn', i+1)}")
            elif curr_side == next_side:
                violations.append(f"ACK from same party as COMMIT at turn {recent[i+1].get('turn', i+1)}")
        
        # NACK should follow COMMIT
        if next_action in ['NACK', '[NACK]'] and curr not in ['COMMIT', '[COMMIT]', 'ACK', '[ACK]']:
            violations.append(f"NACK without COMMIT/ACK context at turn {recent[i+1].get('turn', i+1)}")
        
        # Check for repeated COMMITs without resolution
        if curr in ['COMMIT', '[COMMIT]'] and next_action in ['COMMIT', '[COMMIT]'] and curr_side == next_side:
            violations.append(f"Repeated COMMIT from same party at turn {recent[i+1].get('turn', i+1)}")
    
    # Check for proper conclusion
    if len(recent) >= 2:
        last_two = recent[-2:]
        if last_two[0].get('action') in ['COMMIT', '[COMMIT]'] and last_two[1].get('action') in ['ACK', '[ACK]']:
            if last_two[0].get('side') == last_two[1].get('side'):
                violations.append("Invalid conclusion: ACK from same party as COMMIT")
    
    return len(violations) == 0, violations

def score_violation_severity(violation_type: str) -> int:
    """
    Score the severity of a protocol violation.
    
    Args:
        violation_type: Type of violation
        
    Returns:
        Severity score (1-5)
    """
    severity_map = {
        # Critical violations (5)
        'profanity': ViolationSeverity.CRITICAL.value,
        'personal_attack': ViolationSeverity.CRITICAL.value,
        'terms_forgery': ViolationSeverity.CRITICAL.value,
        
        # High severity (4)
        'tag_in_message': ViolationSeverity.HIGH.value,  # Tags should NEVER appear in messages
        'invalid_state_transition': ViolationSeverity.HIGH.value,
        'missing_terms': ViolationSeverity.HIGH.value,
        'terms_mismatch': ViolationSeverity.HIGH.value,
        'ack_without_commit': ViolationSeverity.HIGH.value,
        'ack_terms_mismatch': ViolationSeverity.HIGH.value,
        'commit_missing_terms': ViolationSeverity.HIGH.value,
        
        # Medium severity (3)
        'missing_action': ViolationSeverity.MEDIUM.value,
        'invalid_action': ViolationSeverity.MEDIUM.value,
        'circular_negotiation': ViolationSeverity.MEDIUM.value,
        'price_out_of_bounds': ViolationSeverity.MEDIUM.value,
        'invalid_commit_timing': ViolationSeverity.MEDIUM.value,
        
        # Low severity (2)
        'excessive_length': ViolationSeverity.LOW.value,
        'stalling': ViolationSeverity.LOW.value,
        'off_topic': ViolationSeverity.LOW.value,
        
        # Warnings (1)
        'minor_format_issue': ViolationSeverity.WARNING.value,
    }
    
    # Default to medium if unknown
    return severity_map.get(violation_type.lower().replace(' ', '_'), ViolationSeverity.MEDIUM.value)

def calculate_protocol_score(violations: List[str], 
                            action_history: List[Dict[str, Any]],
                            deal_completed: bool,
                            message_texts: Optional[List[str]] = None) -> float:
    """
    Calculate overall protocol compliance score for DPO pair selection.
    
    Args:
        violations: List of violation types
        action_history: History of negotiation actions (backend metadata)
        deal_completed: Whether deal was successfully completed
        message_texts: Optional list of actual message texts to check for tag leaks
        
    Returns:
        Score from 0.0 (worst) to 1.0 (best)
    """
    # Start with perfect score
    score = 1.0
    
    # CRITICAL: Check for leaked tags in messages if texts provided
    if message_texts:
        for text in message_texts:
            has_leaked, found_tags = check_for_leaked_tags(text)
            if has_leaked:
                # This is a critical training failure - heavily penalize
                score -= 0.5
                violations.append('tag_in_message')
    
    # Deduct for violations based on severity
    for violation in violations:
        severity = score_violation_severity(violation)
        deduction = severity * 0.1  # Each severity point = 0.1 deduction
        score -= deduction
    
    # Bonus/penalty for deal completion
    if deal_completed:
        # Check if it was a clean completion
        if len(violations) == 0:
            score += 0.2  # Clean deal bonus
        else:
            score += 0.1  # Completed despite violations
    else:
        score -= 0.1  # Penalty for no deal
    
    # Check action sequence validity
    sequence_valid, sequence_violations = validate_action_sequence(action_history)
    if not sequence_valid:
        score -= len(sequence_violations) * 0.05
    
    # Efficiency bonus (fewer turns is better)
    if action_history:
        # Handle both dict and string formats for backward compatibility
        if isinstance(action_history[-1], dict):
            turn_count = action_history[-1].get('turn', len(action_history))
        else:
            turn_count = len(action_history)
        
        if turn_count <= 5 and deal_completed:
            score += 0.1  # Quick negotiation bonus
        elif turn_count > 15:
            score -= 0.05  # Penalty for excessive length
    
    # Clamp to [0, 1]
    return max(0.0, min(1.0, score))

def select_dpo_pairs(episodes: List[Dict[str, Any]], 
                     min_score_gap: float = 0.3,
                     exclude_tag_leaks: bool = True) -> List[Tuple[Dict, Dict]]:
    """
    Select DPO training pairs based on protocol compliance scores.
    
    Args:
        episodes: List of episode dictionaries with metrics
        min_score_gap: Minimum score difference for pair selection
        exclude_tag_leaks: Whether to exclude episodes with tag leaks from training
        
    Returns:
        List of (positive, negative) episode pairs
    """
    # Calculate scores for each episode
    scored_episodes = []
    for episode in episodes:
        violations = episode.get('metrics', {}).get('protocol_violations', [])
        action_history = episode.get('metrics', {}).get('action_history', [])
        deal_completed = episode.get('metrics', {}).get('deal_closed', False)
        
        # Extract message texts if available
        message_texts = []
        if 'negotiation_history' in episode:
            for turn in episode['negotiation_history']:
                if 'text' in turn:
                    message_texts.append(turn['text'])
        
        # Check for tag leaks - these should NEVER be used for training
        if exclude_tag_leaks and message_texts:
            has_leaks = any(check_for_leaked_tags(text)[0] for text in message_texts)
            if has_leaks:
                continue  # Skip this episode entirely
        
        score = calculate_protocol_score(violations, action_history, deal_completed, message_texts)
        scored_episodes.append((score, episode))
    
    # Sort by score
    scored_episodes.sort(key=lambda x: x[0], reverse=True)
    
    # Select pairs with sufficient score gap
    pairs = []
    for i in range(len(scored_episodes)):
        for j in range(i + 1, len(scored_episodes)):
            score_gap = scored_episodes[i][0] - scored_episodes[j][0]
            if score_gap >= min_score_gap:
                # Double-check neither has tag leaks
                positive = scored_episodes[i][1]
                negative = scored_episodes[j][1]
                
                # Only use if both are clean of tag leaks
                pairs.append((positive, negative))
    
    return pairs

def get_violation_statistics(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate statistics about protocol violations across episodes.
    
    Args:
        episodes: List of episode dictionaries
        
    Returns:
        Dictionary of violation statistics
    """
    total_violations = 0
    violation_counts = {}
    episodes_with_violations = 0
    episodes_with_tag_leaks = 0
    severity_distribution = {s.value: 0 for s in ViolationSeverity}
    
    for episode in episodes:
        violations = episode.get('metrics', {}).get('protocol_violations', [])
        
        # Check for tag leaks in messages
        if 'negotiation_history' in episode:
            for turn in episode['negotiation_history']:
                if 'text' in turn:
                    has_leaked, found_tags = check_for_leaked_tags(turn['text'])
                    if has_leaked:
                        episodes_with_tag_leaks += 1
                        violations.append('tag_in_message')
                        break
        
        if violations:
            episodes_with_violations += 1
            total_violations += len(violations)
            
            for violation in violations:
                violation_counts[violation] = violation_counts.get(violation, 0) + 1
                severity = score_violation_severity(violation)
                severity_distribution[severity] = severity_distribution.get(severity, 0) + 1
    
    return {
        'total_violations': total_violations,
        'episodes_with_violations': episodes_with_violations,
        'episodes_with_tag_leaks': episodes_with_tag_leaks,
        'violation_rate': episodes_with_violations / len(episodes) if episodes else 0,
        'tag_leak_rate': episodes_with_tag_leaks / len(episodes) if episodes else 0,
        'avg_violations_per_episode': total_violations / len(episodes) if episodes else 0,
        'violation_types': violation_counts,
        'severity_distribution': severity_distribution,
        'most_common_violation': max(violation_counts.items(), key=lambda x: x[1])[0] if violation_counts else None,
    }

def validate_episode_for_training(episode: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate if an episode is suitable for training.
    
    Args:
        episode: Episode dictionary
        
    Returns:
        (is_suitable, list_of_reasons_if_not)
    """
    reasons = []
    
    # Check for tag leaks - CRITICAL
    if 'negotiation_history' in episode:
        for turn in episode['negotiation_history']:
            if 'text' in turn:
                has_leaked, found_tags = check_for_leaked_tags(turn['text'])
                if has_leaked:
                    reasons.append(f"Contains leaked action tags in messages: {found_tags}")
                    break
    
    # Check for critical violations
    violations = episode.get('metrics', {}).get('protocol_violations', [])
    for violation in violations:
        severity = score_violation_severity(violation)
        if severity == ViolationSeverity.CRITICAL.value:
            reasons.append(f"Contains critical violation: {violation}")
    
    # Check if episode has sufficient content
    # Handle both episode formats - with metrics or with dialogue
    action_history = episode.get('metrics', {}).get('action_history', [])
    dialogue = episode.get('dialogue', [])

    # Check action history if available, otherwise check dialogue length
    if action_history:
        if len(action_history) < 2:
            reasons.append("Insufficient interaction (less than 2 turns)")
    elif dialogue:
        # For exported episodes, check dialogue instead
        if len(dialogue) < 1:  # At least one exchange needed
            reasons.append("Insufficient dialogue content")
    else:
        reasons.append("No dialogue or action history found")

    return len(reasons) == 0, reasons