STRATEGY_SCHEMA = {
    "action_indicator": str,  # PROPOSE, COMMIT, ACK, NACK, NO_DEAL
    "target_price": float,    # Current price target
    "price_adjustment": float, # Delta from previous
    "tone": str,              # aggressive, reasonable, sympathetic, firm
    "tactics": [str],         # ["bundle", "urgency", "sympathy", "competition"]
    "reasoning": str,         # Brief explanation
    "risk_level": str,        # low, medium, high
    "concession_size": str,   # none, small, medium, large
    "focus_points": [str],    # Key points to emphasize
}