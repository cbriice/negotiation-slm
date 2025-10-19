#!/usr/bin/env python3
"""Test script to verify dual price tracking implementation."""

from utils.control_function import NegotiationState, NegotiationParams, ControlFunction

# Create test parameters
params = NegotiationParams(
    beta_b=0.1,  # Buyer personality parameter
    beta_s=0.05  # Seller personality parameter
)

# Create test state with separate prices
state = NegotiationState(
    r=5,  # Turn 5
    Ps=900,  # Seller minimum
    Pb=850,  # Buyer maximum
    buyer_offer=780,  # Buyer's current offer
    seller_offer=920,  # Seller's current offer
    n_b_k=0,
    n_s_k=3,  # 3 extras available
    l_k=1  # 1 extra included
)

# Test the properties
print("Testing NegotiationState with dual prices:")
print(f"  Buyer offer: ${state.buyer_offer}")
print(f"  Seller offer: ${state.seller_offer}")
print(f"  Price gap: ${state.price_gap:.2f}")
print(f"  Midpoint (P_k): ${state.P_k:.2f}")
print(f"  Buyer's active price: ${state.get_active_price_for(True)}")
print(f"  Seller's active price: ${state.get_active_price_for(False)}")
print()

# Test control function with dual prices
control_fn = ControlFunction(params)

print("Testing control function evaluation:")
# Evaluate for buyer
buyer_eval = control_fn.evaluate(state, is_buyer=True)
print(f"Buyer evaluation:")
print(f"  Control value: {buyer_eval['control_value']:.4f}")
print(f"  Recommended action: {buyer_eval['recommended_action']}")
print(f"  Action probabilities:")
for action, prob in buyer_eval['action_probs'].items():
    print(f"    {action}: {prob:.3f}")
print()

# Evaluate for seller
seller_eval = control_fn.evaluate(state, is_buyer=False)
print(f"Seller evaluation:")
print(f"  Control value: {seller_eval['control_value']:.4f}")
print(f"  Recommended action: {seller_eval['recommended_action']}")
print(f"  Action probabilities:")
for action, prob in seller_eval['action_probs'].items():
    print(f"    {action}: {prob:.3f}")
print()

# Test price overlap detection
print("Testing price overlap scenarios:")

# No overlap
state1 = NegotiationState(
    r=10, Ps=900, Pb=850,
    buyer_offer=820, seller_offer=880,
    n_b_k=0, n_s_k=0, l_k=0
)
print(f"  Scenario 1 - Buyer: ${state1.buyer_offer}, Seller: ${state1.seller_offer}")
print(f"    Gap: ${state1.price_gap:.2f}, Overlapping: {state1.buyer_offer >= state1.seller_offer}")

# Overlap - deal zone reached
state2 = NegotiationState(
    r=15, Ps=900, Pb=850,
    buyer_offer=860, seller_offer=855,
    n_b_k=0, n_s_k=0, l_k=0
)
print(f"  Scenario 2 - Buyer: ${state2.buyer_offer}, Seller: ${state2.seller_offer}")
print(f"    Gap: ${state2.price_gap:.2f}, Overlapping: {state2.buyer_offer >= state2.seller_offer}")
print(f"    Deal price (midpoint): ${(state2.buyer_offer + state2.seller_offer) / 2:.2f}")

print("\nDual price tracking implementation completed successfully!")