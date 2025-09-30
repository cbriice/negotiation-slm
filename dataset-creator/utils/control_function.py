"""
Control Function Implementation for Negotiation Dataset Creator.

This module implements the mathematical control function that governs
negotiation flow, as defined in info/control_function.md.
"""

import numpy as np
from typing import Dict, Tuple, Optional, NamedTuple
from dataclasses import dataclass
import warnings


@dataclass
class NegotiationParams:
    """Global parameters for negotiation control."""
    beta_b: float  # Buyer personality parameter
    beta_s: float  # Seller personality parameter
    ALPHA: float = 5.0  # Global exponential scaling parameter
    EPSILON: float = 1e-6  # Global threshold parameter
    R: int = 20  # Max turns

    # Control function weights
    w_g: float = 1  # Price gap weight
    w_z: float = 1  # Price range weight
    w_x: float = 0.2  # Extra items weight

    # Action function parameters
    ha1: float = 1  # Compensate action weight 1
    ha2: float = 0.5  # Close action weight
    hz: float = 1  # Fine-tune action weight
    hc: float = 0.5  # Close action multiplier
    hl: float = 1  # Clarify action weight
    eps: float = 0.01  # Local threshold for clarify


@dataclass
class NegotiationState:
    """Current state of negotiation at turn k."""
    r: int  # Current turn index (0-based)
    Ps: float  # Seller minimum acceptable price
    Pb: float  # Buyer maximum acceptable price
    P_k: float  # Current proposed price at turn k
    n_b_k: int = 0  # Number of extra items for buyer
    n_s_k: int = 0  # Number of extra items for seller
    l_k: int = 0  # Number of extra items included in current offer

    # For tracking completion progress
    prev_s: Optional[float] = None  # Previous completion percentage


class ControlFunction:
    """Implementation of the negotiation control function."""

    def __init__(self, params: NegotiationParams):
        self.params = params

    def compute_base_values(self, state: NegotiationState) -> Dict[str, float]:
        """Compute base values for control function calculation."""
        # Turn normalization
        t_k = state.r / self.params.R if self.params.R > 0 else 0

        # Adjusted prices based on personality
        P_s = state.Ps * (1 + self.params.beta_s)
        P_b = state.Pb * (1 + self.params.beta_b)

        # Price gap
        delta_P = abs(P_s - P_b)

        # Seller and buyer margins
        if delta_P > self.params.EPSILON:
            mstar_s = max(0, state.P_k - P_s)
            mstar_b = max(0, P_b - state.P_k)
            m_s = mstar_s / delta_P
            m_b = mstar_b / delta_P
        else:
            # Prices are essentially equal
            m_s = -1000
            m_b = -1000

        return {
            't_k': t_k,
            'P_s': P_s,
            'P_b': P_b,
            'delta_P': delta_P,
            'm_s': m_s,
            'm_b': m_b
        }

    def compute_subfunctions(self, state: NegotiationState, base_values: Dict[str, float]) -> Dict[str, float]:
        """Compute subfunctions g, z, A, and x."""
        t_k = base_values['t_k']

        # Price gap functions
        g_s = max(0, base_values['m_s'])
        g_b = max(0, base_values['m_b'])

        # Price range indicators
        z_s = t_k * max(0, state.Ps - state.P_k) / state.Ps if state.Ps > 0 else 0
        z_b = t_k * max(0, state.P_k - state.Pb) / state.Pb if state.Pb > 0 else 0

        # Exponential scaling of round pressure (turn-dependent)
        if self.params.ALPHA != 0:
            A_k = (np.exp(self.params.ALPHA * t_k) - 1) / (np.exp(self.params.ALPHA) - 1)
        else:
            A_k = t_k  # Linear fallback if ALPHA is 0

        # Extra items percentage left out
        total_extras = state.n_s_k + state.n_b_k
        if total_extras > 0:
            x = 1 - state.l_k / total_extras
        else:
            x = 0  # No extras available

        return {
            'g_s': g_s,
            'g_b': g_b,
            'z_s': z_s,
            'z_b': z_b,
            'A_k': A_k,
            'x': x
        }

    def compute_control_values(self, subfunctions: Dict[str, float],
                              is_buyer: bool) -> Tuple[float, float, Dict[str, float]]:
        """
        Compute control function values V_k and V_max.

        Returns:
            V_k: Control function value at turn k
            V_max: Maximum possible value
            features: Control features (cg, cz, cx, s)
        """
        A_k = subfunctions['A_k']
        x = subfunctions['x']

        if is_buyer:
            g = subfunctions['g_b']
            z = subfunctions['z_b']
        else:
            g = subfunctions['g_s']
            z = subfunctions['z_s']

        # Compute V_k (turn-dependent control value)
        V_k = A_k * (self.params.w_z * z + self.params.w_g * g) + self.params.w_x * x

        # Compute V_max
        V_max = self.params.w_z + self.params.w_g + self.params.w_x

        # Control features
        cg = A_k * self.params.w_g * g
        cz = A_k * self.params.w_z * z
        cx = self.params.w_x * x

        # Completion percentage
        s = 1 - V_k / V_max if V_max > 0 else 0

        features = {
            'cg': cg,
            'cz': cz,
            'cx': cx,
            's': s,
            'A_k': A_k
        }

        return V_k, V_max, features

    def compute_action_q_values(self, features: Dict[str, float],
                                state: NegotiationState) -> Dict[str, float]:
        """Compute q-values for each action type."""
        cg = features['cg']
        cz = features['cz']
        cx = features['cx']
        s = features['s']
        A_k = features['A_k']

        # Compensate price action
        q_compensate = cz * (1 + self.params.ha1 * A_k)

        # Fine tune price action
        q_finetune = cg * self.params.hz * cz

        # Extra item action
        q_extra = cx

        # Close action
        # Modified formula to reduce q_close dominance
        # Original: q_close = hc * (3 - cg - cz - cx) / 3 + ha2 * A_k
        # Problem: When cg, cz, cx are small, (3 - 0 - 0 - 0)/3 = 1, making q_close always high

        # New approach: q_close should increase when other actions are small AND we're late in negotiation
        # This uses the inverse of the sum of other components, scaled by turn progress
        sum_components = cg + cz + cx
        max_possible = 3.0  # Maximum theoretical value of sum_components

        if sum_components > 0.1:  # If there's meaningful activity in other areas
            # Lower close probability when other actions are active
            q_close = self.params.hc * (1 - sum_components / max_possible) + self.params.ha2 * A_k
        else:
            # Only high close probability if we're late in negotiation
            q_close = self.params.hc * A_k + self.params.ha2 * A_k

        # Clarify action
        if state.prev_s is not None:
            clarify_condition = abs(s - state.prev_s) < self.params.eps
        else:
            clarify_condition = False
        q_clarify = self.params.hl * (1.0 if clarify_condition else 0.0)

        return {
            'compensate': q_compensate,
            'finetune': q_finetune,
            'extra': q_extra,
            'close': q_close,
            'clarify': q_clarify
        }

    def softmax(self, q_values: Dict[str, float], temperature: float = 1.0) -> Dict[str, float]:
        """Apply softmax to q-values to get action probabilities."""
        # Convert to numpy array for computation
        actions = list(q_values.keys())
        values = np.array([q_values[a] for a in actions])

        # Apply temperature scaling
        values = values / temperature

        # Softmax computation with numerical stability
        exp_values = np.exp(values - np.max(values))
        probabilities = exp_values / np.sum(exp_values)

        return dict(zip(actions, probabilities))

    def evaluate(self, state: NegotiationState, is_buyer: bool = True,
                temperature: float = 1.0) -> Dict[str, any]:
        """
        Full evaluation of control function for a given state.

        Returns comprehensive dict with:
            - base_values: Computed base values
            - subfunctions: g, z, A_k, x values
            - control_value: V_k
            - control_max: V_max
            - features: Control features
            - q_values: Action q-values
            - action_probs: Softmax probabilities
            - recommended_action: Highest probability action
        """
        # Compute all intermediate values
        base_values = self.compute_base_values(state)
        subfunctions = self.compute_subfunctions(state, base_values)
        V_k, V_max, features = self.compute_control_values(subfunctions, is_buyer)
        q_values = self.compute_action_q_values(features, state)
        action_probs = self.softmax(q_values, temperature)

        # Get recommended action
        recommended_action = max(action_probs.items(), key=lambda x: x[1])[0]

        return {
            'base_values': base_values,
            'subfunctions': subfunctions,
            'control_value': V_k,
            'control_max': V_max,
            'features': features,
            'q_values': q_values,
            'action_probs': action_probs,
            'recommended_action': recommended_action,
            'is_buyer': is_buyer
        }