"""Properties of the negotiation control function (see docs/control_function.md)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.control_function import ControlFunction, NegotiationParams, NegotiationState

ACTIONS = {"price_adjust", "extra", "close", "clarify"}


def make_state(**overrides):
    state = dict(r=5, Ps=900, Pb=1000, buyer_offer=800, seller_offer=1100, n_b_k=0, n_s_k=2, l_k=0)
    state.update(overrides)
    return NegotiationState(**state)


@pytest.fixture
def cf():
    return ControlFunction(NegotiationParams(beta_b=0.0, beta_s=0.0))


@pytest.mark.parametrize("is_buyer", [True, False])
def test_action_distribution_is_a_probability_distribution(cf, is_buyer):
    probs = cf.evaluate(make_state(), is_buyer=is_buyer)["action_probs"]
    assert set(probs) == ACTIONS
    assert sum(probs.values()) == pytest.approx(1.0)
    assert all(p > 0 for p in probs.values())


def test_round_pressure_rises_from_zero_to_one(cf):
    pressure = [cf.evaluate(make_state(r=r))["subfunctions"]["A_k"] for r in range(cf.params.R + 1)]
    assert pressure[0] == pytest.approx(0.0)
    assert pressure[-1] == pytest.approx(1.0)
    assert all(np.diff(pressure) > 0)


def test_out_of_range_offers_push_toward_price_adjustment(cf):
    # Seller asking above the buyer's max: buyer's range indicator z_b is nonzero
    far = cf.evaluate(make_state(buyer_offer=1200, seller_offer=1300), is_buyer=True)
    assert far["features"]["cz"] > 0
    assert far["recommended_action"] == "price_adjust"



def test_stalled_progress_above_threshold_adds_close_bonus(cf):
    state = make_state(buyer_offer=900, seller_offer=950)
    s = cf.evaluate(state, is_buyer=False)["features"]["s"]
    assert s > cf.params.s_c

    stalled = make_state(buyer_offer=900, seller_offer=950, prev_s=s)
    q_without = cf.evaluate(state, is_buyer=False)["q_values"]["close"]
    q_with = cf.evaluate(stalled, is_buyer=False)["q_values"]["close"]
    assert q_with - q_without == pytest.approx(cf.params.hc2)


def test_softmax_temperature_sharpens_near_completion(cf):
    near = make_state(buyer_offer=900, seller_offer=950)
    result = cf.evaluate(near, is_buyer=False)
    assert result["features"]["s"] > cf.params.s_c
    assert result["action_probs"] == pytest.approx(cf.softmax(result["q_values"], temperature=0.75))


def test_no_extras_means_no_extra_action_value(cf):
    result = cf.evaluate(make_state(n_b_k=0, n_s_k=0))
    assert result["q_values"]["extra"] == 0.0
