import pytest
import torch
from training.curriculum.monitor import CurriculumMonitor

def test_monitor_initialization():
    monitor = CurriculumMonitor()
    assert "basic_conversation" in monitor.validation_prompts
    assert "negotiation" in monitor.validation_prompts
    assert monitor.degeneracy_thresholds["repetition_ratio"] == 0.3

def test_token_collapse_detection():
    monitor = CurriculumMonitor()

    # Test detection of "eller" pattern
    bad_responses = [
        "eller eller eller",
        "40 eller 0000",
        "seller eller"
    ]
    assert monitor._detect_token_collapse(bad_responses) == True

    # Test normal responses pass
    good_responses = [
        "I'd be happy to help with that negotiation.",
        "The price seems reasonable to me.",
        "Let's discuss the terms."
    ]
    assert monitor._detect_token_collapse(good_responses) == False
