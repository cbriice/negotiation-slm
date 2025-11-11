import pytest
from training.curriculum.phase_config import CurriculumPhase, PhaseConfig

def test_phase_config_initialization():
    config = PhaseConfig()
    assert len(config.phases) == 3
    assert CurriculumPhase.PHASE_1_CONTEXTUAL in config.phases
    assert config.get_phase_at_step(0) == CurriculumPhase.PHASE_1_CONTEXTUAL
    assert config.get_phase_at_step(2500) == CurriculumPhase.PHASE_2_GUIDED
    assert config.get_phase_at_step(6000) == CurriculumPhase.PHASE_3_SPECIALIZED

def test_phase_config_data_ratios():
    config = PhaseConfig()
    phase1 = config.phases[CurriculumPhase.PHASE_1_CONTEXTUAL]
    assert phase1["data_ratio"]["conversational"] == 0.5
    assert phase1["data_ratio"]["negotiation"] == 0.5
