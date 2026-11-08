import curriculum_parameters as cp
from training.curriculum.phase_config import PhaseConfig


def test_phase_config_matches_curriculum_parameters():
    config = PhaseConfig()
    assert list(config.phases) == [p["name"] for p in cp.PHASES]
    for phase in cp.PHASES:
        assert config.get_phase_at_step(phase["step_start"]) == phase["name"]
        assert config.get_phase_at_step(phase["step_end"] - 1) == phase["name"]


def test_steps_past_the_end_stay_in_last_phase():
    config = PhaseConfig()
    assert config.get_phase_at_step(cp.PHASES[-1]["step_end"] + 10_000) == cp.PHASES[-1]["name"]


def test_phase_data_ratios_sum_to_one():
    for phase in PhaseConfig().phases.values():
        assert abs(sum(phase["data_ratio"].values()) - 1.0) < 1e-9
