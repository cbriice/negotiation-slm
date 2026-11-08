import pytest
import curriculum_parameters as cp
from training.curriculum.wrapper import CurriculumTrainingWrapper

MODEL = "mistralai/Mistral-7B-v0.1"  # tokenizer only


@pytest.fixture
def wrapper(tiny_datasets, tmp_path):
    conv, neg = tiny_datasets
    return CurriculumTrainingWrapper(
        conversational_path=conv,
        negotiation_path=neg,
        model_name=MODEL,
        output_dir=str(tmp_path / "models"),
    )


def test_wrapper_initialization(wrapper):
    assert wrapper.current_step == 0
    assert wrapper.current_phase == cp.PHASES[0]["name"]
    assert wrapper.recovery_enabled


def test_wrapper_phase_transition_updates_ratio_and_masking(wrapper):
    second = cp.PHASES[1]
    wrapper.update_step(second["step_start"])
    assert wrapper.get_current_phase() == second["name"]
    assert wrapper.data_sampler.current_ratio == second["data_ratio"]
    assert wrapper.masking_collator.masking_config == second["masking"]
