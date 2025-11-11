import pytest
from training.curriculum.wrapper import CurriculumTrainingWrapper
from training.curriculum.phase_config import PhaseConfig

def test_wrapper_initialization():
    wrapper = CurriculumTrainingWrapper(
        conversational_path="data/training_data/conversational-dataset_formatted.jsonl",
        negotiation_path="data/training_data/negotiation-dataset_formatted.jsonl",
        model_name="mistralai/Mistral-7B-v0.1"
    )
    assert wrapper.current_step == 0
    assert wrapper.current_phase is not None
    assert wrapper.recovery_enabled == True

def test_wrapper_phase_transition():
    wrapper = CurriculumTrainingWrapper(
        conversational_path="data/training_data/conversational-dataset_formatted.jsonl",
        negotiation_path="data/training_data/negotiation-dataset_formatted.jsonl",
        model_name="mistralai/Mistral-7B-v0.1"
    )

    # Simulate training progress
    assert wrapper.get_current_phase() == "contextual_learning"
    wrapper.update_step(2500)
    assert wrapper.get_current_phase() == "guided_practice"
