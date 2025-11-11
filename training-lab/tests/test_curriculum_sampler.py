import pytest
from pathlib import Path
from training.curriculum.data_sampler import CurriculumDataSampler

def test_data_sampler_initialization():
    sampler = CurriculumDataSampler(
        conversational_path="data/training_data/conversational-dataset_formatted.jsonl",
        negotiation_path="data/training_data/negotiation-dataset_formatted.jsonl"
    )
    assert sampler.total_examples == 20770  # 10000 + 10770
    assert len(sampler.conversational_data) == 10000
    assert len(sampler.negotiation_data) == 10770

def test_data_sampler_ratio_sampling():
    sampler = CurriculumDataSampler(
        conversational_path="data/training_data/conversational-dataset_formatted.jsonl",
        negotiation_path="data/training_data/negotiation-dataset_formatted.jsonl"
    )

    # Set 50/50 ratio
    sampler.set_ratio({"conversational": 0.5, "negotiation": 0.5})
    batch = sampler.sample_batch(100)

    conv_count = sum(1 for ex in batch if ex["source"] == "conversational")
    assert 40 <= conv_count <= 60  # Allow some variance
