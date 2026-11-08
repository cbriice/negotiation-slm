from training.curriculum.data_sampler import CurriculumDataSampler


def test_data_sampler_loads_and_tags_sources(tiny_datasets):
    conv, neg = tiny_datasets
    sampler = CurriculumDataSampler(conversational_path=conv, negotiation_path=neg)
    assert sampler.total_examples == 100
    assert all(ex["source"] == "conversational" for ex in sampler.conversational_data)
    assert all(ex["source"] == "negotiation" for ex in sampler.negotiation_data)


def test_data_sampler_respects_ratio(tiny_datasets):
    conv, neg = tiny_datasets
    sampler = CurriculumDataSampler(conversational_path=conv, negotiation_path=neg)
    sampler.set_ratio({"conversational": 0.2, "negotiation": 0.8})
    batch = sampler.sample_batch(30)
    assert sum(ex["source"] == "conversational" for ex in batch) == 6
    assert sum(ex["source"] == "negotiation" for ex in batch) == 24
