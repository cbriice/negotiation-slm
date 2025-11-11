import pytest
import torch
from transformers import AutoTokenizer
from training.curriculum.masking_collator import CurriculumMaskingCollator

def test_masking_collator_no_masking():
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    collator = CurriculumMaskingCollator(tokenizer, "contextual_learning")

    example = {
        "input_ids": torch.tensor([1, 2, 3, 4, 5]),
        "source": "conversational"
    }

    collator.set_phase("contextual_learning", {"masking": {"conversational": "none"}})
    result = collator.mask_example(example)

    assert torch.equal(result["labels"], example["input_ids"])

def test_masking_collator_assistant_only():
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    collator = CurriculumMaskingCollator(tokenizer, "specialized_training")

    text = "User: Hello\nAssistant: Hi there"
    input_ids = tokenizer.encode(text, return_tensors="pt")[0]

    example = {
        "input_ids": input_ids,
        "source": "negotiation",
        "text": text
    }

    collator.set_phase("specialized_training", {"masking": {"negotiation": "assistant_only"}})
    result = collator.mask_example(example)

    # Verify User portion is masked (-100)
    labels_str = tokenizer.decode([l if l != -100 else 0 for l in result["labels"]])
    assert "Hello" not in labels_str or result["labels"][labels_str.index("Hello")] == -100
