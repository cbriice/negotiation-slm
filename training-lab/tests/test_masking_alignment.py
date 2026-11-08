"""Loss weights must land on exactly the target turn's tokens.

Regression test: span->token mapping used to re-tokenize text prefixes without special
tokens, while training ids are built with them (BOS at position 0). Every region was
shifted one token left: the newline before the target got weight 1 and the target's
final token got weight 0.
"""

import pytest
import torch
from transformers import AutoTokenizer

from training.curriculum.masking_collator import CurriculumMaskingCollator
from reformat_to_next_utterance import create_masking_regions

TEXT = (
    "---\nsystem: You are an expert negotiator.\n---\n"
    "User: Is the bike still available?\n"
    "Assistant: Yes! Asking $450.\n"
    "User: Would you take $380?\n"
    "Assistant: I could do $420, final offer."
)
WEIGHTS = {"user_weight": 0.05, "assistant_weight": 1.0, "system_weight": 0.0}


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_fast=True)


def weighted(tokenizer, add_special_tokens):
    ids = tokenizer(TEXT, add_special_tokens=add_special_tokens)["input_ids"]
    collator = CurriculumMaskingCollator(tokenizer, "p")
    collator.set_phase("p", {"masking": {"negotiation": WEIGHTS}})
    example = collator.mask_example({
        "input_ids": ids, "text": TEXT, "source": "negotiation",
        "masking_regions": create_masking_regions(TEXT),
    })
    return ids, example["loss_weights"]


@pytest.mark.parametrize("add_special_tokens", [True, False])
def test_target_weight_covers_exactly_the_final_assistant_turn(tokenizer, add_special_tokens):
    ids, weights = weighted(tokenizer, add_special_tokens)
    target_tokens = [i for i, w in zip(ids, weights.tolist()) if w == 1.0]
    assert tokenizer.decode(target_tokens) == TEXT[TEXT.rfind("Assistant:"):]


def test_special_tokens_get_zero_weight(tokenizer):
    ids, weights = weighted(tokenizer, add_special_tokens=True)
    assert ids[0] == tokenizer.bos_token_id
    assert weights[0] == 0.0


def test_mismatched_ids_fail_loudly(tokenizer):
    collator = CurriculumMaskingCollator(tokenizer, "p")
    collator.set_phase("p", {"masking": {"negotiation": WEIGHTS}})
    ids = tokenizer(TEXT)["input_ids"]
    with pytest.raises(RuntimeError, match="do not match"):
        collator.mask_example({
            "input_ids": ids, "text": TEXT.replace("bike", "car"), "source": "negotiation",
            "masking_regions": create_masking_regions(TEXT),
        })
