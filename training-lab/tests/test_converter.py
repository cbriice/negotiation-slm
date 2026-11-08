"""messages_to_sft -> reformat_to_next_utterance on the example packet in schema/."""

import json
from pathlib import Path

from training.converters.messages_to_sft import MessagesToSFTConverter
from reformat_to_next_utterance import split_conversation

EXAMPLE = Path(__file__).resolve().parents[1] / "schema" / "sft_packet_example.jsonl"


def load_example():
    with open(EXAMPLE) as f:
        return json.loads(f.readline())


def test_conversion_layout_and_regions():
    packet = load_example()
    converted = MessagesToSFTConverter(mask_non_assistant=True).convert_packet(packet)
    text, regions = converted["text"], converted["masking_regions"]

    assert text.startswith("---\nsystem:\nassistant_style: bluffing\n")

    turns = [m for m in packet["messages"] if m["role"] != "system"]
    assert converted["meta"]["assistant_turns"] == sum(m["role"] == "assistant" for m in turns)
    assert converted["meta"]["user_turns"] == sum(m["role"] == "user" for m in turns)

    for prev, cur in zip(regions, regions[1:]):
        assert prev["end"] <= cur["start"], "regions overlap"
    assert regions[-1]["end"] <= len(text)

    for region in regions:
        span = text[region["start"]:region["end"]]
        if region["role"] == "assistant":
            assert span.startswith("Assistant: ")
            assert region["mask"] is False
        else:
            assert region["mask"] is True
        if region["role"] == "user":
            assert span.startswith("User: ")


def test_next_utterance_split_unmasks_only_the_final_assistant_turn():
    converted = MessagesToSFTConverter(mask_non_assistant=True).convert_packet(load_example())
    sub_packets = split_conversation({"text": converted["text"], "meta": converted["meta"]})

    assert len(sub_packets) == 3  # 5 assistant turns -> early / mid / final
    for sub in sub_packets:
        assistant = [r for r in sub["masking_regions"] if r["role"] == "assistant"]
        assert [r["mask"] for r in assistant] == [True] * (len(assistant) - 1) + [False]
        target = assistant[-1]
        assert sub["text"][target["start"]:target["end"]].startswith("Assistant: ")
        assert target["end"] == len(sub["text"])  # target turn is the last thing in the example
