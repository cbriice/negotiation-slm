import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data_processing"))

import json
import pytest


@pytest.fixture
def tiny_datasets(tmp_path):
    """Two small JSONL files in the formatted training layout (text + masking_regions)."""
    def write(name, n):
        path = tmp_path / name
        with open(path, "w") as f:
            for i in range(n):
                text = f"---\nsystem: You are an expert negotiator.\n---\nUser: Offer {i}?\nAssistant: Counter {i}."
                f.write(json.dumps({"text": text, "meta": {"assistant_turns": 1}}) + "\n")
        return str(path)
    return write("conversational.jsonl", 40), write("negotiation.jsonl", 60)
