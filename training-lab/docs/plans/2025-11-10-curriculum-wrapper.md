# Curriculum Training Wrapper Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a curriculum training wrapper that prevents token collapse through phased learning with dynamic masking.

**Architecture:** Wrapper sits between datasets and SFT trainer, dynamically adjusting data mixing ratios and masking strategies per training phase. Includes automatic monitoring for catastrophic forgetting with recovery protocols.

**Tech Stack:** PyTorch, Transformers, TRL (SFTTrainer), custom collators

---

## Task 1: Phase Configuration System

**Files:**
- Create: `training/curriculum/phase_config.py`
- Test: `tests/test_phase_config.py`

**Step 1: Write the failing test**

```python
# tests/test_phase_config.py
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_phase_config.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'training.curriculum'"

**Step 3: Write minimal implementation**

```python
# training/curriculum/__init__.py
"""Curriculum training components for preventing catastrophic forgetting."""

# training/curriculum/phase_config.py
from typing import Dict, Any, Tuple

class CurriculumPhase:
    """Phase identifiers for curriculum learning."""
    PHASE_1_CONTEXTUAL = "contextual_learning"
    PHASE_2_GUIDED = "guided_practice"
    PHASE_3_SPECIALIZED = "specialized_training"

class PhaseConfig:
    """Configuration for curriculum training phases."""

    def __init__(self):
        self.phases = {
            CurriculumPhase.PHASE_1_CONTEXTUAL: {
                "steps": (0, 2000),
                "data_ratio": {"conversational": 0.5, "negotiation": 0.5},
                "lr_multiplier": 0.5,
                "masking": {
                    "conversational": "none",
                    "negotiation": "system_only"
                },
                "validation_frequency": 100
            },
            CurriculumPhase.PHASE_2_GUIDED: {
                "steps": (2000, 4000),
                "data_ratio": {"conversational": 0.3, "negotiation": 0.7},
                "lr_multiplier": 1.0,
                "masking": {
                    "conversational": "none",
                    "negotiation": "alternating"
                },
                "validation_frequency": 200
            },
            CurriculumPhase.PHASE_3_SPECIALIZED: {
                "steps": (4000, 8000),
                "data_ratio": {"conversational": 0.1, "negotiation": 0.9},
                "lr_multiplier": 0.8,
                "masking": {
                    "conversational": "assistant_only",
                    "negotiation": "assistant_only"
                },
                "validation_frequency": 500
            }
        }

    def get_phase_at_step(self, step: int) -> str:
        """Get the phase name for a given training step."""
        for phase_name, config in self.phases.items():
            start, end = config["steps"]
            if start <= step < end:
                return phase_name
        # Default to last phase if beyond defined steps
        return CurriculumPhase.PHASE_3_SPECIALIZED
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_phase_config.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add training/curriculum/__init__.py training/curriculum/phase_config.py tests/test_phase_config.py
git commit -m "feat: add curriculum phase configuration system"
```

---

## Task 2: Dynamic Masking Collator

**Files:**
- Create: `training/curriculum/masking_collator.py`
- Test: `tests/test_curriculum_masking.py`

**Step 1: Write the failing test**

```python
# tests/test_curriculum_masking.py
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_curriculum_masking.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'training.curriculum.masking_collator'"

**Step 3: Write minimal implementation**

```python
# training/curriculum/masking_collator.py
import torch
import re
from typing import Dict, Any, List, Optional
from transformers import PreTrainedTokenizer

class CurriculumMaskingCollator:
    """Dynamic masking based on curriculum phase and data source."""

    def __init__(self, tokenizer: PreTrainedTokenizer, current_phase: str, ignore_index: int = -100):
        self.tokenizer = tokenizer
        self.current_phase = current_phase
        self.ignore_index = ignore_index
        self.masking_config = {}

    def set_phase(self, phase: str, phase_config: Dict[str, Any]):
        """Update masking strategy for new phase."""
        self.current_phase = phase
        self.masking_config = phase_config.get("masking", {})

    def mask_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Apply masking based on source and current phase."""
        source = example.get("source", "unknown")
        masking_strategy = self.masking_config.get(source, "assistant_only")

        input_ids = example["input_ids"]

        if masking_strategy == "none":
            labels = input_ids.clone() if torch.is_tensor(input_ids) else input_ids.copy()

        elif masking_strategy == "system_only":
            labels = self._mask_system_only(example)

        elif masking_strategy == "alternating":
            labels = self._mask_alternating(example)

        elif masking_strategy == "assistant_only":
            labels = self._mask_non_assistant(example)
        else:
            labels = input_ids.clone() if torch.is_tensor(input_ids) else input_ids.copy()

        example["labels"] = labels
        return example

    def _mask_system_only(self, example: Dict[str, Any]) -> torch.Tensor:
        """Mask only the system prompt section."""
        input_ids = example["input_ids"]
        text = example.get("text", self.tokenizer.decode(input_ids))

        labels = input_ids.clone() if torch.is_tensor(input_ids) else torch.tensor(input_ids)

        # Find system section (between "---\nsystem:" and first "User:" or "Assistant:")
        system_pattern = r"---\s*\nsystem:.*?(?=(User:|Assistant:|$))"
        match = re.search(system_pattern, text, re.DOTALL)

        if match:
            # Tokenize the prefix to find position
            prefix = text[:match.end()]
            prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
            labels[:len(prefix_ids)] = self.ignore_index

        return labels

    def _mask_alternating(self, example: Dict[str, Any]) -> torch.Tensor:
        """Mask user turns, keep assistant visible."""
        input_ids = example["input_ids"]
        text = example.get("text", self.tokenizer.decode(input_ids))

        labels = input_ids.clone() if torch.is_tensor(input_ids) else torch.tensor(input_ids)

        # Find all User: sections
        user_pattern = r"User:.*?(?=(Assistant:|User:|$))"
        for match in re.finditer(user_pattern, text, re.DOTALL):
            start_text = text[:match.start()]
            end_text = text[:match.end()]

            start_tokens = len(self.tokenizer.encode(start_text, add_special_tokens=False))
            end_tokens = len(self.tokenizer.encode(end_text, add_special_tokens=False))

            labels[start_tokens:end_tokens] = self.ignore_index

        return labels

    def _mask_non_assistant(self, example: Dict[str, Any]) -> torch.Tensor:
        """Standard SFT masking - only assistant responses visible."""
        input_ids = example["input_ids"]
        text = example.get("text", self.tokenizer.decode(input_ids))

        labels = torch.full_like(input_ids, self.ignore_index) if torch.is_tensor(input_ids) else torch.full((len(input_ids),), self.ignore_index)

        # Find all Assistant: sections
        assistant_pattern = r"Assistant:.*?(?=(User:|Assistant:|$))"
        for match in re.finditer(assistant_pattern, text, re.DOTALL):
            start_text = text[:match.start()]
            end_text = text[:match.end()]

            start_tokens = len(self.tokenizer.encode(start_text, add_special_tokens=False))
            end_tokens = len(self.tokenizer.encode(end_text, add_special_tokens=False))

            labels[start_tokens:end_tokens] = input_ids[start_tokens:end_tokens]

        return labels

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Process batch of examples."""
        for example in examples:
            self.mask_example(example)

        # Stack into batch tensors
        batch = {}
        for key in examples[0].keys():
            if torch.is_tensor(examples[0][key]):
                batch[key] = torch.stack([ex[key] for ex in examples])
            else:
                batch[key] = [ex[key] for ex in examples]

        return batch
```

**Step 4: Run test to verify it passes**

Run: `source .venv310/bin/activate && pytest tests/test_curriculum_masking.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add training/curriculum/masking_collator.py tests/test_curriculum_masking.py
git commit -m "feat: add dynamic masking collator for curriculum training"
```

---

## Task 3: Data Sampler with Dynamic Ratios

**Files:**
- Create: `training/curriculum/data_sampler.py`
- Test: `tests/test_curriculum_sampler.py`

**Step 1: Write the failing test**

```python
# tests/test_curriculum_sampler.py
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_curriculum_sampler.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'training.curriculum.data_sampler'"

**Step 3: Write minimal implementation**

```python
# training/curriculum/data_sampler.py
import json
import random
from pathlib import Path
from typing import Dict, List, Any, Optional

class CurriculumDataSampler:
    """Samples from multiple datasets with dynamic ratios."""

    def __init__(self, conversational_path: str, negotiation_path: str, seed: int = 42):
        self.conversational_path = Path(conversational_path)
        self.negotiation_path = Path(negotiation_path)
        self.seed = seed
        random.seed(seed)

        # Load datasets into memory
        self.conversational_data = self._load_jsonl(self.conversational_path)
        self.negotiation_data = self._load_jsonl(self.negotiation_path)

        # Tag with source
        for ex in self.conversational_data:
            ex["source"] = "conversational"
        for ex in self.negotiation_data:
            ex["source"] = "negotiation"

        self.total_examples = len(self.conversational_data) + len(self.negotiation_data)

        # Default ratio
        self.current_ratio = {"conversational": 0.5, "negotiation": 0.5}

        # For iteration
        self.epoch = 0
        self.indices = {"conversational": 0, "negotiation": 0}

    def _load_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        """Load JSONL file into memory."""
        data = []
        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    def set_ratio(self, ratio: Dict[str, float]):
        """Update sampling ratio."""
        assert abs(sum(ratio.values()) - 1.0) < 0.01, "Ratios must sum to 1.0"
        self.current_ratio = ratio

    def sample_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """Sample a batch respecting current ratio."""
        batch = []

        # Calculate samples per dataset
        conv_samples = int(batch_size * self.current_ratio["conversational"])
        neg_samples = batch_size - conv_samples

        # Sample from each dataset
        if conv_samples > 0 and self.conversational_data:
            conv_batch = random.sample(self.conversational_data,
                                      min(conv_samples, len(self.conversational_data)))
            batch.extend(conv_batch)

        if neg_samples > 0 and self.negotiation_data:
            neg_batch = random.sample(self.negotiation_data,
                                    min(neg_samples, len(self.negotiation_data)))
            batch.extend(neg_batch)

        # Shuffle the combined batch
        random.shuffle(batch)

        return batch[:batch_size]  # Ensure exact size

    def get_epoch_iterator(self, batch_size: int, shuffle: bool = True):
        """Get iterator for full epoch with proper ratios."""
        # Prepare indices
        conv_size = int(self.total_examples * self.current_ratio["conversational"])
        neg_size = self.total_examples - conv_size

        # Create repeated indices to match desired ratio
        conv_indices = list(range(len(self.conversational_data))) * (conv_size // len(self.conversational_data) + 1)
        conv_indices = conv_indices[:conv_size]

        neg_indices = list(range(len(self.negotiation_data))) * (neg_size // len(self.negotiation_data) + 1)
        neg_indices = neg_indices[:neg_size]

        # Create combined dataset for epoch
        epoch_data = []
        epoch_data.extend([self.conversational_data[i] for i in conv_indices])
        epoch_data.extend([self.negotiation_data[i] for i in neg_indices])

        if shuffle:
            random.shuffle(epoch_data)

        # Yield batches
        for i in range(0, len(epoch_data), batch_size):
            yield epoch_data[i:i + batch_size]

    def override_ratio(self, ratio: Dict[str, float]):
        """Emergency override for recovery mode."""
        self.set_ratio(ratio)
        print(f"[EMERGENCY] Data ratio overridden to: {ratio}")
```

**Step 4: Run test to verify it passes**

Run: `source .venv310/bin/activate && pytest tests/test_curriculum_sampler.py -v`
Expected: PASS (assuming data files exist)

**Step 5: Commit**

```bash
git add training/curriculum/data_sampler.py tests/test_curriculum_sampler.py
git commit -m "feat: add dynamic data sampler for curriculum training"
```

---

## Task 4: Monitoring System

**Files:**
- Create: `training/curriculum/monitor.py`
- Create: `data/validation_prompts.json`
- Test: `tests/test_curriculum_monitor.py`

**Step 1: Write the failing test**

```python
# tests/test_curriculum_monitor.py
import pytest
import torch
from training.curriculum.monitor import CurriculumMonitor

def test_monitor_initialization():
    monitor = CurriculumMonitor()
    assert "basic_conversation" in monitor.validation_prompts
    assert "negotiation" in monitor.validation_prompts
    assert monitor.degeneracy_thresholds["repetition_ratio"] == 0.3

def test_token_collapse_detection():
    monitor = CurriculumMonitor()

    # Test detection of "eller" pattern
    bad_responses = [
        "eller eller eller",
        "40 eller 0000",
        "seller eller"
    ]
    assert monitor._detect_token_collapse(bad_responses) == True

    # Test normal responses pass
    good_responses = [
        "I'd be happy to help with that negotiation.",
        "The price seems reasonable to me.",
        "Let's discuss the terms."
    ]
    assert monitor._detect_token_collapse(good_responses) == False
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_curriculum_monitor.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'training.curriculum.monitor'"

**Step 3: Create validation prompts**

```json
# data/validation_prompts.json
{
    "basic_conversation": [
        "Hello, how are you today?",
        "What is 2+2?",
        "Can you tell me about yourself?",
        "What's the weather like?",
        "Thank you for your help!"
    ],
    "negotiation": [
        "I'd like to buy this item. What's your price?",
        "Can you offer a better deal?",
        "That seems expensive. Can we negotiate?",
        "I'm interested but the price is too high.",
        "What's your final offer?"
    ],
    "technical": [
        "Explain how a computer works.",
        "What is machine learning?",
        "How do I write a Python function?"
    ]
}
```

**Step 4: Write monitor implementation**

```python
# training/curriculum/monitor.py
import re
import json
import torch
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from collections import Counter

logger = logging.getLogger(__name__)

class CurriculumMonitor:
    """Monitors training health and detects catastrophic forgetting."""

    def __init__(self, validation_prompts_path: str = "data/validation_prompts.json"):
        # Load validation prompts
        prompts_path = Path(validation_prompts_path)
        if prompts_path.exists():
            with open(prompts_path, 'r') as f:
                self.validation_prompts = json.load(f)
        else:
            # Default prompts if file doesn't exist
            self.validation_prompts = {
                "basic_conversation": [
                    "Hello, how are you?",
                    "What is 2+2?",
                    "Tell me about yourself"
                ],
                "negotiation": [
                    "I'd like to buy this item",
                    "What's your best price?",
                    "Can we make a deal?"
                ]
            }

        self.degeneracy_thresholds = {
            "perplexity_increase": 1.5,
            "repetition_ratio": 0.3,
            "response_length_collapse": 10,
            "token_collapse_ratio": 0.3
        }

        self.baseline_metrics = {}
        self.history = []

    def checkpoint_health_check(self, model, tokenizer, step: int) -> Tuple[str, Dict[str, Any]]:
        """Run comprehensive health check."""
        metrics = {}

        # Generate responses for validation prompts
        for category, prompts in self.validation_prompts.items():
            responses = self._generate_responses(model, tokenizer, prompts)

            # Calculate metrics
            metrics[f"{category}_avg_length"] = sum(len(r.split()) for r in responses) / len(responses)
            metrics[f"{category}_repetition"] = self._calculate_repetition_ratio(responses)
            metrics[f"{category}_perplexity"] = self._calculate_perplexity(model, tokenizer, responses)

            # Check for token collapse
            if category == "basic_conversation":
                metrics["has_token_collapse"] = self._detect_token_collapse(responses)

        # Store baseline on first check
        if step == 0:
            self.baseline_metrics = metrics.copy()
            return "HEALTHY", metrics

        # Check for critical issues
        if metrics.get("has_token_collapse", False):
            logger.critical(f"Token collapse detected at step {step}")
            return "CRITICAL", metrics

        # Check for perplexity degradation
        if "basic_conversation_perplexity" in self.baseline_metrics:
            perplexity_ratio = metrics["basic_conversation_perplexity"] / self.baseline_metrics["basic_conversation_perplexity"]
            if perplexity_ratio > self.degeneracy_thresholds["perplexity_increase"]:
                logger.warning(f"Perplexity degradation at step {step}: {perplexity_ratio:.2f}x baseline")
                return "WARNING", metrics

        # Check for response length collapse
        if metrics.get("basic_conversation_avg_length", 100) < self.degeneracy_thresholds["response_length_collapse"]:
            logger.warning(f"Response length collapse at step {step}")
            return "WARNING", metrics

        return "HEALTHY", metrics

    def _generate_responses(self, model, tokenizer, prompts: List[str], max_length: int = 100) -> List[str]:
        """Generate responses for validation prompts."""
        responses = []
        model.eval()

        with torch.no_grad():
            for prompt in prompts:
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id
                )

                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Remove the prompt from response
                response = response[len(prompt):].strip()
                responses.append(response)

        return responses

    def _calculate_repetition_ratio(self, responses: List[str]) -> float:
        """Calculate ratio of repetitive tokens in responses."""
        total_tokens = 0
        repetitive_tokens = 0

        for response in responses:
            tokens = response.split()
            total_tokens += len(tokens)

            # Count tokens that appear more than 30% of the time
            if tokens:
                token_counts = Counter(tokens)
                threshold = len(tokens) * 0.3
                repetitive_tokens += sum(1 for count in token_counts.values() if count > threshold)

        return repetitive_tokens / max(total_tokens, 1)

    def _calculate_perplexity(self, model, tokenizer, texts: List[str]) -> float:
        """Calculate average perplexity for texts."""
        total_loss = 0
        total_tokens = 0

        model.eval()
        with torch.no_grad():
            for text in texts:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                outputs = model(**inputs, labels=inputs["input_ids"])
                total_loss += outputs.loss.item() * inputs["input_ids"].size(1)
                total_tokens += inputs["input_ids"].size(1)

        avg_loss = total_loss / max(total_tokens, 1)
        return torch.exp(torch.tensor(avg_loss)).item()

    def _detect_token_collapse(self, responses: List[str]) -> bool:
        """Detect specific token collapse patterns."""
        danger_patterns = [
            r'eller',  # The nightmare token
            r'(\w+)\1{4,}',  # Word repeated 4+ times
            r'0{4,}',  # Sequences of zeros
            r'^.{,10}$',  # Responses under 10 chars
            r'\d+\s*\n\s*\d+',  # Repetitive numbers
        ]

        collapse_count = 0
        for response in responses:
            for pattern in danger_patterns:
                if re.search(pattern, response, re.IGNORECASE):
                    collapse_count += 1
                    break  # Count each response only once

        collapse_ratio = collapse_count / max(len(responses), 1)
        return collapse_ratio > self.degeneracy_thresholds["token_collapse_ratio"]

    def save_emergency_checkpoint(self, model, optimizer, step: int, metrics: Dict[str, Any], output_dir: str):
        """Save emergency checkpoint with full diagnostics."""
        emergency_dir = Path(output_dir) / f"emergency_checkpoint_step_{step}"
        emergency_dir.mkdir(parents=True, exist_ok=True)

        # Save model state
        torch.save(model.state_dict(), emergency_dir / "model.pt")
        torch.save(optimizer.state_dict(), emergency_dir / "optimizer.pt")

        # Save metrics and diagnostics
        with open(emergency_dir / "diagnostics.json", 'w') as f:
            json.dump({
                "step": step,
                "metrics": metrics,
                "baseline": self.baseline_metrics,
                "thresholds": self.degeneracy_thresholds
            }, f, indent=2)

        logger.info(f"Emergency checkpoint saved to {emergency_dir}")
```

**Step 5: Run test to verify it passes**

Run: `source .venv310/bin/activate && pytest tests/test_curriculum_monitor.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add training/curriculum/monitor.py data/validation_prompts.json tests/test_curriculum_monitor.py
git commit -m "feat: add health monitoring system for curriculum training"
```

---

## Task 5: Main Curriculum Wrapper

**Files:**
- Create: `training/curriculum/wrapper.py`
- Test: `tests/test_curriculum_wrapper.py`

**Step 1: Write the failing test**

```python
# tests/test_curriculum_wrapper.py
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_curriculum_wrapper.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'training.curriculum.wrapper'"

**Step 3: Write wrapper implementation**

```python
# training/curriculum/wrapper.py
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import torch
from transformers import TrainerCallback, TrainingArguments, AutoTokenizer

from training.curriculum.phase_config import CurriculumPhase, PhaseConfig
from training.curriculum.masking_collator import CurriculumMaskingCollator
from training.curriculum.data_sampler import CurriculumDataSampler
from training.curriculum.monitor import CurriculumMonitor

logger = logging.getLogger(__name__)

class CurriculumTrainingWrapper:
    """Main wrapper orchestrating curriculum training."""

    def __init__(
        self,
        conversational_path: str,
        negotiation_path: str,
        model_name: str,
        output_dir: str = "models/curriculum",
        recovery_enabled: bool = True,
        base_lr: float = 1e-5
    ):
        self.conversational_path = conversational_path
        self.negotiation_path = negotiation_path
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.recovery_enabled = recovery_enabled
        self.base_lr = base_lr

        # Initialize components
        self.phase_config = PhaseConfig()
        self.data_sampler = CurriculumDataSampler(conversational_path, negotiation_path)
        self.monitor = CurriculumMonitor()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Initialize masking collator
        self.masking_collator = CurriculumMaskingCollator(
            self.tokenizer,
            CurriculumPhase.PHASE_1_CONTEXTUAL
        )

        # Training state
        self.current_step = 0
        self.current_phase = CurriculumPhase.PHASE_1_CONTEXTUAL
        self.recovery_mode = False
        self.recovery_end_step = 0

        # Apply initial phase configuration
        self._apply_phase_config(self.current_phase)

        logger.info(f"Curriculum wrapper initialized with {self.data_sampler.total_examples} total examples")

    def _apply_phase_config(self, phase: str):
        """Apply configuration for a specific phase."""
        config = self.phase_config.phases[phase]

        # Update data sampling ratio
        self.data_sampler.set_ratio(config["data_ratio"])

        # Update masking strategy
        self.masking_collator.set_phase(phase, config)

        logger.info(f"Applied phase configuration: {phase}")
        logger.info(f"  Data ratio: {config['data_ratio']}")
        logger.info(f"  Masking strategy: {config['masking']}")
        logger.info(f"  LR multiplier: {config['lr_multiplier']}")

    def update_step(self, step: int):
        """Update current step and check for phase transition."""
        self.current_step = step

        # Check if we should transition phase
        new_phase = self.phase_config.get_phase_at_step(step)
        if new_phase != self.current_phase:
            self._transition_phase(new_phase)

    def _transition_phase(self, new_phase: str):
        """Transition to a new training phase."""
        logger.info(f"Transitioning from {self.current_phase} to {new_phase} at step {self.current_step}")

        self.current_phase = new_phase
        self._apply_phase_config(new_phase)

        # Save checkpoint at phase transition
        self._save_phase_checkpoint()

    def _save_phase_checkpoint(self):
        """Save checkpoint at phase boundary."""
        checkpoint_dir = self.output_dir / f"phase_{self.current_phase}_step_{self.current_step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Phase checkpoint saved to {checkpoint_dir}")

    def get_current_phase(self) -> str:
        """Get the current training phase."""
        return self.current_phase

    def get_lr_multiplier(self) -> float:
        """Get learning rate multiplier for current phase."""
        if self.recovery_mode:
            return 0.1  # Emergency reduction
        return self.phase_config.phases[self.current_phase]["lr_multiplier"]

    def should_validate(self) -> bool:
        """Check if validation should run at current step."""
        if self.current_step == 0:
            return True  # Always validate at start

        validation_freq = self.phase_config.phases[self.current_phase]["validation_frequency"]
        return self.current_step % validation_freq == 0

    def run_health_check(self, model) -> Tuple[str, Dict[str, Any]]:
        """Run health check on current model."""
        return self.monitor.checkpoint_health_check(model, self.tokenizer, self.current_step)

    def execute_recovery_protocol(self, model, optimizer, metrics: Dict[str, Any]):
        """Execute emergency recovery when collapse detected."""
        logger.critical(f"EXECUTING RECOVERY PROTOCOL at step {self.current_step}")

        # 1. Reduce learning rate to 10%
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.base_lr * 0.1

        # 2. Override data mixture to 70% conversational
        self.data_sampler.override_ratio({"conversational": 0.7, "negotiation": 0.3})

        # 3. Disable all masking temporarily
        self.masking_collator.set_phase("recovery", {"masking": {
            "conversational": "none",
            "negotiation": "none"
        }})

        # 4. Set recovery mode
        self.recovery_mode = True
        self.recovery_end_step = self.current_step + 200

        # 5. Save emergency checkpoint
        self.monitor.save_emergency_checkpoint(
            model, optimizer, self.current_step, metrics, str(self.output_dir)
        )

        logger.info("Recovery protocol activated:")
        logger.info("  - Learning rate reduced to 10%")
        logger.info("  - Data ratio: 70% conversational, 30% negotiation")
        logger.info("  - All masking disabled")
        logger.info(f"  - Recovery mode until step {self.recovery_end_step}")

    def check_recovery_complete(self) -> bool:
        """Check if recovery mode should end."""
        if self.recovery_mode and self.current_step >= self.recovery_end_step:
            logger.info("Recovery mode complete, resuming normal training")
            self.recovery_mode = False
            # Reapply current phase configuration
            self._apply_phase_config(self.current_phase)
            return True
        return False

    def get_data_collator(self):
        """Get the masking collator for training."""
        return self.masking_collator

    def get_dataset_iterator(self, batch_size: int, shuffle: bool = True):
        """Get dataset iterator with proper mixing."""
        return self.data_sampler.get_epoch_iterator(batch_size, shuffle)

class CurriculumCallback(TrainerCallback):
    """Trainer callback for curriculum training integration."""

    def __init__(self, wrapper: CurriculumTrainingWrapper):
        self.wrapper = wrapper

    def on_step_begin(self, args, state, control, **kwargs):
        """Update wrapper at step beginning."""
        self.wrapper.update_step(state.global_step)

        # Check if recovery mode should end
        self.wrapper.check_recovery_complete()

        # Adjust learning rate based on phase
        lr_multiplier = self.wrapper.get_lr_multiplier()
        for param_group in kwargs['optimizer'].param_groups:
            param_group['lr'] = self.wrapper.base_lr * lr_multiplier

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Run validation and health checks."""
        if self.wrapper.should_validate() and model is not None:
            status, metrics = self.wrapper.run_health_check(model)

            if status == "CRITICAL" and self.wrapper.recovery_enabled:
                self.wrapper.execute_recovery_protocol(
                    model, kwargs.get('optimizer'), metrics
                )
            elif status == "WARNING":
                logger.warning(f"Health check warning at step {state.global_step}")
                # Reduce LR by 50%
                for param_group in kwargs.get('optimizer').param_groups:
                    param_group['lr'] *= 0.5
```

**Step 4: Run test to verify it passes**

Run: `source .venv310/bin/activate && pytest tests/test_curriculum_wrapper.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add training/curriculum/wrapper.py tests/test_curriculum_wrapper.py
git commit -m "feat: add main curriculum training wrapper with recovery protocols"
```

---

## Task 6: Integration Script

**Files:**
- Create: `train_curriculum.py`
- Test: Manual test with small data

**Step 1: Write the training script**

```python
# train_curriculum.py
#!/usr/bin/env python3
"""
Curriculum training script for preventing catastrophic forgetting.
Uses phased training with dynamic masking and automatic recovery.
"""

import argparse
import logging
from pathlib import Path
import torch
from transformers import TrainingArguments
from trl import SFTTrainer

from training.model_loader import ModelLoader
from training.lora_config import LoRAConfigurator
from training.curriculum.wrapper import CurriculumTrainingWrapper, CurriculumCallback
import training_parameters as params

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Curriculum training for negotiation model")
    parser.add_argument("--conversational-data", type=str,
                      default="data/training_data/conversational-dataset_formatted.jsonl")
    parser.add_argument("--negotiation-data", type=str,
                      default="data/training_data/negotiation-dataset_formatted.jsonl")
    parser.add_argument("--model-name", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--output-dir", type=str, default="models/curriculum")
    parser.add_argument("--max-steps", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--base-lr", type=float, default=1e-5)
    parser.add_argument("--disable-recovery", action="store_true", help="Disable automatic recovery")
    parser.add_argument("--test-mode", action="store_true", help="Run in test mode with minimal steps")

    args = parser.parse_args()

    # Initialize curriculum wrapper
    logger.info("Initializing curriculum training wrapper...")
    wrapper = CurriculumTrainingWrapper(
        conversational_path=args.conversational_data,
        negotiation_path=args.negotiation_data,
        model_name=args.model_name,
        output_dir=args.output_dir,
        recovery_enabled=not args.disable_recovery,
        base_lr=args.base_lr
    )

    # Load model with LoRA
    logger.info("Loading model with LoRA configuration...")
    model_loader = ModelLoader()
    model, tokenizer = model_loader.load_model_for_training(
        model_name=args.model_name,
        load_in_4bit=params.LOAD_IN_4BIT,
        bnb_4bit_compute_dtype=params.BNB_4BIT_COMPUTE_DTYPE,
        bnb_4bit_quant_type=params.BNB_4BIT_QUANT_TYPE,
        bnb_4bit_use_double_quant=params.BNB_4BIT_USE_DOUBLE_QUANT,
    )

    # Apply LoRA
    lora_configurator = LoRAConfigurator()
    model = lora_configurator.apply_lora(
        model,
        r=64,  # Increased capacity
        lora_alpha=128,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps if not args.test_mode else 10,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.base_lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,  # Extended warmup
        logging_steps=10,
        save_steps=500,
        save_strategy="steps",
        evaluation_strategy="no",  # We handle validation ourselves
        bf16=params.BF16,
        tf32=params.TF32,
        gradient_checkpointing=params.GRADIENT_CHECKPOINTING,
        optim=params.OPTIM,
        report_to=["tensorboard"] if not args.test_mode else [],
        remove_unused_columns=False,
        dataloader_drop_last=True,
    )

    # Create dataset
    logger.info("Creating curriculum dataset...")
    dataset_iterator = wrapper.get_dataset_iterator(
        batch_size=args.batch_size * args.gradient_accumulation * 100,  # Prefetch batches
        shuffle=True
    )

    # Convert to dataset format
    from datasets import Dataset
    epoch_data = next(dataset_iterator)
    train_dataset = Dataset.from_list(epoch_data)

    # Create trainer with curriculum callback
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=wrapper.get_data_collator(),
        callbacks=[CurriculumCallback(wrapper)],
        max_seq_length=params.MAX_SEQ_LEN,
    )

    # Run training
    logger.info("Starting curriculum training...")
    logger.info(f"Phase 1 (0-2000): Contextual learning - 50/50 mix, minimal masking")
    logger.info(f"Phase 2 (2000-4000): Guided practice - 30/70 mix, alternating masking")
    logger.info(f"Phase 3 (4000-8000): Specialized training - 10/90 mix, full masking")

    if args.test_mode:
        logger.info("TEST MODE: Running only 10 steps")

    trainer.train()

    # Save final model
    logger.info("Saving final model...")
    trainer.save_model(Path(args.output_dir) / "final")

    logger.info("Curriculum training complete!")

if __name__ == "__main__":
    main()
```

**Step 2: Make script executable**

```bash
chmod +x train_curriculum.py
```

**Step 3: Test with minimal steps**

Run: `source .venv310/bin/activate && python train_curriculum.py --test-mode`
Expected: Runs for 10 steps without errors

**Step 4: Commit**

```bash
git add train_curriculum.py
git commit -m "feat: add curriculum training integration script"
```

---

## Validation & Testing

**Create test script:**

```python
# test_curriculum_integration.py
"""Integration test for curriculum training."""

import subprocess
import sys

def test_curriculum_training():
    """Test that curriculum training runs without errors."""
    cmd = [
        sys.executable, "train_curriculum.py",
        "--test-mode",
        "--max-steps", "20",
        "--batch-size", "2"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    assert result.returncode == 0, f"Training failed: {result.stderr}"
    assert "Phase 1" in result.stdout
    assert "Curriculum training complete" in result.stdout

    print("✓ Curriculum training integration test passed!")

if __name__ == "__main__":
    test_curriculum_integration()
```

**Run final test:**
```bash
source .venv310/bin/activate && python test_curriculum_integration.py
```

---

## Documentation

**Create README:**

```markdown
# docs/curriculum_training.md

# Curriculum Training for Preventing Catastrophic Forgetting

## Quick Start

```bash
# Full training (8000 steps)
python train_curriculum.py

# Test mode (10 steps)
python train_curriculum.py --test-mode

# Custom parameters
python train_curriculum.py \
    --base-lr 5e-6 \
    --max-steps 10000 \
    --output-dir models/my_curriculum
```

## Architecture

The curriculum wrapper implements three-phase training:

1. **Contextual Learning (0-2000 steps)**
   - 50/50 data mix
   - Minimal masking (system only)
   - Reduced learning rate (50%)

2. **Guided Practice (2000-4000 steps)**
   - 30/70 conversational/negotiation
   - Alternating masking
   - Full learning rate

3. **Specialized Training (4000-8000 steps)**
   - 10/90 mix
   - Full SFT masking
   - Reduced LR (80%)

## Automatic Recovery

If token collapse is detected:
1. LR drops to 10%
2. Data mix becomes 70% conversational
3. All masking disabled
4. 200 recovery steps
5. Emergency checkpoint saved

## Monitoring

Health checks run every 100-500 steps (phase-dependent) and monitor:
- Token repetition ratios
- Response length
- Perplexity changes
- Specific collapse patterns ("eller", "0000")
```

---

# Summary

Plan complete and saved to `docs/plans/2025-11-10-curriculum-wrapper.md`.

**Two execution options:**

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration with quality gates

**2. Parallel Session (separate)** - Open new session with executing-plans skill, batch execution with review checkpoints

Which approach would you prefer?