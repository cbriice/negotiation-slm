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
