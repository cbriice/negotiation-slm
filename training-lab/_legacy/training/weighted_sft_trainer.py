"""
Custom SFTTrainer that implements weighted loss computation.
Inherits from TRL's SFTTrainer and overrides compute_loss.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, Union, Any, Tuple
from trl import SFTTrainer
import logging

logger = logging.getLogger(__name__)


class WeightedSFTTrainer(SFTTrainer):
    """
    Custom SFTTrainer that applies different loss weights to different parts of the input.

    This trainer expects the data collator to provide a 'loss_weights' tensor
    that specifies the weight for each token in the sequence.
    """

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Compute weighted loss for the model outputs.

        Args:
            model: The model being trained
            inputs: Dictionary containing input tensors
            return_outputs: Whether to return model outputs along with loss

        Returns:
            Loss tensor or tuple of (loss, outputs) if return_outputs=True
        """
        # Extract loss weights if present
        loss_weights = inputs.pop("loss_weights", None)

        # Get model outputs
        outputs = model(**inputs)

        # If no custom weights provided, use standard loss
        if loss_weights is None:
            loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
            return (loss, outputs) if return_outputs else loss

        # Compute custom weighted loss
        logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
        labels = inputs.get("labels")

        if labels is None:
            raise ValueError("Labels must be provided for weighted loss computation")

        # Shift logits and labels for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_weights = loss_weights[..., 1:].contiguous()

        # Flatten tensors for loss computation
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        shift_weights = shift_weights.view(-1)

        # Compute cross entropy loss per token
        loss_per_token = F.cross_entropy(
            shift_logits,
            shift_labels,
            reduction='none',
            ignore_index=-100  # Standard ignore index for padding
        )

        # Apply weights to the loss
        weighted_loss = loss_per_token * shift_weights

        # Count valid tokens (not ignored)
        valid_tokens = (shift_labels != -100).float()
        num_valid_tokens = valid_tokens.sum()

        # Compute mean loss over valid tokens
        if num_valid_tokens > 0:
            loss = weighted_loss.sum() / num_valid_tokens
        else:
            loss = weighted_loss.sum()  # Fallback if no valid tokens

        # Log some statistics periodically
        if self.state.global_step % 100 == 0:
            with torch.no_grad():
                # Count tokens by weight category
                system_tokens = (shift_weights == 0.0).sum().item()
                prompt_tokens = ((shift_weights > 0.0) & (shift_weights < 1.0)).sum().item()
                response_tokens = (shift_weights == 1.0).sum().item()

                logger.debug(f"Step {self.state.global_step} - Token distribution: "
                           f"System: {system_tokens}, Prompt: {prompt_tokens}, Response: {response_tokens}")

                # Compute average loss by category
                if prompt_tokens > 0:
                    prompt_loss = (loss_per_token[shift_weights < 1.0].mean()).item()
                    logger.debug(f"Average prompt loss: {prompt_loss:.4f}")
                if response_tokens > 0:
                    response_loss = (loss_per_token[shift_weights == 1.0].mean()).item()
                    logger.debug(f"Average response loss: {response_loss:.4f}")

        return (loss, outputs) if return_outputs else loss