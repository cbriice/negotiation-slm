"""SFTTrainer subclass that applies per-token loss weights produced by CurriculumMaskingCollator."""

import torch.nn as nn
from trl import SFTTrainer


class WeightedLossSFTTrainer(SFTTrainer):
    """SFTTrainer with weighted loss support for curriculum learning."""

    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute loss with per-token weighting if loss_weights present."""
        loss_weights = inputs.pop("loss_weights", None)
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if loss_weights is None:
            # Standard loss
            loss = outputs.loss if hasattr(outputs, 'loss') and outputs.loss is not None else None
            if loss is None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return (loss, outputs) if return_outputs else loss

        # Weighted loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_weights = loss_weights[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(reduction='none')
        per_token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        per_token_loss = per_token_loss.view(shift_labels.size())
        weighted_loss = per_token_loss * shift_weights
        # Normalize by sum of weights, not total token count
        # This ensures proper loss scaling when most tokens have weight=0
        loss = weighted_loss.sum() / shift_weights.sum()

        return (loss, outputs) if return_outputs else loss
