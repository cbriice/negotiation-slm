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
