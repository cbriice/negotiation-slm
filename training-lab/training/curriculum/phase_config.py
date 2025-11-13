from typing import Dict, Any, Tuple
import curriculum_parameters as cp

class CurriculumPhase:
    """Phase identifiers for curriculum learning."""
    PHASE_1_CONTEXTUAL = "contextual_learning"
    PHASE_2_GUIDED = "guided_practice"
    PHASE_3_SPECIALIZED = "specialized_training"

class PhaseConfig:
    """Configuration for curriculum training phases - loads from curriculum_parameters.py"""

    def __init__(self):
        """Initialize phases from curriculum_parameters.py"""
        self.phases = {}

        # Load phases from parameters file
        for phase in cp.PHASES:
            phase_name = phase["name"]
            self.phases[phase_name] = {
                "steps": (phase["step_start"], phase["step_end"]),
                "data_ratio": phase["data_ratio"],
                "lr_multiplier": phase["lr_multiplier"],
                "masking": phase["masking"],
                "validation_frequency": phase["validation_frequency"]
            }

    def get_phase_at_step(self, step: int) -> str:
        """Get the phase name for a given training step."""
        for phase_name, config in self.phases.items():
            start, end = config["steps"]
            if start <= step < end:
                return phase_name
        # Default to last phase if beyond defined steps
        phase_names = list(self.phases.keys())
        return phase_names[-1] if phase_names else None

    def get_phase_config(self, step: int) -> Dict[str, Any]:
        """Get the full configuration for the phase at a given step."""
        phase_name = self.get_phase_at_step(step)
        return self.phases.get(phase_name, {})
