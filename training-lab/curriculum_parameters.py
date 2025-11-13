"""
Curriculum Training Parameters
Edit this file to customize curriculum phases, steps, and masking weights.
"""

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

# Maximum training steps (total training duration)
MAX_STEPS = 8000

# Quick test configurations (uncomment to use)
# MAX_STEPS = 2000  # Quick validation run (1-2 hours)
# MAX_STEPS = 500   # Ultra-quick diagnostic (10-15 minutes)

# =============================================================================
# CURRICULUM PHASES
# =============================================================================
# Each phase defines:
# - step_start/step_end: When this phase is active
# - data_ratio: Mix of conversational vs negotiation data
# - lr_multiplier: Learning rate adjustment for this phase
# - user_weight: Loss weight for user turns (0.0 = fully masked, 1.0 = full weight)
# - assistant_weight: Loss weight for assistant turns (typically 1.0)
# - system_weight: Loss weight for system prompts (typically 0.0)
# - validation_frequency: Steps between health checks
# =============================================================================

# PHASE 1: Contextual Learning
# Goal: Learn basic conversation patterns and negotiation context
PHASE_1 = {
    "name": "contextual_learning",
    "step_start": 0,
    "step_end": 2000,
    "data_ratio": {
        "conversational": 0.5,
        "negotiation": 0.5
    },
    "lr_multiplier": 0.7,
    "validation_frequency": 100,
    "masking": {
        "conversational": {
            "user_weight": 0.0,      # Fully mask conversational user turns
            "assistant_weight": 1.0,  # Full weight on assistant responses
            "system_weight": 0.0      # Mask system prompts
        },
        "negotiation": {
            "user_weight": 0.05,      # Tiny weight for context learning
            "assistant_weight": 1.0,
            "system_weight": 0.0
        }
    }
}

# PHASE 2: Guided Practice
# Goal: Focus on negotiation behavior while preserving conversational ability
PHASE_2 = {
    "name": "guided_practice",
    "step_start": 2000,
    "step_end": 6000,
    "data_ratio": {
        "conversational": 0.2,
        "negotiation": 0.8
    },
    "lr_multiplier": 1.0,
    "validation_frequency": 200,
    "masking": {
        "conversational": {
            "user_weight": 0.0,
            "assistant_weight": 1.0,
            "system_weight": 0.0
        },
        "negotiation": {
            "user_weight": 0.02,      # Reduced weight
            "assistant_weight": 1.0,
            "system_weight": 0.0
        }
    }
}

# PHASE 3: Specialized Training
# Goal: Pure negotiation specialization
PHASE_3 = {
    "name": "specialized_training",
    "step_start": 6000,
    "step_end": 8000,
    "data_ratio": {
        "conversational": 0.05,
        "negotiation": 0.95
    },
    "lr_multiplier": 0.6,
    "validation_frequency": 500,
    "masking": {
        "conversational": {
            "user_weight": 0.0,
            "assistant_weight": 1.0,
            "system_weight": 0.0
        },
        "negotiation": {
            "user_weight": 0.0,       # Full masking - assistant responses only
            "assistant_weight": 1.0,
            "system_weight": 0.0
        }
    }
}

# =============================================================================
# PHASE COLLECTION
# =============================================================================
# All phases in order - add more phases here if needed
PHASES = [PHASE_1, PHASE_2, PHASE_3]

# =============================================================================
# QUICK TEST PRESETS
# =============================================================================
# Uncomment one of these sections to use a preset configuration

# PRESET: Ultra-Quick Diagnostic (500 steps, ~10-15 minutes)
# Validates that weighted masking fixes the dialog generation problem
DIAGNOSTIC_500 = {
    "MAX_STEPS": 500,
    "PHASES": [
        {"name": "phase1", "step_start": 0, "step_end": 150,
         "data_ratio": {"conversational": 0.5, "negotiation": 0.5},
         "lr_multiplier": 0.7, "validation_frequency": 50,
         "masking": {"negotiation": {"user_weight": 0.05, "assistant_weight": 1.0, "system_weight": 0.0}}},
        {"name": "phase2", "step_start": 150, "step_end": 350,
         "data_ratio": {"conversational": 0.2, "negotiation": 0.8},
         "lr_multiplier": 1.0, "validation_frequency": 50,
         "masking": {"negotiation": {"user_weight": 0.02, "assistant_weight": 1.0, "system_weight": 0.0}}},
        {"name": "phase3", "step_start": 350, "step_end": 500,
         "data_ratio": {"conversational": 0.05, "negotiation": 0.95},
         "lr_multiplier": 0.6, "validation_frequency": 50,
         "masking": {"negotiation": {"user_weight": 0.0, "assistant_weight": 1.0, "system_weight": 0.0}}}
    ]
}

# PRESET: Quick Validation (2000 steps, ~1-2 hours)
# Full curriculum validation with all phase transitions
VALIDATION_2000 = {
    "MAX_STEPS": 2000,
    "PHASES": [
        {"name": "contextual_learning", "step_start": 0, "step_end": 500,
         "data_ratio": {"conversational": 0.5, "negotiation": 0.5},
         "lr_multiplier": 0.7, "validation_frequency": 100,
         "masking": {
             "conversational": {"user_weight": 0.0, "assistant_weight": 1.0, "system_weight": 0.0},
             "negotiation": {"user_weight": 0.05, "assistant_weight": 1.0, "system_weight": 0.0}
         }},
        {"name": "guided_practice", "step_start": 500, "step_end": 1500,
         "data_ratio": {"conversational": 0.2, "negotiation": 0.8},
         "lr_multiplier": 1.0, "validation_frequency": 200,
         "masking": {
             "conversational": {"user_weight": 0.0, "assistant_weight": 1.0, "system_weight": 0.0},
             "negotiation": {"user_weight": 0.02, "assistant_weight": 1.0, "system_weight": 0.0}
         }},
        {"name": "specialized_training", "step_start": 1500, "step_end": 2000,
         "data_ratio": {"conversational": 0.05, "negotiation": 0.95},
         "lr_multiplier": 0.6, "validation_frequency": 500,
         "masking": {
             "conversational": {"user_weight": 0.0, "assistant_weight": 1.0, "system_weight": 0.0},
             "negotiation": {"user_weight": 0.0, "assistant_weight": 1.0, "system_weight": 0.0}
         }}
    ]
}

# =============================================================================
# ACTIVATE PRESET (uncomment to use)
# =============================================================================
# To use a preset, uncomment one of these lines:

MAX_STEPS, PHASES = DIAGNOSTIC_500["MAX_STEPS"], DIAGNOSTIC_500["PHASES"]
# MAX_STEPS, PHASES = VALIDATION_2000["MAX_STEPS"], VALIDATION_2000["PHASES"]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_phase_at_step(step: int) -> dict:
    """Get the phase configuration for a given training step."""
    for phase in PHASES:
        if phase["step_start"] <= step < phase["step_end"]:
            return phase
    # Default to last phase if beyond defined steps
    return PHASES[-1]

def get_masking_weights(step: int, source: str) -> dict:
    """Get masking weights for a specific step and data source."""
    phase = get_phase_at_step(step)
    masking = phase["masking"]

    # Return weights for the specified source (conversational or negotiation)
    if source in masking:
        return masking[source]
    # Fallback to negotiation settings
    return masking.get("negotiation", {
        "user_weight": 0.0,
        "assistant_weight": 1.0,
        "system_weight": 0.0
    })

def print_curriculum_summary():
    """Print a summary of the curriculum configuration."""
    print("\n" + "="*80)
    print("CURRICULUM TRAINING CONFIGURATION")
    print("="*80)
    print(f"Total steps: {MAX_STEPS}")
    print(f"Number of phases: {len(PHASES)}")
    print()

    for i, phase in enumerate(PHASES, 1):
        print(f"Phase {i}: {phase['name']}")
        print(f"  Steps: {phase['step_start']}-{phase['step_end']}")
        print(f"  Data ratio: Conv {phase['data_ratio']['conversational']:.0%} / Neg {phase['data_ratio']['negotiation']:.0%}")
        print(f"  LR multiplier: {phase['lr_multiplier']:.2f}")

        # Print masking weights
        for source, weights in phase['masking'].items():
            user_w = weights.get('user_weight', 0.0)
            asst_w = weights.get('assistant_weight', 1.0)
            print(f"  {source.capitalize()} masking: User {user_w:.2f} / Assistant {asst_w:.2f}")
        print()

    print("="*80 + "\n")

# Print summary when module is imported
if __name__ != "__main__":
    print_curriculum_summary()
