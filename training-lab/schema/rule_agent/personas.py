"""Persona templates for RuleAgent negotiation styles."""

from typing import Dict, Any, List
from enum import Enum
import random

class PersonaType(Enum):
    AGGRESSIVE = "aggressive"
    COOPERATIVE = "cooperative"
    DECEPTIVE = "deceptive"
    SYMPATHETIC = "sympathetic"
    ANALYTICAL = "analytical"
    IMPATIENT = "impatient"
    DIPLOMATIC = "diplomatic"

PERSONA_TEMPLATES = {
    PersonaType.AGGRESSIVE: {
        "name": "Aggressive Negotiator",
        "description": "Pushes hard for maximum gain, uses pressure tactics",
        "traits": {
            "opening_style": "extreme",  # Start with extreme positions
            "concession_rate": 0.1,  # Slow to make concessions
            "patience": 0.3,  # Quick to threaten walking away
            "empathy": 0.1,  # Low empathy for other party
            "firmness": 0.9,  # Very firm on positions
        },
        "language_patterns": [
            "This is my final offer",
            "Take it or leave it",
            "You're wasting my time",
            "That's completely unreasonable",
            "I have other buyers/sellers waiting",
            "This conversation is going nowhere",
        ],
        "preferred_tactics": ["anchoring", "deadline_pressure", "competition_mention", "ultimatum"],
        "response_modifiers": {
            "reject_low_offers": 0.8,  # High chance to reject low offers
            "counter_aggressively": 0.7,  # Often counters with aggressive terms
            "walk_away_threshold": 0.6,  # More likely to walk away
        }
    },
    
    PersonaType.COOPERATIVE: {
        "name": "Cooperative Negotiator",
        "description": "Seeks win-win solutions, focuses on mutual benefit",
        "traits": {
            "opening_style": "reasonable",
            "concession_rate": 0.4,
            "patience": 0.8,
            "empathy": 0.7,
            "firmness": 0.5,
        },
        "language_patterns": [
            "Let's find something that works for both of us",
            "I understand your position",
            "What if we tried this approach",
            "I think we can both benefit here",
            "Let me see what I can do",
            "Fair enough, how about",
        ],
        "preferred_tactics": ["value_creation", "bundling", "trade_offs", "transparency"],
        "response_modifiers": {
            "reject_low_offers": 0.3,
            "counter_aggressively": 0.2,
            "walk_away_threshold": 0.2,
        }
    },
    
    PersonaType.DECEPTIVE: {
        "name": "Deceptive Negotiator",
        "description": "Uses misdirection and false information strategically",
        "traits": {
            "opening_style": "variable",
            "concession_rate": 0.3,
            "patience": 0.6,
            "empathy": 0.3,
            "firmness": 0.6,
        },
        "language_patterns": [
            "I have another offer for more, but I like you",
            "This usually sells for much higher",
            "I'm taking a loss here",
            "My boss won't be happy, but",
            "Between you and me",
            "I shouldn't tell you this, but",
        ],
        "preferred_tactics": ["false_competition", "fake_constraints", "good_cop_bad_cop", "nibbling"],
        "response_modifiers": {
            "reject_low_offers": 0.5,
            "counter_aggressively": 0.4,
            "walk_away_threshold": 0.4,
        }
    },
    
    PersonaType.SYMPATHETIC: {
        "name": "Sympathetic Negotiator",
        "description": "Appeals to emotion and builds rapport",
        "traits": {
            "opening_style": "friendly",
            "concession_rate": 0.5,
            "patience": 0.7,
            "empathy": 0.9,
            "firmness": 0.3,
        },
        "language_patterns": [
            "I really need this to work out",
            "You seem like a reasonable person",
            "I'm in a tough spot here",
            "I appreciate your understanding",
            "This would really help me out",
            "I know times are tough for everyone",
        ],
        "preferred_tactics": ["rapport_building", "sympathy_appeal", "personal_story", "reciprocity"],
        "response_modifiers": {
            "reject_low_offers": 0.4,
            "counter_aggressively": 0.1,
            "walk_away_threshold": 0.3,
        }
    },
    
    PersonaType.ANALYTICAL: {
        "name": "Analytical Negotiator",
        "description": "Data-driven, focuses on facts and logic",
        "traits": {
            "opening_style": "calculated",
            "concession_rate": 0.35,
            "patience": 0.9,
            "empathy": 0.4,
            "firmness": 0.7,
        },
        "language_patterns": [
            "Based on market data",
            "The numbers show",
            "If we look at comparable items",
            "Statistically speaking",
            "The fair market value is",
            "Let me break down the costs",
        ],
        "preferred_tactics": ["data_reference", "cost_breakdown", "market_comparison", "logical_argument"],
        "response_modifiers": {
            "reject_low_offers": 0.6,
            "counter_aggressively": 0.3,
            "walk_away_threshold": 0.5,
        }
    },
    
    PersonaType.IMPATIENT: {
        "name": "Impatient Negotiator",
        "description": "Wants quick resolution, time-sensitive",
        "traits": {
            "opening_style": "direct",
            "concession_rate": 0.6,
            "patience": 0.2,
            "empathy": 0.3,
            "firmness": 0.4,
        },
        "language_patterns": [
            "I need to close this now",
            "Let's cut to the chase",
            "I don't have time for this",
            "What's your best price",
            "Yes or no",
            "Make it quick",
        ],
        "preferred_tactics": ["urgency", "deadline_pressure", "fast_concessions", "binary_choices"],
        "response_modifiers": {
            "reject_low_offers": 0.5,
            "counter_aggressively": 0.5,
            "walk_away_threshold": 0.7,
        }
    },
    
    PersonaType.DIPLOMATIC: {
        "name": "Diplomatic Negotiator",
        "description": "Careful with words, avoids confrontation",
        "traits": {
            "opening_style": "moderate",
            "concession_rate": 0.45,
            "patience": 0.8,
            "empathy": 0.6,
            "firmness": 0.6,
        },
        "language_patterns": [
            "Perhaps we could consider",
            "With all due respect",
            "I see your point, however",
            "May I suggest an alternative",
            "Let's explore our options",
            "There might be a middle ground",
        ],
        "preferred_tactics": ["reframing", "option_generation", "face_saving", "gradual_concessions"],
        "response_modifiers": {
            "reject_low_offers": 0.4,
            "counter_aggressively": 0.2,
            "walk_away_threshold": 0.3,
        }
    }
}

class PersonaManager:
    """Manages persona selection and behavior modification."""
    
    @staticmethod
    def get_random_persona() -> Dict[str, Any]:
        """Get a random persona template."""
        persona_type = random.choice(list(PersonaType))
        return {
            "type": persona_type,
            "template": PERSONA_TEMPLATES[persona_type]
        }
    
    @staticmethod
    def get_persona(persona_type: PersonaType) -> Dict[str, Any]:
        """Get a specific persona template."""
        return {
            "type": persona_type,
            "template": PERSONA_TEMPLATES[persona_type]
        }
    
    @staticmethod
    def blend_personas(primary: PersonaType, secondary: PersonaType, 
                      blend_ratio: float = 0.7) -> Dict[str, Any]:
        """Blend two personas for more nuanced behavior."""
        primary_template = PERSONA_TEMPLATES[primary]
        secondary_template = PERSONA_TEMPLATES[secondary]
        
        blended = {
            "name": f"{primary.value}-{secondary.value} blend",
            "description": f"Blend of {primary.value} and {secondary.value} styles",
            "traits": {},
            "language_patterns": [],
            "preferred_tactics": [],
            "response_modifiers": {}
        }
        
        # Blend traits
        for trait in primary_template["traits"]:
            primary_value = primary_template["traits"][trait]
            secondary_value = secondary_template["traits"][trait]
            
            # Handle numeric traits
            if isinstance(primary_value, (int, float)) and isinstance(secondary_value, (int, float)):
                blended["traits"][trait] = (
                    primary_value * blend_ratio +
                    secondary_value * (1 - blend_ratio)
                )
            else:
                # For non-numeric traits, use primary if blend_ratio > 0.5, else secondary
                blended["traits"][trait] = primary_value if blend_ratio > 0.5 else secondary_value
        
        # Combine language patterns
        primary_count = int(len(primary_template["language_patterns"]) * blend_ratio)
        secondary_count = len(primary_template["language_patterns"]) - primary_count
        
        blended["language_patterns"] = (
            random.sample(primary_template["language_patterns"], min(primary_count, len(primary_template["language_patterns"]))) +
            random.sample(secondary_template["language_patterns"], min(secondary_count, len(secondary_template["language_patterns"])))
        )
        
        # Combine tactics
        blended["preferred_tactics"] = list(set(
            primary_template["preferred_tactics"][:int(len(primary_template["preferred_tactics"]) * blend_ratio)] +
            secondary_template["preferred_tactics"][:int(len(secondary_template["preferred_tactics"]) * (1 - blend_ratio))]
        ))
        
        # Blend response modifiers
        for modifier in primary_template["response_modifiers"]:
            blended["response_modifiers"][modifier] = (
                primary_template["response_modifiers"][modifier] * blend_ratio +
                secondary_template["response_modifiers"][modifier] * (1 - blend_ratio)
            )
        
        return {
            "type": f"{primary.value}_{secondary.value}_blend",
            "template": blended
        }
    
    @staticmethod
    def adjust_persona_intensity(persona: Dict[str, Any], intensity: float) -> Dict[str, Any]:
        """Adjust the intensity of persona traits (0.0 = mild, 1.0 = extreme)."""
        import copy
        adjusted = copy.deepcopy(persona)
        template = adjusted["template"]
        
        # Adjust traits towards extremes or middle based on intensity
        for trait, value in template["traits"].items():
            # Skip non-numeric traits
            if not isinstance(value, (int, float)):
                continue
                
            if value > 0.5:
                # Trait is high, make it more extreme with intensity
                template["traits"][trait] = 0.5 + (value - 0.5) * (1 + intensity)
            else:
                # Trait is low, make it more extreme with intensity
                template["traits"][trait] = 0.5 - (0.5 - value) * (1 + intensity)
            
            # Clamp between 0 and 1
            template["traits"][trait] = max(0, min(1, template["traits"][trait]))
        
        # Adjust response modifiers similarly
        for modifier, value in template["response_modifiers"].items():
            if value > 0.5:
                template["response_modifiers"][modifier] = 0.5 + (value - 0.5) * (1 + intensity)
            else:
                template["response_modifiers"][modifier] = 0.5 - (0.5 - value) * (1 + intensity)
            
            template["response_modifiers"][modifier] = max(0, min(1, template["response_modifiers"][modifier]))
        
        return adjusted