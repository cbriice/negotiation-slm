"""Negotiation tactics library for RuleAgent."""

from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import random

class TacticType(Enum):
    # Opening tactics
    ANCHORING = "anchoring"
    EXTREME_OPENING = "extreme_opening"
    REASONABLE_OPENING = "reasonable_opening"
    
    # Pressure tactics
    DEADLINE_PRESSURE = "deadline_pressure"
    COMPETITION_MENTION = "competition_mention"
    ULTIMATUM = "ultimatum"
    URGENCY = "urgency"
    
    # Value tactics
    VALUE_CREATION = "value_creation"
    BUNDLING = "bundling"
    TRADE_OFFS = "trade_offs"
    NIBBLING = "nibbling"
    
    # Social tactics
    RAPPORT_BUILDING = "rapport_building"
    SYMPATHY_APPEAL = "sympathy_appeal"
    PERSONAL_STORY = "personal_story"
    RECIPROCITY = "reciprocity"
    SOCIAL_PROOF = "social_proof"
    
    # Deceptive tactics
    FALSE_COMPETITION = "false_competition"
    FAKE_CONSTRAINTS = "fake_constraints"
    GOOD_COP_BAD_COP = "good_cop_bad_cop"
    
    # Analytical tactics
    DATA_REFERENCE = "data_reference"
    COST_BREAKDOWN = "cost_breakdown"
    MARKET_COMPARISON = "market_comparison"
    LOGICAL_ARGUMENT = "logical_argument"
    
    # Negotiation flow tactics
    REFRAMING = "reframing"
    OPTION_GENERATION = "option_generation"
    FACE_SAVING = "face_saving"
    GRADUAL_CONCESSIONS = "gradual_concessions"
    FAST_CONCESSIONS = "fast_concessions"
    BINARY_CHOICES = "binary_choices"
    TRANSPARENCY = "transparency"

TACTIC_DEFINITIONS = {
    TacticType.ANCHORING: {
        "name": "Anchoring",
        "description": "Set an extreme initial position to influence perception",
        "when_to_use": "opening",
        "effectiveness": 0.7,
        "templates": [
            "I'm looking for at least ${high_anchor} for this",
            "Similar items sell for ${high_anchor}, but I could consider ${slightly_lower}",
            "My absolute minimum is ${anchor}, and that's already a huge discount",
        ],
        "implementation": {
            "buyer": lambda price: price * 0.5,  # Start at 50% of asking
            "seller": lambda price: price * 1.8,  # Start at 180% of target
        }
    },
    
    TacticType.DEADLINE_PRESSURE: {
        "name": "Deadline Pressure",
        "description": "Create urgency through time constraints",
        "when_to_use": "middle",
        "effectiveness": 0.6,
        "templates": [
            "I need to make a decision by {deadline}",
            "This offer expires in {time_limit}",
            "I have another appointment in {minutes} minutes",
            "After today, the price goes back up",
        ],
        "implementation": {
            "buyer": lambda: f"{random.choice(['30', '60', '90'])} minutes",
            "seller": lambda: f"{random.choice(['today', 'this hour', 'noon'])}",
        }
    },
    
    TacticType.BUNDLING: {
        "name": "Bundling",
        "description": "Combine multiple items or conditions",
        "when_to_use": "middle",
        "effectiveness": 0.65,
        "templates": [
            "What if I also threw in {additional_item}?",
            "I could do {price} if you also take {other_item}",
            "Buy this and I'll give you a discount on {related_item}",
        ],
        "implementation": {
            "buyer": lambda items: f"I'll take both if you do ${sum(items) * 0.85}",
            "seller": lambda items: f"Take all {len(items)} for ${sum(items) * 0.9}",
        }
    },
    
    TacticType.SYMPATHY_APPEAL: {
        "name": "Sympathy Appeal",
        "description": "Appeal to emotions and circumstances",
        "when_to_use": "any",
        "effectiveness": 0.5,
        "templates": [
            "I'm really struggling financially right now",
            "This would mean a lot to my family",
            "I'm trying to make ends meet",
            "It's been a tough year",
        ],
        "implementation": {
            "buyer": lambda: "I'm on a fixed budget",
            "seller": lambda: "I really need this sale",
        }
    },
    
    TacticType.FALSE_COMPETITION: {
        "name": "False Competition",
        "description": "Mention non-existent competing offers",
        "when_to_use": "middle",
        "effectiveness": 0.55,
        "templates": [
            "I have someone else offering ${competitor_price}",
            "Another buyer/seller contacted me with a better deal",
            "I'm talking to three other people about this",
        ],
        "implementation": {
            "buyer": lambda price: f"Another seller offered ${price * 0.9}",
            "seller": lambda price: f"Someone else offered ${price * 1.1}",
        }
    },
    
    TacticType.VALUE_CREATION: {
        "name": "Value Creation",
        "description": "Find ways to expand the pie",
        "when_to_use": "middle",
        "effectiveness": 0.75,
        "templates": [
            "What if we structured this differently?",
            "Could we add {feature} to make this work for both of us?",
            "Let's think creatively about how to make this win-win",
        ],
        "implementation": {
            "buyer": lambda: "What if I paid cash for a discount?",
            "seller": lambda: "I could include free delivery at this price",
        }
    },
    
    TacticType.MARKET_COMPARISON: {
        "name": "Market Comparison",
        "description": "Reference market prices and data",
        "when_to_use": "any",
        "effectiveness": 0.7,
        "templates": [
            "Market price for this is typically ${market_price}",
            "According to recent sales data, these go for ${price_range}",
            "The blue book value is ${reference_price}",
        ],
        "implementation": {
            "buyer": lambda price: f"Market average is ${price * 0.85}",
            "seller": lambda price: f"These typically sell for ${price * 1.15}",
        }
    },
    
    TacticType.GRADUAL_CONCESSIONS: {
        "name": "Gradual Concessions",
        "description": "Make small, decreasing concessions over time",
        "when_to_use": "middle",
        "effectiveness": 0.65,
        "templates": [
            "I could go to ${new_price}, but that's pushing it",
            "Maybe ${small_change} more, but that's really my limit",
            "I'll meet you at ${compromise}, final offer",
        ],
        "implementation": {
            "buyer": lambda last_offer, round: last_offer + (100 / (round + 1)),
            "seller": lambda last_offer, round: last_offer - (100 / (round + 1)),
        }
    },
    
    TacticType.ULTIMATUM: {
        "name": "Ultimatum",
        "description": "Final offer with walk-away threat",
        "when_to_use": "late",
        "effectiveness": 0.5,
        "templates": [
            "This is my absolute final offer: ${price}",
            "${price}, take it or leave it",
            "I'm walking away if you can't do ${price}",
        ],
        "implementation": {
            "buyer": lambda price: price,
            "seller": lambda price: price,
        }
    },
    
    TacticType.REFRAMING: {
        "name": "Reframing",
        "description": "Change the perspective on the negotiation",
        "when_to_use": "any",
        "effectiveness": 0.6,
        "templates": [
            "Think of it as an investment, not a cost",
            "This isn't about price, it's about value",
            "Let's focus on what we both need here",
        ],
        "implementation": {
            "buyer": lambda: "Quality matters more than price",
            "seller": lambda: "This is about finding the right buyer",
        }
    }
}

class TacticLibrary:
    """Manages negotiation tactics and their application."""
    
    def __init__(self):
        self.tactics = TACTIC_DEFINITIONS
        self.tactic_history: List[TacticType] = []
        self.effectiveness_modifiers: Dict[TacticType, float] = {}
    
    def select_tactic(self, 
                     preferred_tactics: List[str],
                     negotiation_stage: str = "middle",
                     used_tactics: List[TacticType] = None) -> Optional[TacticType]:
        """Select an appropriate tactic based on preferences and context."""
        if used_tactics is None:
            used_tactics = []
        
        # Filter tactics by stage and preference
        available_tactics = []
        for tactic_type in TacticType:
            tactic = self.tactics.get(tactic_type)
            if not tactic:
                continue
                
            # Check if tactic matches stage
            when_to_use = tactic.get("when_to_use", "any")
            if when_to_use != "any" and when_to_use != negotiation_stage:
                continue
            
            # Check if tactic is preferred
            if tactic_type.value in preferred_tactics:
                # Avoid repeating the same tactic too soon
                if tactic_type not in used_tactics[-3:]:
                    available_tactics.append(tactic_type)
        
        # If no preferred tactics available, choose from stage-appropriate ones
        if not available_tactics:
            for tactic_type in TacticType:
                tactic = self.tactics.get(tactic_type)
                if tactic and tactic.get("when_to_use", "any") in ["any", negotiation_stage]:
                    if tactic_type not in used_tactics[-2:]:
                        available_tactics.append(tactic_type)
        
        if not available_tactics:
            return None
        
        # Weight by effectiveness
        weights = [self.tactics[t]["effectiveness"] for t in available_tactics]
        return random.choices(available_tactics, weights=weights, k=1)[0]
    
    def apply_tactic(self, 
                    tactic_type: TacticType,
                    side: str,
                    context: Dict[str, Any]) -> str:
        """Generate text using a specific tactic."""
        tactic = self.tactics.get(tactic_type)
        if not tactic:
            return ""
        
        # Select a template
        template = random.choice(tactic["templates"])
        
        # Apply implementation if available
        impl = tactic.get("implementation", {}).get(side)
        if impl and callable(impl):
            # Pass context to implementation
            if "price" in context:
                try:
                    result = impl(context["price"])
                    if isinstance(result, (int, float)):
                        template = template.replace("{price}", f"${result:.0f}")
                        template = template.replace("${high_anchor}", f"${result:.0f}")
                        template = template.replace("${anchor}", f"${result:.0f}")
                        template = template.replace("${slightly_lower}", f"${result * 0.9:.0f}")
                    else:
                        template = template.replace("{time_limit}", str(result))
                        template = template.replace("{deadline}", str(result))
                except:
                    pass
        
        # Fill in remaining placeholders with context
        for key, value in context.items():
            placeholder = f"{{{key}}}"
            if placeholder in template:
                template = template.replace(placeholder, str(value))
        
        # Clean up any remaining placeholders
        template = self._clean_template(template)
        
        return template
    
    def _clean_template(self, template: str) -> str:
        """Remove unfilled placeholders from template."""
        import re
        # Remove ${...} style placeholders
        template = re.sub(r'\$\{[^}]+\}', '[PRICE]', template)
        # Remove {...} style placeholders  
        template = re.sub(r'\{[^}]+\}', '', template)
        return template.strip()
    
    def get_counter_tactic(self, opponent_tactic: TacticType) -> Optional[TacticType]:
        """Suggest a counter-tactic for opponent's move."""
        counters = {
            TacticType.ANCHORING: [TacticType.MARKET_COMPARISON, TacticType.DATA_REFERENCE],
            TacticType.DEADLINE_PRESSURE: [TacticType.DEADLINE_PRESSURE, TacticType.FALSE_COMPETITION],
            TacticType.SYMPATHY_APPEAL: [TacticType.LOGICAL_ARGUMENT, TacticType.MARKET_COMPARISON],
            TacticType.FALSE_COMPETITION: [TacticType.FALSE_COMPETITION, TacticType.URGENCY],
            TacticType.ULTIMATUM: [TacticType.ULTIMATUM, TacticType.REFRAMING],
        }
        
        counter_options = counters.get(opponent_tactic, [])
        if counter_options:
            return random.choice(counter_options)
        return None
    
    def combine_tactics(self, 
                       tactics: List[TacticType],
                       side: str,
                       context: Dict[str, Any]) -> str:
        """Combine multiple tactics into a coherent response."""
        parts = []
        for tactic in tactics[:2]:  # Limit to 2 tactics per turn
            text = self.apply_tactic(tactic, side, context)
            if text:
                parts.append(text)
        
        return " ".join(parts) if parts else ""
    
    def get_stage(self, turn_count: int, max_turns: int) -> str:
        """Determine negotiation stage based on turn count."""
        ratio = turn_count / max_turns
        if ratio < 0.2:
            return "opening"
        elif ratio < 0.8:
            return "middle"
        else:
            return "late"