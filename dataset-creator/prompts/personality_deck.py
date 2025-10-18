"""
personality_deck.py

This file contains classes which define the trait and parameter values for each personality in the deck. 
During selfplay negotiation sims, either side's behavior will be defined by their personality, selected from this "deck" of options. 

"""
from dataclasses import dataclass, field
from math import sqrt
import numpy as np

@dataclass
class PersonalityTraits:
    # traits
    assertiveness: float
    dominance: float
    confidence_self: float
    confidence_deal: float
    cooperativeness: float
    flexibility: float
    patience: float
    empathy: float
    rationality: float
    deceptiveness: float
    risk_tol: float
    persistence: float
    opportunism: float
    strategic: float
    composure: float

    # independent params
    beta: float                         # price threshold adjustment weight (self)
    alpha: float                        # (buyer only) buyer's first offer = seller min * alpha
    temp: float                         # model temp for this personality

    # trait-dependent params
    zeta: float = field(init=False)     # max turns multiplier
    c: float = field(init=False)        # faratin convergence parameter
    eta: float = field(init=False)      # ultimatum parameter (if eta > 0.5, enable ultimatums)
    p: float = field(init=False)        # % chance of price concession on price adjust turn

    def __post_init__(self):
        self.zeta = self.patience
        self.c = sqrt(self.cooperativeness * self.flexibility)
        self.eta = sqrt(self.assertiveness * self.dominance * self.risk_tol)
        self.p = float(np.mean([1 - self.confidence_self, self.cooperativeness, self.flexibility, 1 - self.persistence]))

# -------------------------- Personalities -------------------------------
class StraightToPoint():
    def __init__(self, traits: PersonalityTraits):
        self.description = "A straight to the point negotiator. They just want to close the deal quickly and favorably."

        self.traits = traits
        traits.assertiveness = 0.7
        traits.dominance = 0.6
        traits.confidence_self = 0.8
        traits.confidence_deal = 1
        traits.cooperativeness = 0.7
        traits.flexibility = 0.4
        traits.patience = 0.6
        traits.empathy = 0.2
        traits.rationality = 0.7
        traits.deceptiveness = 0.1
        traits.risk_tol = 0.2
        traits.persistence = 0.5
        traits.opportunism = 0.6
        traits.strategic = 0.5
        traits.composure = 0.9

        self.trait_vector = np.array([
            traits.assertiveness, traits.dominance, traits.confidence_self, traits.confidence_deal, traits.cooperativeness, traits.flexibility, traits.patience,
            traits.empathy, traits.rationality, traits.deceptiveness, traits.risk_tol, traits.persistence, traits.opportunism, traits.strategic, traits.composure
        ])
        
        traits.beta = 0
        traits.alpha = 0.95
        traits.temp = 0.5

class Desperate():
    def __init__(self, traits: PersonalityTraits):
        self.description = "A desperate negotiator. They'll do almost anything to close the deal."

        self.traits = traits
        traits.assertiveness = 0.2
        traits.dominance = 0.1
        traits.confidence_self = 0.3
        traits.confidence_deal = 0.3
        traits.cooperativeness = 1
        traits.flexibility = 0.8
        traits.patience = 0.8
        traits.empathy = 0.6
        traits.rationality = 0.2
        traits.deceptiveness = 0.8
        traits.risk_tol = 0.2
        traits.persistence = 0.3
        traits.opportunism = 1
        traits.strategic = 0.7
        traits.composure = 0.3

        self.trait_vector = np.array([
            traits.assertiveness, traits.dominance, traits.confidence_self, traits.confidence_deal, traits.cooperativeness, traits.flexibility, traits.patience,
            traits.empathy, traits.rationality, traits.deceptiveness, traits.risk_tol, traits.persistence, traits.opportunism, traits.strategic, traits.composure
        ])
        
        traits.beta = 0.3
        traits.alpha = 1
        traits.temp = 0.8

class Skeptic():
    def __init__(self, traits: PersonalityTraits):
        self.description = "A skeptical negotiator. Not easily convinced to change their mind on the deal."

        self.traits = traits
        traits.assertiveness = 0.6
        traits.dominance = 0.6
        traits.confidence_self = 1
        traits.confidence_deal = 0.3
        traits.cooperativeness = 2
        traits.flexibility = 0.6
        traits.patience = 0.7
        traits.empathy = 0.2
        traits.rationality = 0.9
        traits.deceptiveness = 0.3
        traits.risk_tol = 0.4
        traits.persistence = 0.5
        traits.opportunism = 0.5
        traits.strategic = 0.9
        traits.composure = 0.7

        self.trait_vector = np.array([
            traits.assertiveness, traits.dominance, traits.confidence_self, traits.confidence_deal, traits.cooperativeness, traits.flexibility, traits.patience,
            traits.empathy, traits.rationality, traits.deceptiveness, traits.risk_tol, traits.persistence, traits.opportunism, traits.strategic, traits.composure
        ])
        
        traits.beta = 0.1
        traits.alpha = 0.85
        traits.temp = 0.6

class Bluffer():
    def __init__(self, traits: PersonalityTraits):
        self.description = "A bluffing negotiator. You will push strongly for a favorable deal even if it's not super reasonable."

        self.traits = traits
        traits.assertiveness = 1
        traits.dominance = 0.8
        traits.confidence_self = 0.9
        traits.confidence_deal = 0.2
        traits.cooperativeness = 0.3
        traits.flexibility = 0.3
        traits.patience = 0.5
        traits.empathy = 0.4
        traits.rationality = 0.4
        traits.deceptiveness = 0.9
        traits.risk_tol = 0.9
        traits.persistence = 0.8
        traits.opportunism = 1
        traits.strategic = 1
        traits.composure = 0.8

        self.trait_vector = np.array([
            traits.assertiveness, traits.dominance, traits.confidence_self, traits.confidence_deal, traits.cooperativeness, traits.flexibility, traits.patience,
            traits.empathy, traits.rationality, traits.deceptiveness, traits.risk_tol, traits.persistence, traits.opportunism, traits.strategic, traits.composure
        ])
        
        traits.beta = 0.1
        traits.alpha = 0.5
        traits.temp = 0.7

class Confrontational():
    def __init__(self, traits: PersonalityTraits):
        self.description = "A confrontational negotiator. They aggressively push for a good deal using whatever means necessary."

        self.traits = traits
        traits.assertiveness = 0.9
        traits.dominance = 1
        traits.confidence_self = 1
        traits.confidence_deal = 0.9
        traits.cooperativeness = 0.3
        traits.flexibility = 0.2
        traits.patience = 0.6
        traits.empathy = 0.1
        traits.rationality = 0.7
        traits.deceptiveness = 0.6
        traits.risk_tol = 0.8
        traits.persistence = 0.7
        traits.opportunism = 0.6
        traits.strategic = 0.6
        traits.composure = 0.4

        self.trait_vector = np.array([
            traits.assertiveness, traits.dominance, traits.confidence_self, traits.confidence_deal, traits.cooperativeness, traits.flexibility, traits.patience,
            traits.empathy, traits.rationality, traits.deceptiveness, traits.risk_tol, traits.persistence, traits.opportunism, traits.strategic, traits.composure
        ])
        
        traits.beta = 0
        traits.alpha = 0.75
        traits.temp = 0.6

class Haggler():
    def __init__(self, traits: PersonalityTraits):
        self.description = "A haggling negotiator. They don't fold easily and want to get as much from the counterparty as they can."
        
        self.traits = traits
        traits.assertiveness = 0.8
        traits.dominance = 0.7
        traits.confidence_self = 0.9
        traits.confidence_deal = 0.8
        traits.cooperativeness = 0.3
        traits.flexibility = 0.2
        traits.patience = 0.8
        traits.empathy = 0.3
        traits.rationality = 0.4
        traits.deceptiveness = 0.7
        traits.risk_tol = 0.6
        traits.persistence = 1
        traits.opportunism = 0.8
        traits.strategic = 0.7
        traits.composure = 0.8

        self.trait_vector = np.array([
            traits.assertiveness, traits.dominance, traits.confidence_self, traits.confidence_deal, traits.cooperativeness, traits.flexibility, traits.patience,
            traits.empathy, traits.rationality, traits.deceptiveness, traits.risk_tol, traits.persistence, traits.opportunism, traits.strategic, traits.composure
        ])
        
        traits.beta = 0.05
        traits.alpha = 0.75
        traits.temp = 0.7

class WeakEmotional(): 
    def __init__(self, traits: PersonalityTraits):
        self.description = "A weak and emotional negotiator. They just want to reach a deal, and don't like confrontation."

        self.traits = traits
        traits.assertiveness = 0.4
        traits.dominance = 0.3
        traits.confidence_self = 0.5
        traits.confidence_deal = 0.6
        traits.cooperativeness = 0.8
        traits.flexibility = 0.8
        traits.patience = 0.7
        traits.empathy = 1
        traits.rationality = 0.1
        traits.deceptiveness = 0.7
        traits.risk_tol = 0.2
        traits.persistence = 0.5
        traits.opportunism = 0.8
        traits.strategic = 0.2
        traits.composure = 0.1

        self.trait_vector = np.array([
            traits.assertiveness, traits.dominance, traits.confidence_self, traits.confidence_deal, traits.cooperativeness, traits.flexibility, traits.patience,
            traits.empathy, traits.rationality, traits.deceptiveness, traits.risk_tol, traits.persistence, traits.opportunism, traits.strategic, traits.composure
        ])
        
        traits.beta = 0.2
        traits.alpha = 0.8
        traits.temp = 0.8

class Rationalizer():
    def __init__(self, traits: PersonalityTraits):
        self.description = "A rational negotiator. Prefer to use reason and logic to win the negotiation and reach a good deal."

        self.traits = traits
        traits.assertiveness = 0.7
        traits.dominance = 0.7
        traits.confidence_self = 0.7
        traits.confidence_deal = 0.8
        traits.cooperativeness = 0.6
        traits.flexibility = 0.5
        traits.patience = 0.8
        traits.empathy = 0.2
        traits.rationality = 1
        traits.deceptiveness = 0.3
        traits.risk_tol = 0.5
        traits.persistence = 0.6
        traits.opportunism = 0.7
        traits.strategic = 1
        traits.composure = 0.9

        self.trait_vector = np.array([
            traits.assertiveness, traits.dominance, traits.confidence_self, traits.confidence_deal, traits.cooperativeness, traits.flexibility, traits.patience,
            traits.empathy, traits.rationality, traits.deceptiveness, traits.risk_tol, traits.persistence, traits.opportunism, traits.strategic, traits.composure
        ])
        
        traits.beta = 0
        traits.alpha = 0.9
        traits.temp = 0.5

class StoneWall():
    def __init__(self, traits: PersonalityTraits):
        self.description = "A stonewall negotiator. They don't even want to hear it from the counterparty - their offer doesn't need much changing."

        self.traits = traits
        traits.assertiveness = 1
        traits.dominance = 0.7
        traits.confidence_self = 0.9
        traits.confidence_deal = 1
        traits.cooperativeness = 0.2
        traits.flexibility = 0.1
        traits.patience = 0.7
        traits.empathy = 0.1
        traits.rationality = 0.8
        traits.deceptiveness = 0.2
        traits.risk_tol = 0.7
        traits.persistence = 0.4
        traits.opportunism = 0.2
        traits.strategic = 0.3
        traits.composure = 1

        self.trait_vector = np.array([
            traits.assertiveness, traits.dominance, traits.confidence_self, traits.confidence_deal, traits.cooperativeness, traits.flexibility, traits.patience,
            traits.empathy, traits.rationality, traits.deceptiveness, traits.risk_tol, traits.persistence, traits.opportunism, traits.strategic, traits.composure
        ])
        
        traits.beta = 0
        traits.alpha = 0.9
        traits.temp = 0.4

class Opportunist():
    def __init__(self, traits: PersonalityTraits):
        self.description = "An opportunist negotiator. They will jump at a favorable deal as soon as it presents itself."

        self.traits = traits
        traits.assertiveness = 0.5
        traits.dominance = 0.4
        traits.confidence_self = 0.9
        traits.confidence_deal = 0.6
        traits.cooperativeness = 0.4
        traits.flexibility = 0.7
        traits.patience = 1
        traits.empathy = 0.5
        traits.rationality = 0.7
        traits.deceptiveness = 0.9
        traits.risk_tol = 0.5
        traits.persistence = 0.2
        traits.opportunism = 1
        traits.strategic = 0.9
        traits.composure = 0.9

        self.trait_vector = np.array([
            traits.assertiveness, traits.dominance, traits.confidence_self, traits.confidence_deal, traits.cooperativeness, traits.flexibility, traits.patience,
            traits.empathy, traits.rationality, traits.deceptiveness, traits.risk_tol, traits.persistence, traits.opportunism, traits.strategic, traits.composure
        ])
        
        traits.beta = 0.05
        traits.alpha = 0.8
        traits.temp = 0.7

class HighHorse():
    def __init__(self, traits: PersonalityTraits):
        self.description = "A high-and-mighty negotiator. They think the counterparty is being ridiculous for even thinking of trying to undermine them."

        self.traits = traits
        traits.assertiveness = 1
        traits.dominance = 0.9
        traits.confidence_self = 1
        traits.confidence_deal = 0.9
        traits.cooperativeness = 0.1
        traits.flexibility = 0.1
        traits.patience = 0.3
        traits.empathy = 0.1
        traits.rationality = 0.2
        traits.deceptiveness = 0.2
        traits.risk_tol = 0.8
        traits.persistence = 0.3
        traits.opportunism = 0.5
        traits.strategic = 0.2
        traits.composure = 0.8

        self.trait_vector = np.array([
            traits.assertiveness, traits.dominance, traits.confidence_self, traits.confidence_deal, traits.cooperativeness, traits.flexibility, traits.patience,
            traits.empathy, traits.rationality, traits.deceptiveness, traits.risk_tol, traits.persistence, traits.opportunism, traits.strategic, traits.composure
        ])
        
        traits.beta = 0
        traits.alpha = 0.8
        traits.temp = 0.6

class Stubborn():
    def __init__(self, traits: PersonalityTraits):
        self.description = "A stubborn negotiator. They hate to change the terms of the deal from how they started and need a lot of convincing."

        self.traits = traits
        traits.assertiveness = 1
        traits.dominance = 0.6
        traits.confidence_self = 1
        traits.confidence_deal = 1
        traits.cooperativeness = 0.2
        traits.flexibility = 0.1
        traits.patience = 0.8
        traits.empathy = 0.3
        traits.rationality = 0.3
        traits.deceptiveness = 0.2
        traits.risk_tol = 0.6
        traits.persistence = 0.9
        traits.opportunism = 0.5
        traits.strategic = 0.5
        traits.composure = 0.7

        self.trait_vector = np.array([
            traits.assertiveness, traits.dominance, traits.confidence_self, traits.confidence_deal, traits.cooperativeness, traits.flexibility, traits.patience,
            traits.empathy, traits.rationality, traits.deceptiveness, traits.risk_tol, traits.persistence, traits.opportunism, traits.strategic, traits.composure
        ])
        
        traits.beta = 0
        traits.alpha = 0.85
        traits.temp = 0.5


# -------------------------- Personality Deck Selector -------------------------------
import random
from typing import Optional

class PersonalityDeck:
    """Manages selection of personalities from the deck."""

    def __init__(self):
        """Initialize the personality deck with all available personalities."""
        # Create factory functions - each personality will set its own trait values
        # We pass dummy values since each personality class overwrites them
        def create_traits():
            return PersonalityTraits(
                assertiveness=0.5, dominance=0.5, confidence_self=0.5,
                confidence_deal=0.5, cooperativeness=0.5, flexibility=0.5,
                patience=0.5, empathy=0.5, rationality=0.5, deceptiveness=0.5,
                risk_tol=0.5, persistence=0.5, opportunism=0.5, strategic=0.5,
                composure=0.5, beta=0.1, alpha=0.8, temp=0.6
            )

        self.personalities = [
            lambda: StraightToPoint(create_traits()),
            lambda: Desperate(create_traits()),
            lambda: Skeptic(create_traits()),
            lambda: Bluffer(create_traits()),
            lambda: Confrontational(create_traits()),
            lambda: Haggler(create_traits()),
            lambda: WeakEmotional(create_traits()),
            lambda: Rationalizer(create_traits()),
            lambda: StoneWall(create_traits()),
            lambda: Opportunist(create_traits()),
            lambda: HighHorse(create_traits()),
            lambda: Stubborn(create_traits())
        ]

        self.personality_names = [
            "StraightToPoint", "Desperate", "Skeptic", "Bluffer",
            "Confrontational", "Haggler", "WeakEmotional", "Rationalizer",
            "StoneWall", "Opportunist", "HighHorse", "Stubborn"
        ]

    def select_personality(self, specific_name: Optional[str] = None):
        """
        Select a personality from the deck.

        Args:
            specific_name: Optional name of specific personality to select

        Returns:
            Instantiated personality object
        """
        if specific_name:
            # Find personality by name
            try:
                index = self.personality_names.index(specific_name)
                return self.personalities[index]()
            except ValueError:
                print(f"Personality '{specific_name}' not found. Selecting random.")

        # Select random personality
        personality_factory = random.choice(self.personalities)
        return personality_factory()

    def select_two_personalities(self, avoid_same: bool = True):
        """
        Select two personalities for buyer and seller.

        Args:
            avoid_same: If True, ensures different personalities are selected

        Returns:
            Tuple of (buyer_personality, seller_personality)
        """
        if avoid_same and len(self.personalities) > 1:
            indices = random.sample(range(len(self.personalities)), 2)
            buyer = self.personalities[indices[0]]()
            seller = self.personalities[indices[1]]()
        else:
            buyer = self.select_personality()
            seller = self.select_personality()

        return buyer, seller

    def get_personality_by_traits(self, target_traits: np.ndarray, top_k: int = 3):
        """
        Select personality most similar to target trait vector.

        Args:
            target_traits: Target trait vector (15 dimensions)
            top_k: Number of top similar personalities to randomly select from

        Returns:
            Personality object most similar to target traits
        """
        similarities = []

        for personality_factory in self.personalities:
            personality = personality_factory()
            similarity = np.dot(personality.trait_vector, target_traits) / (
                np.linalg.norm(personality.trait_vector) * np.linalg.norm(target_traits)
            )
            similarities.append((personality, similarity))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Select from top k
        top_personalities = [p[0] for p in similarities[:top_k]]
        return random.choice(top_personalities)