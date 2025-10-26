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
        self.eta = round(sqrt(self.assertiveness * self.dominance * self.risk_tol), 3)
        self.p = round(float(np.mean([1 - self.confidence_self, self.cooperativeness, self.flexibility, 1 - self.persistence])), 3)

# -------------------------- Personalities -------------------------------
class StraightToPoint():
    def __init__(self):
        self.description = "A straight to the point negotiator. They just want to close the deal quickly and favorably."
        self.tag = "straightforward"

        self.traits = PersonalityTraits(
            assertiveness=0.7,
            dominance=0.6,
            confidence_self=0.8,
            confidence_deal=1,
            cooperativeness=0.7,
            flexibility=0.4,
            patience=0.6,
            empathy=0.2,
            rationality=0.7,
            deceptiveness=0.1,
            risk_tol=0.2,
            persistence=0.5,
            opportunism=0.6,
            strategic=0.5,
            composure=0.9,
            beta=0,
            alpha=0.95,
            temp=0.5
        )

        self.trait_vector = np.array([
            self.traits.assertiveness, self.traits.dominance, self.traits.confidence_self, self.traits.confidence_deal,
            self.traits.cooperativeness, self.traits.flexibility, self.traits.patience, self.traits.empathy,
            self.traits.rationality, self.traits.deceptiveness, self.traits.risk_tol, self.traits.persistence,
            self.traits.opportunism, self.traits.strategic, self.traits.composure
        ])

class Desperate():
    def __init__(self):
        self.description = "A desperate negotiator. They'll do almost anything to close the deal."
        self.tag = "desperation"

        self.traits = PersonalityTraits(
            assertiveness=0.2,
            dominance=0.1,
            confidence_self=0.3,
            confidence_deal=0.3,
            cooperativeness=1,
            flexibility=0.8,
            patience=0.8,
            empathy=0.6,
            rationality=0.2,
            deceptiveness=0.8,
            risk_tol=0.2,
            persistence=0.3,
            opportunism=1,
            strategic=0.7,
            composure=0.3,
            beta=0.3,
            alpha=1,
            temp=0.8
        )

        self.trait_vector = np.array([
            self.traits.assertiveness, self.traits.dominance, self.traits.confidence_self, self.traits.confidence_deal,
            self.traits.cooperativeness, self.traits.flexibility, self.traits.patience, self.traits.empathy,
            self.traits.rationality, self.traits.deceptiveness, self.traits.risk_tol, self.traits.persistence,
            self.traits.opportunism, self.traits.strategic, self.traits.composure
        ])

class Skeptic():
    def __init__(self):
        self.description = "A skeptical negotiator. Not easily convinced to change their mind on the deal."
        self.tag = "skepticism"

        self.traits = PersonalityTraits(
            assertiveness=0.6,
            dominance=0.6,
            confidence_self=1,
            confidence_deal=0.3,
            cooperativeness=2,
            flexibility=0.6,
            patience=0.7,
            empathy=0.2,
            rationality=0.9,
            deceptiveness=0.3,
            risk_tol=0.4,
            persistence=0.5,
            opportunism=0.5,
            strategic=0.9,
            composure=0.7,
            beta=0.1,
            alpha=0.85,
            temp=0.6
        )

        self.trait_vector = np.array([
            self.traits.assertiveness, self.traits.dominance, self.traits.confidence_self, self.traits.confidence_deal,
            self.traits.cooperativeness, self.traits.flexibility, self.traits.patience, self.traits.empathy,
            self.traits.rationality, self.traits.deceptiveness, self.traits.risk_tol, self.traits.persistence,
            self.traits.opportunism, self.traits.strategic, self.traits.composure
        ])

class Bluffer():
    def __init__(self):
        self.description = "A bluffing negotiator. You will push strongly for a favorable deal even if it's not super reasonable."
        self.tag = "bluffing"

        self.traits = PersonalityTraits(
            assertiveness=1,
            dominance=0.8,
            confidence_self=0.9,
            confidence_deal=0.2,
            cooperativeness=0.3,
            flexibility=0.3,
            patience=0.5,
            empathy=0.4,
            rationality=0.4,
            deceptiveness=0.9,
            risk_tol=0.9,
            persistence=0.8,
            opportunism=1,
            strategic=1,
            composure=0.8,
            beta=0.1,
            alpha=0.5,
            temp=0.7
        )

        self.trait_vector = np.array([
            self.traits.assertiveness, self.traits.dominance, self.traits.confidence_self, self.traits.confidence_deal,
            self.traits.cooperativeness, self.traits.flexibility, self.traits.patience, self.traits.empathy,
            self.traits.rationality, self.traits.deceptiveness, self.traits.risk_tol, self.traits.persistence,
            self.traits.opportunism, self.traits.strategic, self.traits.composure
        ])

class Confrontational():
    def __init__(self):
        self.description = "A confrontational negotiator. They aggressively push for a good deal using whatever means necessary."
        self.tag = "confrontation"

        self.traits = PersonalityTraits(
            assertiveness=0.9,
            dominance=1,
            confidence_self=1,
            confidence_deal=0.9,
            cooperativeness=0.3,
            flexibility=0.2,
            patience=0.6,
            empathy=0.1,
            rationality=0.7,
            deceptiveness=0.6,
            risk_tol=0.8,
            persistence=0.7,
            opportunism=0.6,
            strategic=0.6,
            composure=0.4,
            beta=0,
            alpha=0.75,
            temp=0.6
        )

        self.trait_vector = np.array([
            self.traits.assertiveness, self.traits.dominance, self.traits.confidence_self, self.traits.confidence_deal,
            self.traits.cooperativeness, self.traits.flexibility, self.traits.patience, self.traits.empathy,
            self.traits.rationality, self.traits.deceptiveness, self.traits.risk_tol, self.traits.persistence,
            self.traits.opportunism, self.traits.strategic, self.traits.composure
        ])

class Haggler():
    def __init__(self):
        self.description = "A haggling negotiator. They don't fold easily and want to get as much from the counterparty as they can."
        self.tag = "haggling"

        self.traits = PersonalityTraits(
            assertiveness=0.8,
            dominance=0.7,
            confidence_self=0.9,
            confidence_deal=0.8,
            cooperativeness=0.3,
            flexibility=0.2,
            patience=0.8,
            empathy=0.3,
            rationality=0.4,
            deceptiveness=0.7,
            risk_tol=0.6,
            persistence=1,
            opportunism=0.8,
            strategic=0.7,
            composure=0.8,
            beta=0.05,
            alpha=0.75,
            temp=0.7
        )

        self.trait_vector = np.array([
            self.traits.assertiveness, self.traits.dominance, self.traits.confidence_self, self.traits.confidence_deal,
            self.traits.cooperativeness, self.traits.flexibility, self.traits.patience, self.traits.empathy,
            self.traits.rationality, self.traits.deceptiveness, self.traits.risk_tol, self.traits.persistence,
            self.traits.opportunism, self.traits.strategic, self.traits.composure
        ])

class WeakEmotional():
    def __init__(self):
        self.description = "A weak and emotional negotiator. They just want to reach a deal, and don't like confrontation."
        self.tag = "concession"

        self.traits = PersonalityTraits(
            assertiveness=0.4,
            dominance=0.3,
            confidence_self=0.5,
            confidence_deal=0.6,
            cooperativeness=0.8,
            flexibility=0.8,
            patience=0.7,
            empathy=1,
            rationality=0.1,
            deceptiveness=0.7,
            risk_tol=0.2,
            persistence=0.5,
            opportunism=0.8,
            strategic=0.2,
            composure=0.1,
            beta=0.2,
            alpha=0.8,
            temp=0.8
        )

        self.trait_vector = np.array([
            self.traits.assertiveness, self.traits.dominance, self.traits.confidence_self, self.traits.confidence_deal,
            self.traits.cooperativeness, self.traits.flexibility, self.traits.patience, self.traits.empathy,
            self.traits.rationality, self.traits.deceptiveness, self.traits.risk_tol, self.traits.persistence,
            self.traits.opportunism, self.traits.strategic, self.traits.composure
        ])

class Rationalizer():
    def __init__(self):
        self.description = "A rational negotiator. Prefer to use reason and logic to win the negotiation and reach a good deal."
        self.tag = "rationalization"

        self.traits = PersonalityTraits(
            assertiveness=0.7,
            dominance=0.7,
            confidence_self=0.7,
            confidence_deal=0.8,
            cooperativeness=0.6,
            flexibility=0.5,
            patience=0.8,
            empathy=0.2,
            rationality=1,
            deceptiveness=0.3,
            risk_tol=0.5,
            persistence=0.6,
            opportunism=0.7,
            strategic=1,
            composure=0.9,
            beta=0,
            alpha=0.9,
            temp=0.5
        )

        self.trait_vector = np.array([
            self.traits.assertiveness, self.traits.dominance, self.traits.confidence_self, self.traits.confidence_deal,
            self.traits.cooperativeness, self.traits.flexibility, self.traits.patience, self.traits.empathy,
            self.traits.rationality, self.traits.deceptiveness, self.traits.risk_tol, self.traits.persistence,
            self.traits.opportunism, self.traits.strategic, self.traits.composure
        ])

class StoneWall():
    def __init__(self):
        self.description = "A stonewall negotiator. They don't even want to hear it from the counterparty - their offer doesn't need much changing."
        self.tag = "non-cooperation"

        self.traits = PersonalityTraits(
            assertiveness=1,
            dominance=0.7,
            confidence_self=0.9,
            confidence_deal=1,
            cooperativeness=0.2,
            flexibility=0.1,
            patience=0.7,
            empathy=0.1,
            rationality=0.8,
            deceptiveness=0.2,
            risk_tol=0.7,
            persistence=0.4,
            opportunism=0.2,
            strategic=0.3,
            composure=1,
            beta=0,
            alpha=0.9,
            temp=0.4
        )

        self.trait_vector = np.array([
            self.traits.assertiveness, self.traits.dominance, self.traits.confidence_self, self.traits.confidence_deal,
            self.traits.cooperativeness, self.traits.flexibility, self.traits.patience, self.traits.empathy,
            self.traits.rationality, self.traits.deceptiveness, self.traits.risk_tol, self.traits.persistence,
            self.traits.opportunism, self.traits.strategic, self.traits.composure
        ])

class Opportunist():
    def __init__(self):
        self.description = "An opportunist negotiator. They will jump at a favorable deal as soon as it presents itself."
        self.tag = "opportunism"

        self.traits = PersonalityTraits(
            assertiveness=0.5,
            dominance=0.4,
            confidence_self=0.9,
            confidence_deal=0.6,
            cooperativeness=0.4,
            flexibility=0.7,
            patience=1,
            empathy=0.5,
            rationality=0.7,
            deceptiveness=0.9,
            risk_tol=0.5,
            persistence=0.2,
            opportunism=1,
            strategic=0.9,
            composure=0.9,
            beta=0.05,
            alpha=0.8,
            temp=0.7
        )

        self.trait_vector = np.array([
            self.traits.assertiveness, self.traits.dominance, self.traits.confidence_self, self.traits.confidence_deal,
            self.traits.cooperativeness, self.traits.flexibility, self.traits.patience, self.traits.empathy,
            self.traits.rationality, self.traits.deceptiveness, self.traits.risk_tol, self.traits.persistence,
            self.traits.opportunism, self.traits.strategic, self.traits.composure
        ])

class HighHorse():
    def __init__(self):
        self.description = "A high-and-mighty negotiator. They think the counterparty is being ridiculous for even thinking of trying to undermine them."
        self.tag = "haughtiness"

        self.traits = PersonalityTraits(
            assertiveness=1,
            dominance=0.9,
            confidence_self=1,
            confidence_deal=0.9,
            cooperativeness=0.1,
            flexibility=0.1,
            patience=0.3,
            empathy=0.1,
            rationality=0.2,
            deceptiveness=0.2,
            risk_tol=0.8,
            persistence=0.3,
            opportunism=0.5,
            strategic=0.2,
            composure=0.8,
            beta=0,
            alpha=0.8,
            temp=0.6
        )

        self.trait_vector = np.array([
            self.traits.assertiveness, self.traits.dominance, self.traits.confidence_self, self.traits.confidence_deal,
            self.traits.cooperativeness, self.traits.flexibility, self.traits.patience, self.traits.empathy,
            self.traits.rationality, self.traits.deceptiveness, self.traits.risk_tol, self.traits.persistence,
            self.traits.opportunism, self.traits.strategic, self.traits.composure
        ])

class Stubborn():
    def __init__(self):
        self.description = "A stubborn negotiator. They hate to change the terms of the deal from how they started and need a lot of convincing."
        self.tag = "stubbornness"

        self.traits = PersonalityTraits(
            assertiveness=1,
            dominance=0.6,
            confidence_self=1,
            confidence_deal=1,
            cooperativeness=0.2,
            flexibility=0.1,
            patience=0.8,
            empathy=0.3,
            rationality=0.3,
            deceptiveness=0.2,
            risk_tol=0.6,
            persistence=0.9,
            opportunism=0.5,
            strategic=0.5,
            composure=0.7,
            beta=0,
            alpha=0.85,
            temp=0.5
        )

        self.trait_vector = np.array([
            self.traits.assertiveness, self.traits.dominance, self.traits.confidence_self, self.traits.confidence_deal,
            self.traits.cooperativeness, self.traits.flexibility, self.traits.patience, self.traits.empathy,
            self.traits.rationality, self.traits.deceptiveness, self.traits.risk_tol, self.traits.persistence,
            self.traits.opportunism, self.traits.strategic, self.traits.composure
        ])


# -------------------------- Personality Deck Selector -------------------------------
import random
from typing import Optional

class PersonalityDeck():
    """Manages selection of personalities from the deck."""

    def __init__(self):
        """Initialize the personality deck with all available personalities."""
        # Now each personality class creates its own PersonalityTraits with correct values
        self.personalities = [
            lambda: StraightToPoint(),
            lambda: Desperate(),
            lambda: Skeptic(),
            lambda: Bluffer(),
            lambda: Confrontational(),
            lambda: Haggler(),
            lambda: WeakEmotional(),
            lambda: Rationalizer(),
            lambda: StoneWall(),
            lambda: Opportunist(),
            lambda: HighHorse(),
            lambda: Stubborn()
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
            # When avoid_same is False, select the same personality for both
            selected_personality = self.select_personality()
            # Create a second instance of the same personality type
            personality_class = type(selected_personality)
            buyer = selected_personality
            seller = personality_class()

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