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
    u: float = field(init=False)        # conditional accept paramter (if u > 1, enable conditional accept)
    eta: float = field(init=False)      # ultimatum parameter (if eta > 0.5, enable ultimatums)

    def __post_init__(self):
        self.zeta = self.patience
        self.c = sqrt(self.cooperativeness * self.flexibility)
        self.u = self.c + self.opportunism
        self.eta = sqrt(self.assertiveness * self.dominance * self.risk_tol)

# -------------------------- Personalities -------------------------------
class StraightToPoint():
    def __init__(self, traits: PersonalityTraits):
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
        
        traits.beta = 0.4
        traits.alpha = 1
        traits.temp = 0.8

class Skeptic():
    def __init__(self, traits: PersonalityTraits):
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
        traits.risk_tol = 0.2
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