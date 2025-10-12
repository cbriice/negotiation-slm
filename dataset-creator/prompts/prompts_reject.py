""" 
This file contains the embeddings for reject prompts
embedded vectors traits: 
[assertiveness, dominance, confidence (self), confidence (deal), cooperativeness, flexibility, patience, empathy, rationality, deceptiveness, risk tolerance, persistence, opportunism, strategic, composure]
values bounded to [0, 1] and will be compared against personality trait vectors to find most likely prompt for each personality in given scenario
"""

from dataclasses import dataclass
import numpy as np

@dataclass
class BuyerRejectPrompts:
    rej_1: str = "You're no longer interested in negotiating this deal, but there's no hard feelings."
    rej_2: str = "The seller is stupid for not taking your offer, and you're no longer interested in negotiation."
    rej_3: str = "You're very sorry, but you can't continue negotiating the deal."
    rej_4: str = "You're done negotiating for now, but you could come back if something changed."
    rej_5: str = "You're done with this negotiation, you're gonna go look elsewhere."
    # embeddings
    embedded_rej_1 = np.array([0.5, 0.4, 0.7, 0.5, 0.6, 0.5, 0.7, 0.6, 0.7, 0.2, 0.4, 0.3, 0.4, 0.5, 0.9])
    embedded_rej_2 = np.array([0.8, 0.9, 0.8, 0.3, 0.1, 0.1, 0.2, 0.1, 0.3, 0.4, 0.7, 0.5, 0.3, 0.3, 0.2])
    embedded_rej_3 = np.array([0.2, 0.1, 0.3, 0.4, 0.8, 0.7, 0.6, 0.9, 0.5, 0.2, 0.2, 0.2, 0.3, 0.4, 0.6])
    embedded_rej_4 = np.array([0.5, 0.4, 0.6, 0.6, 0.5, 0.7, 0.6, 0.4, 0.6, 0.3, 0.5, 0.4, 0.8, 0.8, 0.7])
    embedded_rej_5 = np.array([0.9, 0.7, 0.8, 0.7, 0.3, 0.2, 0.4, 0.2, 0.7, 0.2, 0.6, 0.6, 0.5, 0.6, 1])

@dataclass
class SellerRejectPrompts:
    rej_1: str = "Politely say you're no longer interested in negotiating the deal."
    rej_2: str = "You think they're being ridiculous, tell them you're done wasting your time with them."
    rej_3: str = "You're sorry you can't help them out, but you can't keep negotiating."
    rej_4: str = "You won't sell them anything right now, but maybe later they could get a deal."
    rej_5: str = "You're done negotiating, this deal won't work."
    #embeddings
    embedded_rej_1 = np.array([0.6, 0.4, 0.6, 0.5, 0.7, 0.5, 0.7, 0.7, 0.7, 0.1, 0.3, 0.3, 0.4, 0.5, 1])
    embedded_rej_2 = np.array([0.9, 0.8, 0.9, 0.3, 0.1, 0.1, 0.1, 0.1, 0.4, 0.3, 0.7, 0.6, 0.4, 0.3, 0.3])
    embedded_rej_3 = np.array([0.2, 0.1, 0.4, 0.4, 0.9, 0.7, 0.6, 0.9, 0.5, 0.1, 0.2, 0.2, 0.3, 0.3, 0.7])
    embedded_rej_4 = np.array([0.5, 0.5, 0.7, 0.6, 0.4, 0.6, 0.7, 0.3, 0.6, 0.4, 0.5, 0.4, 0.9, 0.8, 0.7])
    embedded_rej_5 = np.array([0.7, 0.6, 0.8, 0.8, 0.2, 0.1, 0.5, 0.1, 0.8, 0.1, 0.7, 0.5, 0.3, 0.5, 0.9])

