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
    prompts = [
        {"prompt": "You're no longer interested in negotiating this deal, but there's no hard feelings.",
         "embedding": np.array([0.5, 0.4, 0.7, 0.5, 0.6, 0.5, 0.7, 0.6, 0.7, 0.2, 0.4, 0.3, 0.4, 0.5, 0.9])},
        {"prompt": "They're stupid for not taking your offer, and you're no longer interested in negotiation.",
         "embedding": np.array([0.8, 0.9, 0.8, 0.3, 0.1, 0.1, 0.2, 0.1, 0.3, 0.4, 0.7, 0.5, 0.3, 0.3, 0.2])},
        {"prompt": "You're very sorry, but you can't continue negotiating the deal.",
         "embedding": np.array([0.2, 0.1, 0.3, 0.4, 0.8, 0.7, 0.6, 0.9, 0.5, 0.2, 0.2, 0.2, 0.3, 0.4, 0.6])},
        {"prompt": "You're done negotiating for now, but you could come back if something changed.",
         "embedding": np.array([0.5, 0.4, 0.6, 0.6, 0.5, 0.7, 0.6, 0.4, 0.6, 0.3, 0.5, 0.4, 0.8, 0.8, 0.7])},
        {"prompt": "You're done with this negotiation, you're gonna go look elsewhere.",
         "embedding": np.array([0.9, 0.7, 0.8, 0.7, 0.3, 0.2, 0.4, 0.2, 0.7, 0.2, 0.6, 0.6, 0.5, 0.6, 1])}
    ]

@dataclass
class SellerRejectPrompts:
    prompts = [
        {"prompt": "Politely say you're no longer interested in negotiating the deal.",
         "embedding": np.array([0.6, 0.4, 0.6, 0.5, 0.7, 0.5, 0.7, 0.7, 0.7, 0.1, 0.3, 0.3, 0.4, 0.5, 1])},
        {"prompt": "You think they're being ridiculous, tell them you're done wasting your time with them.",
         "embedding": np.array([0.9, 0.8, 0.9, 0.3, 0.1, 0.1, 0.1, 0.1, 0.4, 0.3, 0.7, 0.6, 0.4, 0.3, 0.3])},
        {"prompt": "You're sorry you can't help them out, but you can't keep negotiating.",
         "embedding": np.array([0.2, 0.1, 0.4, 0.4, 0.9, 0.7, 0.6, 0.9, 0.5, 0.1, 0.2, 0.2, 0.3, 0.3, 0.7])},
        {"prompt": "You won't sell them anything right now, but maybe later they could get a deal.",
         "embedding": np.array([0.5, 0.5, 0.7, 0.6, 0.4, 0.6, 0.7, 0.3, 0.6, 0.4, 0.5, 0.4, 0.9, 0.8, 0.7])},
        {"prompt": "You're done negotiating, this deal won't work.",
         "embedding": np.array([0.7, 0.6, 0.8, 0.8, 0.2, 0.1, 0.5, 0.1, 0.8, 0.1, 0.7, 0.5, 0.3, 0.5, 0.9])}
    ]