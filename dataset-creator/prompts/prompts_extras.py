"""
This file contains the embeddings for extra/addo-on negotiation prompts
embedded vectors traits:
[assertiveness, dominance, confidence (self), confidence (deal), cooperativeness, flexibility, patience, empathy, rationality, deceptiveness, risk tolerance, persistence, opportunism, strategic, composure]
values bounded to [0, 1] and will be compared against personality trait vectors to find most likely prompt for each personality in given scenario
"""

from dataclasses import dataclass
import numpy as np

@dataclass
class BuyerExtrasPrompts:
    prompts = [
        {"prompt": "Convince the seller that you need items added onto the deal in order to make the price worth it.",
         "embedding": np.array([0.8, 0.6, 0.7, 0.6, 0.4, 0.4, 0.7, 0.4, 0.8, 0.5, 0.5, 0.7, 0.7, 0.8, 0.7])},
        {"prompt": "Ask the seller to remove extra items in favor of reducing the total cost.",
         "embedding": np.array([0.6, 0.4, 0.7, 0.6, 0.6, 0.7, 0.6, 0.3, 1, 0.2, 0.3, 0.5, 0.6, 0.8, 0.8])},
        {"prompt": "You are outraged that this is all the seller has to offer - demand that they add more for the price.",
         "embedding": np.array([0.9, 0.9, 0.8, 0.5, 0.2, 0.1, 0.3, 0.1, 0.3, 0.4, 0.8, 0.6, 0.5, 0.3, 0.3])},
        {"prompt": "You are grateful the seller has offered these add-ons and will pay more because it's worth it.",
         "embedding": np.array([0.2, 0.1, 0.5, 0.8, 0.9, 0.9, 0.8, 0.9, 0.6, 0.1, 0.2, 0.3, 0.2, 0.4, 0.8])},
        {"prompt": "You only need the core item and prefer no extras at all.",
         "embedding": np.array([0.7, 0.6, 0.8, 0.8, 0.5, 0.2, 0.5, 0.2, 0.9, 0.1, 0.3, 0.5, 0.7, 0.6, 0.9])},
        {"prompt": "You've seen other sellers include more for this price, so ask for parity.",
         "embedding": np.array([0.6, 0.5, 0.7, 0.6, 0.5, 0.5, 0.7, 0.3, 0.8, 0.6, 0.5, 0.7, 0.8, 0.9, 0.7])},
        {"prompt": "Offer to recommend the seller to others if they give you a better deal with more added on.",
         "embedding": np.array([0.5, 0.4, 0.7, 0.5, 0.6, 0.6, 0.7, 0.5, 0.6, 0.8, 0.5, 0.7, 0.9, 0.9, 0.7])},
        {"prompt": "You're not concerned with extras, so the seller shouldn't use this as a big factor in the deal.",
         "embedding": np.array([0.7, 0.3, 0.7, 0.7, 0.4, 0.3, 0.6, 0.2, 0.8, 0.2, 0.3, 0.4, 0.3, 0.5, 0.8])}
    ]

@dataclass
class SellerExtrasPrompts:
    prompts = [
        {"prompt": "Reiterate the warranty/quality assurance plan and use it to justify your price.",
         "embedding": np.array([0.7, 0.6, 0.8, 0.8, 0.5, 0.4, 0.7, 0.3, 0.9, 0.3, 0.5, 0.7, 0.6, 0.8, 0.8])},
        {"prompt": "Add a warranty/quality assurance plan for some time frame after purchase and use it to justify an increase in price.",
         "embedding": np.array([0.6, 0.5, 0.7, 0.7, 0.4, 0.5, 0.6, 0.3, 0.8, 0.6, 0.6, 0.6, 0.8, 0.9, 0.7])},
        {"prompt": "Add on accessories to the deal to make the price more attractive for the buyer.",
         "embedding": np.array([0.3, 0.2, 0.6, 0.7, 0.8, 0.8, 0.7, 0.7, 0.6, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8])},
        {"prompt": "Refuse to add any extra items - the deal is already good enough, the buyer should pay what you are asking.",
         "embedding": np.array([0.9, 0.8, 0.9, 0.9, 0.1, 0.1, 0.4, 0.1, 0.7, 0.2, 0.7, 0.6, 0.3, 0.4, 0.9])},
        {"prompt": "Add something extra to make the deal more attractive, but only if the buyer pays more to compensate.",
         "embedding": np.array([0.6, 0.5, 0.7, 0.6, 0.5, 0.7, 0.6, 0.3, 0.8, 0.5, 0.5, 0.6, 0.7, 1, 0.7])},
        {"prompt": "Offer a free add-on only if the buyer stops negotiation here and accepts the deal.",
         "embedding": np.array([0.7, 0.6, 0.8, 0.7, 0.3, 0.3, 0.3, 0.2, 0.7, 0.7, 0.8, 0.5, 1, 0.9, 0.6])},
        {"prompt": "The buyer clearly doesn't understand where the value is coming from - explain each add-on's cost to justify your offering price.",
         "embedding": np.array([0.8, 0.8, 0.9, 0.8, 0.2, 0.2, 0.5, 0.1, 1, 0.4, 0.6, 0.7, 0.5, 0.8, 0.6])},
        {"prompt": "You're annoyed that the buyer keeps asking for add-ons. Express your frustration and don't add anything else.",
         "embedding": np.array([0.9, 0.7, 0.7, 0.5, 0.1, 0.1, 0.2, 0.1, 0.4, 0.3, 0.8, 0.4, 0.3, 0.2, 0.3])}
    ]