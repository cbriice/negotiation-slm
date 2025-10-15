"""
This file contains the embeddings for ultimatum prompts
embedded vectors traits:
[assertiveness, dominance, confidence (self), confidence (deal), cooperativeness, flexibility, patience, empathy, rationality, deceptiveness, risk tolerance, persistence, opportunism, strategic, composure]
values bounded to [0, 1] and will be compared against personality trait vectors to find most likely prompt for each personality in given scenario

NOTE: ultimatum prompts are only enabled for personality parameter `eta` > 0.5. this means that all of these prompts are associated with higher assertiveness, dominance and risk tolerance
"""

from dataclasses import dataclass
import numpy as np

@dataclass
class BuyerUltimatumPrompts:
    # also requires price injection like prompts_price_adjust

    ult_prompts = [
        {"prompt": "You're walking away unless they take your offer of {price}.",
         "embedding": np.array([0.9, 0.8, 0.9, 0.6, 0.1, 0.1, 0.3, 0.1, 0.6, 0.3, 0.9, 0.6, 0.5, 0.7, 0.8])},
        {"prompt": "The only way you'd possibly consider the deal is at {price}, otherwise you're done.",
         "embedding": np.array([1, 0.9, 1, 0.5, 0.1, 0.1, 0.2, 0.1, 0.5, 0.3, 0.9, 0.7, 0.4, 0.6, 0.7])},
        {"prompt": "You're desperate, but you'll look somewhere else if you can't get it for {price}.",
         "embedding": np.array([0.7, 0.6, 0.5, 0.3, 0.2, 0.3, 0.3, 0.2, 0.4, 0.5, 0.8, 0.5, 0.9, 0.5, 0.5])},
        {"prompt": "After considering your options, you won't pursue a deal anymore unless they meet your price of {price}.",
         "embedding": np.array([0.8, 0.7, 0.8, 0.5, 0.2, 0.1, 0.4, 0.1, 1, 0.2, 0.8, 0.5, 0.4, 0.8, 0.8])},
        {"prompt": "You're not stressed, you can find this item elsewhere. Your final offer is {price}.",
         "embedding": np.array([0.8, 0.7, 1, 0.9, 0.1, 0.1, 0.3, 0.1, 0.7, 0.2, 0.9, 0.4, 0.3, 0.6, 1])},
        {"prompt": "You don't have the time to keep negotiating, your final offer is {price}.",
         "embedding": np.array([0.9, 0.8, 0.8, 0.5, 0.1, 0.1, 0.1, 0.1, 0.5, 0.3, 0.8, 0.7, 0.6, 0.5, 0.6])}
    ]


@dataclass
class SellerUltimatumPrompts:
    ult_prompts = [
        {"prompt": "You'll no longer consider a deal unless they offer {price}.",
         "embedding": np.array([0.8, 0.7, 0.9, 0.6, 0.1, 0.1, 0.4, 0.1, 0.7, 0.2, 0.8, 0.5, 0.4, 0.7, 0.9])},
        {"prompt": "You're done listening to them try to scam you - your final offer is {price}.",
         "embedding": np.array([1, 1, 0.9, 0.4, 0.1, 0.1, 0.2, 0.1, 0.3, 0.5, 0.9, 0.6, 0.5, 0.4, 0.4])},
        {"prompt": "Demand they stop wasting your time and pay {price}, otherwise you're done.",
         "embedding": np.array([1, 0.9, 0.8, 0.5, 0.1, 0.1, 0.1, 0.1, 0.5, 0.3, 0.9, 0.7, 0.4, 0.5, 0.5])},
        {"prompt": "You really need this money, and so you can't go lower than {price}. If they don't accept this, you're done.",
         "embedding": np.array([0.6, 0.5, 0.4, 0.3, 0.3, 0.2, 0.3, 0.3, 0.4, 0.4, 0.8, 0.6, 0.9, 0.5, 0.4])},
        {"prompt": "You think the buyer is being ridiculous. Offer them {price} and don't let them try to talk you any lower.",
         "embedding": np.array([0.9, 0.9, 1, 0.8, 0.1, 0.1, 0.3, 0.1, 0.7, 0.3, 0.8, 0.5, 0.3, 0.6, 0.8])},
        {"prompt": "There are others waiting and they'll give you {price}, so this is the buyer's last chance.",
         "embedding": np.array([0.8, 0.7, 1, 0.9, 0.1, 0.1, 0.3, 0.1, 0.6, 0.6, 0.9, 0.4, 0.8, 0.9, 0.9])},
        {"prompt": "This negotiation has been annoying and exhausting. You're not willing to go lower than {price}.",
         "embedding": np.array([0.8, 0.6, 0.6, 0.4, 0.1, 0.1, 0.2, 0.1, 0.4, 0.2, 0.8, 0.8, 0.3, 0.3, 0.3])}
    ]