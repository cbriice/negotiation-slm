"""
This file contains the embeddings for deal close prompts
embedded vectors traits:
[assertiveness, dominance, confidence (self), confidence (deal), cooperativeness, flexibility, patience, empathy, rationality, deceptiveness, risk tolerance, persistence, opportunism, strategic, composure]
values bounded to [0, 1] and will be compared against personality trait vectors to find most likely prompt for each personality in given scenario
"""

from dataclasses import dataclass
import numpy as np

@dataclass
class BuyerClose:
    # prompts from this class require price variable injections same as prompts in prompts/prompts_price_adjust.py

    bc_prompts = [
        {"prompt": "You'll take the deal if they accept your price of {price}.",
         "embedding": np.array([0.8, 0.6, 0.7, 0.7, 0.6, 0.5, 0.7, 0.4, 0.7, 0.3, 0.5, 0.6, 0.6, 0.7, 0.7])},
        {"prompt": "The only way you'll possibly take the deal is if they accept your price of {price}.",
         "embedding": np.array([0.8, 0.7, 0.8, 0.5, 0.3, 0.2, 0.5, 0.2, 0.6, 0.4, 0.7, 0.7, 0.7, 0.6, 0.6])},
        {"prompt": "You'd really appreciate them taking your offer of {price}, and you'll happily take the deal if they do.",
         "embedding": np.array([0.3, 0.2, 0.5, 0.7, 0.9, 0.8, 0.8, 0.8, 0.6, 0.1, 0.3, 0.4, 0.5, 0.5, 0.8])},
        {"prompt": "You don't think their item is worth any more than your offer of {price}, and that you'll take it for that price.",
         "embedding": np.array([0.7, 0.6, 0.8, 0.4, 0.4, 0.3, 0.6, 0.2, 0.9, 0.3, 0.8, 0.6, 0.6, 0.7, 0.3])},
        {"prompt": "You know this would work out if the seller would just meet you at {price}.",
         "embedding": np.array([0.9, 0.4, 0.6, 0.6, 0.7, 0.6, 0.7, 0.5, 0.7, 0.2, 0.4, 0.5, 0.7, 0.6, 0.5])},
         {"prompt": "That price works for you — you'll take the deal at {price}.",
         "embedding": np.array([0.5, 0.4, 0.6, 0.6, 0.7, 0.6, 0.7, 0.6, 0.7, 0.2, 0.5, 0.5, 0.5, 0.4, 0.8])},
        {"prompt": "You’re satisfied with the offer and ready to agree at {price}.",
         "embedding": np.array([0.4, 0.3, 0.6, 0.7, 0.8, 0.7, 0.7, 0.7, 0.7, 0.2, 0.4, 0.5, 0.4, 0.4, 0.9])},
        {"prompt": "You think it’s a fair deal and you’ll accept it for {price}.",
         "embedding": np.array([0.5, 0.4, 0.7, 0.6, 0.8, 0.7, 0.6, 0.6, 0.8, 0.2, 0.4, 0.5, 0.6, 0.5, 0.8])}
    ]

@dataclass
class SellerClose:
    # also require price injections

    sc_prompts = [
        {"prompt": "You'll give it to them right now for {price}.",
         "embedding": np.array([0.8, 0.6, 0.7, 0.8, 0.6, 0.6, 0.4, 0.4, 0.6, 0.3, 0.6, 0.5, 0.8, 0.2, 0.9])},
        {"prompt": "The only way you'd possibly sell to them is if they agree to your price of {price}.",
         "embedding": np.array([0.9, 0.8, 0.9, 0.5, 0.2, 0.1, 0.4, 0.1, 0.6, 0.3, 0.8, 0.7, 0.6, 0.5, 0.6])},
        {"prompt": "You really need the money, and you'll sell it to them if they just buy it at {price}.",
         "embedding": np.array([0.3, 0.2, 0.4, 0.3, 0.8, 0.9, 0.7, 0.6, 0.5, 0.4, 0.3, 0.4, 0.9, 0.7, 0.5])},
        {"prompt": "You think they're insane for offering their price, but you'll sell it to them for {price}.",
         "embedding": np.array([0.8, 0.7, 1, 0.9, 0.3, 0.3, 0.5, 0.2, 0.8, 0.4, 0.6, 0.6, 0.7, 0.6, 0.3])},
        {"prompt": "You don't see why the buyer is being difficult when they could just take it for {price}.",
         "embedding": np.array([0.6, 0.5, 0.7, 0.5, 0.5, 0.4, 0.3, 0.3, 0.7, 0.3, 0.7, 0.7, 0.6, 0.4, 0.6])},
         {"prompt": "That price is fine — you’ll sell it for {price}.",
         "embedding": np.array([0.5, 0.4, 0.6, 0.6, 0.7, 0.6, 0.6, 0.5, 0.7, 0.2, 0.4, 0.4, 0.5, 0.4, 0.8])},
        {"prompt": "You’re happy with that number and agree to sell at {price}.",
         "embedding": np.array([0.4, 0.3, 0.6, 0.7, 0.8, 0.7, 0.6, 0.6, 0.7, 0.2, 0.3, 0.4, 0.4, 0.4, 0.9])},
        {"prompt": "That seems fair — you’ll close the deal for {price}.",
         "embedding": np.array([0.5, 0.4, 0.7, 0.6, 0.8, 0.7, 0.6, 0.6, 0.8, 0.2, 0.4, 0.5, 0.5, 0.5, 0.8])}
    ]