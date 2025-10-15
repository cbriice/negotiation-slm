"""
This file contains the embeddings for deal accept prompts
embedded vectors traits:
[assertiveness, dominance, confidence (self), confidence (deal), cooperativeness, flexibility, patience, empathy, rationality, deceptiveness, risk tolerance, persistence, opportunism, strategic, composure]
values bounded to [0, 1] and will be compared against personality trait vectors to find most likely prompt for each personality in given scenario
"""

from dataclasses import dataclass
import numpy as np

@dataclass
class BuyerDealAccept:
    # favorable accept
    fa_prompts = [
        {"prompt": "You're pleased how smooth the deal went, accept their offer amicably.",
         "embedding": np.array([0.7, 0.6, 0.7, 0.8, 0.9, 0.8, 0.8, 0.7, 0.7, 0.1, 0.3, 0.2, 0.7, 0.3, 0.9])},
        {"prompt": "This is great, accept quickly before the seller changes their mind.",
         "embedding": np.array([0.2, 0.1, 0.6, 0.9, 0.7, 0.7, 0.3, 0.4, 0.5, 0.3, 0.6, 0.2, 1, 0.7, 0.7])},
        {"prompt": "You're grateful for this deal, and you will happily accept.",
         "embedding": np.array([0.3, 0.2, 0.5, 0.8, 1, 0.9, 0.8, 0.9, 0.6, 0.1, 0.2, 0.7, 0.3, 0.4, 0.8])},
        {"prompt": "The seller finally stopped being difficult, now you'll take the deal.",
         "embedding": np.array([1, 0.9, 0.7, 0.7, 0.6, 0.6, 0.5, 0.3, 0.7, 0.2, 0.4, 0.3, 0.6, 0.6, 0.6])},
        {"prompt": "Confirm that you'll accept the deal.",
         "embedding": np.array([0.8, 0.3, 0.7, 0.7, 0.7, 0.7, 0.7, 0.5, 0.8, 0.1, 0.4, 0.4, 0.5, 0.5, 0.8])}
    ]

    # unfavorable accept
    ufa_prompts = [
        {"prompt": "This is a stretch for you financially and you were hoping for cheaper, but you'll accept.",
         "embedding": np.array([0.2, 0.1, 0.4, 0.3, 0.8, 0.9, 0.5, 0.5, 0.6, 0.2, 0.3, 0.3, 0.5, 0.4, 0.6])},
        {"prompt": "Accept the deal begrudgingly - this isn't ideal, but it's the best you can get.",
         "embedding": np.array([0.4, 0.3, 0.5, 0.2, 0.6, 0.7, 0.4, 0.2, 0.7, 0.1, 0.4, 0.2, 0.7, 0.5, 0.4])},
        {"prompt": "You don't want to waste more of your time negotiating, so you'll take this deal even though it's not ideal.",
         "embedding": np.array([0.3, 0.2, 0.6, 0.3, 0.7, 0.6, 0.1, 0.3, 0.8, 0.2, 0.5, 0.2, 1, 0.3, 0.7])},
        {"prompt": "You need the item more than you need the money, so this will work.",
         "embedding": np.array([0.3, 0.2, 0.5, 0.4, 0.8, 0.8, 0.6, 0.4, 0.9, 0.1, 0.6, 0.4, 0.6, 0.8, 0.8])}
    ]

@dataclass
class SellerDealAccept:
    # favorable accept
    fa_prompts = [
        {"prompt": "You're pleased how smooth negotiation went, accept their offer amicably.",
         "embedding": np.array([0.8, 0.6, 0.7, 0.8, 0.9, 0.8, 0.8, 0.7, 0.7, 0.1, 0.3, 0.3, 0.6, 0.5, 0.9])},
        {"prompt": "The buyer is making a great offer for you - accept before they change their mind.",
         "embedding": np.array([0.7, 0.4, 0.7, 0.9, 0.7, 0.7, 0.3, 0.4, 0.6, 0.3, 0.7, 0.2, 1, 0.8, 0.7])},
        {"prompt": "The buyer finally sees reason and is making a good offer that you'll accept.",
         "embedding": np.array([0.9, 0.8, 0.8, 0.8, 0.6, 0.6, 0.6, 0.3, 0.7, 0.2, 0.4, 0.4, 0.6, 0.6, 0.7])},
        {"prompt": "Confirm that you'll accept this offer.",
         "embedding": np.array([0.5, 0.4, 0.7, 0.7, 0.7, 0.7, 0.7, 0.5, 0.8, 0.1, 0.4, 0.4, 0.5, 0.6, 0.8])}
    ]

    # unfavorable accept
    ufa_prompts = [
        {"prompt": "You need the money soon and are sick of negotiating, so this deal is good enough to accept.",
         "embedding": np.array([0.8, 0.2, 0.4, 0.3, 0.7, 0.8, 0.1, 0.4, 0.6, 0.2, 0.2, 0.2, 0.8, 0.5, 0.5])},
        {"prompt": "Accept their offer begrudgingly - this isn't ideal, but it's the best you'll get.",
         "embedding": np.array([0.4, 0.3, 0.5, 0.2, 0.6, 0.7, 0.4, 0.2, 0.7, 0.1, 0.4, 0.3, 0.6, 0.5, 0.4])},
        {"prompt": "You're in a difficult spot financially but this will have to do.",
         "embedding": np.array([0.2, 0.1, 0.5, 0.3, 0.8, 0.9, 0.6, 0.6, 0.5, 0.2, 0.3, 0.3, 0.6, 0.4, 0.6])},
        {"prompt": "This is good enough, you'll accept.",
         "embedding": np.array([0.4, 0.3, 0.7, 0.5, 0.6, 0.6, 0.5, 0.4, 0.6, 0.1, 0.4, 0.4, 0.5, 0.6, 0.9])}
    ]


@dataclass
class BuyerConditionalAccept:
    # prompts from this class require price variable injections same as prompts in prompts/prompts_price_adjust.py

    ca_prompts = [
        {"prompt": "You'll take the deal if they accept your price of {price}.",
         "embedding": np.array([0.8, 0.6, 0.7, 0.7, 0.6, 0.5, 0.7, 0.4, 0.7, 0.3, 0.5, 0.6, 0.6, 0.7, 0.7])},
        {"prompt": "The only way you'll possibly take the deal is if they accept your price of {price}.",
         "embedding": np.array([0.8, 0.7, 0.8, 0.5, 0.3, 0.2, 0.5, 0.2, 0.6, 0.4, 0.7, 0.7, 0.7, 0.6, 0.6])},
        {"prompt": "You'd really appreciate them taking your offer of {price}, and you'll happily take the deal if they do.",
         "embedding": np.array([0.3, 0.2, 0.5, 0.7, 0.9, 0.8, 0.8, 0.8, 0.6, 0.1, 0.3, 0.4, 0.5, 0.5, 0.8])},
        {"prompt": "You don't think their item is worth any more than your offer of {price}, and that you'll take it for that price.",
         "embedding": np.array([0.7, 0.6, 0.8, 0.4, 0.4, 0.3, 0.6, 0.2, 0.9, 0.3, 0.8, 0.6, 0.6, 0.7, 0.3])},
        {"prompt": "You know this would work out if the seller would just meet you at {price}.",
         "embedding": np.array([0.9, 0.4, 0.6, 0.6, 0.7, 0.6, 0.7, 0.5, 0.7, 0.2, 0.4, 0.5, 0.7, 0.6, 0.5])}
    ]

@dataclass
class SellerConditionalAccept:
    # also require price injections

    ca_prompts = [
        {"prompt": "You'll give it to them right now for {price}.",
         "embedding": np.array([0.8, 0.6, 0.7, 0.8, 0.6, 0.6, 0.4, 0.4, 0.6, 0.3, 0.6, 0.5, 0.8, 0.2, 0.9])},
        {"prompt": "The only way you'd possibly sell to them is if they agree to your price of {price}.",
         "embedding": np.array([0.9, 0.8, 0.9, 0.5, 0.2, 0.1, 0.4, 0.1, 0.6, 0.3, 0.8, 0.7, 0.6, 0.5, 0.6])},
        {"prompt": "You really need the money, and you'll sell it to them if they just buy it at {price}.",
         "embedding": np.array([0.3, 0.2, 0.4, 0.3, 0.8, 0.9, 0.7, 0.6, 0.5, 0.4, 0.3, 0.4, 0.9, 0.7, 0.5])},
        {"prompt": "You think they're insane for offering their price, but you'll sell it to them for {price}.",
         "embedding": np.array([0.8, 0.7, 1, 0.9, 0.3, 0.3, 0.5, 0.2, 0.8, 0.4, 0.6, 0.6, 0.7, 0.6, 0.3])},
        {"prompt": "You don't see why the buyer is being difficult when they could just take it for {price}.",
         "embedding": np.array([0.6, 0.5, 0.7, 0.5, 0.5, 0.4, 0.3, 0.3, 0.7, 0.3, 0.7, 0.7, 0.6, 0.4, 0.6])}
    ]