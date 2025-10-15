"""
This file contains the embeddings for clarify prompts
embedded vectors traits:
[assertiveness, dominance, confidence (self), confidence (deal), cooperativeness, flexibility, patience, empathy, rationality, deceptiveness, risk tolerance, persistence, opportunism, strategic, composure]
values bounded to [0, 1] and will be compared against personality trait vectors to find most likely prompt for each personality in given scenario

to avoid writing the same thing 800 times, inject shit to ask about into the prompt based on other negotiation state data
initial idea - categories (hidden fees, etc) are selected randomly but certain categories don't get replaced so that shit isn't repeated. if it turns out this leads to illogical conversation flow sometimes, that can just be filtered out in post for now unless it becomes a problem

NOTE: prompts in this file are not organized by personality theme but they should map to personality themes based on tone and sentiment.
"""

from dataclasses import dataclass
import numpy as np

@dataclass
class BuyerClarifyPrompts:
    # hidden conditions
    hc_prompts = [
        {"prompt": "You want to know if there are any hidden fees or charges not mentioned before.",
         "embedding": np.array([0.5, 0.4, 0.6, 0.5, 0.6, 0.5, 0.7, 0.4, 0.8, 0.3, 0.4, 0.5, 0.6, 0.7, 0.7])},
        {"prompt": "Inquire about any problems with the condition that the seller's hiding from you.",
         "embedding": np.array([0.7, 0.6, 0.7, 0.3, 0.3, 0.3, 0.5, 0.2, 0.7, 0.6, 0.5, 0.6, 0.7, 0.8, 0.6])},
        {"prompt": "You think the seller is hiding something. Demand to see proof of the condition.",
         "embedding": np.array([0.9, 0.8, 0.8, 0.2, 0.2, 0.1, 0.4, 0.1, 0.6, 0.7, 0.7, 0.6, 0.6, 0.7, 0.5])},
        {"prompt": "You're interested in details about the item's {characteristic}, but mention not wanting hidden fees.",
         "embedding": np.array([0.4, 0.3, 0.6, 0.6, 0.5, 0.6, 0.6, 0.5, 0.7, 0.8, 0.4, 0.4, 0.5, 0.6, 0.7])}
    ]

    # context
    con_prompts = [
        {"prompt": "You want to know if there's any incentive outside the current offer for the price to be that high.",
         "embedding": np.array([0.6, 0.5, 0.6, 0.5, 0.4, 0.4, 0.6, 0.3, 0.9, 0.4, 0.5, 0.5, 0.7, 0.8, 0.7])},
        {"prompt": "You want to know if outside circumstances would impact their ability to negotiate price more.",
         "embedding": np.array([0.5, 0.4, 0.6, 0.6, 0.4, 0.5, 0.7, 0.8, 0.7, 0.2, 0.4, 0.5, 0.8, 0.9, 0.7])},
        {"prompt": "You need to know more about {characteristic}.",
         "embedding": np.array([0.8, 0.6, 0.7, 0.5, 0.7, 0.6, 0.7, 0.6, 0.6, 0.5, 0.7, 0.4, 0.5, 0.5, 0.6])},
        {"prompt": "Respond that if it wasn't too big of a deal, you'd like more detail on {characteristic}.",
         "embedding": np.array([0.2, 0.1, 0.4, 0.5, 0.8, 0.8, 0.8, 0.7, 0.6, 0.2, 0.3, 0.3, 0.4, 0.4, 0.8])},
        {"prompt": "You want to know the seller's explanation for why this offer is worth it.",
         "embedding": np.array([0.7, 0.5, 0.8, 0.4, 0.5, 0.4, 0.6, 0.4, 0.9, 0.3, 0.4, 0.5, 0.6, 0.7, 0.7])}
    ]

    # timelines
    tim_prompts = [
        {"prompt": "You want to know how long the seller will keep the listing up.",
         "embedding": np.array([0.6, 0.3, 0.5, 0.6, 0.5, 0.6, 0.8, 0.4, 0.7, 0.3, 0.4, 0.5, 0.7, 0.8, 0.8])},
        {"prompt": "You want to know how many others are trying to buy this right now.",
         "embedding": np.array([0.7, 0.5, 0.7, 0.4, 0.3, 0.4, 0.5, 0.2, 0.6, 0.5, 0.6, 0.6, 0.8, 0.7, 0.5])},
        {"prompt": "Let them know that you'd appreciate them giving you more time to close the deal.",
         "embedding": np.array([0.2, 0.1, 0.4, 0.5, 0.8, 0.7, 0.9, 0.6, 0.5, 0.1, 0.3, 0.3, 0.4, 0.3, 0.7])},
        {"prompt": "You want clarification about delivery times.",
         "embedding": np.array([0.3, 0.2, 0.6, 0.7, 0.6, 0.5, 0.7, 0.5, 0.8, 0.2, 0.5, 0.4, 0.6, 0.6, 0.9])}
    ]

    # location
    loc_prompts = [
        {"prompt": "You need to know if the seller would be open to delivering to your area.",
         "embedding": np.array([0.8, 0.6, 0.5, 0.7, 0.7, 0.8, 0.6, 0.5, 0.6, 0.2, 0.4, 0.4, 0.5, 0.5, 0.8])},
        {"prompt": "You want to know if the seller has any location requirements for this listing.",
         "embedding": np.array([0.6, 0.4, 0.7, 0.5, 0.5, 0.6, 0.7, 0.4, 0.8, 0.3, 0.5, 0.5, 0.6, 0.7, 0.9])},
        {"prompt": "You're curious what location the seller is currently in.",
         "embedding": np.array([0.2, 0.1, 0.4, 0.6, 0.6, 0.5, 0.8, 0.7, 0.5, 0.4, 0.3, 0.3, 0.4, 0.4, 0.7])},
        {"prompt": "You're wondering how far away the seller is from you, and if you could do pickup.",
         "embedding": np.array([0.4, 0.3, 0.6, 0.8, 0.8, 0.7, 0.9, 0.6, 0.7, 0.1, 0.5, 0.6, 0.7, 0.6, 0.8])}
    ]

@dataclass
class SellerClarifyPrompts:
    # hidden conditions
    hc_prompts = [
        {"prompt": "You're wondering if there's anything the buyer hasn't told you that would prevent them from paying full price.",
         "embedding": np.array([0.5, 0.4, 0.6, 0.5, 0.4, 0.5, 0.7, 0.3, 0.8, 0.4, 0.5, 0.5, 0.6, 0.7, 0.7])},
        {"prompt": "You need to make sure the buyer isn't lying about their ability to pay.",
         "embedding": np.array([0.7, 0.6, 0.8, 0.3, 0.2, 0.3, 0.4, 0.2, 0.7, 0.7, 0.6, 0.7, 0.5, 0.6, 0.5])},
        {"prompt": "Make sure the buyer isn't expecting you to add guarantees or incentives that you haven't already mentioned.",
         "embedding": np.array([0.6, 0.8, 0.7, 0.6, 0.3, 0.4, 0.6, 0.4, 0.8, 0.5, 0.5, 0.6, 0.7, 0.8, 0.8])},
        {"prompt": "Confirm what payment method they're using.",
         "embedding": np.array([0.3, 0.5, 0.5, 0.7, 0.6, 0.7, 0.8, 0.5, 0.9, 0.1, 0.3, 0.4, 0.4, 0.5, 0.9])}
    ]

    # context
    con_prompts = [
        {"prompt": "You need to know why the buyer believes their price is more fair.",
         "embedding": np.array([0.6, 0.5, 0.7, 0.4, 0.4, 0.3, 0.6, 0.3, 0.9, 0.3, 0.5, 0.6, 0.7, 0.8, 0.8])},
        {"prompt": "You're wondering what aspect of the deal matters most to the buyer.",
         "embedding": np.array([0.4, 0.3, 0.5, 0.6, 0.5, 0.6, 0.8, 0.7, 0.7, 0.2, 0.4, 0.4, 0.5, 0.6, 0.7])},
        {"prompt": "You're not sure what would make the deal fair to the buyer.",
         "embedding": np.array([0.3, 0.2, 0.4, 0.5, 0.7, 0.8, 0.7, 0.8, 0.6, 0.1, 0.3, 0.3, 0.4, 0.4, 0.6])},
        {"prompt": "You need to know if the buyer is ready to buy or is still considering other options.",
         "embedding": np.array([0.7, 0.6, 0.8, 0.7, 0.3, 0.4, 0.5, 0.2, 0.8, 0.5, 0.6, 0.7, 0.8, 0.9, 0.9])}
    ]

    # timeline
    tim_prompts = [
        {"prompt": "If a deal is reached, you need to know if they're willing to pay right now.",
         "embedding": np.array([0.7, 0.6, 0.8, 0.7, 0.3, 0.2, 0.4, 0.2, 0.8, 0.4, 0.7, 0.8, 0.7, 0.8, 0.9])},
        {"prompt": "You may need more time to consider, and want to know how long the buyer would maintain their offer.",
         "embedding": np.array([0.3, 0.2, 0.5, 0.6, 0.6, 0.7, 0.8, 0.5, 0.6, 0.2, 0.4, 0.3, 0.5, 0.5, 0.7])},
        {"prompt": "Shipping times may vary, so you need to know their preference.",
         "embedding": np.array([0.4, 0.3, 0.6, 0.7, 0.5, 0.6, 0.7, 0.6, 0.8, 0.1, 0.3, 0.4, 0.4, 0.6, 0.8])},
        {"prompt": "You need to know the level of urgency for getting the item.",
         "embedding": np.array([0.5, 0.4, 0.7, 0.5, 0.4, 0.5, 0.6, 0.4, 0.7, 0.3, 0.5, 0.6, 0.6, 0.7, 0.8])},
        {"prompt": "Demand is high for this item, so you want to know how soon the buyer is willing to commit.",
         "embedding": np.array([0.8, 0.7, 0.9, 0.8, 0.2, 0.3, 0.3, 0.1, 0.7, 0.6, 0.8, 0.7, 0.9, 0.9, 0.7])}
    ]

    # location
    loc_prompts = [
        {"prompt": "If the buyer isn't in your preferred location, you need to know ASAP.",
         "embedding": np.array([0.7, 0.6, 0.8, 0.6, 0.3, 0.2, 0.5, 0.1, 0.7, 0.3, 0.6, 0.7, 0.5, 0.6, 0.8])},
        {"prompt": "You're open to meeting in person to make the purchase, but need to know where they're located.",
         "embedding": np.array([0.4, 0.3, 0.5, 0.7, 0.7, 0.8, 0.7, 0.6, 0.6, 0.1, 0.4, 0.4, 0.6, 0.5, 0.7])},
        {"prompt": "You need to know if the buyer expects you to cover shipping or if they're willing to pay for it.",
         "embedding": np.array([0.5, 0.4, 0.6, 0.5, 0.5, 0.6, 0.6, 0.4, 0.8, 0.2, 0.5, 0.5, 0.7, 0.8, 0.9])}
    ]