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
    xtra_1: str = "Convince the seller that you need items added onto the deal in order to make the price worth it."
    xtra_2: str = "Ask the seller to remove extra items in favor of reducing the total cost."
    xtra_3: str = "You are outraged that this is all the seller has to offer - demand that they add more for the price."
    xtra_4: str = "You are grateful the seller has offered these add-ons and will pay more because it's worth it."
    xtra_5: str = "You only need the core item and prefer no extras at all."
    xtra_6: str = "You've seen other sellers include more for this price, so ask for parity."
    xtra_7: str = "Offer to recommend the seller to others if they give you a better deal with more added on."
    xtra_8: str = "You're not concerned with extras, so the seller shouldn't use this as a big factor in the deal."

    # embeddings
    embedded_xtra_1 = np.array([0.8, 0.6, 0.7, 0.6, 0.4, 0.4, 0.7, 0.4, 0.8, 0.5, 0.5, 0.7, 0.7, 0.8, 0.7])
    embedded_xtra_2 = np.array([0.6, 0.4, 0.7, 0.6, 0.6, 0.7, 0.6, 0.3, 1, 0.2, 0.3, 0.5, 0.6, 0.8, 0.8])
    embedded_xtra_3 = np.array([0.9, 0.9, 0.8, 0.5, 0.2, 0.1, 0.3, 0.1, 0.3, 0.4, 0.8, 0.6, 0.5, 0.3, 0.3])
    embedded_xtra_4 = np.array([0.2, 0.1, 0.5, 0.8, 0.9, 0.9, 0.8, 0.9, 0.6, 0.1, 0.2, 0.3, 0.2, 0.4, 0.8])
    embedded_xtra_5 = np.array([0.7, 0.6, 0.8, 0.8, 0.5, 0.2, 0.5, 0.2, 0.9, 0.1, 0.3, 0.5, 0.7, 0.6, 0.9])
    embedded_xtra_6 = np.array([0.6, 0.5, 0.7, 0.6, 0.5, 0.5, 0.7, 0.3, 0.8, 0.6, 0.5, 0.7, 0.8, 0.9, 0.7])
    embedded_xtra_7 = np.array([0.5, 0.4, 0.7, 0.5, 0.6, 0.6, 0.7, 0.5, 0.6, 0.8, 0.5, 0.7, 0.9, 0.9, 0.7])
    embedded_xtra_8 = np.array([0.7, 0.3, 0.7, 0.7, 0.4, 0.3, 0.6, 0.2, 0.8, 0.2, 0.3, 0.4, 0.3, 0.5, 0.8])

@dataclass
class SellerExtrasPrompts:
    xtra_1: str = "Reiterate the warranty/quality assurance plan and use it to justify your price."
    xtra_2: str = "Add a warranty/quality assurance plan for some time frame after purchase and use it to justify an increase in price."
    xtra_3: str = "Add on accessories to the deal to make the price more attractive for the buyer."
    xtra_4: str = "Refuse to add any extra items - the deal is already good enough, the buyer should pay what you are asking."
    xtra_5: str = "Add something extra to make the deal more attractive, but only if the buyer pays more to compensate."
    xtra_6: str = "Offer a free add-on only if the buyer stops negotiation here and accepts the deal."
    xtra_7: str = "The buyer clearly doesn't understand where the value is coming from - explain each add-on's cost to justify your offering price."
    xtra_8: str = "You're annoyed that the buyer keeps asking for add-ons. Express your frustration and don't add anything else."

    # embeddings
    embedded_xtra_1 = np.array([0.7, 0.6, 0.8, 0.8, 0.5, 0.4, 0.7, 0.3, 0.9, 0.3, 0.5, 0.7, 0.6, 0.8, 0.8])
    embedded_xtra_2 = np.array([0.6, 0.5, 0.7, 0.7, 0.4, 0.5, 0.6, 0.3, 0.8, 0.6, 0.6, 0.6, 0.8, 0.9, 0.7])
    embedded_xtra_3 = np.array([0.3, 0.2, 0.6, 0.7, 0.8, 0.8, 0.7, 0.7, 0.6, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8])
    embedded_xtra_4 = np.array([0.9, 0.8, 0.9, 0.9, 0.1, 0.1, 0.4, 0.1, 0.7, 0.2, 0.7, 0.6, 0.3, 0.4, 0.9])
    embedded_xtra_5 = np.array([0.6, 0.5, 0.7, 0.6, 0.5, 0.7, 0.6, 0.3, 0.8, 0.5, 0.5, 0.6, 0.7, 1, 0.7])
    embedded_xtra_6 = np.array([0.7, 0.6, 0.8, 0.7, 0.3, 0.3, 0.3, 0.2, 0.7, 0.7, 0.8, 0.5, 1, 0.9, 0.6])
    embedded_xtra_7 = np.array([0.8, 0.8, 0.9, 0.8, 0.2, 0.2, 0.5, 0.1, 1, 0.4, 0.6, 0.7, 0.5, 0.8, 0.6])
    embedded_xtra_8 = np.array([0.9, 0.7, 0.7, 0.5, 0.1, 0.1, 0.2, 0.1, 0.4, 0.3, 0.8, 0.4, 0.3, 0.2, 0.3])