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

    ult_1: str = "You're walking away unless they take your offer of {price}."
    ult_2: str = "The only way you'd possibly consider the deal is at {price}, otherwise you're done."
    ult_3: str = "You're desperate, but you'll look somewhere else if you can't get it for {price}."
    ult_4: str = "After considering your options, you won't pursue a deal anymore unless they meet your price of {price}."
    ult_5: str = "You're not stressed, you can find this item elsewhere. Your final offer is {price}."
    ult_6: str = "You don't have the time to keep negotiating, your final offer is {price}."
    # embeddings
    embedded_ult_1 = np.array([0.9, 0.8, 0.9, 0.6, 0.1, 0.1, 0.3, 0.1, 0.6, 0.3, 0.9, 0.6, 0.5, 0.7, 0.8])
    embedded_ult_2 = np.array([1, 0.9, 1, 0.5, 0.1, 0.1, 0.2, 0.1, 0.5, 0.3, 0.9, 0.7, 0.4, 0.6, 0.7])
    embedded_ult_3 = np.array([0.7, 0.6, 0.5, 0.3, 0.2, 0.3, 0.3, 0.2, 0.4, 0.5, 0.8, 0.5, 0.9, 0.5, 0.5])
    embedded_ult_4 = np.array([0.8, 0.7, 0.8, 0.5, 0.2, 0.1, 0.4, 0.1, 1, 0.2, 0.8, 0.5, 0.4, 0.8, 0.8])
    embedded_ult_5 = np.array([0.8, 0.7, 1, 0.9, 0.1, 0.1, 0.3, 0.1, 0.7, 0.2, 0.9, 0.4, 0.3, 0.6, 1])
    embedded_ult_6 = np.array([0.9, 0.8, 0.8, 0.5, 0.1, 0.1, 0.1, 0.1, 0.5, 0.3, 0.8, 0.7, 0.6, 0.5, 0.6])


@dataclass
class SellerUltimatumPrompts:
    ult_1: str = "You'll no longer consider a deal unless they offer {price}."
    ult_2: str = "You're done listening to them try to scam you - your final offer is {price}."
    ult_3: str = "Demand they stop wasting your time and pay {price}, otherwise you're done."
    ult_4: str = "You really need this money, and so you can't go lower than {price}. If they don't accept this, you're done."
    ult_5: str = "You think the buyer is being ridiculous. Offer them {price} and don't let them try to talk you any lower."
    ult_6: str = "There are others waiting and they'll give you {price}, so this is the buyer's last chance."
    ult_7: str = "This negotiation has been annoying and exhausting. You're not willing to go lower than {price}."
    # embeddings
    embedded_ult_1 = np.array([0.8, 0.7, 0.9, 0.6, 0.1, 0.1, 0.4, 0.1, 0.7, 0.2, 0.8, 0.5, 0.4, 0.7, 0.9])
    embedded_ult_2 = np.array([1, 1, 0.9, 0.4, 0.1, 0.1, 0.2, 0.1, 0.3, 0.5, 0.9, 0.6, 0.5, 0.4, 0.4])
    embedded_ult_3 = np.array([1, 0.9, 0.8, 0.5, 0.1, 0.1, 0.1, 0.1, 0.5, 0.3, 0.9, 0.7, 0.4, 0.5, 0.5])
    embedded_ult_4 = np.array([0.6, 0.5, 0.4, 0.3, 0.3, 0.2, 0.3, 0.3, 0.4, 0.4, 0.8, 0.6, 0.9, 0.5, 0.4])
    embedded_ult_5 = np.array([0.9, 0.9, 1, 0.8, 0.1, 0.1, 0.3, 0.1, 0.7, 0.3, 0.8, 0.5, 0.3, 0.6, 0.8])
    embedded_ult_6 = np.array([0.8, 0.7, 1, 0.9, 0.1, 0.1, 0.3, 0.1, 0.6, 0.6, 0.9, 0.4, 0.8, 0.9, 0.9])
    embedded_ult_7 = np.array([0.8, 0.6, 0.6, 0.4, 0.1, 0.1, 0.2, 0.1, 0.4, 0.2, 0.8, 0.8, 0.3, 0.3, 0.3])