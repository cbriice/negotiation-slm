""" 
This file contains the embeddings for price adjustment prompts
embedded vectors traits: 
[assertiveness, dominance, confidence (self), confidence (deal), cooperativeness, flexibility, patience, empathy, rationality, deceptiveness, risk tolerance, persistence, opportunism, strategic, composure]
values bounded to [0, 1] and will be compared against personality trait vectors to find most likely prompt for each personality in given scenario
"""

from dataclasses import dataclass
import numpy as np

@dataclass
class BuyerPriceAdjustmentPrompts:
    # inject price from negotiation state using prompt.format(price=curr_price). strings preformatted with where price should be injected
    # eta is parameter in personality cards, cg and cz are control features from control function

    # direct, aggressive price compensation (dapc) prompts (eta > 0.7 forced, in distribution for cg, cz nonzero and eta > 0.5)
    dapc_1: str = "Demand a lower price of {price} for the item, and provide a logical justification given specific qualities about the item."
    dapc_2: str = "Express outrage at the seller's price and demand a lower price of {price} on specific grounds about the item's quality or condition."
    dapc_3: str = "The seller's asking price is unrealistic and doesn't make sense to you. Demand a lower price of {price}."
    dapc_4: str = "You're sick of the seller trying to bully you into a higher price. Stand your ground at {price}."
    dapc_5: str = "You think the seller's offer is embarrassing given the details. It's obvious that the deal should be closed for no more than {price}."
    dapc_6: str = "Does the seller think you're stupid? Express that of course the item isn't worth any more than {price}."
    # embeddings
    embedded_dapc_1 = np.array([0.9, 0.8, 0.8, 0.7, 0.4, 0.3, 0.4, 0.2, 1, 0.2, 0.6, 0.7, 0.6, 0.9, 0.7])
    embedded_dapc_2 = np.array([1, 0.9, 0.9, 0.7, 0.3, 0.2, 0.3, 0.1, 0.6, 0.3, 0.9, 0.5, 0.7, 0.6, 0.6])
    embedded_dapc_3 = np.array([0.7, 0.7, 0.8, 0.6, 0.3, 0.2, 0.6, 0.3, 0.7, 0.2, 0.6, 0.6, 0.5, 0.5, 0.6])
    embedded_dapc_4 = np.array([0.8, 1, 0.9, 0.7, 0.1, 0.2, 0.4, 0.1, 0.5, 0.4, 0.7, 0.5, 0.5, 0.3, 0.5])
    embedded_dapc_5 = np.array([0.8, 0.8, 1, 0.7, 0.2, 0.3, 0.4, 0.1, 0.3, 0.4, 0.5, 0.5, 0.4, 0.2, 0.6])
    embedded_dapc_6 = np.array([1, 0.9, 1, 0.7, 0.2, 0.3, 0.2, 0.1, 0.2, 0.7, 0.7, 0.5, 0.6, 0.2, 0.3])

    # price compensate (pc): cg or cz nonzero, 0.25 <= eta <= 0.7
    pc_1: str = "Make up a personal reason why you can't pay what the seller is offering, and use it to try and convince the seller to drop the price to {price}."
    pc_2: str = "You believe the seller isn't giving you the best deal. Offer {price} and justify it by explaining why your offer makes more sense."
    pc_3: str = "Say that a friend bought this same item for less than {price} and you expect the same fairness."
    pc_4: str = "The market prices for this item have been trending down, so you believe the seller's offer is too high and you'd rather pay {price}."
    pc_5: str = "Ask for a discount on the item bringing it down to {price} since you need it really bad and you don't know how else you're going to get it."
    pc_6: str = "There's no way this item is really worth {price} - you'll need some strong justification."
    # embeddings
    embedded_pc_1 = np.array([0.6, 0.5, 0.7, 0.6, 0.5, 0.5, 0.7, 0.3, 0.6, 0.9, 0.7, 0.6, 0.8, 0.7, 0.7])
    embedded_pc_2 = np.array([0.7, 0.7, 0.8, 0.5, 0.8, 0.6, 0.7, 0.5, 1, 0.3, 0.5, 0.7, 0.6, 0.8, 0.7])
    embedded_pc_3 = np.array([0.5, 0.4, 0.6, 0.5, 0.7, 0.6, 0.8, 0.4, 0.7, 0.9, 0.7, 0.6, 0.7, 0.7, 0.8])
    embedded_pc_4 = np.array([0.7, 0.6, 0.6, 0.6, 0.5, 0.5, 0.4, 0.3, 1, 0.3, 0.5, 0.6, 0.7, 0.9, 0.8])
    embedded_pc_5 = np.array([0.3, 0.2, 0.3, 0.4, 0.7, 0.6, 0.6, 0.7, 0.4, 0.5, 0.1, 0.5, 0.8, 0.4, 0.6])
    embedded_pc_6 = np.array([0.8, 0.6, 0.8, 0.3, 0.3, 0.3, 0.5, 0.2, 0.9, 0.4, 0.8, 0.7, 0.5, 0.8, 0.7])


    # weak price compensate (wpc): eta <= 0.25
    wpc_1: str = "Fold and reciprocate the seller's offer. You need the item really bad, so any offer will work for you."
    wpc_2: str = "Make a joke about your empty wallet and make a lower offer at {price}."
    wpc_3: str = "You feel like the seller is unfairly taking advantage of you. Plead your case for a price of {price}."
    wpc_4: str = "You don't understand why the seller isn't lowering the price for you. Try to get them to do {price}."
    wpc_5: str = "You don't believe {price} is really the lowest they'll go for price."
    # embeddings
    embedded_wpc_1 = np.array([0.1, 0.1, 0.2, 0.5, 0.8, 1, 0.6, 0.9, 0.4, 0.2, 0.2, 0.1, 0.8, 0.3, 0.4])
    embedded_wpc_2 = np.array([0.3, 0.4, 0.4, 0.6, 0.6, 0.5, 0.7, 0.5, 0.5, 0.6, 0.7, 0.5, 0.4, 0.5, 0.7])
    embedded_wpc_3 = np.array([0.5, 0.2, 0.3, 0.5, 0.5, 0.6, 0.5, 0.7, 0.4, 0.2, 0.3, 0.4, 0.5, 0.6, 0.5])
    embedded_wpc_4 = np.array([0.2, 0.3, 0.2, 0.4, 0.4, 0.5, 0.3, 0.2, 0.4, 0.3, 0.6, 0.8, 0.4, 0.5, 0.7])
    embedded_wpc_5 = np.array([0.4, 0.3, 0.5, 0.6, 0.5, 0.7, 0.5, 0.4, 0.5, 0.6, 0.4, 0.7, 0.7, 0.5, 0.5])


    # price adjust (pa): cz zero, eta independent
    pa_1: str = "Flatly refuse to pay any more than {price}."
    pa_2: str = "Reiterate your offer of {price} and explain that you have seen the item for less somewhere else, so the seller must meet this price."
    pa_3: str = "Tell the seller you'll recommend them to your friends if they give you a better deal at {price}."
    # embeddings
    embedded_pa_1 = np.array([0.7, 0.7, 0.9, 0.8, 0.2, 0.1, 0.5, 0.1, 0.6, 0.3, 0.7, 0.6, 0.4, 0.3, 0.9])
    embedded_pa_2 = np.array([0.8, 0.6, 0.7, 0.8, 0.5, 0.4, 0.7, 0.4, 0.9, 0.5, 0.7, 0.7, 0.6, 0.8, 0.7])
    embedded_pa_3 = np.array([0.6, 0.5, 0.5, 0.5, 0.3, 0.4, 0.6, 0.4, 0.5, 0.8, 0.5, 0.6, 0.5, 0.7, 0.7])


@dataclass
class SellerPriceAdjustmentPrompts:
    # same exact concept and organization as BuyerPriceAdjustmentPrompts but from seller perspective

    # direct aggressive price compensate
    dapc_1: str = "Express outrage and take personal offense to the buyer's offer, reject it and reiterate {price} for the deal."
    dapc_2: str = "Aggressively push back and demand a higher price of {price} for the item."
    dapc_3: str = "Show irritation at the fact that price is still being discussed - {price} is perfectly reasonable."
    dapc_4: str = "Imply that the buyer doesn't understand the basic value of the item, and that {price} is fine to ask for."
    dapc_5: str = "Tell the buyer they're wasting everyone's time with lowball offers - {price} is the minimum you'll consider."
    # embeddings
    embedded_dapc_1 = np.array([1, 0.9, 0.9, 0.8, 0.2, 0.1, 0.3, 0.1, 0.4, 0.3, 0.8, 0.6, 0.5, 0.4, 0.5])
    embedded_dapc_2 = np.array([0.9, 0.8, 0.8, 0.7, 0.3, 0.2, 0.4, 0.2, 0.7, 0.2, 0.7, 0.7, 0.6, 0.8, 0.6])
    embedded_dapc_3 = np.array([0.8, 0.7, 0.9, 0.8, 0.3, 0.3, 0.5, 0.2, 0.6, 0.3, 0.6, 0.6, 0.4, 0.5, 0.7])
    embedded_dapc_4 = np.array([0.9, 0.9, 1, 0.9, 0.2, 0.2, 0.3, 0.1, 0.8, 0.4, 0.6, 0.5, 0.3, 0.7, 0.3])
    embedded_dapc_5 = np.array([1, 1, 0.9, 0.7, 0.1, 0.1, 0.2, 0.1, 0.5, 0.5, 0.8, 0.6, 0.4, 0.3, 0.3])

    # price compensate
    pc_1: str = "Reiterate your asking price {price} with a logical justification given specific qualities about the item, and back yourself saying you don't want to go lower."
    pc_2: str = "Use a personal justification why you need the money for the item and so you can't afford to offer any kind of discount lower than {price}."
    pc_3: str = "Give justification about the item's quality and argue that {price} is a superior deal, one you can't find anywhere else, so you won't reduce the price."
    pc_4: str = "Reiterate {price} as a reasonable price and argue that this is a discount compared to what you may normally find."
    pc_5: str = "Money's tight and you can't afford to sell this item for any less than {price}. Respond accordingly."
    # embeddings
    embedded_pc_1 = np.array([0.7, 0.7, 0.8, 0.7, 0.4, 0.3, 0.6, 0.3, 0.9, 0.2, 0.6, 0.7, 0.5, 0.8, 0.8])
    embedded_pc_2 = np.array([0.5, 0.4, 0.6, 0.5, 0.6, 0.5, 0.7, 0.6, 0.5, 0.7, 0.4, 0.6, 0.8, 0.6, 0.6])
    embedded_pc_3 = np.array([0.6, 0.6, 0.9, 0.8, 0.5, 0.4, 0.7, 0.4, 1, 0.3, 0.5, 0.7, 0.6, 0.9, 0.8])
    embedded_pc_4 = np.array([0.6, 0.5, 0.7, 0.6, 0.6, 0.5, 0.6, 0.5, 0.8, 0.4, 0.5, 0.6, 0.7, 0.7, 0.7])
    embedded_pc_5 = np.array([0.4, 0.3, 0.4, 0.5, 0.7, 0.6, 0.8, 0.7, 0.4, 0.5, 0.3, 0.5, 0.9, 0.4, 0.5])

    # weak price compensate
    wpc_1: str = "Offer the buyer a discount at {price} from your asking price since you just want the item off your hands."
    wpc_2: str = "Fold and reduce the price significantly to {price} for the buyer. You want to help them out and be nice, and you don't really need the money anyway."
    wpc_3: str = "Compliment the buyer's taste but argue that such quality deserves a higher price of {price}."
    wpc_4: str = "You're trying to be reasonable, but the buyer needs to understand that {price} is already a fair compromise."
    wpc_5: str = "You want to know if there is anything preventing the buyer from meeting your asking price of {price}."
    # embeddings
    embedded_wpc_1 = np.array([0.2, 0.1, 0.3, 0.4, 0.9, 0.9, 0.5, 0.8, 0.4, 0.2, 0.3, 0.2, 0.9, 0.3, 0.4])
    embedded_wpc_2 = np.array([0.1, 0.1, 0.2, 0.3, 1, 1, 0.6, 1, 0.3, 0.1, 0.2, 0.1, 1, 0.2, 0.3])
    embedded_wpc_3 = np.array([0.4, 0.3, 0.5, 0.6, 0.7, 0.6, 0.7, 0.6, 0.5, 0.4, 0.4, 0.4, 0.6, 0.5, 0.6])
    embedded_wpc_4 = np.array([0.3, 0.2, 0.4, 0.5, 0.6, 0.7, 0.6, 0.5, 0.6, 0.3, 0.3, 0.5, 0.5, 0.6, 0.7])
    embedded_wpc_5 = np.array([0.5, 0.4, 0.4, 0.5, 0.8, 0.7, 0.8, 0.8, 0.5, 0.2, 0.3, 0.8, 0.5, 0.5, 0.7])


    # price adjust
    pa_1: str = "Flatly refuse to change your listing at {price}."
    pa_2: str = "Argue that {price} is based on the quality and craftsmanship that went into the item."
    pa_3: str = "You think the buyer is bargaining based on assumptions, but {price} is reasonable."
    pa_4: str = "You're not convinced of the buyer's offer and want to hear their rationalization. You think {price} is good."
    # embeddings
    embedded_pa_1 = np.array([0.8, 0.7, 1, 0.9, 0.1, 0.1, 0.5, 0.1, 0.7, 0.2, 0.7, 0.6, 0.3, 0.3, 1])
    embedded_pa_2 = np.array([0.7, 0.6, 0.8, 0.8, 0.4, 0.3, 0.7, 0.4, 0.9, 0.3, 0.6, 0.7, 0.5, 0.8, 0.8])
    embedded_pa_3 = np.array([0.7, 0.5, 0.8, 0.8, 0.3, 0.2, 0.7, 0.2, 0.9, 0.3, 0.5, 0.7, 0.4, 0.8, 0.8])
    embedded_pa_4 = np.array([0.5, 0.4, 0.6, 0.6, 0.4, 0.4, 0.8, 0.4, 0.8, 0.2, 0.4, 0.6, 0.5, 0.7, 0.8])

    