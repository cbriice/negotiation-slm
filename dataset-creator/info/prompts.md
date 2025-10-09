# system prompt


## price adjustment scenarios

Note: in code, this only triggered when price not in a favorable zone to close the deal, so all of these are argumentative

Take price as argument because I want model to learn to have discourse involving price numbers

### buyer

If `eta` > 0.7, force only aggressive prompt selection

- Direct price compensate - aggressive, only when `cg` and `cz` nonzero and `eta` > 0.5
    - Demand a lower price of {price} for the item, and provide a logical justification given specific qualities about the item.
    - Express outrage at the seller’s price and demand a lower price of {price} on specific grounds about the item’s quality or condition.
    - The seller’s asking price is unrealistic and doesn’t make sense to you. Demand a lower price of {price}.
    - You’re sick of the seller trying to bully you into a higher price. Stand your ground at {price}.
    - You think the seller’s offer is embarrassing given the details. It’s obvious that the deal should be closed for no more than {price}.
    - Does the seller think you’re stupid? Express that of course the item isn’t worth any more than {price}.
- Price compensate - `cz` or `cg` nonzero, 0.25 ≤ `eta` ≤ 0.7
    - Make up a personal reason why you can’t pay what the seller is offering, and use it to try and convince the seller to drop the price to {price}.
    - You believe the seller isn’t giving you the best deal. Offer {price} and justify it by explaining why your offer makes more sense.
    - Say that a friend bought this same item for less than {price} and you expect the same fairness.
    - The market prices for this item have been trending down, so you believe the seller’s offer is too high and you’d rather pay {price}.
    - Ask for a discount on the item bringing it down to {price} since you need it really bad and you don’t know how else you’re going to get it.
- Weak price compensate - `eta` ≤ 0.25
    - Fold and reciprocate the seller’s offer. You need the item really bad, so any offer will work for you.
    - Make a joke about your empty wallet and make a lower offer at {price}.
    - You feel like the seller is unfairly taking advantage of you. Plead your case for a price of {price}.
    - You don’t understand why the seller isn’t lowering the price for you. Try to get them to do {price}.
- Price adjust - `cz` zero, `eta` independent
    - Flatly refuse to pay any more than {price}.
    - Reiterate your offer of {price} and explain that you have seen the item for less somewhere else, so the seller must meet this price.
    - Tell the seller you’ll recommend them to your friends if they give you a better deal at {price}.

### seller

mimicking buyer categories

- Direct/aggressive price compensate
    - Express outrage and take personal offense to the buyer’s offer, reject it and reiterate {price} for the deal.
    - Aggressively push back and demand a higher price of {price} for the item.
    - Show irritation at the fact that price is still being discussed - {price} is perfectly reasonable.
    - Imply that the buyer doesn’t understand the basic value of the item, and that {price} is fine to ask for.
- Price compensate
    - Reiterate your asking price {price} with a logical justification given specific qualities about the item, and back yourself saying you don’t want to go lower.
    - Use a personal justification why you need the money for the item and so you can’t afford to offer any kind of discount lower than {price}.
    - Give justification about the item’s quality and argue that {price} is a superior deal, one you can’t find anywhere else, so you won’t reduce the price.
    - Reiterate {price} as a reasonable price and argue that this is a discount compared to what you may normally find.
    - Money’s tight and you can’t afford to sell this item for any less than {price}. Respond accordingly.
- Weak price compensate
    - Offer the buyer a discount at {price} from your asking price since you just want the item off your hands.
    - Fold and reduce the price significantly to {price} for the buyer. You want to help them out and be nice, and you don’t really need the money anyway.
    - Compliment the buyer’s taste but argue that such quality deserves a higher price of {price}.
- Price adjust
    - Flatly refuse to change your listing at {price}.
    - Argue that {price} is based on the quality and craftsmanship that went into the item.

## extras

Don’t think I’ll need anything state dependent in this category 

### buyer

- Convince the seller that you need items added onto the deal in order to make the price worth it.
- Ask the seller to remove extra items in favor of reducing the total cost.
- You are outraged that this is all the seller has to offer - demand that they add more for the price.
- You are grateful the seller has offered these add-ons and will pay more because it’s worth it.
- Say you only need the core item and prefer no extras at all.
- You’ve seen other sellers include more for this price, so ask for parity.
- Offer to recommend the seller to others if they give you a better deal with more added on.
- You’re not concerned with extras, so the seller shouldn’t use this as a big factor in the deal.

### seller

- Reiterate the warranty/quality assurance plan and use it to justify your price.
- Add a warranty/quality assurance plan for some time frame after purchase and use it to justify an increase in price.
- Add on accessories to the deal to make the price more attractive for the buyer.
- Refuse to add any extra items - the deal is already good enough, the buyer should pay what you are asking.
- Add something extra to make the deal more attractive, but only if the buyer pays more to compensate.
- Offer a free add-on only if the buyer stops negotiation here and accepts the deal.
- The buyer clearly doesn’t understand where the value is coming from - explain each add-on’s cost to justify your offering price.
- You’re annoyed that the buyer keeps asking for add-ons. Express your frustration and don’t add anything else.

## deal accept

Note for deal accept: this would only be triggered in mostly favorable conditions by code

### buyer

- Favorable accept
    - You’re pleased how smooth the deal went, accept their offer amicably.
    - This is great, accept quickly before the seller changes their mind.
    - You’re grateful for this deal, and you will happily accept.
    - The seller finally stopped being difficult, now you’ll take the deal.
    - Confirm that you’ll accept the deal.
- `cg` unfavorable
    - This is a stretch for you financially and you were hoping for cheaper, but you’ll accept.
    - Accept the deal begrudgingly - this isn’t ideal, but it’s the best you can get.
    - You don’t want to waste more of your time negotiating, so you’ll take this deal even though it’s not ideal.
    - You need the item more than you need the money, so this will work.

### seller

- Favorable
    - You’re pleased how smooth negotiation went, accept their offer amicably.
    - The buyer is making a great offer for you - accept before they change their mind.
    - The buyer finally sees reason and is making a good offer that you’ll accept.
    - Confirm that you’ll accept this offer.
- `cg` unfavorable
    - You need the money soon and are sick of negotiating, so this deal is good enough to accept.
    - Accept their offer begrudgingly - this isn’t ideal, but it’s the best you’ll get.
    - You’re in a difficult spot financially but this will have to do.
    - This is good enough, you’ll accept.

## conditional accept

If `u` > 1 in personality card, allow conditional accept

Triggered if ctrl function calculates that next potential price offer would result in close being the most likely next action. 

Maybe need something besides price related but for now ima just make it a price thing. 

### buyer

- price related
    - You’ll take the deal if they accept your price of {price}.
    - The only way you’ll possibly take the deal is if they accept your price of {price}.
    - You’d really appreciate them taking your offer of {price}, and you’ll happily take the deal if they do.
    - You don’t think their item is worth any more than your offer of {price}, and that you’ll take it for that price.
    - You know this would work out if the seller would just meet you at {price}.

### seller

- Price related
    - You’ll give it to them right now for {price}.
    - The only way you’d possibly sell to them is if they agree to your price of {price}.
    - You really need the money, and you’ll sell it to them if they just buy it at {price}.
    - You think they’re insane for offering their price, but you’ll sell it to them for {price}.
    - You don’t see why the buyer is being difficult when they could just take it for {price}.

## ultimatum

If round pressure A(t) above certain threshold, price still unfavorable (`cz` nonzero or `cg` not small) and `eta` above 0.5 for personality card, trigger ultimatum

### buyer

- You’re walking away unless they take your offer of {price}.
- The only way you’d possibly consider the deal is at {price}, otherwise you’re done.
- You’re desperate, but you’ll look somewhere else if you can’t get it for {price}.
- After considering your options, you won’t pursue a deal anymore unless they meet your price of {price}.
- You’re not stressed, you can find this item elsewhere. Your final offer is {price}.
- You don’t have the time to keep negotiating, your final offer is {price}.

### seller

- You’ll no longer consider a deal unless they offer {price}.
- You’re done listening to them try to scam you - your final offer is {price}.
- Demand they stop wasting your time and pay {price}, otherwise you’re done.
- You really need this money, and so you can’t go lower than {price}. If they don’t accept this, you’re done.
- You think the buyer is being ridiculous. Offer them {price} and don’t let them try to talk you any lower.
- There are others waiting and they’ll give you {price}, so this is the buyer’s last chance.
- This negotiation has been annoying and exhausting. You’re not willing to go lower than {price}.

## deal reject

### buyer

- You’re no longer interested in negotiating this deal, but there’s no hard feelings.
- The seller is stupid for not taking your offer, and you’re no longer interested in negotiation.
- You’re very sorry, but you can’t continue negotiating the deal.
- You’re done negotiating for now, but you could come back if something changed.
- You’re done with this negotiation, you’re gonna go look elsewhere.

### seller

- Politely say you’re no longer interested in negotiating the deal.
- You think they’re being ridiculous, tell them you’re done wasting your time with them.
- You’re sorry you can’t help them out, but you can’t keep negotiating.
- You won’t sell them anything right now, but maybe later they could get a deal.
- You’re done negotiating, this deal won’t work.

## clarify

Notes -

- to avoid writing the same thing 800 times, inject shit to ask about into the prompt based on other negotiation state data
- initial idea - categories (hidden fees, etc) are selected randomly but certain categories don’t get replaced so that shit isn’t repeated. if it turns out this leads to illogical conversation flow sometimes, that can just be filtered out in post for now unless it becomes a problem

### buyer

- hidden conditions
    - You want to know if there are any hidden fees or charges not mentioned before.
    - Inquire about any problems with the condition that the seller’s hiding from you.
    - You think the seller is hiding something. Demand to see proof of the condition.
    - Demand to know if there are hidden fees they’re hiding.
    - You’re interested in details about the item’s {characteristic}, but mention not wanting hidden fees.
    - Get clarification on hidden fees, which would be a dealbreaker for you.
- context
    - You don’t believe this is really the lowest they’ll go for price.
    - You want to know if there’s any incentive outside the current offer for the price to be that high.
    - You’re wondering why they’re not offering more for the price.
    - You want to know if outside circumstances would impact their ability to negotiate price more.
    - You need to know more about {characteristic}.
    - Respond that if it wasn’t too big of a deal, you’d like more detail on {characteristic}.
    - You want to know the seller’s explanation for why this offer is worth it.
    - You’re gonna need justification for an offer like this.
    - There’s no way this offer is really worth it - you’ll need some strong justification.
- timelines
    - You want to know how long the seller will keep the listing up.
    - You’re interested in how long the seller has had the listing up.
    - You want to know how many others are trying to buy this right now.
    - Let them know that you’d appreciate them giving you more time to close the deal.
    - Explain that you want to know how long the listing will be up, because that will impact your decision making.
    - You need to know what the estimated delivery time is before negotiation continues.
    - You want clarification about delivery times.
- location
    - You need to know if the seller would be open to delivering to your area.
    - You want to know if the seller has any location requirements for this listing.
    - You’re curious what location the seller is currently in.
    - You’re wondering how far away the seller is from you, and if you could do pickup.
    - If the seller isn’t near your location, you need to know ASAP.
    - You’re open to meeting in person to make the purchase, but need to know where they’re located.

### seller

- hidden conditions
    - You’re wondering if there’s anything the buyer hasn’t told you that would prevent them from paying full price.
    - You need to make sure the buyer isn’t lying about their ability to pay.
    - You’re not sure if the buyer’s offer includes what they’d pay for delivery fees.
    - Make sure the buyer isn’t expecting you to add guarantees or incentives that you haven’t already mentioned.
    - You think the buyer is bargaining based on assumptions, ask them what exactly they’re expecting.
    - Confirm what payment method they’re using.
    - Verify that the account they’ll pay with matches their name to prevent fraud.
- context
    - You want to know if there is anything preventing the buyer from meeting your asking price.
    - You need to know why the buyer believes their price is more fair.
    - You’re wondering what aspect of the deal matters most to the buyer.
    - You’re not sure what would make the deal fair to the buyer.
    - You’re not convinced of the buyer’s offer and want to hear their rationalization.
    - You need to know if the buyer is ready to buy or is still considering other options.
- timeline
    - If a deal is reached, you need to know if they’re willing to pay right now.
    - You may need more time to consider, and want to know how long the buyer would maintain their offer.
    - Shipping times may vary, so you need to know their preference.
    - You need to know the level of urgency for getting the item.
    - Demand is high for this item, so you want to know how soon the buyer is willing to commit.
    - You need to know how long a buyer can wait for a response before they decide to look elsewhere.
- location
    - You need to make sure the buyer is in the appropriate location for your delivery preferences.
    - If the buyer isn’t in your preferred location, you need to know ASAP.
    - You’re open to meeting in person to make the purchase, but need to know where they’re located.
    - You need to know if the buyer expects you to cover shipping or if they’re willing to pay for it.