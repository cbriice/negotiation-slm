# seed_template.py: template for simulation seeds. contents of context field should follow context field template
# comment format: option1/option2/... where only one option should be chosen 

seed_template = {
    "seed_id": 000,
    "category": "",
    "item": "",
    "quantity": "",             # single/per unit (listing multiple)/group (listing group)
    "condition": "",            # new/used/refurbished/custom made

    "buyer_max": 000,
    "seller_min": 000,          # NOTE: only ~15-20% of seeds should have no overlap between buyer max and seller min; these are the highest difficulty for negotiation
    "seller_list": 000,

    "extras_available": [""],
    "locality": "",             # urban/suburban/rural/small town
    "context": {},
    "notes": ""
}

# what can go in the context section. 
context_options = { 
    # --- NOTE: these should be a part of context for every seed ---
    "market_type": "",          # retail/wholesale/marketplace/individual -> business/business -> individual/business -> business/rental
    "timeline": "",             # urgent/neutral/long term (listing timeline)
    "communication_style": "",  # regular/casual/formal/technical/businesslike      | NOTE: distribution % in prompt database should follow: 40%/15%/15%/15%/15%

    # --- NOTE: randomly sample 1-2 of these MAXIMUM per seed to add variety to dialog themes ---
    "listing_urgency": "",      # high/medium
    "scarcity": "",             # high/medium/low
    "competing_offer": 000,     # +/- x * seller_list; x in [0.05, 0.15]
    "risk_of_defect": "",       # high/medium/low
    "returning_customer": "",   # true if used
    "location_match": "",       # far/neutral/close
    "info_advantage": ""        # buyer/seller  |  who has more information about deal?
}