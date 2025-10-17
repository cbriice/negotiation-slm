"""
seed_initializer.py - loads seed into a class for formatting consistency.
"""

class NegotiationSeed():
    def __init__(self, seed: dict):
        self.seed_id = seed["seed_id"]
        self.category = seed["category"]
        self.item = seed["item"]
        self.quantity = seed["quantity"]
        self.condition = seed["condition"]

        self.buyer_max = seed["buyer_max"]
        self.seller_min = seed["seller_min"]
        self.seller_list = seed["seller_list"]

        self.extras_available = seed["extras_available"]
        self.locality = seed["locality"]
        self.context = seed["context"]
        self.notes = seed["notes"]