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

    def to_dict(self):
        """Convert NegotiationSeed to dictionary for JSON serialization."""
        return {
            "seed_id": self.seed_id,
            "category": self.category,
            "item": self.item,
            "quantity": self.quantity,
            "condition": self.condition,
            "buyer_max": self.buyer_max,
            "seller_min": self.seller_min,
            "seller_list": self.seller_list,
            "extras_available": self.extras_available,
            "locality": self.locality,
            "context": self.context,
            "notes": self.notes
        }