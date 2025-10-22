#!/usr/bin/env python3
import json

# Read the tech_seeds.json file
with open('seeds/tech_seeds.json', 'r') as f:
    data = json.load(f)

# Update competing_offer values
updated_count = 0
for item in data:
    if 'context' in item and 'competing_offer' in item['context']:
        # Get the current decimal value (e.g., -0.05)
        x = item['context']['competing_offer']
        # Get the seller_list price
        seller_list = item['seller_list']
        # Calculate the actual competing offer price
        # competing_offer = seller_list + x * seller_list = seller_list * (1 + x)
        competing_offer = seller_list * (1 + x)
        # Round to nearest integer for cleaner prices
        item['context']['competing_offer'] = round(competing_offer)
        updated_count += 1
        print(f"Seed {item['seed_id']}: {x:.2%} of {seller_list} = {round(competing_offer)}")

# Save the updated data back to the file
with open('seeds/tech_seeds.json', 'w') as f:
    json.dump(data, f, indent=2)

print(f"\nSuccessfully updated {updated_count} competing_offer values in tech_seeds.json")