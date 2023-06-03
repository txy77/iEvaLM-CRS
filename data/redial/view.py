import json

with open('item_ids.json', encoding='utf-8') as f:
    item_ids = json.load(f)

print(len(item_ids))
