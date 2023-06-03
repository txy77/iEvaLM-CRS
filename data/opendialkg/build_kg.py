import json
from collections import defaultdict

kg = defaultdict(list)
entity2id = defaultdict(lambda: len(entity2id))
relation2id = defaultdict(lambda: len(relation2id))

with open('id2info.json', encoding='utf-8') as f:
    id2info = json.load(f)
    for info_dict in id2info.values():
        item = info_dict['name']
        for attr, value in info_dict.items():
            if attr == 'name':
                continue
            if isinstance(value, list):
                for v in value:
                    kg[entity2id[item]].append((relation2id[attr], entity2id[v]))
            else:
                kg[entity2id[item]].append((relation2id[attr], entity2id[value]))

print(len(kg), len(entity2id), len(relation2id))

with open('kg.json', 'w', encoding='utf-8') as f:
    json.dump(kg, f, ensure_ascii=False)
with open('entity2id.json', 'w', encoding='utf-8') as f:
    json.dump(entity2id, f, ensure_ascii=False)
with open('relation2id.json', 'w', encoding='utf-8') as f:
    json.dump(relation2id, f, ensure_ascii=False)

item_ids = set()
with open('data.jsonl', encoding='utf-8') as f:
    for line in f:
        line = json.loads(line)
        for turn in line['dialog']:
            for item in turn['item']:
                if item in entity2id:
                    item_ids.add(entity2id[item])
print(len(item_ids))
item_ids = sorted(item_ids)
with open('item_ids.json', 'w', encoding='utf-8') as f:
    json.dump(item_ids, f, ensure_ascii=False)
