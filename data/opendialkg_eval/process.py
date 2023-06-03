import json
import os
import re

from copy import copy

with open('../opendialkg/train_data_id.json', 'r', encoding="utf-8") as f:
    train_data_id = json.load(f)
train_data_id_set = set(train_data_id)
print(len(train_data_id_set))

with open('../opendialkg/valid_data_id.json', 'r', encoding="utf-8") as f:
    valid_data_id = json.load(f)
valid_data_id_set = set(valid_data_id)
print(len(valid_data_id_set))

with open('../opendialkg/test_data_id.json', 'r', encoding="utf-8") as f:
    test_data_id = json.load(f)
test_data_id_set = set(test_data_id)
print(len(test_data_id_set))

with open('../opendialkg/data.jsonl', 'r', encoding="utf-8") as f, open('train_data_processed.jsonl', 'w', encoding="utf-8") as train_w, open('valid_data_processed.jsonl', 'w', encoding="utf-8") as valid_w, open('test_data_processed.jsonl', 'w', encoding="utf-8") as test_w:
    lines = f.readlines()
    for line in lines:
        dialog = json.loads(line)
        context_list = []
        entity_list = []
        for message in dialog['dialog']:
            role = message['role']
            text = message['text']
            # mask_text = message['text_template']
            entity_turn = message['entity']
            item_turn = message['item']
            dialog_turn_id = str(dialog['dialog_id']) + '_' + str(message['turn_id'])
            
            if dialog_turn_id in train_data_id_set:
                data = {
                    'dialog_id': dialog['dialog_id'],
                    'turn_id': message['turn_id'],
                    'context': copy(context_list),
                    'entity': copy(entity_list),
                    'rec': copy(item_turn),
                    'resp': text,
                }
                train_w.write(json.dumps(data, ensure_ascii=False) + '\n')
                
            elif dialog_turn_id in valid_data_id_set:
                data = {
                    'dialog_id': dialog['dialog_id'],
                    'turn_id': message['turn_id'],
                    'context': copy(context_list),
                    'entity': copy(entity_list),
                    'rec': copy(item_turn),
                    'resp': text,
                }
                valid_w.write(json.dumps(data, ensure_ascii=False) + '\n')
            
            elif dialog_turn_id in test_data_id_set:
                data = {
                    'dialog_id': dialog['dialog_id'],
                    'turn_id': message['turn_id'],
                    'context': copy(context_list),
                    'entity': copy(entity_list),
                    'rec': copy(item_turn),
                    'resp': text,
                }
                test_w.write(json.dumps(data, ensure_ascii=False) + '\n')
            
            context_list.append(text)
            entity_list.extend(entity_turn)
            
with open('train_data_processed.jsonl', 'r', encoding="utf-8") as f:
    lines = f.readlines()
    print('train:', len(lines))

with open('valid_data_processed.jsonl', 'r', encoding="utf-8") as f:
    lines = f.readlines()
    print('valid:', len(lines))

with open('test_data_processed.jsonl', 'r', encoding="utf-8") as f:
    lines = f.readlines()
    print('test:', len(lines))