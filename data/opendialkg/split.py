import json
import random
from copy import copy
from tqdm import tqdm


def process_data(data_file_path):
    global dialog_id

    dialog_list = []
    data_list = []
    with open(data_file_path, encoding='utf-8') as f:
        for line in tqdm(f):
            line = json.loads(line)
            context_text_list = []
            context_text_template_list = []
            context_entity_list = []
            context_item_list = []
            turn_id = 0

            dialog = {'dialog_id': dialog_id, 'dialog': []}

            for turn in line:
                turn['turn_id'] = turn_id
                dialog['dialog'].append(turn)

                text = turn['text']
                text_template = turn['text_template']
                role = turn['role']
                # item = [title2id[title] for title in turn['item']]
                item = turn['item']
                entity = turn['entity']

                if role == 'assistant':
                    flag = True
                    if len(context_text_list) == 0:
                        context_text_list.append('')
                        turn_id += 1
                        flag = False

                    if len(item) > 0 and flag is True:
                        data = {
                            'dialog_id': dialog_id,
                            'turn_id': turn_id,
                            'context': copy(context_text_list),
                            'context_template': copy(context_text_template_list),
                            'context_entity': copy(context_entity_list),
                            'context_item': copy(context_item_list),
                            'resp': text,
                            'item': item
                        }
                        data_list.append(data)
                        # out_file.write(json.dumps(data, ensure_ascii=False) + '\n')

                context_text_list.append(text)
                context_text_template_list.append(text_template)
                context_entity_list.extend(entity)
                context_item_list.extend(item)
                turn_id += 1

            dialog_id += 1
            dialog_list.append(dialog)

    return data_list, dialog_list


if __name__ == '__main__':
    dialog_id = 0

    random.seed(42)

    with open('id2info.json', encoding='utf-8') as f:
        id2info = json.load(f)
    with open('title2id.json', encoding='utf-8') as f:
        title2id = json.load(f)

    data_file_path = 'dialog_movie.jsonl'
    movie_data_list, movie_dialog_list = process_data(data_file_path)

    data_file_path = 'dialog_Books.jsonl'
    book_data_list, book_dialog_list = process_data(data_file_path)

    all_dialog_list = movie_dialog_list + book_dialog_list
    with open('data.jsonl', 'w', encoding='utf-8') as f:
        for dialog in all_dialog_list:
            f.write(json.dumps(dialog, ensure_ascii=False) + '\n')

    all_data_list = movie_data_list + book_data_list
    test_data_list = random.sample(all_data_list, int(len(all_data_list) * 0.15))
    test_data_list = sorted(test_data_list, key=lambda x: x['dialog_id'])

    print(len(test_data_list))

    test_data_dialog_id_set = {f"{data['dialog_id']}_{data['turn_id']}" for data in test_data_list}
    rest_data_list = []
    for data in all_data_list:
        if f"{data['dialog_id']}_{data['turn_id']}" not in test_data_dialog_id_set:
            rest_data_list.append(data)
    assert len(rest_data_list) + len(test_data_list) == len(all_data_list)

    random.shuffle(rest_data_list)
    train_data_list, valid_data_list = rest_data_list[int(0.15 * len(all_data_list)):], rest_data_list[:int(0.15 * len(all_data_list))]
    assert len(valid_data_list) == len(test_data_list)

    train_data_id_list = [f"{data['dialog_id']}_{data['turn_id']}" for data in train_data_list]
    with open('train_data_id.json', 'w', encoding='utf-8') as f:
        json.dump(train_data_id_list, f, ensure_ascii=False)

    valid_data_id_list = [f"{data['dialog_id']}_{data['turn_id']}" for data in valid_data_list]
    with open('valid_data_id.json', 'w', encoding='utf-8') as f:
        json.dump(valid_data_id_list, f, ensure_ascii=False)

    test_data_id_list = [f"{data['dialog_id']}_{data['turn_id']}" for data in test_data_list]
    with open('test_data_id.json', 'w', encoding='utf-8') as f:
        json.dump(test_data_id_list, f, ensure_ascii=False)

    # with open('train_data.jsonl', 'w', encoding='utf-8') as f:
    #     for data in train_data_list:
    #         f.write(json.dumps(data, ensure_ascii=False) + '\n')
    #
    # with open('valid_data.jsonl', 'w', encoding='utf-8') as f:
    #     for data in valid_data_list:
    #         f.write(json.dumps(data, ensure_ascii=False) + '\n')
    #
    # with open('test_data.jsonl', 'w', encoding='utf-8') as f:
    #     for data in test_data_list:
    #         f.write(json.dumps(data, ensure_ascii=False) + '\n')

    # cnt = 0
    # with open('../../model/chat/data/opendialkg/test_data_processed.jsonl', encoding='utf-8') as f:
    #     for line in f:
    #         data = json.loads(line)
    #         data_id = f"{data['dialog_id']}_{data['turn_id']}"
    #         if data_id not in test_data_dialog_id_set:
    #             print(data_id)
    #         cnt += 1
    # assert cnt == len(test_data_list)
