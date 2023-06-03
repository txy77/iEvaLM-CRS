import argparse
import copy
import json
import os
import random
import time
import typing
import warnings
import tiktoken

import openai
from loguru import logger
from tenacity import Retrying, retry_if_not_exception_type, _utils
from tenacity.stop import stop_base
from tenacity.wait import wait_base

import sys
sys.path.append("..")

from src.model.recommender import RECOMMENDER

warnings.filterwarnings('ignore')

def get_exist_dialog_set():
    exist_id_set = set()
    for file in os.listdir(save_dir):
        file_id = os.path.splitext(file)[0]
        exist_id_set.add(file_id)
    return exist_id_set


def my_before_sleep(retry_state):
    logger.debug(
        f'Retrying: attempt {retry_state.attempt_number} ended with: {retry_state.outcome}, spend {retry_state.seconds_since_start} in total')


class my_wait_exponential(wait_base):
    def __init__(
        self,
        multiplier: typing.Union[int, float] = 1,
        max: _utils.time_unit_type = _utils.MAX_WAIT,  # noqa
        exp_base: typing.Union[int, float] = 2,
        min: _utils.time_unit_type = 0,  # noqa
    ) -> None:
        self.multiplier = multiplier
        self.min = _utils.to_seconds(min)
        self.max = _utils.to_seconds(max)
        self.exp_base = exp_base

    def __call__(self, retry_state: "RetryCallState") -> float:
        if retry_state.outcome == openai.error.Timeout:
            return 0

        try:
            exp = self.exp_base ** (retry_state.attempt_number - 1)
            result = self.multiplier * exp
        except OverflowError:
            return self.max
        return max(max(0, self.min), min(result, self.max))


class my_stop_after_attempt(stop_base):
    """Stop when the previous attempt >= max_attempt."""

    def __init__(self, max_attempt_number: int) -> None:
        self.max_attempt_number = max_attempt_number

    def __call__(self, retry_state: "RetryCallState") -> bool:
        if retry_state.outcome == openai.error.Timeout:
            retry_state.attempt_number -= 1
        return retry_state.attempt_number >= self.max_attempt_number

def annotate_completion(prompt, logit_bias=None):
    if logit_bias is None:
        logit_bias = {}

    request_timeout = 20
    for attempt in Retrying(
            reraise=True,
            retry=retry_if_not_exception_type((openai.error.InvalidRequestError, openai.error.AuthenticationError)),
            wait=my_wait_exponential(min=1, max=60), stop=(my_stop_after_attempt(8))
    ):
        with attempt:
            response = openai.Completion.create(
                model='text-davinci-003', prompt=prompt, temperature=0, max_tokens=128, stop='Recommender',
                logit_bias=logit_bias,
                request_timeout=request_timeout,
            )['choices'][0]['text']
        request_timeout = min(300, request_timeout * 2)

    return response


def get_instruction(dataset):
    if dataset == 'redial_eval':
        item_with_year = True
        init_ask_instruction = '''To recommend me items that I will accept, you can choose one of the following options.
A: ask my preference for genre
B: ask my preference for actor
C: ask my preference for director
D: I can directly give recommendations
Please enter the option character. Please only response a character.'''
        ask_instruction = '''To recommend me items that I will accept, you can choose one of the following options.
A: ask my preference for genre
B: ask my preference for actor
C: ask my preference for director
D: I can directly give recommendations
You have selected {}, do not repeat them. Please enter the option character.'''
        option2attr = {
            'A': 'genre',
            'B': 'star',
            'C': 'director',
            'D': 'recommend'
        }
        option2temaplte = {
            'A': 'Which genre do you like?',
            'B': 'Which star do you like?',
            'C': 'Which director do you like?',
        }
    elif dataset == 'opendialkg_eval':
        item_with_year = False
        init_ask_instruction = '''To recommend me items that I will accept, you can choose one of the following options.
A: ask my preference for genre
B: ask my preference for actor
C: ask my preference for director
D: ask my preference for writer
E: I can directly give recommendations
Please enter the option character. Please only response a character.'''
        ask_instruction = '''To recommend me items that I will accept, you can choose one of the following options.
A: ask my preference for genre
B: ask my preference for actor
C: ask my preference for director
D: ask my preference for writer
E: I can directly give recommendations
You have selected {}, do not repeat them. Please enter the option character.'''
        option2attr = {
            'A': 'genre',
            'B': 'actor',
            'C': 'director',
            'D': 'writer',
            'E': 'recommend'
        }
        option2temaplte = {
            'A': 'Which genre do you like?',
            'B': 'Which actor do you like?',
            'C': 'Which director do you like?',
            'D': 'Which writer do you like?',
        }
    else:
        raise Exception('do not support this dataset')

    if item_with_year is True:
        rec_instruction = 'Please give me 10 recommendations according to my preference (Format: no. title (year if exists). No other things except the movie list in your response).'
    else:
        rec_instruction = 'Please give me 10 recommendations according to my preference (Format: no. title. No other things except the item list in your response). You can recommend mentioned items in our dialog.'

    return init_ask_instruction, ask_instruction, rec_instruction, option2attr, option2temaplte


def get_model_args(model_name):
    if model_name == 'kbrd':
        args_dict = {
            'debug': args.debug, 'kg_dataset': args.kg_dataset, 'hidden_size': args.hidden_size,
            'entity_hidden_size': args.entity_hidden_size, 'num_bases': args.num_bases,
            'rec_model': args.rec_model, 'conv_model': args.conv_model,
            'context_max_length': args.context_max_length, 'entity_max_length': args.entity_max_length, 'tokenizer_path': args.tokenizer_path,
            'encoder_layers': args.encoder_layers, 'decoder_layers': args.decoder_layers, 'text_hidden_size': args.text_hidden_size,
            'attn_head': args.attn_head, 'resp_max_length': args.resp_max_length,
            'seed':args.seed
        }
    elif model_name == 'barcor':
        args_dict = {
            'debug': args.debug, 'kg_dataset': args.kg_dataset, 'rec_model': args.rec_model, 'conv_model': args.conv_model, 'context_max_length': args.context_max_length,
            'resp_max_length': args.resp_max_length, 'tokenizer_path': args.tokenizer_path, 'seed': args.seed
        }
    elif model_name == 'unicrs':
        args_dict = {
            'debug': args.debug, 'seed': args.seed, 'kg_dataset': args.kg_dataset, 'tokenizer_path': args.tokenizer_path,
            'context_max_length': args.context_max_length, 'entity_max_length': args.entity_max_length, 'resp_max_length': args.resp_max_length,
            'text_tokenizer_path': args.text_tokenizer_path,
            'rec_model': args.rec_model, 'conv_model': args.conv_model, 'model': args.model, 'num_bases': args.num_bases, 'text_encoder': args.text_encoder
        }
    elif model_name == 'chatgpt':
        args_dict = {
            'seed': args.seed, 'debug': args.debug, 'kg_dataset': args.kg_dataset
        }
    
    return args_dict


if __name__ == '__main__':
    local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key')
    parser.add_argument('--dataset', type=str, choices=['redial_eval', 'opendialkg_eval'])
    parser.add_argument('--turn_num', type=int, default=5)
    parser.add_argument('--crs_model', type=str, choices=['kbrd', 'barcor', 'unicrs', 'chatgpt'])
    
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--kg_dataset', type=str, choices=['redial', 'opendialkg'])
    
    # model_detailed
    parser.add_argument('--hidden_size', type=int)
    parser.add_argument('--entity_hidden_size', type=int)
    parser.add_argument('--num_bases', type=int, default=8)
    parser.add_argument('--context_max_length', type=int)
    parser.add_argument('--entity_max_length', type=int)
    
    # model
    parser.add_argument('--rec_model', type=str)
    parser.add_argument('--conv_model', type=str)
    
    # conv
    parser.add_argument('--tokenizer_path', type=str)
    parser.add_argument('--encoder_layers', type=int)
    parser.add_argument('--decoder_layers', type=int)
    parser.add_argument('--text_hidden_size', type=int)
    parser.add_argument('--attn_head', type=int)
    parser.add_argument('--resp_max_length', type=int)
    
    # prompt
    parser.add_argument('--model', type=str)
    parser.add_argument('--text_tokenizer_path', type=str)
    parser.add_argument('--text_encoder', type=str)

    args = parser.parse_args()
    openai.api_key = args.api_key
    save_dir = f'../save_{args.turn_num}/ask/{args.crs_model}/{args.dataset}'
    os.makedirs(save_dir, exist_ok=True)

    random.seed(args.seed)

    # recommender
    recommendation_template = "I would recommend the following items:\n\n{}"

    # recommender
    model_args = get_model_args(args.crs_model)
    recommender = RECOMMENDER(crs_model=args.crs_model, **model_args)

    # seeker
    init_ask_instruction, ask_instruction, rec_instruction, option2attr, option2template = get_instruction(args.dataset)
    options = list(option2attr.keys())

    # scorer
    persuasiveness_template = '''Does the explanation make you want to accept the recommendation? Please give your score.
If mention one of [{}], give 2.
Else if you think recommended items are worse than [{}], give 0.
Else if you think recommended items are comparable to [{}] according to the explanation, give 1.
Else if you think recommended items are better than [{}] according to the explanation, give 2.
Only answer the score number.'''
    encoding = tiktoken.encoding_for_model("text-davinci-003")
    logit_bias = {encoding.encode(str(score))[0]: 10 for score in range(3)}

    with open(f'../data/{args.kg_dataset}/entity2id.json', 'r', encoding="utf-8") as f:
        entity2id = json.load(f)
    id2entity = {}
    for k, v in entity2id.items():
        id2entity[int(v)] = k
    entity_list = list(entity2id.keys())
    
    name2id = {}
    
    with open(f'../data/{args.kg_dataset}/id2info.json', 'r', encoding="utf-8") as f:
        id2info = json.load(f)

    for k, v in id2info.items():
        name2id[v['name']] = k
    
    dialog_id2data = {}
    with open(f'../data/{args.dataset}/test_data_processed.jsonl', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            dialog_id = str(line['dialog_id']) + '_' + str(line['turn_id'])
            dialog_id2data[dialog_id] = line

    dialog_id_set = set(dialog_id2data.keys()) - get_exist_dialog_set()
    while len(dialog_id_set) > 0:
        
        print(len(dialog_id_set))
        dialog_id = random.choice(tuple(dialog_id_set))

        data = dialog_id2data[dialog_id]
        conv_dict = copy.deepcopy(data)  # for model
        goal_item_list = [f'"{item}"' for item in conv_dict['rec']]
        goal_item_str = ', '.join(goal_item_list)
        rec_labels = [name2id[rec] for rec in data['rec']]

        context_dict = []  # for save
        for i, text in enumerate(conv_dict['context']):
            if len(text) == 0:
                continue
            if i % 2 == 0:
                role_str = 'user'
            else:
                role_str = 'assistant'
            context_dict.append({
                'role': role_str,
                'content': text
            })

        # dialog state
        rec_success = False
        asked_options = []
        option2index = {
            'A': 0,
            'B': 1,
            'C': 2,
            'D': 3,
            'E': 4
        }
        if args.kg_dataset == 'redial':
            state = [0, 0, 0, 0]
        elif args.kg_dataset == 'opendialkg':
            state = [0, 0, 0, 0, 0]

        for i in range(0, args.turn_num):
            # seeker
            # choose option
            
            if args.crs_model == 'chatgpt':
                conv_dict['context'].append(init_ask_instruction)

            # recommender
            # options (list of str): available options, generate one of them
            gen_inputs, recommender_text = recommender.get_conv(conv_dict)
            if args.crs_model != 'chatgpt':
                recommender_choose = recommender.get_choice(gen_inputs, options, state)
            else:
                recommender_choose = recommender.get_choice(gen_inputs, options, state, conv_dict)
            selected_option = recommender_choose 

            if selected_option == options[-1]:  # choose to rec
                # recommender
                rec_items, rec_truth = recommender.get_rec(conv_dict)
                rec_pred = rec_items[0]

                rec_items_str = ''
                for j, rec_item in enumerate(rec_pred[:50]):
                    rec_items_str += f"{i + 1}: {id2entity[rec_item]}\n"
                recommender_text = recommendation_template.format(rec_items_str)

                # judge whether success
                for rec_label in rec_truth:
                    if rec_label in rec_pred:
                        rec_success = True
                        break

                context_dict.append({
                    'role': 'assistant',
                    'content': recommender_text,
                    'rec_items': rec_pred,
                    'rec_success': rec_success,
                    'option': selected_option
                })
                conv_dict['context'].append(recommender_text)

                # seeker
                if rec_success is True:
                    seeker_text = "That's perfect, thank you!"
                else:
                    seeker_text = "I don't like them."

                context_dict.append({
                    'role': 'user',
                    'content': seeker_text
                })
                conv_dict['context'].append(seeker_text)

            else:  # choose to ask
                recommender_text = option2template[selected_option]
                context_dict.append({
                    'role': 'assistant',
                    'content': recommender_text,
                    'option': selected_option
                })
                conv_dict['context'].append(recommender_text)
            
                # seeker
                ask_attr = option2attr[selected_option]
                
                # update state
                state[option2index[selected_option]] = -1e5
                
                ans_attr_list = []
                for label_id in rec_labels:
                    if str(label_id) in id2info and ask_attr in id2info[str(label_id)]:
                        ans_attr_list.extend(id2info[str(label_id)][ask_attr])
                if len(ans_attr_list) > 0:
                    seeker_text = ', '.join(list(set(ans_attr_list)))
                else:
                    seeker_text = 'Sorry, no information about this, please choose another option.'

                context_dict.append({
                    'role': 'user',
                    'content': seeker_text,
                    'entity': ans_attr_list,
                })
                conv_dict['context'].append(seeker_text)
                conv_dict['entity'] += ans_attr_list

            if rec_success is True:
                break

        # score persuasiveness
        # seeker_prompt = ''
        # for turn_dict in context_dict:
        #     if turn_dict['role'] == 'user':
        #         role_str = 'Seeker'
        #     else:
        #         role_str = 'Recommender'
        #     seeker_prompt += f'{role_str}: {turn_dict["content"]}\n'
        # persuasiveness_str = persuasiveness_template.format(goal_item_str, goal_item_str, goal_item_str,
        #                                                          goal_item_str)
        # prompt_str_for_persuasiveness = seeker_prompt + persuasiveness_str
        # prompt_str_for_persuasiveness += '\nSeeker:'

        # persuasiveness_score = annotate_completion(prompt_str_for_persuasiveness, logit_bias).strip()

        # save
        conv_dict['context'] = context_dict
        data['simulator_dialog'] = conv_dict

        with open(f'{save_dir}/{dialog_id}.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        dialog_id_set -= get_exist_dialog_set()
            
