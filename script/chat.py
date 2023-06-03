import argparse
import copy
import json
import os
import re
import random
import time
import typing
import warnings
import tiktoken

import openai
import nltk
from loguru import logger
from thefuzz import fuzz
from tenacity import Retrying, retry_if_not_exception_type, _utils
from tenacity.stop import stop_base
from tenacity.wait import wait_base

import sys
sys.path.append("..")

from src.model.utils import get_entity
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
    if dataset.startswith('redial'):
        item_with_year = True
    elif dataset.startswith('opendialkg'):
        item_with_year = False

    if item_with_year is True:
        recommender_instruction = '''You are a recommender chatting with the user to provide recommendation. You must follow the instructions below during chat.
If you do not have enough information about user preference, you should ask the user for his preference.
If you have enough information about user preference, you can give recommendation.'''
        seeker_instruction_template = '''You are a seeker chatting with a recommender for recommendation. Your target items: {}. You must follow the instructions below during chat.
If the recommender recommend {}, you should accept.
If the recommender recommend other items, you should refuse them and provide the information about {}. You should never directly tell the target item title.
If the recommender asks for your preference, you should provide the information about {}. You should never directly tell the target item title.

'''
    else:
        recommender_instruction = '''You are a recommender chatting with the user to provide recommendation. You must follow the instructions below during chat.
If you do not have enough information about user preference, you should ask the user for his preference.
If you have enough information about user preference, you can give recommendation.'''
        
        seeker_instruction_template = '''You are a seeker chatting with a recommender for recommendation. Your target items: {}. You must follow the instructions below during chat.
If the recommender recommend {}, you should accept.
If the recommender recommend other items, you should refuse them and provide the information about {}. You should never directly tell the target item title.
If the recommender asks for your preference, you should provide the information about {}. You should never directly tell the target item title.

'''

    return recommender_instruction, seeker_instruction_template

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
    else:
        raise Exception('do not support this model')
    
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
    save_dir = f'../save_{args.turn_num}/chat/{args.crs_model}/{args.dataset}'
    os.makedirs(save_dir, exist_ok=True)
    
    random.seed(args.seed)
    
    encoding = tiktoken.encoding_for_model("text-davinci-003")
    logit_bias = {encoding.encode(str(score))[0]: 10 for score in range(3)}
    
    # recommender
    model_args = get_model_args(args.crs_model)
    recommender = RECOMMENDER(crs_model=args.crs_model, **model_args)

    recommender_instruction, seeker_instruction_template = get_instruction(args.dataset)

    with open(f'../data/{args.kg_dataset}/entity2id.json', 'r', encoding="utf-8") as f:
        entity2id = json.load(f)
    
    id2entity = {}
    for k, v in entity2id.items():
        id2entity[int(v)] = k
    entity_list = list(entity2id.keys())
    
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
        conv_dict = copy.deepcopy(data) # for model
        context = conv_dict['context']

        goal_item_list = [f'"{item}"' for item in conv_dict['rec']]
        goal_item_str = ', '.join(goal_item_list)
        seeker_prompt = seeker_instruction_template.format(goal_item_str, goal_item_str, goal_item_str, goal_item_str)
        context_dict = [] # for save

        for i, text in enumerate(context):
            if len(text) == 0:
                continue
            if i % 2 == 0:
                role_str = 'user'
                seeker_prompt += f'Seeker: {text}\n'
            else:
                role_str = 'assistant'
                seeker_prompt += f'Recommender: {text}\n'
            context_dict.append({
                'role': role_str,
                'content': text
            })
            
        rec_success = False
        recommendation_template = "I would recommend the following items: {}:"

        for i in range(0, args.turn_num):
            # rec only
            rec_items, rec_labels = recommender.get_rec(conv_dict)
            
            for rec_label in rec_labels:
                if rec_label in rec_items[0]:
                    rec_success = True
                    break
            # rec only
            _, recommender_text = recommender.get_conv(conv_dict)
            
            # barcor
            if args.crs_model == 'barcor':
                recommender_text = recommender_text.lstrip('System;:')
                recommender_text = recommender_text.strip()
            
            # unicrs
            if args.crs_model == 'unicrs':
                if args.dataset.startswith('redial'):
                    movie_token = '<movie>'
                else:
                    movie_token = '<mask>'
                recommender_text = recommender_text[recommender_text.rfind('System:') + len('System:') + 1 : ]
                for i in range(str.count(recommender_text, movie_token)):
                    recommender_text = recommender_text.replace(movie_token, id2entity[rec_items[i]], 1)
                recommender_text = recommender_text.strip()
            
            if rec_success == True or i == args.turn_num - 1:
                rec_items_str = ''
                for j, rec_item in enumerate(rec_items[0][:50]):
                    rec_items_str += f"{j+1}: {id2entity[rec_item]}\n"
                recommendation_template = recommendation_template.format(rec_items_str)
                recommender_text = recommendation_template + recommender_text
            
            # public 
            recommender_resp_entity = get_entity(recommender_text, entity_list)
            
            conv_dict['context'].append(recommender_text)
            conv_dict['entity'] += recommender_resp_entity
            conv_dict['entity'] = list(set(conv_dict['entity']))
            
            context_dict.append({
                'role': 'assistant',
                'content': recommender_text,
                'entity': recommender_resp_entity,
                'rec_items': rec_items[0],
                'rec_success': rec_success
            })
            
            seeker_prompt += f'Recommender: {recommender_text}\nSeeker:'
            
            # seeker
            year_pattern = re.compile(r'\(\d+\)')
            goal_item_no_year_list = [year_pattern.sub('', rec_item).strip() for rec_item in goal_item_list]
            seeker_text = annotate_completion(seeker_prompt).strip()
            
            seeker_response_no_movie_list = []
            for sent in nltk.sent_tokenize(seeker_text):
                use_sent = True
                for rec_item_str in goal_item_list + goal_item_no_year_list:
                    if fuzz.partial_ratio(rec_item_str.lower(), sent.lower()) > 90:
                        use_sent = False
                        break
                if use_sent is True:
                    seeker_response_no_movie_list.append(sent)
            seeker_response = ' '.join(seeker_response_no_movie_list)
            if not rec_success:
                seeker_response = 'Sorry, ' + seeker_response
            seeker_prompt += f' {seeker_response}\n'
            
            # public
            seeker_resp_entity = get_entity(seeker_text, entity_list)
            
            context_dict.append({
                'role': 'user',
                'content': seeker_text,
                'entity': seeker_resp_entity,
            })
            
            conv_dict['context'].append(seeker_text)
            conv_dict['entity'] += seeker_resp_entity
            conv_dict['entity'] = list(set(conv_dict['entity']))
            
            if rec_success:
                break
        
        # score persuativeness
        conv_dict['context'] = context_dict
        data['simulator_dialog'] = conv_dict
        persuasiveness_template = '''Does the explanation make you want to accept the recommendation? Please give your score.
If mention one of [{}], give 2.
Else if you think recommended items are worse than [{}], give 0.
Else if you think recommended items are comparable to [{}] according to the explanation, give 1.
Else if you think recommended items are better than [{}] according to the explanation, give 2.
Only answer the score number.'''

        persuasiveness_template = persuasiveness_template.format(goal_item_str, goal_item_str, goal_item_str, goal_item_str)
        prompt_str_for_persuasiveness = seeker_prompt + persuasiveness_template
        prompt_str_for_persuasiveness += "\nSeeker:"
        persuasiveness_score = annotate_completion(prompt_str_for_persuasiveness, logit_bias).strip()
        
        data['persuasiveness_score'] = persuasiveness_score
        
        # save
        with open(f'{save_dir}/{dialog_id}.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        dialog_id_set -= get_exist_dialog_set()
        