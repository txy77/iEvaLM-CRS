import json
import argparse
import re
import os
from tqdm import tqdm

import sys
sys.path.append("..")

from src.model.metric import RecMetric

datasets = ['redial_eval', 'opendialkg_eval']
models = ['kbrd', 'barcor', 'unicrs', 'chatgpt']


# compute rec recall
def rec_eval(turn_num, mode):

    for dataset in datasets:
        
        with open(f"../data/{dataset.split('_')[0]}/entity2id.json", 'r', encoding="utf-8") as f:
            entity2id = json.load(f)
        
        for model in models:
            metric = RecMetric([1, 10, 25, 50])
            persuatiness = 0
            save_path = f"../save_{turn_num}/{mode}/{model}/{dataset}" # data loaded path
            result_path = f"../save_{turn_num}/result/{mode}/{model}"
            os.makedirs(result_path, exist_ok=True)
            if os.path.exists(save_path) and len(os.listdir(save_path)) > 0:
                path_list = os.listdir(save_path)
                print(f"turn_num: {turn_num}, mode: {mode} model: {model} dataset: {dataset}", len(path_list))
                
                for path in tqdm(path_list):
                    with open(f"{save_path}/{path}", 'r', encoding="utf-8") as f:
                        data = json.load(f)
                        if mode == 'chat':
                            persuasiveness_score = data['persuasiveness_score']
                            persuatiness += float(persuasiveness_score)
                        PE_dialog = data['simulator_dialog']
                        rec_label = data['rec']
                        rec_label = [entity2id[rec] for rec in rec_label if rec in entity2id]
                        contexts = PE_dialog['context']
                        for context in contexts[::-1]:
                            if 'rec_items' in context:
                                rec_items = context['rec_items']
                                metric.evaluate(rec_items, rec_label)
                                break
                    
                report = metric.report()
                
                print('r1:', f"{report['recall@1']:.3f}", 'r10:', f"{report['recall@10']:.3f}", 'r25:', f"{report['recall@25']:.3f}", 'r50:', f"{report['recall@50']:.3f}", 'count:', report['count'])
                if mode == 'chat':
                    persuativeness_score = persuatiness / len(path_list)
                    print(f"{persuativeness_score:.3f}")
                    report['persuativeness'] = persuativeness_score
                
                with open(f"{result_path}/{dataset}.json", 'w', encoding="utf-8") as w:
                    w.write(json.dumps(report))
                        
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--turn_num', type=int)
    parser.add_argument('--mode', type=str)
    args = parser.parse_args()
    rec_eval(args.turn_num, args.mode)