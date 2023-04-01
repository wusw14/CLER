import os
import argparse
import json
import sys
import torch
import numpy as np
import pandas as pd 
import random
import warnings

from utils import *
from dataset import GTDatasetWithLabel
from runner import train

warnings.filterwarnings('ignore')

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="../data4/")
    parser.add_argument("--dataset", type=str, default="wdc/shoes")
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--add_token", type=bool, default=True)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--finetuning", dest="finetuning", action="store_true")
    parser.add_argument("--save_model", dest="save_model", action="store_true")
    parser.add_argument("--logdir", type=str, default="checkpoints/")
    parser.add_argument("--CLlogdir", type=str, default="CL-sep-sup_0104")
    parser.add_argument("--lm", type=str, default='roberta')
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    parser.add_argument("--total_budget", type=int, default=500)
    parser.add_argument("--warmup_budget", type=int, default=400)
    parser.add_argument("--active_budget", type=int, default=100)
    parser.add_argument("--warmup_epochs", type=int, default=20)
    parser.add_argument("--topK", type=int, default=5)
    parser.add_argument("--balance", type=bool, default=False)
    parser.add_argument("--valid_size", type=int, default=200)
    parser.add_argument("--validation_with_pseudo", type=bool, default=False)
    
    
    hp = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = hp.gpu

    dataset = hp.dataset
    dataset_dict = {
        'AG': 'Amazon-Google',\
        'BR': 'BeerAdvo-RateBeer',\
        'DA': 'DBLP-ACM',\
        'DS': 'DBLP-Scholar',\
        'FZ': 'Fodors-Zagats',\
        'IA': 'iTunes-Amazon',\
        'WA': 'Walmart-Amazon',\
        'AB': 'Abt-Buy'
    }
    dataset = dataset_dict.get(dataset, dataset)

    if 'wdc' in dataset:
        hp.path = "../../data4"
    else:
        hp.path = "../../data4/ER-Magellan"
        if 'Abt' in dataset:
            hp.dataset = os.path.join("Textual", dataset)
        else:
            hp.dataset = os.path.join("Structured", dataset)    

    # set seeds
    seed = hp.run_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    topkA, distA, topkB, distB, distAll = get_topK_sim(hp)
    candidates_dict = filter_test(topkA, hp)
    num1 = 0
    for key, values in candidates_dict.items():
        num1 += len(values)
    print('Candidates after blocking: ', num1)
    candidates_dict = filter_by_rule(candidates_dict, hp)
    num2 = 0
    for key, values in candidates_dict.items():
        num2 += len(values)
    print('Candidates after checking by rules: ', num2)
    print(num1, num2)
    # samples = []
    # for idxA, idxB_list in enumerate(topkA):
    #     for idxB in idxB_list:
    #         samples.append([idxA, idxB])
    # samples = np.array(samples)
    # df1 = pd.DataFrame({'ltable_id': samples[:,0], 'rtable_id': samples[:,1], 'flag': [1]*len(samples)})
    # df1 = df1.set_index(['ltable_id', 'rtable_id'])
    # path = os.path.join(hp.path, hp.dataset)
    # if 'wdc' in path:
    #     df = pd.read_csv(os.path.join(path, 'train.csv.xlarge'), sep = ',')
    #     df = df.append(pd.read_csv(os.path.join(path, 'valid.csv.xlarge'), sep = ','))
    # else:
    #     df = pd.read_csv(os.path.join(path, 'train.csv'), sep = ',')
    #     df = df.append(pd.read_csv(os.path.join(path, 'valid.csv'), sep = ','))
    # df = df.append(pd.read_csv(os.path.join(path, 'test.csv'), sep = ','))
    # df = df.set_index(['ltable_id', 'rtable_id'])
    # intersection_num = len(df1.join(df, how = 'inner'))
    # print('Intersection num: %d, Samples after blocking: %d, Samples by Magellan %d.' %(intersection_num, len(df1), len(df)))
    # print(intersection_num, len(df1), len(df))
