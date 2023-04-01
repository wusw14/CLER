import os
import time
import warnings
import sys 

dataset = sys.argv[1]

blocker = 'sentbert'
# K = int(sys.argv[3])
K = 10
size = int(sys.argv[3])
for run_id in range(5): 
    cmd = """python -u test_dynamic.py --fp16 --total_budget %d --gpu %s --dataset %s --topK %d --run_id %d --ckpt_type %s """ % (size, sys.argv[2], dataset, K, run_id, sys.argv[4])
    # cmd = """python -u test_magellan.py --fp16 --total_budget %d --gpu %s --dataset %s --topK %d --run_id %d""" % (size, sys.argv[2], dataset, K, run_id)
    print(cmd)
    os.system(cmd)