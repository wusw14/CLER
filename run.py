import os
import time
import warnings
import sys 

dataset = sys.argv[1]

blocker = 'sentbert'
# K = int(sys.argv[3])
K = 10
size = int(sys.argv[3])
for run_id in range(1,5): 
    cmd = """python -u train.py --fp16 --lr 1e-5 --total_budget %d --gpu %s --dataset %s --run_id %d  --batch_size 64 --save_model --blocker %s --topK %d --active_budget 50 --logdir 0208_ckpt --valid_size 100 """ % (size, sys.argv[2], dataset, run_id, blocker, K)
    # cmd += ' >> logs/%s/%s.log' %(blocker, dataset)
    print(cmd)
    os.system(cmd)

# for run_id in range(5):
    # cmd = """python -u test.py --fp16 --total_budget %d --gpu %s --dataset %s --topK %d --run_id %d""" % (size, sys.argv[2], dataset, K, run_id)
    # print(cmd)
    # os.system(cmd)