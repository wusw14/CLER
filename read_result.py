import numpy as np 
from collections import defaultdict
import sys 
import os 

def read_log(logdir, datalist, budget, K):
    result_BK = defaultdict(list)
    result_f1 = defaultdict(list)
    for data in datalist:
        filepath = os.path.join(logdir, '%s%s.log' %(data, budget[:-2]))
        if os.path.exists(filepath) == False:
            continue
        f = open(filepath, 'r')
        while True:
            line = f.readline()
            if len(line) <= 0:
                break 
            if ('%s Final Test BK Recall %s' %(K, budget)) in line:
                result_BK[data].append([float(v) for v in line.strip().split()[-6:]])
            elif ('%s Final Test F1 %s' %(K, budget)) in line:
                result_f1[data].append([float(v) for v in line.strip().split()[-6:]])
        f.close()
    return result_BK, result_f1

# K = sys.argv[1]
K = str(10)
num_iter = sys.argv[1]
K_list = [1,5,10,20,50,100]
for budget in [500, 1000, 1500, 2000]:
    datalist = ['AG', 'DA', 'DS', 'FZ', 'WA', 'AB', 'M']
    indexK = [10, 5, 20, 5, 10, 10, 20]
    result_BK, result_f1 = read_log('logs', datalist, str(budget), K)
    result = []
    for i, data in enumerate(datalist):
        try:
            if len(result_f1[data]) == 0:
                continue
            index = K_list.index(indexK[i])
            tmp = result_f1[data][-5:]
            tmp = sorted(tmp)
            avg = list(np.nanmean(tmp, 0) * 100)
            result.append(str(round(avg[index], 2)))
        except:
            result.append(' ')
    print(' '.join(result))