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
                result_BK[data].append(float(line.strip().split()[-1]))
            elif ('%s Final Test F1 %s' %(K, budget)) in line:
                result_f1[data].append(float(line.strip().split()[-1]))
        f.close()
    return result_BK, result_f1

K = sys.argv[1]
for budget in [500, 1000, 1500, 2000]:
    datalist = ['AG', 'DA', 'DS', 'FZ', 'WA', 'AB', 'M']
    result_BK, result_f1 = read_log('test0329', datalist, str(budget), K)

    result = []
    for i, data in enumerate(datalist):
        try:
            tmp = result_f1[data]
            tmp = sorted(tmp)
            tmp = np.nanmean(tmp[1:-1])
            # tmp = np.nanmean(tmp)
            result.append(str(round(100*float(tmp), 2)))
        except:
            result.append(' ')
    print(' '.join(result))

print()

for budget in [500, 1000, 1500, 2000]:
    datalist = ['AG', 'DA', 'DS', 'FZ', 'WA', 'AB', 'M']
    result_BK, result_f1 = read_log('test0329', datalist, str(budget), K)    
    result = []
    for i, data in enumerate(datalist):
        try:
            tmp = result_BK[data]
            tmp = sorted(tmp)
            tmp = np.nanmean(tmp[1:-1])
            # tmp = np.nanmean(tmp)
            result.append(str(round(100*float(tmp), 2)))
        except:
            result.append(' ')
    print(' '.join(result))    