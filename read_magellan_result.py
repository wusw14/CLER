import numpy as np 
from collections import defaultdict
import sys 
import os 

def read_log(logdir, datalist, budget, K):
    result_avg = {}
    result_std = {}
    for data in datalist:
        filepath = os.path.join(logdir, '%s%s.log' %(data, str(budget)[:-2]))
        if os.path.exists(filepath) == False:
            continue
        f = open(filepath, 'r')
        while True:
            line = f.readline()
            if len(line) <= 0:
                break 
            if ('%s Final Test F1 %s' %(K, budget)) in line:
                result_avg[data] = str(float(line.strip().split()[-1]) * 100)
                result_std[data] = line.strip().split()[-1]
        f.close()
    return result_avg, result_std

K = sys.argv[1]
resutl1_list, resutl2_list = [], []
for budget in [500, 1000, 1500, 2000]:
    datalist = ['AG', 'DA', 'DS', 'FZ', 'WA', 'AB', 'M']
    result_avg, result_std = read_log('mag_logs', datalist, str(budget), K)

    result1, result2 = [], []
    for data in datalist:
        try:
            result1.append(result_avg.get(data, ''))
            result2.append(result_std.get(data, ''))
        except:
            result1.append(' ')
            result2.append(' ')
    resutl1_list.append(' '.join(result1))
    resutl2_list.append(' '.join(result2))

for v in resutl1_list:
    print(v)
print('='*20)
for v in resutl2_list:
    print(v)