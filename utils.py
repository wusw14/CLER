import pandas as pd
import numpy as np
import os 
import random 
import re
from sklearn.metrics import f1_score, precision_score, recall_score
from collections import defaultdict
import torch 
from dataset import AugDataset, SingleEntityDataset
from model import * 
import os 
import torch 
from sentence_transformers import util, SentenceTransformer
from torch.utils import data
from sklearn.cluster import MiniBatchKMeans

def select4BK(pseudo_MC_df):
    # ['ltable_id', 'rtable_id', 'avg_score', 'num_BK']
    df = pseudo_MC_df[pseudo_MC_df.avg_score > 0.5]
    # left max
    df1 = df.reset_index().groupby('ltable_id').apply(lambda x: x.sort_values(by = 'avg_score', ascending = False)['rtable_id'].values[0])
    df2 = df.reset_index().groupby('rtable_id').apply(lambda x: x.sort_values(by = 'avg_score', ascending = False)['ltable_id'].values[0])
    df1 = pd.DataFrame(df1).reset_index()
    df1.columns = ['ltable_id', 'rtable_id']
    df2 = pd.DataFrame(df2).reset_index()
    df2.columns = ['rtable_id', 'ltable_id']
    df3 = df1.append(df2)
    df3 = df3.drop_duplicates()
    df3['label'] = 1
    return df3.values 
    
def update_pseudoBK_df(pseudo_BK_df, blocker_pseudo_pos, blocker_pseudo_neg):
    pseudo_BK_df = None
    pseudo_new = np.concatenate([blocker_pseudo_pos, blocker_pseudo_neg], 0)
    df_new = pd.DataFrame({'ltable_id': pseudo_new[:,0], 'rtable_id': pseudo_new[:,1], 'label1': pseudo_new[:,2]})
    if pseudo_BK_df is None:
        pseudo_BK_df = pd.DataFrame(df_new)
        pseudo_BK_df = pseudo_BK_df.set_index(['ltable_id', 'rtable_id'])
        pseudo_BK_df.columns = ['label']
        pseudo_BK_df['numBK'] = 1
        pseudo_BK_df['iter'] = 1
        return pseudo_BK_df
    else:
        cur_iter = pseudo_BK_df['iter'].max() + 1
        df_new = df_new.set_index(['ltable_id', 'rtable_id'])
        df = pseudo_BK_df.join(df_new, how = 'outer')
        
        dforg = df[pd.isna(df.label1)][['label', 'numBK', 'iter']]
        dforg = dforg[dforg.iter >= cur_iter - 1]
        # dforg = dforg[df.numBK>1]
        
        dfnew = df[pd.isna(df.label)][['label1']]
        dfnew.columns = ['label']
        dfnew['numBK'] = 1
        dfnew['iter'] = cur_iter

        dfboth = df[(pd.isna(df.label)==False) & (pd.isna(df.label1)==False)]
        dfboth = dfboth[dfboth.label==dfboth.label1]
        dfboth = dfboth[['label', 'numBK']]
        dfboth['numBK'] = dfboth['numBK'] + 1
        dfboth['iter'] = cur_iter

        df_all = dforg.append(dfnew)
        df_all = df_all.append(dfboth)
    return df_all

def update_pseudoMC_df(pseudo_MC_df, pseudo_MC_cur):
    pseudo_MC_df = None
    if pseudo_MC_df is None:
        if pseudo_MC_cur is None:
            return None
        pseudo_MC_df = pd.DataFrame(pseudo_MC_cur)
        pseudo_MC_df.columns = ['avg_score']
        pseudo_MC_df['numMC'] = 1
        pseudo_MC_df['iter'] = 1
        return pseudo_MC_df
    else:
        cur_iter = pseudo_MC_df['iter'].max() + 1
        df = pseudo_MC_df.join(pseudo_MC_cur, how = 'outer')
        
        dforg = df[pd.isna(df.score)][['avg_score', 'numMC', 'iter']]
        dforg = dforg[dforg.iter >= cur_iter - 1]
        # dforg = dforg[df.numMC > 1]
        
        dfnew = df[pd.isna(df.avg_score)][['score']]
        dfnew.columns = ['avg_score']
        dfnew['numMC'] = 1
        dfnew['iter'] = cur_iter
        
        dfboth = df[(pd.isna(df.score)==False) & (pd.isna(df.avg_score)==False)]
        dfboth['flag'] = ((dfboth['avg_score'] - 0.5) * (dfboth['score'] - 0.5)) > 0
        dfboth = dfboth[dfboth.flag == 1]
        dfboth['avg_score'] = (dfboth['avg_score'] * dfboth['numMC'] + dfboth['score']) / (dfboth['numMC'] + 1)
        dfboth['numMC'] = dfboth['numMC'] + 1
        dfboth = dfboth[['avg_score', 'numMC']]
        df['iter'] = cur_iter
        
        df_all = dforg.append(dfnew)
        df_all = df_all.append(dfboth)
        return df_all

def evaluate_pseudo(data, annotated_df, hp):
    gt = load_gt(hp)
    data = np.array(data)
    df = pd.DataFrame({'ltable_id': data[:,0], 'rtable_id': data[:,1], 'pred': data[:,2]})
    df = df.set_index(['ltable_id', 'rtable_id'])
    if annotated_df is not None:
        df = df.join(annotated_df)
        df = df[pd.isna(df.label)]
        df = df[['pred']]
    df = df.join(gt)
    df = df.fillna(0)
    y_pred = np.array(df['pred'].values, int)
    y_true = np.array(df['label'].values, int)
    if len(data[data[:,2]==1]) == 0:
        pos_pre, pos_rec = 0, 0
    else:
        pos_pre = precision_score(y_true, y_pred, pos_label = 1)
        pos_rec = recall_score(y_true, y_pred, pos_label = 1)
    if len(data[data[:,2]==0]) == 0:
        neg_pre, neg_rec = 0, 0
    else:
        neg_pre = precision_score(y_true, y_pred, pos_label = 0)
        neg_rec = recall_score(y_true, y_pred, pos_label = 0)
    return pos_pre, pos_rec, neg_pre, neg_rec

def update_annotated_df(annotated_df, annotated_pos, annotated_neg):
    annotated = np.concatenate([annotated_pos, annotated_neg], 0)
    new_annotated_df = pd.DataFrame({'ltable_id': annotated[:,0], 'rtable_id': annotated[:,1], 'label': annotated[:,2]})
    new_annotated_df = new_annotated_df.set_index(['ltable_id', 'rtable_id'])
    if annotated_df is None:
        return new_annotated_df
    else:
        annotated_df = annotated_df.append(new_annotated_df)
        return annotated_df

def filter_test_from_candidates(topkA, annotated_df, test_idxs):
    ltable_id, rtable_id = [], []
    for idxA, idxB_list in enumerate(topkA):
        if test_idxs is not None and idxA in test_idxs:
            continue
        ltable_id.extend([idxA] * len(idxB_list))
        rtable_id.extend(list(idxB_list))
    cand_df = pd.DataFrame({'ltable_id': ltable_id, 'rtable_id': rtable_id, 'flag':[1]*len(ltable_id)})
    cand_df = cand_df.set_index(['ltable_id', 'rtable_id'])
    if annotated_df is not None:
        cand_df = cand_df.join(annotated_df)
        cand_df = cand_df[pd.isna(cand_df['label'])]
        cand_df = cand_df[['flag']]
    cand_df = cand_df.reset_index()
    return cand_df.values, cand_df[['ltable_id', 'rtable_id']].values

def update_annotated_dict(annotated_dict, annotated_pos, annotated_neg):
    for a, b, _ in annotated_pos:
        annotated_dict[a].append(b)
    for a, b, _ in annotated_neg:
        annotated_dict[a].append(b)
    return annotated_dict

def filter_from_candidates(topkA, annotated_dict, hp, filter_test = False):
    candidates_dict = defaultdict(list)
    for idxA, idxB_list in enumerate(topkA):
        cands = list(idxB_list)
        for v in annotated_dict.get(idxA, []):
            try:
                cands.remove(v)
            except:
                pass 
        candidates_dict[idxA] = cands
    if filter_test:
        test_data = pd.read_csv(os.path.join(hp.path, hp.dataset, 'test_'+hp.blocker_type+'.csv')).values 
        for a, b, _ in test_data:
            try:
                candidates_dict[a].remove(b)
            except:
                pass
    return candidates_dict

def load_candidates_from_train(filename):
    candidates_dict = defaultdict(list)
    samples = pd.read_csv(filename).values
    for a, b, _ in samples:
        candidates_dict[a].append(b)
    return candidates_dict

def get_scores(samples, all_pairs, scores):
    df = pd.DataFrame({'ltable_id': all_pairs[:,0], 'rtable_id': all_pairs[:, 1], 'score': scores})
    df1 = pd.DataFrame({'ltable_id': samples[:,0], 'rtable_id': samples[:, 1], 'flag': [1]*len(samples)})
    df = df.set_index(['ltable_id', 'rtable_id'])
    df1 = df1.set_index(['ltable_id', 'rtable_id'])
    df1 = df1.join(df)
    return df1['score'].values

def MC_self_check(pseudo_pos_matcher, pseudo_neg_matcher, pos_score, neg_score):
    def maxscore(dfsub):
        dfsub = dfsub.sort_values(by = 'score', ascending = False)
        num = len(dfsub)
        return [num, dfsub['rtable_id'].values[0], dfsub['score'].values[0]]
    def filter_unsure(x):
        if len(x) <= 4:
            return -1
        return float((x['score']>0.5).sum())/len(x)    
    print('='*50, pseudo_pos_matcher.shape, pseudo_neg_matcher.shape)
    if len(pseudo_pos_matcher) == 0 and len(pseudo_neg_matcher) == 0:
        return None
    elif len(pseudo_pos_matcher) == 0:
        df_neg = pd.DataFrame({'ltable_id': pseudo_neg_matcher[:,0], 'rtable_id': pseudo_neg_matcher[:,1], 'score': neg_score})
        df_neg = df_neg.set_index(['ltable_id', 'rtable_id'])
        return df_neg
    df_pos = pd.DataFrame({'ltable_id': pseudo_pos_matcher[:,0], 'rtable_id': pseudo_pos_matcher[:,1], 'score': pos_score})
    df_neg = pd.DataFrame({'ltable_id': pseudo_neg_matcher[:,0], 'rtable_id': pseudo_neg_matcher[:,1], 'score': neg_score})
    
    df_pos_summ = df_pos.groupby('ltable_id').apply(maxscore)
    print('='*20, 'pos', len(df_pos_summ))
    lids = np.reshape(np.array(list(df_pos_summ.index)), [-1, 1])
    values = np.array(list(df_pos_summ.values))
    pos_np = np.concatenate([lids, values], 1) # [N, 4]
    pos_np_single = pos_np[pos_np[:,1] == 1]
    df_pos_single = pd.DataFrame({'ltable_id': pos_np_single[:,0], 'rtable_id': pos_np_single[:,2], 'score': pos_np_single[:,3]})
    pos_np_multi = pos_np[pos_np[:,1] > 1]

    ltable_ids_checked = []
    pairs = []
    cnt = 0
    for lid in pos_np_multi[:,0]:
        flag = True
        rtable_ids = pseudo_pos_matcher[pseudo_pos_matcher[:,0]==lid][:,1]
        rids_intersec = None
        lids_union = set()
        for rid in rtable_ids:
            lids = pseudo_pos_matcher[pseudo_pos_matcher[:,1]==rid][:,0]
            lids_union = lids_union | set(lids)
        for lid1 in lids_union:
            rtable_ids = pseudo_pos_matcher[pseudo_pos_matcher[:,0]==lid1][:,1]
            if rids_intersec is None:
                rids_intersec = set(rtable_ids)
            else:
                rids_intersec = set(rtable_ids) & set(rids_intersec)
                if len(rids_intersec) == 0:
                    break
        if len(rids_intersec) > 0:
            for rid in rids_intersec:
                pairs.append([lid, rid])
            cnt += 1

    print('='*20, 'shared rtable_ids', cnt)
    if len(pairs) > 0:
        pairs = np.array(pairs)
        df_lid = pd.DataFrame({'ltable_id': pairs[:,0], 'rtable_id': pairs[:,1], 'flag': [1] * len(pairs)})
        df_lid = df_lid.set_index(['ltable_id', 'rtable_id'])
        df_pos = df_pos.set_index(['ltable_id', 'rtable_id'])
        df_pos = df_lid.join(df_pos, how = 'inner')
        df_pos = df_pos.reset_index()[['ltable_id', 'rtable_id', 'score']]
        df_pos = df_pos_single.append(df_pos)
        df_pos = df_pos.drop_duplicates()
        df = df_pos.append(df_neg)
    else:
        df = df_pos_single.append(df_neg)

    df = df.set_index(['ltable_id', 'rtable_id'])
    
    return df

def gen_pseudo_matcher(candidates_scores, candidated_pairs, annotated_scores, annotated_data):
    annotated_data = np.array(annotated_data)
    annotated_scores = np.array(annotated_scores)
    candidated_pairs = np.array(candidated_pairs)
    candidates_scores = np.array(candidates_scores)
    pos50 = max(np.median(annotated_scores[annotated_data[:,2]==1]), 0.75)
    neg50 = min(np.median(annotated_scores[annotated_data[:,2]==0]), 0.25)
    pseudo_pos_matcher = np.array([[a,b,1] for a,b in candidated_pairs[candidates_scores > pos50]])
    pseudo_neg_matcher = np.array([[a,b,0] for a,b in candidated_pairs[candidates_scores < neg50]])
    pos_score = candidates_scores[candidates_scores > pos50]
    neg_score = candidates_scores[candidates_scores < neg50]
    df_pseudo = MC_self_check(pseudo_pos_matcher, pseudo_neg_matcher, pos_score, neg_score)
    df_score = pd.DataFrame({'ltable_id': candidated_pairs[:,0], 'rtable_id': candidated_pairs[:,1], 'score': candidates_scores})
    df_score = df_score.set_index(['ltable_id', 'rtable_id'])
    print('*'*20, 'before check: pos/neg', len(pseudo_pos_matcher), len(pseudo_neg_matcher), 'after check: pos/neg', (df_pseudo['score']>0.5).sum(), (df_pseudo['score']<=0.5).sum())
    return df_pseudo, df_score, pos50, neg50

def getdist(dist_matrix, pairs):
    dists = []
    for l, r in pairs:
        dists.append(dist_matrix[l][r])
    return dists

def valid_gap(valid_df):
    def cal_gap(df_sub):
        pos = df_sub[df_sub.label==1]
        if len(pos) == 0:
            return -1
        pos_sim = np.min(pos['sim'].values)
        neg = df_sub[df_sub.label==0]['sim'].values
        if len(neg) > 0:
            neg_sim = np.max(neg)
            return pos_sim - neg_sim
        else:
            return -1
    gaps = valid_df.groupby('ltable_id').apply(cal_gap).values
    if len(gaps[gaps>0]) > 0:
        print('min gap', np.min(gaps[gaps>0]))
        gap = np.percentile(gaps[gaps>0], 10)
    else:
        gap = 0.01
    return gap

def valid_pos_thr(valid_df):
    return np.percentile(valid_df[valid_df.label == 1]['sim'].values, 25)

def gen_pseudo_blocker(topkA, topkB, distA, distAll, valid_df):
    valid_dist = getdist(distAll, valid_df[['ltable_id', 'rtable_id']].values)
    valid_df['sim'] = valid_dist
    gap = valid_gap(valid_df)
    # pos_thr = valid_pos_thr(valid_df)
    # print('*'*20, 'gen_pseudo_blocker gap', gap, 'pos_thr', pos_thr)
    print('*'*20, 'gen_pseudo_blocker gap', gap)

    pos_pairs, neg_pairs = [], []
    for idxA, (idxB_list, distB_list) in enumerate(zip(topkA, distA)):
        idxB = idxB_list[0]
        dist1 = distB_list[0]
        # if dist1 >= pos_thr and topkB[idxB][0] == idxA:
        if topkB[idxB][0] == idxA:
            pos_pairs.append([idxA, idxB, 1])
            for idxB, dist in zip(idxB_list[1:], distB_list[1:]):
                # neg_pairs.append([idxA, idxB, 0])
                if dist1 - dist >= gap:
                    neg_pairs.append([idxA, idxB, 0])
    print('*'*20, 'pseudo by BK, pos/neg', len(pos_pairs), len(neg_pairs))
    return pos_pairs, neg_pairs

def filter_by_rule(candidates_dict, hp):
    entityA = pd.read_csv(os.path.join(hp.path, hp.dataset, 'tableA.csv'), index_col = 0)
    if 'wdc' in hp.dataset:
        entityB = pd.read_csv(os.path.join(hp.path, hp.dataset, 'tableA.csv'), index_col = 0)
    else:
        entityB = pd.read_csv(os.path.join(hp.path, hp.dataset, 'tableB.csv'), index_col = 0)
    columns = list(entityA.columns)
    entityA_vals = entityA.values 
    entityB_vals = entityB.values
    str_overlap_col = ['title', 'description', 'manufacturer', 'Beer_Name', 'Brew_Factory_Name', 'name']
    num_equal_col = ['year']
    num_sim_col = ['price']
    candidates_dict_new = {}
    for idxA, idxB_list in candidates_dict.items():
        A_attrs = entityA_vals[idxA]
        filtered_Blist = []
        for idxB in idxB_list:
            B_attrs = entityB_vals[idxB]
            for i, (col, A_attr, B_attr) in enumerate(zip(columns, A_attrs, B_attrs)):
                flag = True
                if type(A_attr) == float and np.isnan(A_attr) or type(B_attr) == float and np.isnan(B_attr):
                    continue
                if col in str_overlap_col:
                    A_attr = A_attr.lower().split()
                    B_attr = B_attr.lower().split()
                    if len(set(A_attr) & set(B_attr)) == 0:
                        flag = False
                        break 
                elif col in num_equal_col:
                    if type(A_attr) != str and type(B_attr) != str and A_attr > 0 and B_attr > 0 and A_attr != B_attr:
                        flag = False
                        break 
                elif col in num_sim_col:
                    if type(A_attr) != str and type(B_attr) != str and A_attr > 0 and B_attr > 0 and (A_attr/B_attr > 2 or B_attr/A_attr > 2):
                        flag = False
                        break 
            if flag:
                filtered_Blist.append(idxB)
        if len(filtered_Blist) > 0:
            candidates_dict_new[idxA] = filtered_Blist
    return candidates_dict_new

def split(pos_data, neg_data, ratio = 0.25):
    np.random.shuffle(pos_data)
    np.random.shuffle(neg_data)
    train_set = np.concatenate([pos_data[int(len(pos_data)*ratio):], neg_data[int(len(neg_data)*ratio):]], 0)
    valid_set = np.concatenate([pos_data[:int(len(pos_data)*ratio)], neg_data[:int(len(neg_data)*ratio)]], 0)
    np.random.shuffle(train_set)
    np.random.shuffle(valid_set)
    return train_set, valid_set

def filter_annotated(pseudo, annotated_df):
    pseudo = np.array(pseudo)
    if annotated_df is not None:
        df = pd.DataFrame({'ltable_id': pseudo[:,0], 'rtable_id': pseudo[:,1], 'label1': pseudo[:,2]})
        df = df.set_index(['ltable_id', 'rtable_id'])
        df = df.join(annotated_df)
        df = df[pd.isna(df.label)]
        df = df[['label1']]
        return df.reset_index().values 
    else:
        return pseudo

def select4MC(pseudo_BK_df, pseudo_MC_df, pseudo_MC_score, annotated_data, GtPosNum, GTNegNum):
    if pseudo_MC_df is None:
        return pseudo_BK_df[['label']].reset_index().values
    df1 = pseudo_BK_df.join(pseudo_MC_score, how = 'inner')
    df1['label2'] = df1['score'] > 0.5
    df1 = df1[df1.label == df1.label2]
    df1_pos, df1_neg = df1[df1['label']==1], df1[df1['label']==0]
    # if GTNegNum / float(GtPosNum) < len(df1_neg) / float(len(df1_pos)):
    #     num = int(float(len(df1_pos)) * GTNegNum / float(GtPosNum))
    #     # df1_neg = df1_neg.sort_values(by = 'score')
    #     # df1_neg = df1_neg.iloc[:num]
    #     df1_neg = df1_neg.sample(n=num)
    df1 = df1_pos.append(df1_neg)
    df1 = df1[['label']]

    pseudo_MC_df = pseudo_MC_df[['avg_score']]
    df2 = pseudo_MC_df.join(pseudo_BK_df)
    df2 = df2[pd.isna(df2.label)]
    df2['label'] = df2['avg_score'] > 0.5
    df2['weights'] = (0.5 - df2['avg_score']).abs()
    df2_pos = df2[df2['label']==1]
    df2_neg = df2[df2['label']==0]
    # if len(df2_pos) > len(df1_pos):
    #     # df2_pos = df2_pos.sort_values(by = 'weights', ascending = False).iloc[:len(df1_pos)]
    #     df2_pos = df2_pos.sample(n=len(df1_pos), weights='weights')
    # if len(df2_neg) > GTNegNum / float(GtPosNum) * len(df2_pos):
    #     num = int(GTNegNum / float(GtPosNum) * len(df2_pos))
    #     # df2_neg = df2_neg.sort_values(by = 'weights', ascending = False).iloc[:num]
    #     df2_neg = df2_neg.sample(n=num, weights='weights')
    df2 = df2_pos.append(df2_neg)
    df2 = df2[['label']]
    df = df1.append(df2)
    # df = df1

    print('='*10, 'pseudo from BK:', len(df1), 'pseudo from MC:', len(df2))
    
    annotated_data = np.array(annotated_data)
    df3 = pd.DataFrame({'ltable_id': annotated_data[:,0], 'rtable_id': annotated_data[:,1], 'label3': annotated_data[:,2]})
    df3 = df3.set_index(['ltable_id', 'rtable_id'])
    df = df.join(df3)
    df = df[pd.isna(df['label3'])][['label']]
    return df.reset_index().values

def gen_labeling_matcher(candidated_pairs, candidates_scores, budget, hp, pos_thr, neg_thr, ratio = 0.5):
    df = pd.DataFrame({'idx': list(range(len(candidated_pairs))), 'score': candidates_scores})
    idxs = []
    # each_budget = int(budget / 2)
    each_budget = int(budget * ratio)
    idxs.extend(list(df[df.score>=0.5].sort_values(by=['score'])['idx'].values[:each_budget]))
    # if len(df[(df.score>0.5)&(df.score<pos_thr)]) > each_budget:
    #     cands = np.random.choice(df[(df.score>0.5)&(df.score<pos_thr)]['idx'].values, each_budget, replace = False)
    #     idxs.extend(list(cands))
    # else:
    #     idxs.extend(list(df[df.score>=0.5].sort_values(by=['score'])['idx'].values[:each_budget]))
    each_budget = budget - len(idxs)
    idxs.extend(list(df[df.score<0.5].sort_values(by=['score'], ascending=False)['idx'].values[:each_budget]))
    # if len(df[(df.score<0.5)&(df.score>neg_thr)]) > each_budget:
    #     cands = np.random.choice(df[(df.score<0.5)&(df.score>neg_thr)]['idx'].values, each_budget, replace = False)
    #     idxs.extend(list(cands))
    # else:
    #     idxs.extend(list(df[df.score<0.5].sort_values(by=['score'], ascending=False)['idx'].values[:each_budget]))

    annotated_pairs = []
    candidated_pairs = np.array(candidated_pairs)
    for (a, b) in candidated_pairs[idxs]:
        annotated_pairs.append([a, b])       
    gt = load_gt(hp)
    annotated_pos, annotated_neg = annotate(annotated_pairs, gt)   
    return annotated_pos, annotated_neg

def gen_labeling_pseudo_matcher(candidated_pairs, candidates_scores, candidates_dict, budget, hp):
    candidated_pairs = np.array(candidated_pairs)
    candidated_pairs, candidates_scores = zip(*sorted(zip(candidated_pairs, candidates_scores), key = lambda x:x[1]))
    candidates_scores = np.array(candidates_scores)
    candidated_pairs = np.array(candidated_pairs)
    pseudo_pos_matcher = [(a,b,1) for (a,b) in candidated_pairs[candidates_scores > 0.75]]
    pseudo_neg_matcher = [(a,b,0) for (a,b) in candidated_pairs[candidates_scores < 0.25]]
    candidated_pos = candidated_pairs[candidates_scores >= 0.5]
    candidated_neg = candidated_pairs[candidates_scores < 0.5]
    annotated_pairs = []
    if budget > 0:
        for (a, b) in candidated_pos:
            annotated_pairs.append([a, b])
            try:
                candidates_dict[a].remove(b)
            except:
                pass
            if len(annotated_pairs) >= budget/2:
                break 
        for (a, b) in candidated_neg[::-1]:
            annotated_pairs.append([a, b])
            try:
                candidates_dict[a].remove(b)
            except:
                pass
            if len(annotated_pairs) >= budget:
                break 
        gt = load_gt(hp)
        annotated_pos, annotated_neg = annotate(annotated_pairs, gt)    
    else:
        annotated_pos, annotated_neg = [], []
    return candidates_dict, annotated_pos, annotated_neg, pseudo_pos_matcher, pseudo_neg_matcher

def transform_candidates_dict2pairs(candidates_dict):
    candidated_pairs = []
    for idxA, idxB_list in candidates_dict.items():
        for idxB in idxB_list:
            candidated_pairs.append([idxA, idxB])
    return candidated_pairs

def filter_valid(annotated_df, valid_data, only_pos = False):
    valid_df = pd.DataFrame({'ltable_id': valid_data[:,0], 'rtable_id': valid_data[:,1], 'flag': valid_data[:,2]})
    valid_df = valid_df.set_index(['ltable_id', 'rtable_id'])
    if only_pos:
        df = (annotated_df[annotated_df.label==1]).join(valid_df)
    else:
        df = annotated_df.join(valid_df)
    df = df[pd.isna(df['flag'])]
    df = df[['label']]
    annotated_data = df.reset_index().values
    print('Overall Annotated Data %d, after filtering %d' %(len(annotated_df), len(annotated_data)))
    return annotated_data

def prepare_validation_set(topKA, idxs, gt, budget):
    total_samples = []
    for idx in idxs:
        idxB_list = topKA[idx]
        if len(idxB_list) > 5:
            idxB_list = list(idxB_list[:2]) + list(np.random.choice(idxB_list[2:], 3, replace = False))
        for idxB in idxB_list:
            total_samples.append([idx, idxB])
            if len(total_samples) >= budget:
                break
        if len(total_samples) >= budget:
            break        
    total_samples = np.array(total_samples)
    annotated_pos, annotated_neg = annotate(total_samples, gt)   
    return annotated_pos, annotated_neg

def gen_annotation_candidates(candidates_dict, budget):
    pairs = []
    all_annotated = False
    while len(pairs) < budget and all_annotated == False:
        all_annotated = True
        idxA_list = np.random.choice(list(candidates_dict.keys()), min(budget * 2, len(candidates_dict)), replace = False)
        for idxA in idxA_list:
            idxB_list = candidates_dict[idxA]
            if len(idxB_list) > 0:
                all_annotated = False
                pairs.append([idxA, idxB_list[0]])
                del candidates_dict[idxA][0]
            if len(pairs) == budget:
                break
    return candidates_dict, pairs, all_annotated

def annotate(samples, gt):
    ltable_id, rtable_id = zip(*samples)
    samples_df = pd.DataFrame({'ltable_id': ltable_id, 'rtable_id': rtable_id})
    samples_df['flag'] = 1
    samples_df = samples_df.set_index(['ltable_id', 'rtable_id'])
    df = samples_df.join(gt)
    df = df.fillna(0)
    df = df.reset_index()
    df = df[['ltable_id', 'rtable_id', 'label']]
    values = df.values 
    values = np.array(values, dtype = int)
    pos = values[values[:,2] == 1]
    neg = values[values[:,2] == 0]
    return pos, neg

def load_gt(hp):
    gt = pd.read_csv(os.path.join(hp.path, hp.dataset, 'matches.csv'), index_col=0) # ltable_id,rtable_id,label
    if 'wdc' in hp.dataset:
        gt = gt[gt['ltable_id'] != gt['rtable_id']]
    gt = gt.set_index(['ltable_id', 'rtable_id'])
    return gt

def gen_blocker_pseudo(topkA, topkB):
    pos_pairs, neg_pairs = [], []
    for idxA, idxB_list in enumerate(topkA):
        if len(idxB_list) > 2:
            idxB = idxB_list[0]
            try:
                if topkB[idxB][0] == idxA:
                    pos_pairs.append([idxA, idxB, 1])
                    for idxB in idxB_list[2:]:
                        neg_pairs.append([idxA, idxB, 0])
            except:
                pass
    # np.random.shuffle(pos_pairs)
    # np.random.shuffle(neg_pairs)
    return pos_pairs, neg_pairs

def balance(annotated_pos, annotated_neg, pseudo_pos, pseudo_neg, ratio):
    if len(pseudo_pos) > len(annotated_pos) * ratio * 3:
        pseudo_pos = pseudo_pos[:int(len(annotated_pos)*3*ratio)]
    if len(pseudo_neg) > len(annotated_neg) * ratio * 3:
        pseudo_neg = pseudo_neg[:int(len(annotated_neg)*3*ratio)] 
    if len(pseudo_pos) == 0:
        pseudo_neg = []
    return pseudo_pos, pseudo_neg

def evaluate_blocker_pseudo(annotated_pos, annotated_neg, pseudo_pos, pseudo_neg):
    annotated = np.concatenate([annotated_pos, annotated_neg], 0)
    pseudo = np.concatenate([pseudo_pos, pseudo_neg], 0)
    annotated_df = pd.DataFrame({'ltable_id': annotated[:,0], 'rtable_id': annotated[:,1], 'label': annotated[:,2]})
    pseudo_df = pd.DataFrame({'ltable_id': pseudo[:,0], 'rtable_id': pseudo[:,1], 'pseudo': pseudo[:,2]})
    annotated_df = annotated_df.set_index(['ltable_id', 'rtable_id'])
    pseudo_df = pseudo_df.set_index(['ltable_id', 'rtable_id'])
    df = pseudo_df.join(annotated_df, how = 'inner')
    df = df.fillna(0)
    f1 = f1_score(df['label'].values, df['pseudo'].values)
    return f1

def gen_blocker_labeling_pseudo(candidates_dict, topkA, topkB, distAll, hp, budget, eval_blocker_pseudo = False):
    blocker_quality = 1.0
    if eval_blocker_pseudo:
        pseudo_pos, pseudo_neg = gen_blocker_pseudo(topkA, topkB)
    # annotation
    if budget > 0:
        candidates_dict, pairs, all_annotated = gen_annotation_candidates(candidates_dict, budget)
        gt = load_gt(hp)
        annotated_pos, annotated_neg = annotate(pairs, gt)
        if eval_blocker_pseudo:
            blocker_quality = evaluate_blocker_pseudo(annotated_pos, annotated_neg, pseudo_pos, pseudo_neg)
        # pseudo labeling
        pseudo_pos, pseudo_neg = gen_blocker_pseudo(topkA, topkB)
        return candidates_dict, annotated_pos, annotated_neg, pseudo_pos, pseudo_neg, all_annotated, blocker_quality
    else:
        pseudo_pos, pseudo_neg = gen_blocker_pseudo(topkA, topkB)
        return pseudo_pos, pseudo_neg

def get_test_ABdict(hp):
    test_idxAB = defaultdict(list)
    test = pd.read_csv(os.path.join(hp.path, hp.dataset, 'test.csv')).values 
    for a, b, _ in test:
        test_idxAB[a].append(b)
    return test_idxAB

def filter_test(topkA, hp):
    candidates_dict = {}
    test_idxAB = get_test_ABdict(hp) # {idxA: [idxB]}
    for idxA, B_list in enumerate(topkA):
        test_idxB = test_idxAB.get(idxA, [])
        filtered_idxB = []
        for idxB in B_list:
            if idxB not in test_idxB:
                filtered_idxB.append(idxB)
        candidates_dict[idxA] = filtered_idxB
    return candidates_dict

def load_model(hp):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    EmbModel = CLSepModel(device=device, lm=hp.lm)
    EmbModel.load_state_dict(torch.load(os.path.join('../SentEmb', hp.CLlogdir, 'random', hp.dataset, str(0), 'model.pt'))['model'])
    EmbModel = EmbModel.cuda()
    return EmbModel

def get_SentEmb(EmbModel, dataset):
    padder = dataset.pad
    data_iter = data.DataLoader(dataset=dataset, batch_size=256, shuffle=False, num_workers=0, collate_fn=padder)
    EmbModel.eval()
    # get emb
    emb = []
    with torch.no_grad():
        for batch in data_iter:
            x1, x1_mask = batch 
            x1, x1_mask = x1.to(EmbModel.device), x1_mask.to(EmbModel.device)
            emb1 = EmbModel.get_emb(x1, x1_mask)
            emb.extend(emb1.cpu().numpy())
    return emb

def cal_cosine_sim(embeddingA, embeddingB, topK = 5):
    embeddingA = torch.tensor(embeddingA).cuda()
    embeddingB = torch.tensor(embeddingB).cuda()
    sim_score = util.pytorch_cos_sim(embeddingA, embeddingB)
    distA, topkA = torch.topk(sim_score, k=topK, dim=1) # topkA [sizeA, hp.K]   
    distB, topkB = torch.topk(sim_score.T, k=topK, dim=1) # topkA [sizeA, hp.K] 
    distA = distA.cpu().numpy()
    topkA = topkA.cpu().numpy()
    distB = distB.cpu().numpy()
    topkB = topkB.cpu().numpy()    
    return topkA, distA, topkB, distB, sim_score.cpu().numpy()

def get_topK_sim(hp):
    attr_listA, entity_listA = read_entity(os.path.join(hp.path, hp.dataset), table='tableA', shuffle=False)
    if 'wdc' in hp.dataset:
        attr_listB, entity_listB = read_entity(os.path.join(hp.path, hp.dataset), table='tableA', shuffle=False)
    else:
        attr_listB, entity_listB = read_entity(os.path.join(hp.path, hp.dataset), table='tableB', shuffle=False)
    datasetA = SingleEntityDataset(entity_listA, attr_listA, lm='sent-bert', max_len=128, add_token=hp.add_token)
    datasetB = SingleEntityDataset(entity_listB, attr_listB, lm='sent-bert', max_len=128, add_token=hp.add_token)
    # EmbModel = load_model(hp)
    # embeddingA = get_SentEmb(EmbModel, datasetA)
    # embeddingB = get_SentEmb(EmbModel, datasetB)
    EmbModel = SentenceTransformer('stsb-roberta-base')
    embeddingA = EmbModel.encode(datasetA.entitytext, batch_size=512)
    embeddingB = EmbModel.encode(datasetB.entitytext, batch_size=512)
    topkA, distA, topkB, distB, sim_score = cal_cosine_sim(embeddingA, embeddingB, hp.topK)
    return topkA, distA, topkB, distB, sim_score

def worker_init(worker_init):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def evaluate(y_truth, y_pred):
    """
    Evaluate model.
    """
    precision = precision_score(y_truth, y_pred)
    recall = recall_score(y_truth, y_pred)
    f1 = f1_score(y_truth, y_pred)
    return precision, recall, f1

def load_attributes(path):
    attr_listA, entity_listA = read_entity(path, table='tableA', shuffle=False)
    if 'wdc' in path:
        attr_listB, entity_listB = read_entity(path, table='tableA', shuffle=False)
    else:
        attr_listB, entity_listB = read_entity(path, table='tableB', shuffle=False)
    return attr_listA, entity_listA, attr_listB, entity_listB

def read_entity(dataset, table=None, shuffle=True):
    """
    Read entities from tables.
    """
    if table is None:
        df = pd.read_csv(os.path.join(dataset, 'tableA.csv'), sep = ',', index_col=0)
        if 'wdc' not in dataset:
            df = df.append(pd.read_csv(os.path.join(dataset, 'tableB.csv'), sep = ',', index_col=0))
    else:
        df = pd.read_csv(os.path.join(dataset, table + '.csv'), sep = ',', index_col=0)

    entity_list = df.values
    if shuffle:
        np.random.shuffle(entity_list)
    return list(df.columns), list(entity_list)
