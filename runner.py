import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import sklearn.metrics as metrics
import argparse

from torch.utils import data
from transformers import AutoModel, AdamW, get_linear_schedule_with_warmup
from tensorboardX import SummaryWriter
from apex import amp
from model import * 
from utils import * 
from torch.utils.data import DataLoader
from dataset import GTDatasetWithLabel, AugDataset, AugDatasetWithLabel, SingleEntityDataset, GTDatasetWithLabelWeights
# import matplotlib.pyplot as plt
from sklearn import metrics

def pred_scores(model, dataset):
    iterator = DataLoader(dataset=dataset, batch_size=128, collate_fn=dataset.pad)
    model.eval()
    scores_all = []
    for i, batch in enumerate(iterator):
        x, _, y = batch   
        logits = model(x)
        scores = logits.softmax(-1)[:,1]
        scores_all.extend(scores.detach().cpu().numpy().tolist())
    return scores_all

def get_emb_and_pred_scores(model, dataset):
    iterator = DataLoader(dataset=dataset, batch_size=128, collate_fn=dataset.pad)
    model.eval()
    scores_all, emb_all = [], []
    for i, batch in enumerate(iterator):
        x, _, y = batch   
        emb, logits = model.get_emb_and_score(x)
        scores = logits.softmax(-1)[:,1]
        scores_all.extend(scores.detach().cpu().numpy().tolist())
        emb_all.append(emb.detach().cpu().numpy())
    return scores_all, np.concatenate(emb_all, 0)

def eval_matcher(model, dataset, if_valid = False):
    iterator = DataLoader(dataset=dataset, batch_size=128, collate_fn=dataset.pad)
    model.eval()
    y_truth, y_pre = [], []
    valid_loss, num = 0.0, 0
    criterion = nn.CrossEntropyLoss()
    for i, batch in enumerate(iterator):
        x, _, y = batch   
        logits = model(x)
        loss = criterion(logits, y.cuda())
        valid_loss += loss.item() * len(x)
        num += len(x)
        scores = logits.argmax(-1)
        for item in y.cpu().numpy().tolist():
            y_truth.append(item)
        for item in scores.cpu().numpy().tolist():
            y_pre.append(item)
    valid_loss = valid_loss / num
    precision, recall, F1 = evaluate(y_truth, y_pre)
    if if_valid:
        return valid_loss, precision, recall, F1
    else:
        return precision, recall, F1

def train_matcher(model, train_set, optimizer, hp, scheduler=None, fp16=True):
    iterator = DataLoader(dataset=train_set, batch_size=hp.batch_size, shuffle=True,\
                          num_workers=0, worker_init_fn=worker_init,\
                          collate_fn=train_set.pad)
    criterion = nn.CrossEntropyLoss(reduction = 'none')
    model.train()
    train_loss, num = 0.0, 0
    for batch in iterator:
        x, _, y, w = batch
        # forward
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y.cuda())
        loss = torch.mean(loss * w.cuda())
        # back propagation
        if fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        train_loss += loss.item() * np.sum(w.cpu().detach().numpy())
        num += np.sum(w.cpu().detach().numpy())
    return train_loss/num

def train_blocker(blocker, train_set, bk_opt, hp, scheduler=None):
    f = lambda x: torch.exp(x / 0.05)
    iterator = DataLoader(dataset=train_set, batch_size=hp.batch_size, shuffle=True, num_workers=0, worker_init_fn=worker_init, collate_fn=train_set.pad)    
    blocker.train()
    total_loss, total_num = 0.0, 0.0
    w_list = []
    for batch in iterator:
        x1, x1_mask, x2, x2_mask, w = batch
        w_list.extend(w.detach().numpy())
        emb1 = blocker(x1, x1_mask)
        emb2 = blocker(x2, x2_mask)
        scores_pos = torch.mm(emb1, emb2.t())
        scores_pos = f(scores_pos)
        w = w.cuda()
        loss = ((-torch.log(scores_pos.diag() / (scores_pos.sum(1)))) * w).sum() / w.sum()
        if hp.fp16:
            with amp.scale_loss(loss, bk_opt) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        bk_opt.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * (w.sum()).item()
        total_num += len(emb1)
        del loss
    # w_list = np.array(w_list)
    # print('w=2', np.sum(w_list==2), 'w=1', np.sum(w_list==1), 'w=0.5', np.sum(w_list==0.5))
    # exit()
    return total_loss / total_num

def eval_blocker(blocker, blocker_valid):
    iterator = DataLoader(dataset=blocker_valid, batch_size=128, collate_fn=blocker_valid.pad)
    blocker.eval()
    scores, y_truth = [], []
    for i, batch in enumerate(iterator):
        x1, x1_mask, x2, x2_mask, y = batch   
        x1_emb = blocker.get_emb(x1, x1_mask).detach().cpu().numpy()
        x2_emb = blocker.get_emb(x2, x2_mask).detach().cpu().numpy()
        batch_scores = np.sum(x1_emb * x2_emb, 1)
        scores.extend(list(1 / (1 + np.exp(-batch_scores))))
        y_truth.extend(y.detach().cpu().numpy())
    bestF1 = 0.0
    for th in np.arange(0,1,0.05):
        pred_list = [1 if s > th else 0 for s in scores]
        f1 = f1_score(y_truth, pred_list)
        if f1 > bestF1:
            bestF1 = f1 
    return bestF1

def eval_sentbert(valid_data, sim_scores):
    scores, y_truth = [], []
    for e1, e2, y in valid_data:
        scores.append(1 / (1 + np.exp(-sim_scores[e1][e2])))
        y_truth.append(y)
    bestF1 = 0.0
    for th in np.arange(0,1,0.05):
        pred_list = [1 if s > th else 0 for s in scores]
        f1 = f1_score(y_truth, pred_list)
        if f1 > bestF1:
            bestF1 = f1 
    return bestF1    

def printAndWrite(run_tag, name, data_dict, active_iteration, writer):
    for k, v in data_dict.items():
        if type(v) != int:
            data_dict[k] = round(v, 5)
    print('[Tensorboard]', name, 'Iteration', active_iteration, data_dict)
    writer.add_scalars(run_tag+name, data_dict, active_iteration)

def train(hp):
    '''
        Initialize the datasets
    '''
    path = os.path.join(hp.path, hp.dataset)
    attr_listA, entity_listA, attr_listB, entity_listB = load_attributes(path)
    datasetA = SingleEntityDataset(entity_listA, attr_listA, lm='sent-bert', max_len=128, add_token=hp.add_token)
    datasetB = SingleEntityDataset(entity_listB, attr_listB, lm='sent-bert', max_len=128, add_token=hp.add_token)
    test_idxs = pd.read_csv(os.path.join(hp.path, hp.dataset, 'test_idxs.csv'), index_col=0)['ltable_id'].values
    annotated_df, pseudo_BK_df, pseudo_MC_df = None, None, None
    writer = SummaryWriter(log_dir=hp.logdir)
    run_tag = '_'.join(['dataset='+hp.dataset.split('/')[-1],'topK='+str(hp.topK),'size='+str(hp.total_budget),'runid='+str(hp.run_id)])
    if os.path.exists(os.path.join('checkpoints', hp.dataset, str(hp.topK), str(hp.total_budget), str(hp.run_id))) == False:
        os.makedirs(os.path.join('checkpoints', hp.dataset, str(hp.topK), str(hp.total_budget), str(hp.run_id)))
    else:
        print('Already got the model')

    '''
        Initialize blocker
    '''
    blocker = CLSepModel(lm='sent-bert')
    blocker = blocker.cuda()
    bk_opt = AdamW(blocker.parameters(), lr=hp.lr)
    if hp.fp16:
        blocker, bk_opt = amp.initialize(blocker, bk_opt, opt_level='O2')
    
    '''
        Warm-up model training with pseudo labels
    '''
    topkA, distA, topkB, distB, distAll = get_topK_sim(hp)
    warmup_iteration = 0
    valid_df = pd.read_csv(os.path.join(hp.path, hp.dataset, 'valid0207.csv'))
    blocker_pseudo_pos, blocker_pseudo_neg = gen_pseudo_blocker(topkA, topkB, distA, distAll, valid_df)
    pseudo_BK_df = update_pseudoBK_df(pseudo_BK_df, blocker_pseudo_pos, blocker_pseudo_neg)
    printAndWrite(run_tag, '/BKCollection',  {'pos_num': pseudo_BK_df['label'].sum(), 'neg_num': (pseudo_BK_df['label']==0).sum()}, 0, writer)
    train_data, valid_data = split(blocker_pseudo_pos, blocker_pseudo_neg, ratio = 0.25)
    train_data = np.concatenate([train_data, [[1]]*len(train_data)], 1)
    train_set = GTDatasetWithLabelWeights(train_data, entity_listA, entity_listB, attr_listA, lm=hp.lm, concat=True, shuffle=False)
    valid_set = GTDatasetWithLabel(valid_data, entity_listA, entity_listB, attr_listA, lm=hp.lm, concat=True, shuffle=False)
    
    '''
        Warm-up Stage
        generate pseudo lables for training and validation
    '''
    best_valid_F1 = -1
    while True:
        warmup_iteration += 1
        model = DittoConcatModel(lm = hp.lm)
        model = model.cuda()
        optimizer = AdamW(model.parameters(), lr=hp.lr)
        if hp.fp16:
            model, optimizer = amp.initialize(model, optimizer, opt_level='O2')
        epoch = 0
        train_loss_list = []
        while True:
            epoch += 1
            torch.cuda.empty_cache()
            train_loss = train_matcher(model, train_set, optimizer, hp, scheduler=None)
            train_loss_list.append(train_loss)
            torch.cuda.empty_cache()
            valid_loss, valid_precision, valid_recall, valid_F1 = eval_matcher(model, valid_set, if_valid=True)
            printAndWrite(run_tag, '/WarmUpLoss_lr=%.3g' %(hp.lr), {'train_loss': train_loss, 'valid_loss': valid_loss}, epoch, writer)
            printAndWrite(run_tag, '/WarmUpValid_lr=%.3g' %(hp.lr), {'pre': valid_precision, 'rec': valid_recall, 'F1': valid_F1}, epoch, writer)         
            if valid_F1 >= best_valid_F1:
                best_valid_F1 = valid_F1
                if hp.save_model:
                    torch.save(model.state_dict(), os.path.join('checkpoints', hp.dataset, str(hp.topK), str(hp.total_budget), str(hp.run_id), 'matcher_model.pt'))
            elif epoch >= 20 or epoch >= 10 and (best_valid_F1 >= 0.5 or best_valid_F1 == 0 and train_loss_list[0] - np.mean(train_loss_list) <= 0.01):
                break 
        if best_valid_F1 >= 0.5 or warmup_iteration >= 3:
            break
        hp.lr = hp.lr * 0.7
    try:
        model.load_state_dict(torch.load(os.path.join('checkpoints', hp.dataset, str(hp.topK), str(hp.total_budget), str(hp.run_id), 'matcher_model.pt')))
        if hp.run_id==0 and hp.total_budget in [500, 2000]:
            torch.save(model.state_dict(), os.path.join('checkpoints', hp.dataset, str(hp.topK), str(hp.total_budget), str(hp.run_id), 'Iter0_matcher_model.pt'))
    except:
        pass

    '''
        Prepare Validation Set
    '''
    test_idxs = np.array(pd.read_csv(os.path.join(hp.path, hp.dataset, 'test_idxs.csv'), index_col=0)['ltable_id'].values, int)
    valid_data = pd.read_csv(os.path.join(hp.path, hp.dataset, 'valid0207.csv')).values
    annotated_pos, annotated_neg = valid_data[valid_data[:,2]==1], valid_data[valid_data[:,2]==0]
    annotated_df = update_annotated_df(annotated_df, annotated_pos, annotated_neg)
    valid_pos_num, valid_neg_num = len(annotated_pos), len(annotated_neg)
    valid_data = np.concatenate([annotated_pos, annotated_neg], 0)
    printAndWrite(run_tag, '/Validation Set', {'pos': len(annotated_pos), 'neg': len(annotated_neg)}, 0, writer)
    valid_set = GTDatasetWithLabel(valid_data, entity_listA, entity_listB, attr_listA, lm=hp.lm, concat=True, shuffle=False)
    blocker_valid = GTDatasetWithLabel(valid_data, entity_listA, entity_listB, attr_listA, lm='sent-bert', max_len=128, concat=False, shuffle=False)
    remaining_budget = hp.total_budget
    
    pos_pre, pos_rec, neg_pre, neg_rec = evaluate_pseudo(np.concatenate([blocker_pseudo_pos, blocker_pseudo_neg], 0), annotated_df, hp)
    printAndWrite(run_tag, '/PseudoByBKQlt',  {'Positive_Precision': pos_pre, 'Positive_Recall': pos_rec, 'Negative_Precision': neg_pre, 'Negative_Recall': neg_rec}, 0, writer)

    ''' evaluate sentence bert as the initial metric '''
    best_BK_valid_F1 = eval_sentbert(valid_data, distAll)
    printAndWrite(run_tag, '/blocker_metric', {'BK_valid_F1': best_BK_valid_F1, 'best_BK_valid_F1': best_BK_valid_F1}, 0, writer)    
    ''' evalute matcher after warm-up on the validation set '''
    valid_precision, valid_recall, best_MC_valid_F1 = eval_matcher(model, valid_set)
    printAndWrite(run_tag, '/matcher_metric', {'valid_precision': valid_precision, 'valid_recall':valid_recall, 'MC_valid_F1': best_MC_valid_F1}, 0, writer)

    '''
        Active Learning Stage
    '''
    active_iteration = 0
    blocker_epoch, matcher_epoch = 0, 0
    hp.active_budget = int(hp.total_budget / 10)
    flag_matcher = True
    while True:
        active_iteration += 1
        budget = min(remaining_budget, hp.active_budget)   
        
        ''' Generate pseudo labels by matcher '''
        try:
            model.load_state_dict(torch.load(os.path.join('checkpoints', hp.dataset, str(hp.topK), str(hp.total_budget), str(hp.run_id), 'matcher_model.pt')))
        except:
            pass
        valid_scores = pred_scores(model, valid_set)
        if flag_matcher:
            cand_dataset_full, candidated_pairs_full = filter_test_from_candidates(topkA, annotated_df, None)
            cand_dataset_full = GTDatasetWithLabel(cand_dataset_full, entity_listA, entity_listB, attr_listA, lm=hp.lm, concat=True, shuffle=False)
            candidates_scores_full = pred_scores(model, cand_dataset_full)
            pseudo_MC_cur, pseudo_MC_score, pos50, neg50 = gen_pseudo_matcher(candidates_scores_full, candidated_pairs_full, valid_scores, valid_data)
            printAndWrite(run_tag, '/MC_Threshold', {'pos50':pos50, 'neg50':neg50}, active_iteration, writer)  
            pseudo_MC_np = pseudo_MC_cur.reset_index()[['ltable_id','rtable_id', 'score']].values
            pseudo_MC_np[:,2] = pseudo_MC_np[:,2] > 0.5
            pos_pre, pos_rec, neg_pre, neg_rec = evaluate_pseudo(pseudo_MC_np, annotated_df, hp)
            printAndWrite(run_tag, '/pseudoByMCQlt', {'Positive_Precision1': pos_pre, 'Positive_Recall1': pos_rec, 'Negative_Precision1': neg_pre, 'Negative_Recall1': neg_rec}, active_iteration, writer)
            pseudo_MC_df = update_pseudoMC_df(pseudo_MC_df, pseudo_MC_cur)    
            printAndWrite(run_tag, '/MCCollection',  {'pos_num1': (pseudo_MC_df['avg_score']>0.5).sum(), 'neg_num1': (pseudo_MC_df['avg_score']<0.5).sum()}, active_iteration, writer) 

        ''' annotation '''
        if budget > 0:
            annotated_data = filter_valid(annotated_df, valid_data)
            cand_dataset_filtered, candidated_pairs_filtered = filter_test_from_candidates(topkA, annotated_df, test_idxs)
            candidates_scores_filtered = get_scores(candidated_pairs_filtered, candidated_pairs_full, candidates_scores_full)
            annotated_pos, annotated_neg = gen_labeling_matcher(candidated_pairs_filtered, candidates_scores_filtered, budget, hp, pos50, neg50)
            annotated_df = update_annotated_df(annotated_df, annotated_pos, annotated_neg)
            remaining_budget -= (len(annotated_pos) + len(annotated_neg))
            printAndWrite(run_tag, '/annotation', {'Annotated_Pos': len(annotated_pos), 'Annotated_Neg': len(annotated_neg)}, active_iteration, writer)
        
        '''
            train data for blocker
        '''
        blocker_train_data = []
        annotated_data = filter_valid(annotated_df, valid_data, only_pos = True)
        annotated_data = np.concatenate([annotated_data, [[2]]*len(annotated_data)], 1)
        if pseudo_MC_df is not None and len(pseudo_MC_df[pseudo_MC_df.avg_score > 0.5]) > 0:
            pseudo_MC_np = select4BK(pseudo_MC_df)
            pos_pre, pos_rec, neg_pre, neg_rec = evaluate_pseudo(pseudo_MC_np, annotated_df, hp)
            pseudo_MC_np = np.concatenate([pseudo_MC_np, [[1.0]]*len(pseudo_MC_np)], 1)
            printAndWrite(run_tag, '/pseudo4BKQlt', {'Positive_Precision': pos_pre, 'Positive_Recall': pos_rec}, active_iteration, writer)
            printAndWrite(run_tag, '/pseudo4BKNum', {'Pos Num': len(pseudo_MC_np[pseudo_MC_np[:,2]==1])}, active_iteration, writer)
            if len(pseudo_MC_np) > 0:
                blocker_train_data.append(pseudo_MC_np)
        if len(annotated_data) > 0:
            blocker_train_data.append(annotated_data)
        if len(blocker_train_data) > 0:
            blocker_train_data = np.concatenate(blocker_train_data, 0)
        printAndWrite(run_tag, '/BKtrain', {'Num': len(blocker_train_data)}, active_iteration, writer)

        '''
            training blocker
        '''
        blocker_train = AugDatasetWithLabel(blocker_train_data, entity_listA, entity_listB, attr_listA, lm='sent-bert', max_len=128, add_token=True, concat=False, same_table=False, aug_type=hp.aug_type)
        if active_iteration == 1 or remaining_budget <= 0:
            epochs = 20
        else:
            epochs = 5
        flag = False
        last_BK_F1 = -1
        for epoch in range(1, epochs + 1):
            blocker_epoch += 1
            torch.cuda.empty_cache()
            train_loss = train_blocker(blocker, blocker_train, bk_opt, hp, scheduler=None)
            torch.cuda.empty_cache()      
            BK_valid_F1 = eval_blocker(blocker, blocker_valid)
            if (BK_valid_F1 > best_BK_valid_F1) or (flag == False and BK_valid_F1 == best_BK_valid_F1):
                best_BK_valid_F1 = BK_valid_F1
                if hp.save_model:
                    torch.save(blocker.state_dict(), os.path.join('checkpoints', hp.dataset, str(hp.topK), str(hp.total_budget), str(hp.run_id), 'blocker_model.pt'))   
                    if hp.run_id==0 and hp.total_budget in [500, 2000]:
                        torch.save(blocker.state_dict(), os.path.join('checkpoints', hp.dataset, str(hp.topK), str(hp.total_budget), str(hp.run_id), 'Iter'+str(active_iteration)+'_blocker_model.pt'))
                flag = True
            if remaining_budget <= 0 and last_BK_F1 < BK_valid_F1:
                last_BK_F1 = BK_valid_F1
                torch.save(blocker.state_dict(), os.path.join('checkpoints', hp.dataset, str(hp.topK), str(hp.total_budget), str(hp.run_id), 'last_blocker_model.pt'))
            printAndWrite(run_tag, '/blocker_train_loss', {'train_loss': train_loss}, blocker_epoch, writer)
            printAndWrite(run_tag, '/blocker_metric', {'BK_valid_F1': BK_valid_F1, 'best_BK_valid_F1': best_BK_valid_F1}, blocker_epoch, writer)

        ''' Generate pseudo labels by blocker and update pseudo labels by matcher'''
        try:
            blocker.load_state_dict(torch.load(os.path.join('checkpoints', hp.dataset, str(hp.topK), str(hp.total_budget), str(hp.run_id), 'blocker_model.pt')))
        except:
            pass
        if flag:
            # update pseudo labels by blocker
            embeddingA = get_SentEmb(blocker, datasetA)
            embeddingB = get_SentEmb(blocker, datasetB)
            topkA, distA, topkB, distB, distAll = cal_cosine_sim(embeddingA, embeddingB, hp.topK)
            blocker_pseudo_pos, blocker_pseudo_neg = gen_pseudo_blocker(topkA, topkB, distA, distAll, valid_df)
            pos_pre, pos_rec, neg_pre, neg_rec = evaluate_pseudo(np.concatenate([blocker_pseudo_pos, blocker_pseudo_neg], 0), annotated_df, hp)
            pseudo_BK_df = update_pseudoBK_df(pseudo_BK_df, blocker_pseudo_pos, blocker_pseudo_neg)
            printAndWrite(run_tag, '/PseudoByBKQlt',  {'Positive_Precision': pos_pre, 'Positive_Recall': pos_rec, 'Negative_Precision': neg_pre, 'Negative_Recall': neg_rec}, active_iteration, writer)
            printAndWrite(run_tag, '/BKCollection',  {'pos_num': pseudo_BK_df['label'].sum(), 'neg_num': (pseudo_BK_df['label']==0).sum()}, active_iteration, writer)
            # update pseudo labels by matcher
            cand_dataset_full, candidated_pairs_full = filter_test_from_candidates(topkA, annotated_df, None)
            cand_dataset_full = GTDatasetWithLabel(cand_dataset_full, entity_listA, entity_listB, attr_listA, lm=hp.lm, concat=True, shuffle=False)
            candidates_scores_full = pred_scores(model, cand_dataset_full)
            pseudo_MC_cur, pseudo_MC_score, pos50, neg50 = gen_pseudo_matcher(candidates_scores_full, candidated_pairs_full, valid_scores, valid_data)
            pseudo_MC_np = pseudo_MC_cur.reset_index()[['ltable_id','rtable_id', 'score']].values
            pseudo_MC_np[:,2] = pseudo_MC_np[:,2] > 0.5
            pos_pre, pos_rec, neg_pre, neg_rec = evaluate_pseudo(pseudo_MC_np, annotated_df, hp)
            printAndWrite(run_tag, '/pseudoByMCQlt', {'Positive_Precision2': pos_pre, 'Positive_Recall2': pos_rec, 'Negative_Precision2': neg_pre, 'Negative_Recall2': neg_rec}, active_iteration, writer)
            pseudo_MC_df = update_pseudoMC_df(pseudo_MC_df, pseudo_MC_cur)    
            printAndWrite(run_tag, '/MCCollection',  {'pos_num2': (pseudo_MC_df['avg_score']>0.5).sum(), 'neg_num2': (pseudo_MC_df['avg_score']<0.5).sum()}, active_iteration, writer) 

        '''
            train data for matcher
        '''
        annotated_data = filter_valid(annotated_df, valid_data)
        if len(annotated_data) > 0:
            annotated_data = np.concatenate([annotated_data, [[1.0]]*len(annotated_data)], 1)
        GTposNum = len(annotated_data[annotated_data[:,2]==1])
        pseudo4MC = select4MC(pseudo_BK_df, pseudo_MC_df, pseudo_MC_score, annotated_df.reset_index().values, GTposNum, len(annotated_data)-GTposNum)
        if pseudo4MC is not None and len(pseudo4MC[pseudo4MC[:,2]==1]) > 0:
            pos_pre, pos_rec, neg_pre, neg_rec = evaluate_pseudo(pseudo4MC, annotated_df, hp)
            printAndWrite(run_tag, '/pseudo4MCNum', {'pos_num': len(pseudo4MC[pseudo4MC[:,2]==1]), 'neg_num': len(pseudo4MC[pseudo4MC[:,2]==0])}, active_iteration, writer)
            printAndWrite(run_tag, '/pseudo4MCQlt', {'Positive_Precision': pos_pre, 'Positive_Recall': pos_rec, 'Negative_Precision': neg_pre, 'Negative_Recall': neg_rec}, active_iteration, writer)
            w = min(1, float(len(annotated_data)) / len(pseudo4MC))
            # w = 1.0
            pseudo_data = np.concatenate([pseudo4MC, [[w]]*len(pseudo4MC)], 1)
            if len(annotated_data) > 0:
                train_data = np.concatenate([annotated_data, pseudo_data], 0)
            else:
                train_data = pseudo_data
        else:
            train_data = annotated_data

        '''
            training matcher
        '''
        train_set = GTDatasetWithLabelWeights(train_data, entity_listA, entity_listB, attr_listA, lm=hp.lm, concat=True, shuffle=False)
        print('Summary of Training Set: annotated pos/neg = %d/%d, pseudo pos/neg = %d/%d, Remaining budget %d' %((annotated_df['label']==1).sum()-valid_pos_num, (annotated_df['label']==0).sum()-valid_neg_num, len(pseudo4MC[pseudo4MC[:,2]==1]), len(pseudo4MC[pseudo4MC[:,2]==0]), remaining_budget))
        if remaining_budget <= 0:
            epochs = 40
        else:
            epochs = 5        
        early_stop = 0
        try:
            model.load_state_dict(torch.load(os.path.join('checkpoints', hp.dataset, str(hp.topK), str(hp.total_budget), str(hp.run_id), 'matcher_model.pt')))
        except:
            pass
        flag_matcher = False
        last_MC_F1 = -1
        for epoch in range(1, epochs + 1):
            matcher_epoch += 1
            torch.cuda.empty_cache()
            train_loss = train_matcher(model, train_set, optimizer, hp, scheduler=None)
            torch.cuda.empty_cache()
            valid_loss, valid_precision, valid_recall, MC_valid_F1 = eval_matcher(model, valid_set, if_valid=True)
            if remaining_budget <= 0 and last_MC_F1 < MC_valid_F1:
                torch.save(model.state_dict(), os.path.join('checkpoints', hp.dataset, str(hp.topK), str(hp.total_budget), str(hp.run_id), 'last_matcher_model.pt'))
                last_MC_F1 = MC_valid_F1
            if MC_valid_F1 > best_MC_valid_F1 or (flag_matcher == False and MC_valid_F1 == best_MC_valid_F1):
                flag_matcher = True
                best_MC_valid_F1 = MC_valid_F1
                if hp.save_model:
                    torch.save(model.state_dict(), os.path.join('checkpoints', hp.dataset, str(hp.topK), str(hp.total_budget), str(hp.run_id), 'matcher_model.pt'))
                    if hp.run_id==0 and hp.total_budget in [500, 2000]:
                        torch.save(model.state_dict(), os.path.join('checkpoints', hp.dataset, str(hp.topK), str(hp.total_budget), str(hp.run_id), 'Iter'+str(active_iteration)+'_matcher_model.pt'))
                early_stop = 0
            elif MC_valid_F1 > 0 and epoch >= 5:
                early_stop += 1
                # if early_stop > 10:
                #     break
            printAndWrite(run_tag, '/matcher_loss', {'train_loss': train_loss, 'valid_loss': valid_loss}, matcher_epoch, writer)
            printAndWrite(run_tag, '/matcher_metric',  {'valid_precision': valid_precision, 'valid_recall':valid_recall, 'MC_valid_F1': MC_valid_F1, 'best_MC_valid_F1': best_MC_valid_F1}, matcher_epoch, writer)

        if remaining_budget <= 0:
            break 
    