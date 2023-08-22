from utils import *
from model import *
from apex import amp
from transformers import AutoModel, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
import argparse
from dataset import GTDatasetWithLabel

def get_topK(embeddingA, embeddingB, topK = 100):
    embeddingA = torch.tensor(embeddingA).cuda()
    embeddingB = torch.tensor(embeddingB).cuda()
    sim_score = util.pytorch_cos_sim(embeddingA, embeddingB)
    distA, topkA = torch.topk(sim_score, k=topK, dim=1) # topkA [sizeA, K]
    topkA = topkA.cpu().numpy()
    distA = distA.cpu().numpy()
    sim_score = sim_score.cpu().numpy()
    return topkA, distA, sim_score

def get_emb(hp):
    path = os.path.join(hp.path, hp.dataset)
    attr_listA, entity_listA, attr_listB, entity_listB = load_attributes(path)
    datasetA = SingleEntityDataset(entity_listA, attr_listA, lm='sent-bert', max_len=128, add_token=hp.add_token)
    datasetB = SingleEntityDataset(entity_listB, attr_listB, lm='sent-bert', max_len=128, add_token=hp.add_token)
    blocker = CLSepModel(lm='sent-bert')
    blocker = blocker.cuda()
    bk_opt = AdamW(blocker.parameters(), lr=hp.lr)
    if hp.fp16:
        blocker, bk_opt = amp.initialize(blocker, bk_opt, opt_level='O2')
    if hp.ckpt_type == 'best':
        blocker.load_state_dict(torch.load(os.path.join('checkpoints', hp.dataset, str(hp.topK), str(hp.total_budget), str(hp.run_id), 'blocker_model.pt')))
    else:
        blocker.load_state_dict(torch.load(os.path.join('checkpoints', hp.dataset, str(hp.topK), str(hp.total_budget), str(hp.run_id), 'last_blocker_model.pt')))
    embeddingA = get_SentEmb(blocker, datasetA)
    embeddingB = get_SentEmb(blocker, datasetB)
    return embeddingA, embeddingB

def load_matcher(hp):
    model = DittoConcatModel(lm = hp.lm)
    model = model.cuda()
    optimizer = AdamW(model.parameters(), lr=hp.lr)
    if hp.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O2')
    if hp.ckpt_type == 'best':
        model.load_state_dict(torch.load(os.path.join('checkpoints', hp.dataset, str(hp.topK), str(hp.total_budget), str(hp.run_id), 'matcher_model.pt')))
    else:
        model.load_state_dict(torch.load(os.path.join('checkpoints', hp.dataset, str(hp.topK), str(hp.total_budget), str(hp.run_id), 'last_matcher_model.pt')))
    return model

def gen_testset(topkA, distA, test_idxs, start, end):
    test_data = []
    for idxA in test_idxs:
        for idxB, dist in zip(topkA[idxA][start:end], distA[idxA][start:end]):
            test_data.append([idxA, idxB, dist])
    test_data = np.array(test_data)
    return test_data

def pred(model, dataset):
    iterator = DataLoader(dataset=dataset, batch_size=128, collate_fn=dataset.pad)
    model.eval()
    y_truth, y_pre, y_scores = [], [], []
    e1_list, e2_list = [], []
    for i, batch in enumerate(iterator):
        x, _, y = batch   
        x = x.cuda()
        logits = model(x)
        scores = logits.argmax(-1)
        for item in scores.cpu().numpy().tolist():
            y_pre.append(item)
        y_scores.extend(logits.softmax(-1)[:,1].cpu().detach().numpy().tolist())
    return np.array(y_pre), np.array(y_scores)

def load_test_gt(hp):
    ''' load gt only for test idxs '''
    gt = load_gt(hp)
    gt = gt.reset_index()
    gt = gt.set_index('ltable_id')
    test_idxs = pd.read_csv(os.path.join(hp.path, hp.dataset, 'test_idxs.csv'), index_col=0)
    test_idxs['flag'] = 1 
    test_idxs = test_idxs.set_index('ltable_id')
    df = test_idxs.join(gt, how = 'inner')
    df = df.fillna(0)
    df = df[['rtable_id', 'label']]
    df = df.reset_index()
    df = df.set_index(['ltable_id', 'rtable_id'])
    return df 

def valid_gap(valid_df, p):
    def cal_gap(df_sub):
        pos = df_sub[df_sub.label==1]
        if len(pos) == 0:
            return -1
        pos_sim = np.min(pos['sim'].values)
        neg = df_sub[df_sub.label==0]['sim'].values
        neg_sim = np.max(neg)
        return pos_sim - neg_sim
    gaps = valid_df.groupby('ltable_id').apply(cal_gap).values
    return np.percentile(gaps[gaps>0], p)

def update(df_all, k, test_idxs, gap, min_sim, thr):
    rm_testids = set()
    for lid in test_idxs:
        dfsub = df_all[df_all.ltable_id==lid] # [ltable_id, rtable_id, sim, pred]
        dfsub = dfsub.sort_values(by = 'sim', ascending = False)
        sims = dfsub['sim'].values
        if (((dfsub['pred']==1).sum() > 0) and (np.sum(dfsub['pred'].values[-k:]) == 0)) or (((dfsub['pred']==1).sum() == 0) and (len(dfsub)==50 or (sims[-1] < min_sim))):
            rm_testids.add(lid)

    return test_idxs - rm_testids

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
    parser.add_argument("--blocker_type", type=str, default='sentbert') # sentbert/magellan
    parser.add_argument("--validation_with_pseudo", type=bool, default=False)
    parser.add_argument("--aug_type", type=str, default='random')
    parser.add_argument("--num_iter", type=int, default=5)
    parser.add_argument("--p", type=int, default=10)
    parser.add_argument("--ckpt_type", type=str, default='last')
    
    hp = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = hp.gpu

    alpha = 1.65
    dataset = hp.dataset
    dataset_dict = {
        'AG': 'Amazon-Google',\
        'BR': 'BeerAdvo-RateBeer',\
        'DA': 'DBLP-ACM',\
        'DS': 'DBLP-Scholar',\
        'FZ': 'Fodors-Zagats',\
        'IA': 'iTunes-Amazon',\
        'WA': 'Walmart-Amazon',\
        'AB': 'Abt-Buy',
        'M': 'monitor'
    }
    dataset = dataset_dict.get(dataset, dataset)

    if 'wdc' in dataset:
        hp.path = "../data4"
    elif 'camera1' in dataset or 'monitor' in dataset:
        hp.path = '../data4/Alaska'
        hp.dataset = dataset
    else:
        hp.path = "../data4/ER-Magellan"
        if 'Abt' in dataset:
            hp.dataset = os.path.join("Textual", dataset)
        else:
            hp.dataset = os.path.join("Structured", dataset)    
    
    gt = load_test_gt(hp)
    test_idxs = set(list(pd.read_csv(os.path.join(hp.path, hp.dataset, 'test_idxs.csv'), index_col=0)['ltable_id'].values))
    attr_listA, entity_listA, attr_listB, entity_listB = load_attributes(os.path.join(hp.path, hp.dataset))
    model = load_matcher(hp)

    all_BK_rec_list, all_f1_list, all_pre_list, all_rec_list = [], [], [], []
    embeddingA, embeddingB = get_emb(hp)
    topkA, distA, sim_score = get_topK(embeddingA, embeddingB, topK = min(500, len(embeddingB)))
    start, end, k = 0, 5, 5
    valid_df = pd.read_csv(os.path.join(hp.path, hp.dataset, 'valid0207.csv'))
    valid_set = GTDatasetWithLabel(valid_df.values, entity_listA, entity_listB, attr_listA, lm=hp.lm, concat=True, shuffle=False)
    y_pred, y_scores = pred(model, valid_set)
    valid_df['score'] = y_scores
    thr = np.percentile(valid_df[(valid_df.label==0)&(valid_df.score<0.5)]['score'].values, 100-hp.p)
    print('neg thr', thr)
    print('pos thr',valid_df[(valid_df.label==0)]['score'].min())
    
    sim = []
    for l, r, _ in valid_df[['ltable_id', 'rtable_id', 'label']].values:
        sim.append(sim_score[l][r])
    valid_df['sim'] = sim
    gap = valid_gap(valid_df, hp.p)
    pos_sim = valid_df[valid_df.label==1]['sim'].values
    min_sim = np.min(pos_sim)
    print('gap', gap, 'min sim', min_sim, 'std sim', np.std(pos_sim))

    df_all = None
    iter = 0
    while len(test_idxs) > 0:
        iter += 1
        print(iter, len(test_idxs))
        test_data = gen_testset(topkA, distA, test_idxs, start, end)
        test_set = GTDatasetWithLabel(test_data, entity_listA, entity_listB, attr_listA, lm=hp.lm, concat=True, shuffle=False)
        y_pred, y_scores = pred(model, test_set)
        df_test = pd.DataFrame({'ltable_id': test_data[:,0], 'rtable_id': test_data[:,1], 'sim': test_data[:,2], 'pred': y_pred, 'score': y_scores})
        if df_all is None:
            df_all = df_test
        else:
            df_all = df_all.append(df_test)
        test_idxs = update(df_all, k, test_idxs, gap, min_sim-alpha*np.std(pos_sim), thr)              
        start = end
        end += k
        if start >= min(len(topkA[0]), 50):
            break
    
    df_all = df_all.set_index(['ltable_id', 'rtable_id'])
    df1 = gt.join(df_all)
    BK_recall = len(df1[(df1.label==1) & (pd.isna(df1.pred)==False)]) / float(len(df1[(df1.label==1)]))
    # matcher precision recall
    df2 = df_all.join(gt)
    df2 = df2.fillna(0)
    pre = precision_score(df2['label'].values, df2['pred'].values)
    rec = recall_score(df2['label'].values, df2['pred'].values)
    rec = BK_recall * rec 
    try:
        f1 = (2 * pre * rec) / (pre + rec)
    except:
        f1 = 0

print('run_id_'+str(hp.run_id), hp.topK, 'Final BK size', hp.total_budget, len(df_all))
print('run_id_'+str(hp.run_id), hp.topK, 'Final Test BK Recall', hp.total_budget, BK_recall)
print('run_id_'+str(hp.run_id), hp.topK, 'Final Test F1', hp.total_budget, f1)
print('run_id_'+str(hp.run_id), hp.topK, 'Final Test Precision', hp.total_budget, pre)
print('run_id_'+str(hp.run_id), hp.topK, 'Final Test Recall', hp.total_budget, rec)