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
    _, topkA = torch.topk(sim_score, k=topK, dim=1) # topkA [sizeA, K]
    topkA = topkA.cpu().numpy()
    return topkA

def get_emb(hp):
    path = os.path.join(hp.path, hp.dataset)
    attr_listA, entity_listA, attr_listB, entity_listB = load_attributes(path)
    datasetA = SingleEntityDataset(entity_listA, attr_listA, lm='sent-bert', max_len=128, add_token=hp.add_token)
    datasetB = SingleEntityDataset(entity_listB, attr_listB, lm='sent-bert', max_len=128, add_token=hp.add_token)
    try:
        blocker = CLSepModel(lm='sent-bert')
        blocker = blocker.cuda()
        bk_opt = AdamW(blocker.parameters(), lr=hp.lr)
        if hp.fp16:
            blocker, bk_opt = amp.initialize(blocker, bk_opt, opt_level='O2')
        blocker.load_state_dict(torch.load(os.path.join('checkpoints', hp.dataset, str(hp.topK), str(hp.total_budget), str(hp.run_id), 'last_blocker_model.pt')))
        embeddingA = get_SentEmb(blocker, datasetA)
        embeddingB = get_SentEmb(blocker, datasetB)
    except:
        EmbModel = SentenceTransformer('stsb-roberta-base')
        embeddingA = EmbModel.encode(datasetA.entitytext, batch_size=512)
        embeddingB = EmbModel.encode(datasetB.entitytext, batch_size=512)        
    return embeddingA, embeddingB

def load_matcher(hp):
    model = DittoConcatModel(lm = hp.lm)
    model = model.cuda()
    optimizer = AdamW(model.parameters(), lr=hp.lr)
    if hp.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O2')
    model.load_state_dict(torch.load(os.path.join('checkpoints', hp.dataset, str(hp.topK), str(hp.total_budget), str(hp.run_id), 'last_matcher_model.pt')))
    return model

def gen_testset(hp, topkA, K = 10):
    test_idxs = pd.read_csv(os.path.join(hp.path, hp.dataset, 'test_idxs.csv'), index_col=0)['ltable_id'].values
    test_data = []
    for idxA in test_idxs:
        for idxB in topkA[idxA][:K]:
            test_data.append([idxA, idxB, 1])
    test_data = np.array(test_data)
    return test_data

def pred(model, dataset):
    iterator = DataLoader(dataset=dataset, batch_size=128, collate_fn=dataset.pad)
    model.eval()
    y_truth, y_pre, y_scores = [], [], []
    for i, batch in enumerate(iterator):
        x, _, y = batch   
        x = x.cuda()
        logits = model(x)
        scores = logits.argmax(-1)
        for item in scores.cpu().numpy().tolist():
            y_pre.append(item)
        y_scores.extend(logits.softmax(-1)[:,1].cpu().detach().numpy().tolist())
    return y_pre, y_scores

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

    all_BK_rec_list, all_f1_list, all_pre_list, all_rec_list = [], [], [], []
    embeddingA, embeddingB = get_emb(hp)
    topkA = get_topK(embeddingA, embeddingB, topK = 10)
    test_data = gen_testset(hp, topkA, K = 10)
    attr_listA, entity_listA, attr_listB, entity_listB = load_attributes(os.path.join(hp.path, hp.dataset))
    
    model = load_matcher(hp)
    test_set = GTDatasetWithLabel(test_data, entity_listA, entity_listB, attr_listA, lm=hp.lm, concat=True, shuffle=False)
    y_pred, y_scores = pred(model, test_set)
    df_pre = pd.DataFrame({'ltable_id': test_data[:,0], 'rtable_id': test_data[:,1], 'pred': y_pred, 'score': y_scores})
    df_pre = df_pre.set_index(['ltable_id', 'rtable_id'])
    df_pre.to_csv('debug1.csv')
    BK_rec_list, f1_list, pre_list, rec_list = [], [], [], []
    # for K in [1,5,10,20,50,100]:
    for K in [10]:
        test_data = gen_testset(hp, topkA, K = K)
        df_test = pd.DataFrame({'ltable_id': test_data[:,0], 'rtable_id': test_data[:,1], 'flag': test_data[:,2]})
        df_test = df_test.set_index(['ltable_id', 'rtable_id'])
        df_test = df_test.join(df_pre)
        df_test = df_test[['pred']]
        df_test = df_test.fillna(0)
        print('='*10, 'test', len(df_test))
        # blocker recall
        df1 = gt.join(df_test)
        BK_recall = len(df1[(df1.label==1) & (pd.isna(df1.pred)==False)]) / float(len(df1[(df1.label==1)]))
        # matcher precision recall
        df2 = df_test.join(gt)
        df2 = df2.fillna(0)
        pre = precision_score(df2['label'].values, df2['pred'].values)
        rec = recall_score(df2['label'].values, df2['pred'].values)
        rec = BK_recall * rec 
        try:
            f1 = (2 * pre * rec) / (pre + rec)
        except:
            f1 = 0
        BK_rec_list.append(BK_recall)
        f1_list.append(f1)
        rec_list.append(rec)
        pre_list.append(pre)
        # print('BK_recall', BK_recall)
        # print(pre, rec, f1)

    # print(' '.join(BK_rec_list))
    # print(' '.join(f1_list))
    all_BK_rec_list.append(BK_rec_list)
    all_f1_list.append(f1_list)
    all_pre_list.append(pre_list)
    all_rec_list.append(rec_list)
avg_BK_rec = [str(v) for v in np.nanmean(all_BK_rec_list, 0)]
avg_f1 = [str(v) for v in np.nanmean(all_f1_list, 0)]
avg_pre = [str(v) for v in np.nanmean(all_pre_list, 0)]
avg_rec = [str(v) for v in np.nanmean(all_rec_list, 0)]
print('run_id_'+str(hp.run_id), hp.topK, 'Final Test BK Recall', hp.total_budget, ' '.join(avg_BK_rec))
print('run_id_'+str(hp.run_id), hp.topK, 'Final Test F1', hp.total_budget, ' '.join(avg_f1))
print('run_id_'+str(hp.run_id), hp.topK, 'Final Test Precision', hp.total_budget, ' '.join(avg_pre))
print('run_id_'+str(hp.run_id), hp.topK, 'Final Test Recall', hp.total_budget, ' '.join(avg_rec))
# for f1 in all_f1_list:
#     print(f1)