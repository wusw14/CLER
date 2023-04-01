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
    y_truth, y_pre = [], []
    for i, batch in enumerate(iterator):
        x, _, y = batch   
        logits = model(x)
        scores = logits.argmax(-1)
        for item in scores.cpu().numpy().tolist():
            y_pre.append(item)
    return y_pre

def load_test_gt(hp):
    ''' load gt only for test idxs '''
    gt = load_gt(hp)
    gt = gt.reset_index()
    gt = gt.set_index('ltable_id')
    test_idxs = pd.read_csv(os.path.join(hp.path, hp.dataset, 'test_idxs.csv'), index_col=0)
    test_idxs['flag'] = 1 
    test_idxs = test_idxs.set_index('ltable_id')
    df = test_idxs.join(gt)
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

    attr_listA, entity_listA, attr_listB, entity_listB = load_attributes(os.path.join(hp.path, hp.dataset))
    test_data = pd.read_csv(os.path.join(hp.path, hp.dataset, 'test_magellan.csv')).values
    test_set = GTDatasetWithLabel(test_data, entity_listA, entity_listB, attr_listA, lm=hp.lm, concat=True, shuffle=False)
    
    all_BK_rec_list, all_f1_list, all_pre_list, all_rec_list = [], [], [], []
    model = load_matcher(hp)
    y_pred = pred(model, test_set)
    pre = precision_score(test_data[:,2], y_pred)
    rec = recall_score(test_data[:,2], y_pred)
    try:
        f1 = (2 * pre * rec) / (pre + rec)
    except:
        f1 = 0            
    all_f1_list.append(f1)
    all_pre_list.append(pre)
    all_rec_list.append(rec)

    print('run_id_'+str(hp.run_id), hp.topK, 'Final Test F1', hp.total_budget, f1)
    print('run_id_'+str(hp.run_id), hp.topK, 'Final Test Precision', hp.total_budget, pre)
    print('run_id_'+str(hp.run_id), hp.topK, 'Final Test Recall', hp.total_budget, rec)