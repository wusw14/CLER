from sentence_transformers import util, SentenceTransformer
import torch
import pandas as pd 
import os
import numpy as np 
from torch.utils import data
from transformers import AutoTokenizer
import sys 
from model import CLSepModel

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# map lm name to huggingface's pre-trained model names
lm_mp = {'roberta': 'roberta-base',
         'distilbert': 'distilbert-base-uncased',
         'sent-bert': 'sentence-transformers/stsb-roberta-base'}

def get_tokenizer(lm):
    if lm in lm_mp:
        return AutoTokenizer.from_pretrained(lm_mp[lm])
    else:
        return AutoTokenizer.from_pretrained(lm)

class SingleEntityDataset(data.Dataset):
    def __init__(self, data, attr_list, lm='roberta', max_len=256, add_token=True):
        self.tokenizer = get_tokenizer(lm)
        self.data = data
        self.attr_list = attr_list
        self.max_len = max_len
        self.add_token = add_token
        self.entitytext = self.combine_token_feature_text()

    def __len__(self):
        """Return the size of the dataset."""
        return len(self.data)
    
    def attr2text(self, attr_list, entity):
        text = []
        for i, e in enumerate(entity):
            if len(str(e)) > 0 and (type(e) == str or type(e) == float and np.isnan(e) == False):
                if self.add_token:  
                    text.append('COL %s VAL %s' %(attr_list[i], str(e)))
                else:
                    text.append(str(e))         
        text = ' '.join(text)
        return text 
    
    def combine_token_feature_text(self):
        entity_text = []
        for entity in self.data:
            entity = list(entity) # list of attribute values
            entity = [str(e) for e in entity]
            entity_text.append(self.attr2text(list(self.attr_list.copy()), list(entity)))
        return entity_text
    
    def __getitem__(self, idx):
        """Return a tokenized item of the dataset.

        Args:
            idx (int): the index of the item

        Returns:
            List of int: token ID's of the entity
            List of int: mask of the entity
        """
        text = self.entitytext[idx]
        x1 = self.tokenizer(text=text, max_length=self.max_len, truncation=True)
        return x1['input_ids'], x1['attention_mask']

    @staticmethod
    def pad(batch):
        x, x_mask = zip(*batch)
        maxlen = max([len(xi) for xi in x])
        x = [xi + [0]*(maxlen - len(xi)) for xi in x]
        x_mask = [xi + [0]*(maxlen - len(xi)) for xi in x_mask]
        return torch.LongTensor(x), torch.LongTensor(x_mask)

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

def load_gt(path, dataset):
    gt = pd.read_csv(os.path.join(path, dataset, 'matches.csv'), index_col=0) # ltable_id,rtable_id,label
    if 'wdc' in dataset:
        gt = gt[gt['ltable_id'] != gt['rtable_id']]
    gt = gt.set_index(['ltable_id', 'rtable_id'])
    return gt

def load_test_gt(path, dataset):
    ''' load gt only for test idxs '''
    gt = load_gt(path, dataset)
    gt = gt.reset_index()
    gt = gt.set_index('ltable_id')
    test_idxs = pd.read_csv(os.path.join(path, dataset, 'test_idxs.csv'), index_col=0)
    test_idxs['flag'] = 1 
    test_idxs = test_idxs.set_index('ltable_id')
    df = test_idxs.join(gt, how = 'inner')
    df = df.fillna(0)
    df = df[['rtable_id', 'label']]
    df = df.reset_index()
    df = df.set_index(['ltable_id', 'rtable_id'])
    return df 

def load_model(path, dataset, K, budget, run_id, iter):
    dataset_category = path.split('/')[-1]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    EmbModel = CLSepModel(device=device, lm='sent-bert')
    EmbModel.load_state_dict(torch.load(os.path.join('checkpoints', dataset, str(K), budget, str(run_id), 'Iter'+str(iter)+'_blocker_model.pt')))
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

def get_emb(path, dataset, K, budget, run_id, iter):
    attr_listA, entity_listA = read_entity(os.path.join(path, dataset), table='tableA', shuffle=False)
    if 'wdc' in dataset:
        attr_listB, entity_listB = read_entity(os.path.join(path, dataset), table='tableA', shuffle=False)
    else:
        attr_listB, entity_listB = read_entity(os.path.join(path, dataset), table='tableB', shuffle=False)
    datasetA = SingleEntityDataset(entity_listA, attr_listA, lm='sent-bert', max_len=128, add_token=True)
    datasetB = SingleEntityDataset(entity_listB, attr_listB, lm='sent-bert', max_len=128, add_token=True)
    EmbModel = load_model(path, dataset, K, budget, run_id, iter)
    embeddingA = get_SentEmb(EmbModel, datasetA)
    embeddingB = get_SentEmb(EmbModel, datasetB)
    return embeddingA, embeddingB

# def get_emb(path, dataset, K, budget, run_id):
#     attr_listA, entity_listA = read_entity(os.path.join(path, dataset), table='tableA', shuffle=False)
#     if 'wdc' in dataset:
#         attr_listB, entity_listB = read_entity(os.path.join(path, dataset), table='tableA', shuffle=False)
#     else:
#         attr_listB, entity_listB = read_entity(os.path.join(path, dataset), table='tableB', shuffle=False)
#     datasetA = SingleEntityDataset(entity_listA, attr_listA, lm='sent-bert', max_len=128, add_token=True)
#     datasetB = SingleEntityDataset(entity_listB, attr_listB, lm='sent-bert', max_len=128, add_token=True)
#     EmbModel = SentenceTransformer('stsb-roberta-base')
#     embeddingA = EmbModel.encode(datasetA.entitytext, batch_size=512)
#     embeddingB = EmbModel.encode(datasetB.entitytext, batch_size=512)
#     return embeddingA, embeddingB

def get_topK(embeddingA, embeddingB, topK = 5):
    embeddingA = torch.tensor(embeddingA).cuda()
    embeddingB = torch.tensor(embeddingB).cuda()
    sim_score = util.pytorch_cos_sim(embeddingA, embeddingB)
    _, topkA = torch.topk(sim_score, k=topK, dim=1) # topkA [sizeA, K]
    topkA = topkA.cpu().numpy()
    return topkA

def cal_recall(topkA, gt, test_idxs):
    gt = gt[gt.label == 1]
    ltable_ids, rtable_ids = [], []
    for idxA, idxB_list in enumerate(topkA):
        if test_idxs is not None and idxA not in test_idxs:
            continue
        ltable_ids.extend([idxA] * len(idxB_list))
        rtable_ids.extend(list(idxB_list))
    df = pd.DataFrame({'ltable_id': ltable_ids, 'rtable_id': rtable_ids, 'flag': [1]*len(ltable_ids)})
    df = df.set_index(['ltable_id', 'rtable_id'])
    df = gt.join(df)
    recall = 1 - pd.isna(df['flag']).sum() / float(len(df))
    return recall

def cal_posrate(topkA, gt):
    ltable_ids, rtable_ids = [], []
    for idxA, idxB_list in enumerate(topkA):
        ltable_ids.extend([idxA] * len(idxB_list))
        rtable_ids.extend(list(idxB_list))
    df = pd.DataFrame({'ltable_id': ltable_ids, 'rtable_id': rtable_ids, 'flag': [1]*len(ltable_ids)})
    df = df.set_index(['ltable_id', 'rtable_id'])
    df = df.join(gt)
    ratio = len(df[df['label']==1]) / float(len(df))
    return ratio    

dataset = sys.argv[1]

dataset_dict = {
    'AG': 'Amazon-Google',\
    'BR': 'BeerAdvo-RateBeer',\
    'DA': 'DBLP-ACM',\
    'DS': 'DBLP-Scholar',\
    'FZ': 'Fodors-Zagats',\
    'IA': 'iTunes-Amazon',\
    'WA': 'Walmart-Amazon',\
    'AB': 'Abt-Buy'
}
dataset = dataset_dict.get(dataset, dataset)
# K = int(sys.argv[2])
K = 10
if dataset == 'AB':
    path = '../data4/ER-Magellan'
    dataset = os.path.join('Textual', dataset)
elif dataset == 'camera1' or dataset == 'monitor':
    path = '../data4/Alaska'
else:
    path = '../data4/ER-Magellan'
    dataset = os.path.join('Structured', dataset)

gt = load_test_gt(path, dataset)
test_idxs = pd.read_csv(os.path.join(path, dataset, 'test_idxs.csv'), index_col=0).values
iter_recall_dict = {}
budget = int(sys.argv[2])
run_id = 0
for iter in range(1, 11):
    recall_list_all = []
    # try:
    if os.path.exists(os.path.join('checkpoints', dataset, str(K), str(budget), str(run_id), 'Iter'+str(iter)+'_blocker_model.pt')) == False:
        continue
    embeddingA, embeddingB = get_emb(path, dataset, K, str(budget), run_id, iter)
    recall_list = []
    ratio_list = []
    topkA = get_topK(embeddingA, embeddingB, 50)
    for k in [1,2,5,10,20,50]:
        recall = cal_recall(topkA[:,:k], gt, test_idxs)
        recall_list.append(str(recall))
    iter_recall_dict[iter] = recall_list
    print('iter', iter, ' '.join(recall_list))
    # except:
    #     pass

result = None
for iter in range(1, 11):
    try:
        result = iter_recall_dict[iter]
    except:
        pass
    print(sys.argv[1], budget, iter, ' '.join(result))