import torch

from torch.utils import data
from transformers import AutoTokenizer

import random
import numpy as np 
from utils import *

# map lm name to huggingface's pre-trained model names
lm_mp = {'roberta': 'roberta-base',
         'distilbert': 'distilbert-base-uncased',
         'sent-bert': 'sentence-transformers/stsb-roberta-base'}

def augment(attr_list, feat_list, aug_type = 'identical'):
    def token_len(feat_list):
        feat_tokens, feat_len = [], []
        for v in feat_list:
            feat_tokens.append(v.split(' '))
            feat_len.append(len(feat_tokens[-1]))
        return feat_tokens, feat_len

    # ['identical', 'swap_token', 'del_token', 'swap_col', 'del_col', 'shuffle_token', 'shuffle_col']
    if aug_type == 'del_token':
        span_len = random.randint(1, 2)
        feat_tokens, feat_len = token_len(feat_list)
        try_time = 0
        while True:
            try_time += 1
            if try_time > 5:
                break
            idx = random.randint(0, len(feat_tokens)-1)
            if feat_len[idx] > span_len:
                pos = random.randint(0, feat_len[idx] - span_len)
                feat_tokens[idx] = feat_tokens[idx][:pos] + feat_tokens[idx][pos+span_len:]
                break
        feat_list = [' '.join(tokens) for tokens in feat_tokens]                        
    elif aug_type == 'swap_token':
        span_len = random.randint(2, 4)
        feat_tokens, feat_len = token_len(feat_list)
        try_time = 0
        while True:
            try_time += 1
            if try_time > 5:
                break
            idx = random.randint(0, len(feat_tokens)-1)
            if feat_len[idx] >= span_len:
                pos = random.randint(0, feat_len[idx] - span_len)
                subattr = feat_tokens[idx][pos:pos+span_len]
                np.random.shuffle(subattr)
                feat_tokens[idx] = feat_tokens[idx][:pos] + subattr + feat_tokens[idx][pos+span_len:]
                break
        feat_list = [' '.join(tokens) for tokens in feat_tokens]
    elif aug_type == 'swap_col':
        idx1 = random.randint(0, len(feat_list)-1)
        idx2 = random.randint(0, len(feat_list)-1)
        feat_list[idx1], feat_list[idx2] = feat_list[idx2], feat_list[idx1]
        attr_list[idx1], attr_list[idx2] = attr_list[idx2], attr_list[idx1]
    elif aug_type == 'del_col':
        idx = random.randint(0, len(feat_list)-1)
        del feat_list[idx]
        del attr_list[idx]
    elif aug_type == 'shuffle_col':
        shuffled_idx = np.array(list(range(len(attr_list))))
        np.random.shuffle(shuffled_idx)
        attr_list = list(np.array(attr_list)[shuffled_idx])
        feat_list = list(np.array(feat_list)[shuffled_idx])
    elif aug_type == 'shuffle_token':
        feat_tokens, feat_len = token_len(feat_list)
        idx = random.randint(0, len(feat_tokens)-1)
        np.random.shuffle(feat_tokens[idx])
        feat_list = [' '.join(tokens) for tokens in feat_tokens]

    return attr_list, feat_list

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

class AugDataset(data.Dataset):
    """
        for contrastive learning
        EM dataset: generate augmented text pair
        return (x1, x1_mask, x2, x2_mask) or (x1, x1_mask)  
    """

    def __init__(self, data, attr_list, lm='roberta', max_len=256, add_token=True, aug=True, aug_type='random', aug_both=False):
        self.tokenizer = get_tokenizer(lm)
        self.data = data
        self.attr_list = attr_list
        self.max_len = max_len
        self.add_token = add_token
        self.aug = aug
        self.aug_type = aug_type
        self.aug_both = aug_both

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
            List of int: token ID's of the original entity
            List of int: token ID's of the augmented entity
        """
        entity = list(self.data[idx]) # list of attribute values
        entity = [str(e) for e in entity]

        if self.aug == False:
            org_text = self.attr2text(list(self.attr_list.copy()), list(entity))
            x1 = self.tokenizer(text=org_text, max_length=self.max_len, truncation=True)
            return x1['input_ids'], x1['attention_mask']

        if self.aug_type != 'random':
            aug_type = self.aug_type
        else:
            if len(self.attr_list) > 1:
                # aug_type = random.choice(['identical', 'swap_token', 'del_token', 'shuffle_token', 'swap_col', 'del_col', 'shuffle_col'])
                aug_type = random.choice(['shuffle_token', 'shuffle_col', 'del_token'])
            else:
                # aug_type = random.choice(['identical', 'del_token', 'swap_token', 'shuffle_token'])
                aug_type = random.choice(['del_token', 'shuffle_token'])
        if self.aug_both:
            aug_attr1, aug_entity1 = augment(list(self.attr_list.copy()), list(entity.copy()), aug_type)
            aug_text1 = self.attr2text(aug_attr1, aug_entity1)
        else:
            aug_text1 = self.attr2text(list(self.attr_list.copy()), list(entity))
        x1 = self.tokenizer(text=aug_text1, max_length=self.max_len, truncation=True)
        pos_attr, pos_entity = augment(list(self.attr_list.copy()), list(entity.copy()), aug_type)
        aug_text2 = self.attr2text(pos_attr, pos_entity)
        x2 = self.tokenizer(text=aug_text2, max_length=self.max_len, truncation=True)

        return x1['input_ids'], x1['attention_mask'], x2['input_ids'], x2['attention_mask']

    @staticmethod
    def pad(batch):
        """Merge a list of dataset items into a train/test batch
        Args:
            batch (list of tuple): a list of dataset items
        Returns:
            LongTensor: x1 of shape (batch_size, seq_len)
            LongTensor: x2 of shape (batch_size, seq_len).
                        Elements of x1 and x2 are padded to the same length
            LongTensor: a batch of labels, (batch_size,)
        """
        if len(batch[0]) == 4:
            x1, x1_mask, x2, x2_mask = zip(*batch)
            maxlen1 = max([len(x) for x in x1])
            maxlen2 = max([len(x) for x in x2])
            maxlen = max(maxlen1, maxlen2)
            x1 = [xi + [0]*(maxlen - len(xi)) for xi in x1]
            x2 = [xi + [0]*(maxlen - len(xi)) for xi in x2]
            x1_mask = [xi + [0]*(maxlen - len(xi)) for xi in x1_mask]
            x2_mask = [xi + [0]*(maxlen - len(xi)) for xi in x2_mask]
            return torch.LongTensor(x1), torch.LongTensor(x1_mask), torch.LongTensor(x2), torch.LongTensor(x2_mask)
        else:
            x, x_mask = zip(*batch)
            maxlen = max([len(xi) for xi in x])
            x = [xi + [0]*(maxlen - len(xi)) for xi in x]
            x_mask = [xi + [0]*(maxlen - len(xi)) for xi in x_mask]
            return torch.LongTensor(x), torch.LongTensor(x_mask)

class AugDatasetWithLabel(AugDataset):
    """
        EM dataset: generate augmented text pair
        Keeping the positive labeled pair 
        return (x1, x1_mask, x2, x2_mask) or (x1, x1_mask)  
    """
    def __init__(self, samples, entityA, entityB=None, attr_list=None, lm='roberta', max_len=256, add_token=True, concat=False, shuffle=False, same_table=False, aug_type='random'):
        self.tokenizer = get_tokenizer(lm)
        self.entityA = entityA
        self.entityB = entityB
        self.attr_list = attr_list
        self.max_len = max_len
        self.add_token = add_token
        self.concat = concat
        self.samples = samples # only positive pairs
        self.shuffle = shuffle
        self.same_table = same_table
        self.aug_type = aug_type
        self.paired_pos()
    
    def paired_pos(self):
        self.A_pos = defaultdict(list)
        self.B_pos = defaultdict(list)
        self.A_neg = defaultdict(list)
        self.B_neg = defaultdict(list)
        for e1, e2, y, w in self.samples:
            if y == 1:
                self.A_pos[e1].append([e2, w])
                self.B_pos[e2].append([e1, w])
            else:
                self.A_neg[e1].append([e2, w])
                self.B_neg[e2].append([e1, w])

    def __len__(self):
        """Return the size of the dataset."""
        if self.same_table == False:
            return len(self.entityA) + len(self.entityB)
        else:
            return len(self.entityA)

    def __getitem__(self, idx):
        """Return a tokenized item of the dataset.

        Args:
            idx (int): the index of the item

        Returns:
            List of int: token ID's of the original entity
            List of int: token ID's of the augmented entity
        """
        if idx >= len(self.entityA):
            e2 = idx - len(self.entityA)
            entity = self.entityB[e2]
            entity = [str(e) for e in entity]
            
            cand = self.B_pos.get(e2, [])
            if len(cand) > 0:
                idx = np.random.choice(range(len(cand)))
                pos_entity = self.entityA[int(cand[idx][0])]
                w = cand[idx][1]
            else:
                pos_entity = None
                w = 1
        else:
            e1 = idx
            entity = self.entityA[idx]
            entity = [str(e) for e in entity]
            
            cand = self.A_pos.get(e1, [])
            if len(cand) > 0:
                idx = np.random.choice(range(len(cand)))
                pos_entity = self.entityB[int(cand[idx][0])]
                w = cand[idx][1]
            else:
                pos_entity = None
                w = 1

        if pos_entity is not None:
            pos_attr = list(self.attr_list.copy())
        else:
            if self.aug_type != 'random':
                aug_type = self.aug_type
            else:
                if len(self.attr_list) > 1:
                    aug_type = random.choice(['shuffle_token', 'shuffle_col', 'del_token'])
                else:
                    aug_type = random.choice(['del_token', 'shuffle_token'])
            pos_attr, pos_entity = augment(list(self.attr_list.copy()), list(entity.copy()), aug_type)

        org_text = self.attr2text(list(self.attr_list.copy()), list(entity))
        x1 = self.tokenizer(text=org_text, max_length=self.max_len, truncation=True)
        pos_text = self.attr2text(pos_attr, pos_entity)
        x2 = self.tokenizer(text=pos_text, max_length=self.max_len, truncation=True)

        return x1['input_ids'], x1['attention_mask'], x2['input_ids'], x2['attention_mask'], w

    @staticmethod
    def pad(batch):
        x1, x1_mask, x2, x2_mask, w = zip(*batch)
        maxlen1 = max([len(x) for x in x1])
        maxlen2 = max([len(x) for x in x2])
        x1 = [xi + [0]*(maxlen1 - len(xi)) for xi in x1]
        x2 = [xi + [0]*(maxlen2 - len(xi)) for xi in x2]
        x1_mask = [xi + [0]*(maxlen1 - len(xi)) for xi in x1_mask]
        x2_mask = [xi + [0]*(maxlen2 - len(xi)) for xi in x2_mask]
        return torch.LongTensor(x1), torch.LongTensor(x1_mask), torch.LongTensor(x2), torch.LongTensor(x2_mask), torch.FloatTensor(w)

class GTDatasetWithLabel(data.Dataset):
    """
        for supervised learning
        EM dataset: load the labeled data
        return (x1, x1_mask, x2, x2_mask, y) or (x1, x1_mask, y)  
    """
    def __init__(self, samples, entityA, entityB, attr_list, lm='roberta', max_len=256, add_token=True, concat=False, shuffle=False, random_neg=False):
        self.tokenizer = get_tokenizer(lm)
        self.entityA = entityA
        self.entityB = entityB
        self.attr_list = attr_list
        self.max_len = max_len
        self.add_token = add_token
        self.concat = concat
        self.samples = self.get_text_samples(samples)
        self.shuffle = shuffle
        self.random_neg = random_neg
    
    def __len__(self):
        """Return the size of the dataset."""
        return len(self.samples)
    
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

    def get_text_samples(self, samples):
        samples_text = []
        for e1, e2, y in samples:
            entity1 = self.entityA[int(e1)]
            entity2 = self.entityB[int(e2)]
            text1 = self.attr2text(self.attr_list, entity1)
            text2 = self.attr2text(self.attr_list, entity2)            
            samples_text.append([text1, text2, y, e1, e2])
        return samples_text
    
    def __getitem__(self, idx):
        text1, text2, y, e1, e2 = self.samples[idx]

        if self.random_neg and y == 0:
            text2 = self.attr2text(self.attr_list, random.choice(self.entityB))

        if self.shuffle and random.randint(0, 1):
            text1, text2 = text2, text1
        if self.concat:
            x = self.tokenizer(text=text1, text_pair=text2, max_length=self.max_len, add_special_tokens=True, truncation=True)
            return x['input_ids'], x['attention_mask'], y, e1, e2
        else:
            x1 = self.tokenizer(text=text1, max_length=self.max_len, add_special_tokens=True, truncation=True)
            x2 = self.tokenizer(text=text2, max_length=self.max_len, add_special_tokens=True, truncation=True)
            return x1['input_ids'], x1['attention_mask'], x2['input_ids'], x2['attention_mask'], y, e1, e2
    @staticmethod
    def pad(batch):
        if len(batch[0]) == 7:
            x1, x1_mask, x2, x2_mask, y, e1, e2 = zip(*batch)
            maxlen1 = max([len(x) for x in x1])
            maxlen2 = max([len(x) for x in x2])
            maxlen = max(maxlen1, maxlen2)
            x1 = [xi + [0]*(maxlen - len(xi)) for xi in x1]
            x2 = [xi + [0]*(maxlen - len(xi)) for xi in x2]
            x1_mask = [xi + [0]*(maxlen - len(xi)) for xi in x1_mask]
            x2_mask = [xi + [0]*(maxlen - len(xi)) for xi in x2_mask]
            return torch.LongTensor(x1), torch.LongTensor(x1_mask), torch.LongTensor(x2), torch.LongTensor(x2_mask), torch.LongTensor(y)
        else:
            x, x_mask, y, e1, e2 = zip(*batch)
            maxlen = max([len(xi) for xi in x])
            x = [xi + [0]*(maxlen - len(xi)) for xi in x]
            x_mask = [xi + [0]*(maxlen - len(xi)) for xi in x_mask]
            return torch.LongTensor(x), torch.LongTensor(x_mask), torch.LongTensor(y)

class GTDatasetWithLabelWeights(data.Dataset):
    """
        for supervised learning
        EM dataset: load the labeled data
        return (x1, x1_mask, x2, x2_mask, y) or (x1, x1_mask, y)  
    """
    def __init__(self, samples, entityA, entityB, attr_list, lm='roberta', max_len=256, add_token=True, concat=False, shuffle=False, random_neg=False):
        self.tokenizer = get_tokenizer(lm)
        self.entityA = entityA
        self.entityB = entityB
        self.attr_list = attr_list
        self.max_len = max_len
        self.add_token = add_token
        self.concat = concat
        self.samples = self.get_text_samples(samples)
        self.shuffle = shuffle
        self.random_neg = random_neg
    
    def __len__(self):
        """Return the size of the dataset."""
        return len(self.samples)
    
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

    def get_text_samples(self, samples):
        samples_text = []
        for e1, e2, y, w in samples:
            entity1 = self.entityA[int(e1)]
            entity2 = self.entityB[int(e2)]
            text1 = self.attr2text(self.attr_list, entity1)
            text2 = self.attr2text(self.attr_list, entity2)            
            samples_text.append([text1, text2, y, w])
        return samples_text
    
    def __getitem__(self, idx):
        text1, text2, y, w = self.samples[idx]

        if self.random_neg and y == 0:
            text2 = self.attr2text(self.attr_list, random.choice(self.entityB))

        if self.shuffle and random.randint(0, 1):
            text1, text2 = text2, text1
        if self.concat:
            x = self.tokenizer(text=text1, text_pair=text2, max_length=self.max_len, add_special_tokens=True, truncation=True)
            return x['input_ids'], x['attention_mask'], y, w
        else:
            x1 = self.tokenizer(text=text1, max_length=self.max_len, add_special_tokens=True, truncation=True)
            x2 = self.tokenizer(text=text2, max_length=self.max_len, add_special_tokens=True, truncation=True)
            return x1['input_ids'], x1['attention_mask'], x2['input_ids'], x2['attention_mask'], y, w
    @staticmethod
    def pad(batch):
        if len(batch[0]) == 6:
            x1, x1_mask, x2, x2_mask, y, w = zip(*batch)
            maxlen1 = max([len(x) for x in x1])
            maxlen2 = max([len(x) for x in x2])
            maxlen = max(maxlen1, maxlen2)
            x1 = [xi + [0]*(maxlen - len(xi)) for xi in x1]
            x2 = [xi + [0]*(maxlen - len(xi)) for xi in x2]
            x1_mask = [xi + [0]*(maxlen - len(xi)) for xi in x1_mask]
            x2_mask = [xi + [0]*(maxlen - len(xi)) for xi in x2_mask]
            return torch.LongTensor(x1), torch.LongTensor(x1_mask), torch.LongTensor(x2), torch.LongTensor(x2_mask), torch.LongTensor(y), torch.FloatTensor(w)
        else:
            x, x_mask, y, w = zip(*batch)
            maxlen = max([len(xi) for xi in x])
            x = [xi + [0]*(maxlen - len(xi)) for xi in x]
            x_mask = [xi + [0]*(maxlen - len(xi)) for xi in x_mask]
            return torch.LongTensor(x), torch.LongTensor(x_mask), torch.LongTensor(y), torch.FloatTensor(w)