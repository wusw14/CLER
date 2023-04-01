import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import numpy as np 

lm_mp = {'roberta': 'roberta-base',
         'distilbert': 'distilbert-base-uncased',
         'sent-bert': 'sentence-transformers/stsb-roberta-base'}

class CLConcatModel(nn.Module):
    def __init__(self, device='cuda', lm='roberta'):
        super().__init__()

        if lm in lm_mp:
            self.bert = AutoModel.from_pretrained(lm_mp[lm])
            print(lm_mp[lm])
        else:
            self.bert = AutoModel.from_pretrained(lm)
            print(lm)

        self.device = device

        # linear layer
        hidden_size = self.bert.config.hidden_size
        self.fc = torch.nn.Linear(hidden_size, 1)
    
    def forward(self, x1, x2):
        """Encode the left, right, and the concatenation of left+right.

        Args:
            x1 (LongTensor): a batch of ID's
            x2 (LongTensor): a batch of ID's 
        Returns:
            Tensor: binary prediction
        """
        batch_size = x1.shape[0]
        x1 = x1.to(self.device) # (batch_size, seq_len)
        x2 = x2.to(self.device)
        x1 = x1.repeat_interleave(batch_size, dim=0)
        x2 = x2.repeat(batch_size, 1) # [B*B, L]
        x = torch.concat([x1, x2], -1) # [B*B, 2L]
        hidden = self.bert(x)[0][:,0,:] # [B*B, size]
        scores = self.fc(hidden) # [B*B, 1]
        scores = torch.reshape(scores, [-1, batch_size]) # [B, B]

        return scores
    
    def infer(self, x1, x2):
        x = torch.concat([x1, x2], -1)
        x = x.to(self.device) # [B, 2L]
        hidden = self.bert(x)[0][:,0,:]
        scores = self.fc(hidden)
        scores = torch.reshape(scores, [-1])
        return scores

class CLSepModel(nn.Module):
    """A baseline model for EM."""

    def __init__(self, device='cuda', lm='roberta', pooling='cls'):
        super().__init__()

        if lm in lm_mp:
            self.bert = AutoModel.from_pretrained(lm_mp[lm])
            print(lm_mp[lm])
        else:
            self.bert = AutoModel.from_pretrained(lm)
            print(lm)

        self.device = device
        self.pooling = pooling

        # linear layer
        hidden_size = self.bert.config.hidden_size
        self.fc = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, x1, x1_mask):
        """Encode the left, right, and the concatenation of left+right.

        Args:
            x1 (LongTensor): a batch of ID's
            x2 (LongTensor, optional): a batch of ID's (augmented)

        Returns:
            Tensor: binary prediction
        """
        x1 = x1.to(self.device) # (batch_size, seq_len)
        x1_mask = x1_mask.to(self.device)
        h1 = self.bert(x1)[0] # [B, L, K]
        if self.pooling == 'mean':
            x1_mask_expanded = x1_mask.unsqueeze(-1).expand(h1.size()).to(torch.float16)
            emb1 = torch.sum(h1 * x1_mask_expanded, 1) / torch.clamp(x1_mask_expanded.sum(1), min=1e-9)
            emb1 = self.fc(emb1)
        else:
            emb1 = self.fc(h1[:,0,:])

        emb1 = F.normalize(emb1)

        return emb1
    
    def get_emb(self, x, x_mask):
        x = x.to(self.device) # (batch_size, seq_len)
        x_mask = x_mask.to(self.device)
        h = self.bert(x)[0]
        if self.pooling == 'mean':
            x_mask_expanded = x_mask.unsqueeze(-1).expand(h.size()).to(torch.float16)
            emb = torch.sum(h * x_mask_expanded, 1) / torch.clamp(x_mask_expanded.sum(1), min=1e-9)
        else:
            emb = h[:,0,:]
        # emb = self.fc(emb)
        emb = F.normalize(emb)
        return emb

class DittoConcatModel(nn.Module):
    """A baseline model for EM."""

    def __init__(self, device='cuda', lm='roberta'):
        super().__init__()

        if lm in lm_mp:
            self.bert = AutoModel.from_pretrained(lm_mp[lm])
            print(lm_mp[lm])
        else:
            self.bert = AutoModel.from_pretrained(lm)
            print(lm)

        self.device = device

        # linear layer
        hidden_size = self.bert.config.hidden_size
        self.fc = torch.nn.Linear(hidden_size, 2)

    def forward(self, x1):
        """Encode the left, right, and the concatenation of left+right.

        Args:
            x1 (LongTensor): a batch of ID's
            x2 (LongTensor, optional): a batch of ID's (augmented)

        Returns:
            Tensor: binary prediction
        """
        x1 = x1.to(self.device) # (batch_size, seq_len)
        h1 = self.bert(x1)[0][:,0,:] # [B, K]
        pred = self.fc(h1)
            
        # return pred
        return pred
    
    def get_emb_and_score(self, x1):
        x1 = x1.to(self.device) # (batch_size, seq_len)
        h1 = self.bert(x1)[0][:,0,:] # [B, K]
        pred = self.fc(h1)
        return h1, pred        

class DittoSepModel(nn.Module):
    """A baseline model for EM."""

    def __init__(self, device='cuda', lm='roberta'):
        super().__init__()
        print('*'*20, "DittoSepModel", '*'*20)

        if lm in lm_mp:
            self.bert = AutoModel.from_pretrained(lm_mp[lm])
            print(lm_mp[lm])
        else:
            self.bert = AutoModel.from_pretrained(lm)
            print(lm)

        self.device = device

        # linear layer
        hidden_size = self.bert.config.hidden_size
        self.fc = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, x1, x2=None):
        """Encode the left, right, and the concatenation of left+right.

        Args:
            x1 (LongTensor): a batch of ID's
            x2 (LongTensor, optional): a batch of ID's (augmented)

        Returns:
            Tensor: binary prediction
        """
        x1 = x1.to(self.device) # (batch_size, seq_len)
        x2 = x2.to(self.device) # (batch_size, seq_len)
        # hidden = self.bert(torch.cat((x1, x2), -1))[0][:, 0, :]
        enc = self.bert(torch.cat((x1, x2)))[0]
        batch_size = len(x1)
        # h1 = torch.mean(enc[:batch_size], 1) # (batch_size, emb_size)
        # h2 = torch.mean(enc[batch_size:], 1) # (batch_size, emb_size)
        self.h1 = enc[:batch_size][:,0,:] # [B, N1, K]
        self.h2 = enc[batch_size:][:,0,:] # [B, N2, K]
        # self.h1 = (self.fc(self.h1)).tanh()
        # self.h2 = (self.fc(self.h2)).tanh()
        CosSim = F.cosine_similarity(self.h1, self.h2)
        # scores = CosSim.sigmoid()

        # cross attention
        # self.weight1 = torch.matmul(self.h1, torch.permute(self.h2, (0,2,1))).softmax(2) # [B, N1, N2]
        # self.h12 = torch.matmul(self.weight1, self.h2) # [B, N1, K]
        # self.weight2 = torch.matmul(self.h2, torch.permute(self.h1, (0,2,1))).softmax(2) # [B, N2, N1]
        # self.h21 = torch.matmul(self.weight2, self.h1)
        # self.h1_final = (self.h1 + self.h12) / 2
        # self.h2_final = (self.h2 + self.h21) / 2
        # hidden = torch.mean(torch.concat([self.h1_final, self.h2_final], 1), 1)
        # hidden = torch.concat([self.h1, self.h2], -1)
        # hidden = h1 * h2
        # pred = self.fc(hidden)
            
        # return pred 
        return CosSim

class DittoModel(nn.Module):
    """A baseline model for EM."""

    def __init__(self, device='cuda', lm='roberta', alpha_aug=0.8):
        super().__init__()
        if lm in lm_mp:
            self.bert = AutoModel.from_pretrained(lm_mp[lm])
        else:
            self.bert = AutoModel.from_pretrained(lm)

        self.device = device
        self.alpha_aug = alpha_aug

        # linear layer
        hidden_size = self.bert.config.hidden_size
        self.fc = torch.nn.Linear(hidden_size, 2)


    def forward(self, x1, x2=None):
        """Encode the left, right, and the concatenation of left+right.

        Args:
            x1 (LongTensor): a batch of ID's
            x2 (LongTensor, optional): a batch of ID's (augmented)

        Returns:
            Tensor: binary prediction
        """
        x1 = x1.to(self.device) # (batch_size, seq_len)
        if x2 is not None:
            # MixDA
            x2 = x2.to(self.device) # (batch_size, seq_len)
            enc = self.bert(torch.cat((x1, x2)))[0][:, 0, :]
            batch_size = len(x1)
            enc1 = enc[:batch_size] # (batch_size, emb_size)
            enc2 = enc[batch_size:] # (batch_size, emb_size)

            aug_lam = np.random.beta(self.alpha_aug, self.alpha_aug)
            enc = enc1 * aug_lam + enc2 * (1.0 - aug_lam)
        else:
            enc = self.bert(x1)[0][:, 0, :]

        return self.fc(enc) # .squeeze() # .sigmoid()