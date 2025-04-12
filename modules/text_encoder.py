#import ipdb

import torch
import torch.nn as nn
import torch.nn.functional as F

#import copy
#import math
import numpy as np

from modules.transformer import MultiHeadedAttention, PositionalEncoding, PositionwiseFeedForward, Encoder, \
    EncoderLayer, Embeddings, SublayerConnection, clones


class TextEncoder(nn.Module):
    def __init__(self, d_model, d_ff, num_layers,tgt_vocab,dropout,h=8):
        super(TextEncoder, self).__init__()
        # TODO:
        #  将eos,pad的index改为参数输入
        self.eos_idx = 0
        self.pad_idx = 0
        self.separator_id = 1 

        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.encoder = Encoder(EncoderLayer(d_model, attn, ff, dropout), num_layers)
       
    def prepare_mask(self, seq):
        seq_mask = (seq.data != self.eos_idx) & (seq.data != self.pad_idx)
        seq_mask[:, 0] = 1  # bos
        seq_mask = seq_mask.unsqueeze(-2)
        return seq_mask
    
    def find_sentence_boundaries(self,token_ids, separator_id=9):
        separator_positions = [i for i, token in enumerate(token_ids) if token == separator_id]
        sentence_boundaries = []
        start = 0
        for pos in separator_positions:
            sentence_boundaries.append((start, pos))
            start = pos + 1
        if start < len(token_ids):
            sentence_boundaries.append((start, len(token_ids)))
        return sentence_boundaries
    

    def forward(self, src,tgt_embed):
        src_mask = self.prepare_mask(src)
        batch_size=src.size(0)
        feats = self.encoder(tgt_embed(src), src_mask)
        pooled_output = feats[:, 0, :]
        all_sentence_features = []
        for i in range(batch_size):
            sentence_boundaries = self.find_sentence_boundaries(src[i].tolist(), self.separator_id)
            sample_sentence_features = []

            
            for start, end in sentence_boundaries:
                if end!=60:#if dataset=mimic  end=100
                    sentence_feat = feats[i, start:end, :]
                    sentence_pooled = torch.mean(sentence_feat, dim=0)
                    sample_sentence_features.append(sentence_pooled)
                elif end==60 and src[i][59]!=0:#if dataset=mimic  end=100
                    sentence_feat = feats[i, start:end, :]
                    sentence_pooled = torch.mean(sentence_feat, dim=0)
                    sample_sentence_features.append(sentence_pooled)
            all_sentence_features.append(sample_sentence_features)

        return feats, pooled_output, all_sentence_features


