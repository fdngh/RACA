from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from .att_model import pack_wrapper, AttModel
from modules.transformer import (
    MultiHeadedAttention, PositionalEncoding, PositionwiseFeedForward,
    Encoder, EncoderLayer, Embeddings, SublayerConnection, clones,
    Decoder, DecoderLayer
)
from modules.text_encoder import TextEncoder
from RL.agent1 import local_agent
from RL.PPO import ppo
from .att1 import ESF


class OrgansEmbedding(nn.Module):

    def __init__(self, args):
        super(OrgansEmbedding, self).__init__()
        self.args = args

        # Set embedding dimension based on dataset
        if self.args.dataset_name == 'iu_xray':
            embedding_dim = 60
        else:
            embedding_dim = 158

        # Embedding layer for 12 organ types
        self.lut = nn.Embedding(12, embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, x):

        return self.lut(x) * math.sqrt(self.embedding_dim)


def subsequent_mask(size):

    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class Transformer(nn.Module):


    def __init__(self, args, visual_encoder, decoder, src_embed, tgt_embed, organs_embedding, local_agent, ESF,
                 t_encoder):
        super(Transformer, self).__init__()
        self.encoder = visual_encoder
        self.locate_agent = local_agent
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.organs_embedding = organs_embedding
        self.esf = ESF
        self.tencoder = t_encoder

        # Projection layer for contrast learning
        self.projection = nn.Linear(512, args.d_dubi)
        self.temperature = args.tt

    def global_info_nce_loss(self, text_features, image_features):

        # Project features to common space
        text_embeddings = self.projection(text_features)
        image_embeddings = self.projection(image_features)

        # Calculate similarity matrix with temperature scaling
        similarity_matrix = torch.matmul(text_embeddings, image_embeddings.t()) / 0.07

        # Create labels for contrastive learning (diagonal is positive)
        labels = torch.arange(similarity_matrix.shape[0], device=similarity_matrix.device)

        # Calculate bidirectional contrastive loss
        loss_t2i = F.cross_entropy(similarity_matrix, labels)
        loss_i2t = F.cross_entropy(similarity_matrix.t(), labels)

        return (loss_t2i + loss_i2t) / 2

    def local_info_nce_loss(self, text_features, image_features, alignment_labels):

        batch_size = len(text_features)
        num_organs = len(image_features)

        # List of organ names for reference
        organs = ['Lung fields', 'Airways', "Heart", "Pulmonary vasculature", 'Mediastinum', 'Spine', 'Abdomen',
                  'Diaphragm', 'Shoulders', 'Ribs', 'Clavicles', 'Costophrenic angles']

        total_loss = 0
        valid_samples = 0

        for i in range(batch_size):
            sample_text_features = text_features[i]
            num_sentences = len(sample_text_features)

            if len(sample_text_features) > 0:
                # Project features to common space
                text_embeddings = self.projection(torch.stack(sample_text_features))
                sample_image_embeddings = torch.cat(
                    [self.projection(organ_features[i, :]).unsqueeze(0) for organ_features in image_features])

                # Calculate cosine similarity with temperature scaling
                similarity = F.cosine_similarity(text_embeddings.unsqueeze(1),
                                                 sample_image_embeddings.unsqueeze(0),
                                                 dim=2) / self.temperature

                # Create alignment label matrix based on provided labels
                labels = torch.zeros((num_sentences, num_organs), device=text_embeddings.device)
                for sent_idx_str, organ_names in alignment_labels[i].items():
                    sent_idx = int(sent_idx_str[1:]) - 1
                    if sent_idx < num_sentences:
                        for organ_name in organ_names:
                            if organ_name in organs:
                                organ_idx = organs.index(organ_name)
                                labels[sent_idx, organ_idx] = 1

                # Calculate contrastive loss using positive and negative pairs
                positive_pairs = similarity[labels.bool()].mean()
                negative_pairs = similarity[~labels.bool()].mean()
                loss = -torch.log(
                    torch.exp(positive_pairs) / (torch.exp(positive_pairs) + torch.exp(negative_pairs) + 1e-8))

                if not torch.isnan(loss):
                    total_loss += loss
                    valid_samples += 1
            else:
                # Zero loss for empty samples
                total_loss += torch.tensor(0.0, device=image_features[0].device)

        if valid_samples > 0:
            # Average loss over batch size
            avg_loss = total_loss / batch_size
            return avg_loss
        else:
            # Return zero loss if all samples are invalid
            return torch.tensor(0.0, device=image_features[0].device)

    def forward(self, args, src, tgt, src_mask, tgt_mask, sentence_label, device, seq):

        # Encode image features
        image_feature = self.encode(src, src_mask)  # [B,P,D]

        # Create organ tensor for localization
        organs_tensor = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=torch.long).to(device)
        organs_tensor_embedding = self.organs_embedding(organs_tensor)

        # Initialize RL agent
        self.locate_agent.init_transition_dict()
        local_list = []
        pooled_local_list = []

        # Get localization actions from RL agent
        local_action, local_transition_dic, selected_patches = self.locate_agent.forward(organs_tensor_embedding,
                                                                                         image_feature)

        # Encode text features
        feats, pooled_output, sentence_features = self.tencoder(seq, self.tgt_embed)

        # Calculate global contrastive loss
        pooled_image_feature = torch.mean(image_feature, dim=1)
        global_loss = self.global_info_nce_loss(pooled_output, pooled_image_feature)

        # Process localized features
        for i in range(len(local_action)):
            lo_action1 = local_action[i].unsqueeze(-1)
            patch_f = lo_action1 * image_feature
            local_list.append(patch_f)
            pooled_local_list.append(torch.mean(patch_f, dim=1))

        # Apply enhanced self-attention fusion
        im_feature1 = self.esf(image_feature, local_list)

        # Calculate local contrastive loss
        local_loss = self.local_info_nce_loss(sentence_features, pooled_local_list, sentence_label)

        # Decode output
        output = self.decode(im_feature1, src_mask, tgt, tgt_mask)

        return output, local_transition_dic, global_loss, local_loss, selected_patches, local_action

    def encode(self, src, src_mask):

        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, hidden_states, src_mask, tgt, tgt_mask):

        return self.decoder(self.tgt_embed(tgt), hidden_states, src_mask, tgt_mask)


class EncoderDecoder(AttModel):


    def make_model(self, tgt_vocab, args):

        # Initialize RL parameters
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.hidden_dim = args.hidden_dim
        self.gamma = args.gam
        self.dataname = args.dataset_name
        self.lmbda = args.lmbda
        self.epochs = 1
        self.eps = args.eps
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.att = args.d_att

        # Set dimensions based on dataset
        if self.dataname == 'iu_xray':
            state_dim1 = 256
            action_dim1 = 98
        else:
            state_dim1 = 256
            action_dim1 = 49

        # Initialize PPO agent
        self.agent1 = ppo(state_dim1, self.hidden_dim, action_dim1, self.actor_lr, self.critic_lr, self.lmbda,
                          self.epochs, self.eps, self.gamma, self.device)

        # Create model components
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.num_heads, self.d_model)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        position = PositionalEncoding(self.d_model, self.dropout)

        # Assemble the full model
        model = Transformer(
            args,
            Encoder(EncoderLayer(self.d_model, c(attn), c(ff), self.dropout), self.num_layers),
            Decoder(DecoderLayer(self.d_model, c(attn), c(attn), c(ff), self.dropout), self.num_layers),
            lambda x: x,  # Source embedding function
            nn.Sequential(Embeddings(self.d_model, tgt_vocab), c(position)),  # Target embedding
            OrgansEmbedding(args),  # Organ embedding
            local_agent(args, self.agent1),  # Localization agent
            ESF(self.d_model, self.att),  # Enhanced Self-attention Fusion
            TextEncoder(self.d_model, self.d_ff, args.tnum_layers, tgt_vocab, args.dropout_t)  # Text encoder
        )

        # Initialize parameters
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        return model

    def __init__(self, args, tokenizer):

        super(EncoderDecoder, self).__init__(args, tokenizer)
        self.args = args

        # Set model parameters
        self.num_layers = args.num_layers
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_heads = args.num_heads
        self.dropout = args.dropout

        # Create model
        tgt_vocab = self.vocab_size + 1
        self.model = self.make_model(tgt_vocab, args)
        self.logit = nn.Linear(args.d_model, tgt_vocab)

    def init_hidden(self, bsz):

        return []

    def _analyze(self, args, fc_feats, att_feats, seq, sentence_label, device, image_id, att_masks=None):

        att_feats, seq1, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks, seq)

        # No gradient computation for analysis
        with torch.no_grad():
            out, _, _, _, selected_patches, local_action = self.model(
                args, att_feats, seq1, att_masks, seq_mask, sentence_label, device, seq)
            outputs = F.log_softmax(self.logit(out), dim=-1)

            # Save selected patches information if available
            if local_action is not None:
                organs = ['Lung fields', 'Airways', "Heart", "Pulmonary vasculature", 'Mediastinum', 'Spine', 'Abdomen',
                          'Diaphragm', 'Shoulders', 'Ribs', 'Clavicles', 'Costophrenic angles']
                data = []
                for i in range(12):
                    data.append([image_id, organs[i], local_action[i]])
                df = pd.DataFrame(data, columns=['Image ID', 'Organ', 'Selected Patches'])

                # Append to existing CSV or create new one
                if not os.path.exists('selected_patches.csv'):
                    df.to_csv('selected_patches.csv', index=False)
                else:
                    df.to_csv('selected_patches_wu.csv', mode='a', header=False, index=False)

        return outputs

    def _prepare_feature(self, fc_feats, att_feats, att_masks):

        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks)
        memory = self.model.encode(att_feats, att_masks)
        return fc_feats[..., :1], att_feats[..., :1], memory, att_masks

    def _prepare_feature_forward(self, att_feats, att_masks=None, seq=None):

        # Clip features to maximum length
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        # Apply attention embedding
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        # Create attention masks if not provided
        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        att_masks = att_masks.unsqueeze(-2)

        # Process sequence if provided
        if seq is not None:
            # Crop the last token
            seq = seq[:, :-1]
            seq_mask = (seq.data > 0)
            seq_mask[:, 0] += True  # Always attend to first token
            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
        else:
            seq_mask = None

        return att_feats, seq, att_masks, seq_mask

    def _forward(self, args, fc_feats, att_feats, seq, sentence_label, device, att_masks=None):

        att_feats, seq1, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks, seq)
        out, t1, g_loss, l_loss, _, _ = self.model(
            args, att_feats, seq1, att_masks, seq_mask, sentence_label, device, seq)
        outputs = F.log_softmax(self.logit(out), dim=-1)
        return (outputs, t1, g_loss, l_loss)

    def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask):

        # Concatenate previous tokens with current token
        if len(state) == 0:
            ys = it.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)

        # Create organ embeddings
        organs_tensor = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=torch.long).to(memory.device)
        organs_tensor_embedding = self.model.organs_embedding(organs_tensor)

        # Get localization actions
        local_list = []
        selected_patches = []
        local_action, _, _ = self.model.locate_agent.forward(organs_tensor_embedding, memory)

        # Extract selected patch indices
        selected_patches = [action.nonzero().squeeze().tolist() for action in local_action]

        # Apply attention to selected patches
        for s in range(len(local_action)):
            lo_action1 = local_action[s].unsqueeze(-1)
            patch_f = lo_action1 * memory
            local_list.append(patch_f)

        # Fuse features with enhanced self-attention
        im_feature = self.model.esf(memory, local_list)

        # Decode next token
        out = self.model.decode(im_feature, mask, ys, subsequent_mask(ys.size(1)).to(memory.device))

        return out[:, -1], [ys.unsqueeze(0)], selected_patches