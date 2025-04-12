import torch
import torch.nn as nn
import torch.nn.functional as F


class ESF(nn.Module):

    def __init__(self, feature_dim=512, head_dim=256):
        super(ESF, self).__init__()
        self.feature_dim = feature_dim
        self.head_dim = head_dim

        # Linear projections for attention mechanism
        self.global_query = nn.Linear(feature_dim, head_dim)
        self.global_key = nn.Linear(feature_dim, head_dim)
        self.global_value = nn.Linear(feature_dim, head_dim)

        # Dimension reduction layer (from 12 dimensions to 1)
        self.dimension_reduction = nn.Linear(12, 1)

        # Feature fusion network
        self.fusion = nn.Sequential(
            nn.Linear(head_dim * 2, feature_dim),
            nn.ReLU()
        )

    def forward(self, global_feature, local_features):


        local_features_stacked = torch.stack(local_features, dim=1)
        local_features_mean = self.dimension_reduction(local_features_stacked.transpose(1, 3)).squeeze(-1)
        local_features_mean = local_features_mean.transpose(1, 2)

        # ----------------- Global-to-Local Attention -----------------
        # Project global features to key and value spaces
        global_key = self.global_key(global_feature)
        global_value = self.global_value(global_feature)
        local_query = self.global_query(local_features_stacked)

        # Compute attention scores
        attention_scores = torch.matmul(local_query, global_key.unsqueeze(1).transpose(-2, -1))
        attention_scores = attention_scores / (self.head_dim ** 0.5)
        attention_map = F.softmax(attention_scores, dim=-1)
        weighted_global_feature = torch.matmul(attention_map, global_value.unsqueeze(1))

        # ----------------- Local-to-Global Attention -----------------
        global_query = self.global_query(global_feature)
        local_keys = self.global_key(local_features_stacked)
        local_values = self.global_value(local_features_stacked)
        local_to_global_scores = torch.matmul(global_query.unsqueeze(1), local_keys.transpose(-2, -1))
        local_to_global_scores = local_to_global_scores / (self.head_dim ** 0.5)
        local_to_global_attention = F.softmax(local_to_global_scores, dim=-1)
        local_to_global_feature = torch.matmul(local_to_global_attention, local_values).squeeze(1)

        # ----------------- Feature Fusion -----------------
        weighted_global_feature_mean = self.dimension_reduction(weighted_global_feature.transpose(1, 3)).squeeze(-1)
        weighted_global_feature_mean = weighted_global_feature_mean.transpose(1, 2)
        local_to_global_feature_mean = local_to_global_feature
        combined_feature = torch.cat([local_to_global_feature_mean, weighted_global_feature_mean], dim=-1)
        fused_feature = self.fusion(combined_feature) + local_features_mean + global_feature

        return fused_feature