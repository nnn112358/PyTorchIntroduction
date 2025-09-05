""" 本コードは Wide&Deep モデル実装の参考例です。
"""

import torch
import torch.nn as nn

class WideDeep(nn.Module):
    def __init__(self, num_wide_feat, deep_feat_sizes, 
        deep_feat_dims, nhiddens):

        super(WideDeep, self).__init__()

        self.num_wide_feat = num_wide_feat
        self.deep_feat_sizes = deep_feat_sizes
        self.deep_feat_dims = deep_feat_dims
        self.nhiddens = nhiddens

        # Deep モデルの埋め込み部
        self.embeds = nn.ModuleList()
        for deep_feat_size, deep_feat_dim in \
            zip(deep_feat_sizes, deep_feat_dims):
            self.embeds.append(nn.Embedding(deep_feat_size, 
                deep_feat_dim))

        self.deep_input_size = sum(deep_feat_dims)

        # Deep モデルの線形部 
        self.linears = nn.ModuleList()
        in_size = self.deep_input_size
        for out_size in nhiddens:
            self.linears.append(nn.Linear(in_size, out_size))
            in_size = out_size

        # Wide/Deep 共通の線形部 
        self.proj = nn.Linear(in_size + num_wide_feat, 1)

    def forward(self, wide_input, deep_input):
        
        # Wide 入力: N×W（N はミニバッチ、W は特徴数）
        # Deep 入力: N×D（D は特徴数）
        embed_feats = []
        for i in range(deep_input.size(1)):
            embed_feats.append(self.embeds[i](deep_input[:, i]))
        deep_feats = torch.cat(embed_feats, 1)
        
        # Deep 特徴の変換
        for layer in self.linears:
            deep_feats = layer(deep_feats)
            deep_feats = torch.relu(deep_feats)
        print(wide_input.shape, deep_feats.shape)

        # Wide/Deep 特徴の結合
        wide_deep_feats = torch.cat([wide_input, deep_feats], -1)
        return torch.sigmoid(self.proj(wide_deep_feats)).squeeze()
