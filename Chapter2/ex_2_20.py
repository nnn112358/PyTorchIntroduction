""" 本モジュールは他の Python スクリプトや Python 対話環境でも利用できます。
    （# 以降はコメントのため無視して構いません）
"""

import torch
import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self, ndim):
        super(LinearModel, self).__init__()
        self.ndim = ndim

        self.weight = nn.Parameter(torch.randn(ndim, 1)) # 重みを定義
        self.bias = nn.Parameter(torch.randn(1)) # バイアスを定義

    def forward(self, x):
        # 線形モデル y = Wx + b を定義
        return x.mm(self.weight) + self.bias
