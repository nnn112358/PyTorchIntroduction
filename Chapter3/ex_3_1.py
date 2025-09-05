""" このコードは関数シグネチャの例であり、実行は想定していません。
"""

# クラスシグネチャ
torch.nn.Linear(in_features, out_features, bias=True)

# クラス使用方法
import torch.nn as nn
ndim = ... # 入力特徴の次元数（整数）
lm = nn.Linear(ndim, 1)

# クラス使用例
import torch
import torch.nn as nn
lm = nn.Linear(5, 10) # 入力特徴 5、出力特徴 10
t = torch.randn(4, 5) # ミニバッチ 4、特徴 5
lm(t).shape # ミニバッチ 4、特徴 10
