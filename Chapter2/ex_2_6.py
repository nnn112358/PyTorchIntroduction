""" 以下のコードの実行結果を再現するには、PyTorch をインストールした後、
    システムのコマンドラインで `python` を実行して対話モードに入り、>>> プロンプトで次のコードを入力してください。
    （# 以降はコメントのため無視して構いません）
    このサンプルはコード 2.6 と 2.7 を統合した内容です。
"""
import numpy as np # NumPy をインポート
import torch # PyTorch をインポート

t = torch.randn(3,3) # 正規乱数に従うテンソル t を生成
print(t)
torch.zeros_like(t) # 要素が全て 0 のテンソル（t と同形状）
torch.ones_like(t)  # 要素が全て 1 のテンソル（t と同形状）
torch.rand_like(t)  # [0,1) の一様分布に従うテンソル（t と同形状）
torch.randn_like(t) # 標準正規分布に従うテンソル（t と同形状）

t.new_tensor([1,2,3]).dtype # Python のリストからテンソルを生成（単精度浮動小数点）
t.new_zeros(3, 3) # 同じ dtype で全要素 0 のテンソル
t.new_ones(3,3)  # 同じ dtype で全要素 1 のテンソル
