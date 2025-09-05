""" 以下のコードの実行結果を再現するには、PyTorch をインストールした後、
    システムのコマンドラインで `python` を実行して対話モードに入り、>>> プロンプトで次のコードを入力してください。
    （# 以降はコメントのため無視して構いません）
"""

import torch

t1 = torch.randn(3, 3, requires_grad=True) # 3×3 のテンソルを定義
t1
t2 = t1.pow(2).sum() # 全要素の二乗和を計算
t2.backward() # 逆伝播
t1.grad # 勾配は元の要素の2倍
t2 = t1.pow(2).sum() # 再度、二乗和を計算
t2.backward() # 再度、逆伝播
t1.grad # 勾配が蓄積
t1.grad.zero_() # 単一テンソルの勾配を 0 にする
