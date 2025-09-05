""" 以下のコードの実行結果を再現するには、PyTorch をインストールした後、
    システムのコマンドラインで `python` を実行して対話モードに入り、>>> プロンプトで次のコードを入力してください。
    （# 以降はコメントのため無視して構いません）
"""

import torch

t1 = torch.randn(3, 3, requires_grad=True) # t1 テンソルを初期化
t2 = t1.pow(2).sum() # t1 から t2 を計算
torch.autograd.grad(t2, t1) # t2 の t1 に対する勾配を求める
