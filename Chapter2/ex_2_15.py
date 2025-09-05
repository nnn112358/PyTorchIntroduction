""" 以下のコードの実行結果を再現するには、PyTorch をインストールした後、
    システムのコマンドラインで `python` を実行して対話モードに入り、>>> プロンプトで次のコードを入力してください。
    （# 以降はコメントのため無視して構いません）
"""

import torch

a = torch.randn(2,3,4) # ランダムにテンソルを生成
b = torch.randn(2,4,3)
a.bmm(b) # バッチ行列積の結果
torch.einsum("bnk,bkl->bnl", a, b) # einsum 関数の結果（上と同じ）
