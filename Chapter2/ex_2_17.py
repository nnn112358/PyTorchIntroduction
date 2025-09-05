""" 以下のコードの実行結果を再現するには、PyTorch をインストールした後、
    システムのコマンドラインで `python` を実行して対話モードに入り、>>> プロンプトで次のコードを入力してください。
    （# 以降はコメントのため無視して構いません）
"""

import torch

t = torch.rand(3, 4) # ランダムにテンソルを生成
t.shape
t.unsqueeze(-1).shape # 最後の次元を拡張
t.unsqueeze(-1).unsqueeze(-1).shape # さらに最後の次元を拡張
t = torch.rand(1,3,4,1) # 2 つの次元サイズが 1 のテンソル
t.shape
t.squeeze().shape # サイズ 1 の次元をすべて圧縮
