""" 以下のコードの実行結果を再現するには、PyTorch をインストールした後、
    システムのコマンドラインで `python` を実行して対話モードに入り、>>> プロンプトで次のコードを入力してください。
    （# 以降はコメントのため無視して構いません）
"""

import torch

t1 = torch.randn(3, 3, requires_grad=True) # t1 テンソルを初期化
t2 = t1.sum()
t2 # t2 の計算は計算グラフを構築し、出力には grad_fn が付く
with torch.no_grad():
    t3 = t1.sum()
t3 # t3 の計算は計算グラフを構築せず、出力には grad_fn が付かない
t1.sum() # 既存の計算グラフを維持
t1.sum().detach() # 既存の計算グラフから分離
