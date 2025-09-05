""" 以下のコードの実行結果を再現するには、PyTorch をインストールした後、
    システムのコマンドラインで `python` を実行して対話モードに入り、>>> プロンプトで次のコードを入力してください。
    （# 以降はコメントのため無視して構いません）
    ex_2_20.py で定義した LinearModel を利用
"""

import torch
import torch.nn as nn
from ex_2_20 import LinearModel

lm = LinearModel(5) # 線形モデルを定義
x = torch.randn(4, 5) # モデル入力を定義
lm(x) # モデルから出力を取得
lm.named_parameters() # 名前付きパラメータのイテレータ
list(lm.named_parameters()) # イテレータをリストへ変換
lm.parameters() # 名前なしパラメータのイテレータ
list(lm.parameters()) # イテレータをリストへ変換
lm.cuda() # モデルパラメータを GPU へ移動
list(lm.parameters()) # 表示して GPU 上（device='cuda:0'）であることを確認
lm.half() # モデルパラメータを半精度へ変換
list(lm.parameters()) # 表示して半精度（dtype=torch.float16）であることを確認
