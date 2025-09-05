""" 以下のコードの実行結果を再現するには、PyTorch をインストールした後、
    システムのコマンドラインで `python` を実行して対話モードに入り、>>> プロンプトで次のコードを入力してください。
    （# 以降はコメントのため無視して構いません）
    ex_2_20.py で定義した LinearModel を利用
"""

import torch
import torch.nn as nn
from ex_2_20 import LinearModel

lm = LinearModel(5) # 線形回帰モデル（特徴数 5）を定義
x = torch.randn(4, 5) # ランダム入力（ミニバッチサイズ 4）
lm(x) # 各ミニバッチの出力を得る
