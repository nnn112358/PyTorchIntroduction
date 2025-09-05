""" 以下のコードの実行結果を再現するには、PyTorch をインストールした後、
    システムのコマンドラインで `python` を実行して対話モードに入り、>>> プロンプトで次のコードを入力してください。
    （# 以降はコメントのため無視して構いません）
    ex_2_20.py で定義した LinearModel を利用
"""

import torch
import torch.nn as nn
from ex_2_20 import LinearModel

lm = LinearModel(5) # 線形モデルを定義
lm.state_dict() # state_dict を取得
t = lm.state_dict() # state_dict を保存
lm = LinearModel(5) # 線形モデルを再定義
lm.state_dict() # 新しい state_dict（元と異なる）
lm.load_state_dict(t) # 以前の state_dict を読み込み
lm.state_dict() # パラメータが更新された
