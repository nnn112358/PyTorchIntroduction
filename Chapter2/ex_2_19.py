""" 以下のコードの実行結果を再現するには、PyTorch をインストールした後、
    システムのコマンドラインで `python` を実行して対話モードに入り、>>> プロンプトで次のコードを入力してください。
    （# 以降はコメントのため無視して構いません）
    本コードはサンプルであり、実行には実装の詳細が必要です。
"""

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, ...): # クラスの初期化。... は引数を表す
        super(Model, self).__init__()
        ... # 引数に基づきサブモジュールを定義
    
    def forward(self, ...): # forward の入力パラメータ（一般にテンソル等）
        ret = ... # 入力とサブモジュールから出力テンソルを計算
        return ret
