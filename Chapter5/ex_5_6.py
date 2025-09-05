""" 以下のコードの実行結果を再現するには、PyTorch をインストールした後、
    システムのコマンドラインで `python` を実行して対話モードに入り、>>> プロンプトで次のコードを入力してください。
    （# 以降はコメントのため無視して構いません）
"""

import torch
import torch.nn as nn

embedding = nn.Embedding(10, 4)
embedding.weight
input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
embedding(input)
embedding = nn.Embedding(10, 4, padding_idx=0) # 10×4 の埋め込みテンソルを定義（インデックス0のベクトルは0）
embedding.weight
input = torch.LongTensor([[0,2,0,5]])
embedding(input)
