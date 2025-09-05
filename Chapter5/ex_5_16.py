""" 以下のコードの実行結果を再現するには、PyTorch をインストールした後、
    システムのコマンドラインで `python` を実行して対話モードに入り、>>> プロンプトで次のコードを入力してください。
    （# 以降はコメントのため無視して構いません）
"""

import torch
import torch.nn as nn

rnn = nn.RNNCell(10, 20)
input = torch.randn(6, 3, 10)
hx = torch.randn(3, 20)
output = []
for i in range(6):
    hx = rnn(input[i], hx)
    output.append(hx)

rnn = nn.LSTMCell(10, 20)
input = torch.randn(6, 3, 10)
hx = torch.randn(3, 20)
cx = torch.randn(3, 20)
output = []
for i in range(6):
    hx, cx = rnn(input[i], (hx, cx))
    output.append(hx)

rnn = nn.GRUCell(10, 20)
input = torch.randn(6, 3, 10)
hx = torch.randn(3, 20)
output = []
for i in range(6):
    hx = rnn(input[i], hx)
    output.append(hx)
