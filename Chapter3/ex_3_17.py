""" このファイルのコードは `python ex_3_17.py` で実行できます。
   （# 以降はコメントのため無視して構いません）
"""

import torch.nn as nn

# パターン1: 引数で順序モジュールを構築
model = nn.Sequential(
          nn.Conv2d(1,20,5),
          nn.ReLU(),
          nn.Conv2d(20,64,5),
          nn.ReLU()
        )

print(model)

# パターン2: OrderedDict で順序モジュールを構築
model = nn.Sequential(OrderedDict([
          ('conv1', nn.Conv2d(1,20,5)),
          ('relu1', nn.ReLU()),
          ('conv2', nn.Conv2d(20,64,5)),
          ('relu2', nn.ReLU())
        ]))

print(model)
