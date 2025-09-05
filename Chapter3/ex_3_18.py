""" このファイルのコードは `python ex_3_18.py` で実行できます。
   （# 以降はコメントのため無視して構いません）
"""

import torch.nn as nn

# ModuleList の使用例
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

    def forward(self, x):
        # ModuleList の反復と使用は通常のリストと同様
        for i, l in enumerate(self.linears):
            x = self.linears[i // 2](x) + l(x)
        return x

# ModuleDict の使用例
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.choices = nn.ModuleDict({
                'conv': nn.Conv2d(10, 10, 3),
                'pool': nn.MaxPool2d(3)
        })
        self.activations = nn.ModuleDict([
                ['lrelu', nn.LeakyReLU()],
                ['prelu', nn.PReLU()]
        ])

    def forward(self, x, choice, act):
        x = self.choices[choice](x)
        x = self.activations[act](x)
        return x
