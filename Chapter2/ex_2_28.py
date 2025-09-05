""" 本コードはサンプルで、オプティマイザの使い方を示す。
"""

import torch
optim = torch.optim.SGD([
    {'params': model.base.parameters()},
    {'params': model.classifier.parameters(), 'lr': 1e-3}
], lr=1e-2, momentum=0.9)
