""" 本コードは LeNet モデルの学習過程を定義します。
"""

import torch
import torch.nn as nn
from model import LeNet

# ... 学習用データローダの定義は省略（詳細はコード 4.3 を参照）

model = LeNet() # LeNet モデルを定義
model.train() # 学習モードへ切り替え
lr = 0.01 # 学習率
criterion = nn.CrossEntropyLoss() # 損失関数
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, 
    weight_decay=5e-4) # SGD オプティマイザ

train_loss = 0
correct = 0
total = 0

for batch_idx, (inputs, targets) in enumerate(data_train_loader):

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    train_loss += loss.item()
    _, predicted = outputs.max(1)
    total += targets.size(0)
    correct += predicted.eq(targets).sum().item()

    print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
