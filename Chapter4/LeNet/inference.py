""" 本コードは LeNet モデルの推論手順を定義します。
"""

import torch
import torch.nn as nn
from model import LeNet

# ... テスト用データローダの定義は省略（コード 4.3 を参照）

# save_info = { # 保存する情報
#    "iter_num": iter_num,  # 反復回数 
#    "optimizer": optimizer.state_dict(), # オプティマイザの state_dict
#    "model": model.state_dict(), # モデルの state_dict
# }
 
model_path = "./model.pth" # モデルが model.pth に保存されていると仮定
save_info = torch.load(model_path) # モデルを読み込み
model = LeNet() # LeNet モデルを定義
criterion = nn.CrossEntropyLoss() # 損失関数を定義
model.load_state_dict(save_info["model"]) # モデルパラメータを読み込み
model.eval() # 評価モードへ切り替え

test_loss = 0
correct = 0
total = 0
with torch.no_grad(): # 計算グラフを無効化
    for batch_idx, (inputs, targets) in enumerate(data_test_loader):

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
