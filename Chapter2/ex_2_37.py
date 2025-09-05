""" このコードは `python ex_2_37.py` で実行できます（TensorBoard が必要）。
    （# 以降はコメントのため無視して構いません）
    ex_2_20.py で定義した LinearModel を利用
"""


from sklearn.datasets import load_boston
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn

from ex_2_20 import LinearModel

boston = load_boston()
lm = LinearModel(13)
criterion = nn.MSELoss()
optim = torch.optim.SGD(lm.parameters(), lr=1e-6)
data = torch.tensor(boston["data"], requires_grad=True, dtype=torch.float32)
target = torch.tensor(boston["target"], dtype=torch.float32)
writer = SummaryWriter() # TensorBoard 出力クラスを定義
for step in range(10000):
    predict = lm(data)
    loss = criterion(predict, target)
    writer.add_scalar("Loss/train", loss, step) # 損失を記録
    writer.add_histogram("Param/weight", lm.weight, step) # 重みのヒストグラム
    writer.add_histogram("Param/bias", lm.bias, step) # バイアスのヒストグラム
    if step and step % 1000 == 0 :
        print("Loss: {:.3f}".format(loss.item()))
    optim.zero_grad()
    loss.backward()
    optim.step()
