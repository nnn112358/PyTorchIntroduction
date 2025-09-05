""" このコードは `python ex_2_27.py` で実行できます（scikit-learn が必要）。
    （# 以降はコメントのため無視して構いません）
"""

from sklearn.datasets import load_boston
boston = load_boston()

lm = LinearModel(13)
criterion = nn.MSELoss()
optim = torch.optim.SGD(lm.parameters(), lr=1e-6) # オプティマイザを定義
data = torch.tensor(boston["data"], requires_grad=True, dtype=torch.float32)
target = torch.tensor(boston["target"], dtype=torch.float32)

for step in range(10000):
    predict = lm(data) # 予測を計算
    loss = criterion(predict, target) # 損失を計算
    if step and step % 1000 == 0 :
        print("Loss: {:.3f}".format(loss.item()))
    optim.zero_grad() # 勾配を 0 に
    loss.backward() # 逆伝播
    optim.step()
# 出力例：
# Loss: 150.855
# Loss: 113.852
# …
# Loss: 98.165
# Loss: 97.907
