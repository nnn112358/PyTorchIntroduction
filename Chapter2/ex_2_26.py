""" 以下のコードの実行結果を再現するには、PyTorch をインストールした後、
    システムのコマンドラインで `python` を実行して対話モードに入り、>>> プロンプトで次のコードを入力してください。
    （# 以降はコメントのため無視して構いません）
"""

import torch

mse = nn.MSELoss() # 平方誤差損失を初期化
t1 = torch.randn(5, requires_grad=True) # テンソル t1 をランダム生成
t2 = torch.randn(5, requires_grad=True) # テンソル t2 をランダム生成
mse(t1, t2) # t1 と t2 の二乗誤差損失を計算
t1 = torch.randn(5, requires_grad=True) # テンソル t1 をランダム生成
t1s = torch.sigmoid(t1)
t2 = torch.randint(0, 2, (5, )).float() # 0/1 の整数列を生成し float に変換
bce(t1s, t2) # 2クラスの交差エントロピーを計算
bce_logits = nn.BCEWithLogitsLoss() # ロジット版 BCE 損失
bce_logits(t1, t2) # 交差エントロピー（前と一致）
N=10 # クラス数
t1 = torch.randn(5, N, requires_grad=True) # 予測テンソルを生成
t2 = torch.randint(0, N, (5, )) # 目標テンソルを生成
t1s = torch.nn.functional.log_softmax(t1, -1) # 予測の LogSoftmax
nll = nn.NLLLoss() # NLL 損失
nll(t1s, t2) # 損失を計算
ce = nn.CrossEntropyLoss() # 交差エントロピー損失
ce(t1, t2) # NLL と一致する結果
