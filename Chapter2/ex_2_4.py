""" 以下のコードの実行結果を再現するには、PyTorch をインストールした後、
    システムのコマンドラインで `python` を実行して対話モードに入り、>>> プロンプトで次のコードを入力してください。
    （# 以降はコメントのため無視して構いません）
"""

import numpy as np # NumPy をインポート
import torch # PyTorch をインポート
torch.tensor([1,2,3,4]) # Python リストを PyTorch テンソルへ変換
torch.tensor([1,2,3,4]).dtype # 查看テンソルデータクラス型
torch.tensor([1,2,3,4], dtype=torch.float32) # dtype を float32 に指定
torch.tensor([1,2,3,4], dtype=torch.float32).dtype  # 查看テンソルデータクラス型
torch.tensor(range(10)) # イテレータをテンソルへ変換
np.array([1,2,3,4]).dtype # NumPy 配列の dtype を確認
torch.tensor(np.array([1,2,3,4])) # NumPy 配列から PyTorch テンソルへ変換
torch.tensor(np.array([1,2,3,4])).dtype # 変換後テンソルの dtype
torch.tensor([1.0, 2.0, 3.0, 4.0]).dtype # PyTorch の既定浮動小数は float32
torch.tensor(np.array([1.0, 2.0, 3.0, 4.0])).dtype # NumPy の既定浮動小数は float64
torch.tensor([[1,2], [3,4,5]]) # ネストしたリストからの作成（誤り: サブリストの長さ不一致）
torch.tensor([[1,2,3], [4,5,6]]) # ネストしたリストからの作成（正: 2×3 行列）
torch.randn(3,3).to(torch.int) # torch.float -> torch.int（.int() も可）
torch.randint(0, 5, (3,3)).to(torch.float) # torch.int64 -> torch.float（.float() も可）
