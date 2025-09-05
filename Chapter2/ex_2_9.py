""" 以下のコードの実行結果を再現するには、PyTorch をインストールした後、
    システムのコマンドラインで `python` を実行して対話モードに入り、>>> プロンプトで次のコードを入力してください。
    （# 以降はコメントのため無視して構いません）
"""

import torch

t = torch.randn(3,4,5) # 3×4×5 のテンソルを生成
t.ndimension() # 次元数を取得
t.nelement() # テンソルの総要素数を取得
t.size() # 各次元のサイズ（メソッド）
t.shape # 各次元のサイズ（属性）
t.size(0) # 次元 0 のサイズ（メソッド）
t = torch.randn(12) # 長さ 12 のベクトルを生成
t.view(3, 4) # ベクトルを 3×4 の行列へ変形
t.view(4, 3) # ベクトルを 4×3 の行列へ変形
t.view(-1, 4) # 先頭次元を -1（自動計算）にして変形
t # view は基礎データを変えず、view 後の変更は元テンソルに反映
t.view(4, 3)[0, 0] = 1.0
t.data_ptr() # テンソルのデータポインタを取得
t.view(3,4).data_ptr() # データポインタは変わらない
t.view(4,3).data_ptr() # 同上、変わらない
t.view(3,4).contiguous().data_ptr() # 同上、変わらない
t.view(4,3).contiguous().data_ptr() # 同上、変わらない
t.view(3,4).transpose(0,1).data_ptr() # 転置でストライドが入れ替わる
t.view(3,4).transpose(0,1).contiguous().data_ptr() # ストライド不整合のため再配置
