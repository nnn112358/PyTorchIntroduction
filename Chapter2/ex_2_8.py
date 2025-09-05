""" 以下のコードの実行結果を再現するには、PyTorch をインストールした後、
    システムのコマンドラインで `python` を実行して対話モードに入り、>>> プロンプトで次のコードを入力してください。
    （# 以降はコメントのため無視して構いません）
    注意: 一部のコードは GPU を要求し、複数 GPU を前提とするものもあります。
"""

import torch # PyTorch をインポート

torch.randn(3, 3, device="cpu")    # CPU 上にテンソルを作成
torch.randn(3, 3, device="cuda:0") # GPU 0 上にテンソルを作成
torch.randn(3, 3, device="cuda:1") # GPU 1 上にテンソルを作成
torch.randn(3, 3, device="cuda:1").device            # 現在のデバイスを取得
torch.randn(3, 3, device="cuda:1").cpu().device       # GPU1 -> CPU へ移動
torch.randn(3, 3, device="cuda:1").cuda(1).device     # デバイスを保持（GPU1 のまま）
torch.randn(3, 3, device="cuda:1").cuda(0).device     # GPU1 -> GPU0 へ移動
torch.randn(3, 3, device="cuda:1").to("cuda:0").device # GPU1 -> GPU0 へ移動
