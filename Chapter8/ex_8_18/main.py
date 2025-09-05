import torch
# 静的モデルの保存と読み込み
from torchvision.models import resnet18
m = resnet18(pretrained=True)
# 動的モデルを静的グラフへ変換
static_model = torch.jit.trace(m, torch.randn(1, 3, 224, 224))
# モデルを保存
torch.jit.save(static_model, "resnet18.pt")
# モデルを読み込み
static_model = torch.load("resnet18.pt")

# ONNX へエクスポート
from torchvision.models import resnet18
# `pip install onnx` で Python インターフェイスをインストール
import onnx
m = resnet18(pretrained=True)
torch.onnx.export(m, torch.randn(1, 3, 224, 224), 
                  "resnet18.onnx", verbose=True)
# onnx でモデルを読み込む
m = onnx.load("resnet18.onnx")
# モデルの正当性を検証
onnx.checker.check_model(m)
# 計算グラフを表示
onnx.helper.printable_graph(m.graph)
