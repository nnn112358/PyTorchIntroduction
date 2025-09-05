# 静的ロード
import torch
import gelu

# 同様に gelu = GELU.apply でもこの活性化関数を利用できる
class GELU(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        return gelu.forward(input)
    
    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.input
        return gelu.backward(grad_output, input)

# 動的ロード
import torch
from torch.utils.cpp_extension import load

# PyTorch が自動ビルドして対応するモジュールを生成
gelu = load(name="gelu", sources=["gelu/gelu.cc"])

class GELU(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        return gelu.forward(input)
    
    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.input
        return gelu.backward(grad_output, input)
