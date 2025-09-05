""" 本コードはフック関数のデモです。
"""

# モジュール実行前の forward フック
# nn.Module のインスタンスを定義
module = ...
def hook(module, input):
    # モジュールの重みや入力に対する処理
    # 返り値は変更後テンソルまたは None
    return input
handle = module.register_forward_pre_hook(hook)

# モジュール実行後の forward フック
# nn.Module のインスタンスを定義
module = ...
def hook(module, input, output):
    # モジュールの重みや入出力に対する処理
    # 返り値は変更後テンソルまたは None
    return output
handle = module.register_forward_hook(hook)

# モジュール実行後の backward フック
# nn.Module のインスタンスを定義
module = ...
def hook(module, grad_input, grad_output):
    # モジュールの重みや入出力の勾配に対する処理
    # 返り値は変更後テンソルまたは None
    return output
handle = module.register_backward_hook(hook)

# フックの使用例
import torch
import torch.nn as nn
def print_pre_shape(module, input):
    print("前フック")
    print(module.weight.shape)
    print(input[0].shape)
def print_post_shape(module, input, output):
    print("後フック")
    print(module.weight.shape)
    print(input[0].shape)
    print(output[0].shape)
def print_grad_shape(module, grad_input, grad_output):
    print("勾配フック")
    print(module.weight.grad.shape)
    print(grad_input[0].shape)
    print(grad_output[0].shape)
conv = nn.Conv2d(16, 32, kernel_size=(3,3))
handle1 = conv.register_forward_pre_hook(print_pre_shape)
handle2 = conv.register_forward_hook(print_post_shape)
handle3 = conv.register_backward_hook(print_grad_shape)
input = torch.randn(4, 16, 128, 128, requires_grad=True)
ret = conv(input)
