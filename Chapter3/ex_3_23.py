""" このコードは関数シグネチャの例であり、実行は想定していません。
"""
# ゲイン係数の計算（関数シグネチャ）
torch.nn.init.calculate_gain(nonlinearity, param=None)

# ゲイン係数を計算して表示
gain = nn.init.calculate_gain('leaky_relu', 0.2)
print(gain)

# パラメータ初期化（関数シグネチャ）
torch.nn.init.uniform_(tensor, a=0.0, b=1.0)
torch.nn.init.normal_(tensor, mean=0.0, std=1.0)
torch.nn.init.ones_(tensor)
torch.nn.init.zeros_(tensor)
torch.nn.init.xavier_uniform_(tensor, gain=1.0)
torch.nn.init.xavier_normal_(tensor, gain=1.0)
torch.nn.init.kaiming_uniform_(tensor, a=0, mode='fan_in',
    nonlinearity='leaky_relu')
torch.nn.init.kaiming_normal_(tensor, a=0, mode='fan_in',
    nonlinearity='leaky_relu')
