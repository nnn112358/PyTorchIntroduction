""" このコードは関数シグネチャの例であり、実行は想定していません。
"""

# バッチ正規化
class torch.nn.BatchNorm1d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
class torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
class torch.nn.BatchNorm3d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

# グループ正規化
class torch.nn.GroupNorm(num_groups, num_channels, eps=1e-05, affine=True)

# インスタンス正規化
class torch.nn.InstanceNorm1d(num_features, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
class torch.nn.InstanceNorm2d(num_features, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
class torch.nn.InstanceNorm3d(num_features, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)

# 層正規化
class torch.nn.LocalResponseNorm(size, alpha=0.0001, beta=0.75, k=1.0)

# 局所応答正規化
class torch.nn.LocalResponseNorm(size, alpha=0.0001, beta=0.75, k=1.0)
