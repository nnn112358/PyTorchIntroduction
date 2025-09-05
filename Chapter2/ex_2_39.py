""" このコードは関数シグネチャの例であり、実行は想定していません。
"""

torch.nn.DataParallel(module, device_ids=None, output_device=None, dim=0)
model = … # モデルを定義
model = model.cuda()
model = nn.DataParallel(model, device_ids=[0, 1, 2, 3]) # データ並列
output = model(input_var)
