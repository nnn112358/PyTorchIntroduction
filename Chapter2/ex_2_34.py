""" このコードは関数シグネチャの例であり、実行は想定していません。
"""

torch.save(obj, f, pickle_module=pickle, pickle_protocol=2)
torch.load(f, map_location=None, pickle_module=pickle, **pickle_load_args)
