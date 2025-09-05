""" このコードは関数シグネチャの例であり、実行は想定していません。
"""

save_info = { # 保存する情報
    "iter_num": iter_num,  # 反復回数 
    "optimizer": optimizer.state_dict(), # オプティマイザの state_dict
    "model": model.state_dict(), # モデルの state_dict
}
# 情報を保存
torch.save(save_info, save_path)
# 情報を読み込み
save_info = torch.load(save_path)
optimizer.load_state_dict(save_info["optimizer"])
model.load_state_dict(sae_info["model"])
