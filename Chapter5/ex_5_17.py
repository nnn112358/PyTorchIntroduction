""" このコードは関数のシグネチャと使用方法を示すためのサンプルで、実行は想定していません
"""

class torch.nn.MultiheadAttention(embed_dim, num_heads,
    dropout=0.0, bias=True, add_bias_kv=False,
    add_zero_attn=False, kdim=None, vdim=None)

# 対応する forward メソッドの定義
forward(query, key, value, key_padding_mask=None,
    need_weights=True, attn_mask=None)
