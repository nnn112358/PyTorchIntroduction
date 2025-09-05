""" このコードは関数のシグネチャと使用方法を示すためのサンプルで、実行は想定していません
"""

class torch.nn.TransformerEncoderLayer(d_model,
    nhead, dim_feedforward=2048, dropout=0.1)
# TransformerEncoderLayer の forward メソッドの定義
forward(src, src_mask=None, src_key_padding_mask=None)

class torch.nn.TransformerDecoderLayer(d_model,
    nhead, dim_feedforward=2048, dropout=0.1)
# TransformerDecoderLayer の forward メソッドの定義
forward(tgt, memory, tgt_mask=None, memory_mask=None,
    tgt_key_padding_mask=None, memory_key_padding_mask=None)
