""" このコードは関数のシグネチャと使用方法を示すためのサンプルで、実行は想定していません
"""

class torch.nn.TransformerEncoder(encoder_layer, num_layers, norm=None)
# TransformerEncoder の forward メソッドの定義
forward(src, mask=None, src_key_padding_mask=None)

class torch.nn.TransformerDecoder(decoder_layer, num_layers, norm=None)
# TransformerDecoder の forward メソッドの定義
forward(tgt, memory, tgt_mask=None, memory_mask=None,
    tgt_key_padding_mask=None, memory_key_padding_mask=None)

class torch.nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6,
    num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
    custom_encoder=None, custom_decoder=None)
# Transformer の forward メソッドの定義
forward(src, tgt, src_mask=None, tgt_mask=None, memory_mask=None,
    src_key_padding_mask=None, tgt_key_padding_mask=None,
    memory_key_padding_mask=None)
