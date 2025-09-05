""" このコードは関数のシグネチャと使用方法を示すためのサンプルで、実行は想定していません
"""

# classmethod: クラスメソッド。nn.Embedding のインスタンスを返す
torch.nn.Embedding.from_pretrained(embeddings, freeze=True,
    padding_idx=None, max_norm=None, norm_type=2.0, 
    scale_grad_by_freq=False, sparse=False)
