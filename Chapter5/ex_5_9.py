""" このコードは関数のシグネチャと使用方法を示すためのサンプルで、実行は想定していません
"""

torch.nn.utils.rnn.pack_padded_sequence(input, lengths, 
    batch_first=False, enforce_sorted=True)
torch.nn.utils.rnn.pad_packed_sequence(sequence, batch_first=False,
    padding_value=0.0, total_length=None)