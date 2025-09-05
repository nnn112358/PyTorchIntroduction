""" 本コードは WaveNet の参考実装です。
"""

import torch
import torch.nn as nn

# 因果畳み込みモジュール
class CausalConv(nn.Module):

    def __init__(self, residual_channels, gate_channels, kernel_size,
                 local_channels, dropout=0.05, dilation=1, bias=True):

        super(CausalConv, self).__init__()
        self.dropout = dropout

        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(residual_channels, gate_channels, 
                              kernel_size, padding=padding,
                              dilation=dilation, bias=bias)

        self.conv1x1_local = Conv1d1x1(local_channels,
                                       gate_channels, bias=False)
        gate_out_channels = gate_channels // 2
        self.conv1x1_out = Conv1d1x1(gate_out_channels, 
                                     residual_channels, bias=bias)
        self.conv1x1_skip = Conv1d1x1(gate_out_channels,
                                      residual_channels, bias=bias)

    def forward(self, x, x_local):

        # x は音声信号、x_local はメル特徴を x と同次元へアップサンプルした結果
        # 入力 x のサイズは N×C×T（N: バッチ、C: 特徴、T: 長さ）
        # x_local のサイズは x と同じ

        residual = x
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 因果畳み込み
        x = self.conv(x)
        x = x[:, :, :residual.size(-1)]

        # 因果畳み込みの出力を分割
        a, b = x.split(x.size(-1) // 2, dim=-1)
        # ローカル特徴で変調
        c = self.conv1x1_local(x_local)
        ca, cb = c.split(c.size(-1) // 2, dim=-1)
        a, b = a + ca, b + cb

        x = torch.tanh(a) * torch.sigmoid(b)

        s = self.conv1x1_skip(x)
        x = self.conv1x1_out(x)

        x = (x + residual) * math.sqrt(0.5)
        return x, s

# WaveNetモデルコード
class WaveNet(nn.Module):

    def __init__(self, out_channels=256, layers=20,
                 layers_per_stack = 2,
                 residual_channels=512,
                 gate_channels=512,
                 mel_channels = 80,
                 mel_kernel = 1024,
                 mel_stride = 256,
                 skip_out_channels=512,
                 kernel_size=3, dropout= 0.05,
                 local_channels=512):

        super(WaveNet, self).__init__()

        self.out_channels = out_channels
        self.local_channels = local_channels
        self.first_conv = nn.Conv1d(out_channels, residual_channels, 1)

        self.conv_layers = nn.ModuleList()
        for layer in range(layers):
            dilation = 2**(layer % layers_per_stack)
            conv = CausalConv(residual_channels, gate_channels, kernel_size,
                              local_channels, dropout, dilation, True)
            self.conv_layers.append(conv)
        self.last_conv_layers = nn.ModuleList([
            nn.ReLU(inplace=True),
            nn.Conv1d(skip_out_channels, skip_out_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(skip_out_channels, out_channels, 1),
        ])

        self.upsample_net = nn.ConvTranspose1d(mel_channels, gate_channels, 
                                               mel_kernel, mel_stride)

    def forward(self, x, x_local):

        # x は音声信号、x_local はメル特徴
        B, _, T = x.size()
        # 特徴をアップサンプルし、音声と同じ長さの信号を出力
        c = self.upsample_net(x_local)
        x = self.first_conv(x)
        skips = 0
        for f in self.conv_layers:
            x, h = f(x, c, g_bct)
            skips += h
        skips *= math.sqrt(1.0 / len(self.conv_layers))

        x = skips
        for f in self.last_conv_layers:
            x = f(x)

        # 各強度の確率を出力
        x = F.softmax(x, dim=1)
        return x
