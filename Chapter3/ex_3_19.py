""" このファイルのコードは `python ex_3_19.py` で実行できます。
   （# 以降はコメントのため無視して構いません）
"""

import torch
import torch.nn as nn

class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# モデルの特徴抽出モジュールを表示
from torchvision.models import alexnet
model = alexnet(pretrained=True)
print(model.features)

# 異なる層の特徴抽出モジュールの構築
conv1 = nn.Sequential(*model.features[:1])
conv2 = nn.Sequential(*model.features[:4])
conv3_1 = nn.Sequential(*model.features[:7])
conv3_2 = nn.Sequential(*model.features[:9])
conv3_3 = nn.Sequential(*model.features[:11])

# 入力画像テンソル（1×3×224×224）に対する特徴テンソルを出力
feat1 = conv1(img)
feat2 = conv2(img)
feat3_1 = conv3_1(img)
feat3_2 = conv3_2(img)
feat3_3 = conv3_3(img)
print(feat1.shape)
print(feat2.shape)
print(feat3_1.shape)
print(feat3_2.shape)
print(feat3_3.shape)
