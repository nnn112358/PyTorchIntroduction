""" 本コードは DC-GAN 実装の参考例です。
"""

import torch
import torch.nn as nn

#  生成器の定義
class Generator(nn.Module):
    def __init__(self, nz, ngf):
        super(Generator, self).__init__()
        self.main = nn.Sequential(

            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.main(input)
        return output

# 識別器（Discriminator）の定義
class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(

            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):

        output = self.main(input)
        return output.view(-1, 1).squeeze(1)

# DCGAN の学習コード

def train():
    # nz: 潜在変数の次元
    # ngf, ncf: 生成器/識別器の特徴次元
    netG = Generator(nz, ngf).to(device)
    netD = Discriminator(nc, ndf).to(device)
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr,
        betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr,
        betas=(opt.beta1, 0.999))
    real_label = 1 # 実画像のラベル
    fake_label = 0 # 生成画像のラベル
    for epoch in range(opt.niter):
        for i, data in enumerate(dataloader, 0):
            # 識別器の勾配を 0 に
            netD.zero_grad()
            real_cpu = data[0].to(device)
            batch_size = real_cpu.size(0)
            # 入力データを定義
            label = torch.full((batch_size,), real_label, device=device)
            output = netD(real_cpu)
            # 実画像に対する識別器の損失
            errD_real = criterion(output, label)
            # 実画像に対して逆伝播
            errD_real.backward()
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach())
            # 生成画像に対する識別器の損失
            errD_fake = criterion(output, label)
            # 生成画像に対して逆伝播
            errD_fake.backward()
            # 識別器の総損失＝実画像損失＋生成画像損失
            errD = errD_real + errD_fake
            # 識別器を最適化
            optimizerD.step()        
    
            # 生成器の勾配を 0 に
            netG.zero_grad()
            # ここでは real_label（前段は fake_label）
            label.fill_(real_label)
            output = netD(fake)
            # 実画像に対する識別器の損失
            errG = criterion(output, label)
            # 生成器に対して逆伝播
            errG.backward()
            # 生成器を最適化
            optimizerG.step()
