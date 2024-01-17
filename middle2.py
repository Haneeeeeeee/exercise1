import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torchvision.utils as vutils
import os
import time

# 设置参数
batch_size = 64#批量大小，表示在更新网络权重之前要处理的训练样本数量
image_size = 64#图像大小，输入图像调整为64x64像素
nz = 100  #噪声向量的维度，用于生成器输入
ngf = 64  #生成器特征映射的深度
ndf = 64  #判别器特征映射的深度
num_epochs = 20#迭代次数，表示整个训练数据集将被遍历几次
lr = 0.0002#学习率
beta1 = 0.5#Adam优化器的参数，用于计算梯度及其平方的指数移动平均，是一个超参数
ngpu = 1

#加载并转换数据集
dataset = dset.CIFAR10(root="./data", download=True,
                       transform=transforms.Compose([
                           transforms.Resize(image_size),#图像大小调整为64x64
                           transforms.ToTensor(),#将图像转化为张量
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                       ]))

#创建数据加载器
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

#设备配置
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

#生成器，神经网络结构是CNN卷积神经网络
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        #生成器的网络结构
        self.main = nn.Sequential(
            #全连接层并转化为多维数据
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            #上一层的输出形状: (ngf*8) x 4 x 4，卷积转置
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            #上一层的输出形状: (ngf*4) x 8 x 8，卷积转置
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            #上一层的输出形状: (ngf*2) x 16 x 16，卷积转置
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            #上一层的输出形状: (ngf) x 32 x 32，卷积转置，并输出3通道图像
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()#使用Tanh使输出转置到[1,1]，与图像标准化匹配
            #最终输出形状: 3 x 64 x 64
        )

    def forward(self, input):
        #前向传播
        return self.main(input)

#判别器定义
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        #判别器网络结构
        self.main = nn.Sequential(
            #输入形状: 3 x 64 x 64，将3通道的图像转换到ndf通道的图像
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),#使用LeakyReLu激活函数
            #上一层的输出形状: (ndf) x 32 x 32，卷积层
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            #上一层的输出形状: (ndf*2) x 16 x 16，卷积层
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            #上一层的输出形状: (ndf*4) x 8 x 8，卷积层
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            #上一层的输出形状: (ndf*8) x 4 x 4，同为卷积层，但是输出是一个单值（真或假）
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()#使用Sigmoid激活函数将输出值规范到[0,1]
        )

    def forward(self, input):
        #判别器的前向传播
        return self.main(input).view(-1, 1).squeeze(1)

#创建生成器和判别器实例，.to(device)用于指定配置进行训练
netG = Generator(ngpu).to(device)
netD = Discriminator(ngpu).to(device)

#权重的初始化函数
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)#卷积层的权重进行正态分布初始化
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)#对批量标准层的权重进行正态分布初始化
        nn.init.constant_(m.bias.data, 0)#偏置量设置为0

#应用权重初始化到判别器和生成器
netG.apply(weights_init)
netD.apply(weights_init)

#定义损失函数和优化器
criterion = nn.BCELoss()#使用二进制交叉熵损失
#将Adam改为AdamW，起到更好的泛化作用
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

#训练循环
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        #训练判别器
        netD.zero_grad()
        #将真实图像移至设备
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), 1, dtype=torch.float, device=device)
        #判别器在真实图像的输出
        output = netD(real_cpu)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()
        #生成假图像以训练判别器识别它们
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(0)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        #训练生成器
        netG.zero_grad()
        label.fill_(1)#生成器要做的就是让判别器将其输出设置为真实的
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        #打印训练状态
        if i % 50 == 0:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
    #每个epoch结束时保存模型状态
    timestamp = time.strftime("%Y%m%d-%H%M%S")#增加时间戳，在多次训练不必移除已经得出的图像
    torch.save(netG.state_dict(), f'./output/netG_epoch_{epoch}.pth')
    torch.save(netD.state_dict(), f'./output/netD_epoch_{epoch}.pth')

    #保存生成的图像
    if not os.path.exists('./output'):
        os.makedirs('./output')
    vutils.save_image(real_cpu, f"./output/real_samples_epoch_{epoch}.png", normalize=True)
    fake = netG(noise)
    vutils.save_image(fake.detach(), f"./output/fake_samples_epoch_{epoch}.png", normalize=True)
