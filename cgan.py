import pandas as pd
import torch.nn as nn
# 读取CSV文件
#file_path = '../data/tall.csv'
#df = pd.read_csv(file_path)  # 指定第一列为索引列

# 查看数据
#print(df.head())

# 假设最后一列是标签列，其余列为特征列
# features = df.iloc[:, 1:]  # 特征列
# labels = df.iloc[:, 0]  # 标签列

# 查看特征和标签
# print("Features:")
# print(features.head())
# print("Labels:")
# print(labels)



################################################

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 标准化特征
# scaler = StandardScaler()
# features_scaled = scaler.fit_transform(features)

# 将数据划分为训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)
#
# # 查看划分后的数据
# print("Training features shape:", X_train.shape)
# print("Testing features shape:", X_test.shape)
# print("Training labels shape:", y_train.shape)
# print("Testing labels shape:", y_test.shape)




import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
#
# # 创建数据加载器
# train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train.values).long())
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
#
# test_dataset = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test.values).long())
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
#
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

######################################
class Generator(nn.Module):
    def __init__(self, latent_dim, n_classes, img_shape):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(n_classes, n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim + n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )
        self.img_shape = img_shape  # 明确保存img_shape

    def forward(self, noise, labels):
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.img_shape)  # 使用self.img_shape
        return img

class Discriminator(nn.Module):
    def __init__(self, n_classes, img_shape):
        super(Discriminator, self).__init__()
        self.label_embedding = nn.Embedding(n_classes, n_classes)

        self.model = nn.Sequential(
            nn.Linear(n_classes + int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, img, labels):
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity
    ################################################
import torch
import torch.nn as nn
import numpy as np
#
# # 配置参数
# opt = {
#     'latent_dim': 100,  # 噪声维度
#     'n_classes': 2,     # 类别数量（0和1）
#     'img_shape': (1, 30)  # 图像形状（假设为单一通道的50维向量）
# }
#
# # 创建模型实例
# generator = Generator(opt['latent_dim'], opt['n_classes'], opt['img_shape']).to(device)
# discriminator = Discriminator(opt['n_classes'], opt['img_shape']).to(device)
#
# # 定义损失函数和优化器
# adversarial_loss = torch.nn.BCEWithLogitsLoss()
# optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
# optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
#
# # 训练CGAN
# num_epochs = 100
# for epoch in range(num_epochs):
#     for i, (real_imgs, real_labels) in enumerate(train_loader):
#         batch_size = real_imgs.size(0)
#
#         # 将数据移动到指定设备
#         real_imgs = real_imgs.to(device)
#         real_labels = real_labels.to(device)
#
#         # 创建真实的和假的标签
#         valid = torch.ones(batch_size, 1).to(device)
#         fake = torch.zeros(batch_size, 1).to(device)
#
#         # 生成噪声
#         z = torch.randn(batch_size, opt['latent_dim']).to(device)
#         gen_labels = torch.randint(0, opt['n_classes'], (batch_size,)).to(device)
#
#         # 训练生成器
#         optimizer_G.zero_grad()
#         gen_imgs = generator(z, gen_labels)
#         validity = discriminator(gen_imgs, gen_labels)
#         g_loss = adversarial_loss(validity, valid)
#         g_loss.backward()
#         optimizer_G.step()
#
#         # 训练判别器
#         optimizer_D.zero_grad()
#         real_validity = discriminator(real_imgs, real_labels)
#         real_loss = adversarial_loss(real_validity, valid)
#         fake_validity = discriminator(gen_imgs.detach(), gen_labels)
#         fake_loss = adversarial_loss(fake_validity, fake)
#         d_loss = (real_loss + fake_loss) / 2
#         d_loss.backward()
#         optimizer_D.step()
#
#         # 打印训练进度
#         print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
#               (epoch, num_epochs, i, len(train_loader), d_loss.item(), g_loss.item()))