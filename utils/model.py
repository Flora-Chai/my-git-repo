"""ResNet1818/34/50/101/152 in Pytorch."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
class LeNet5(nn.Module):
    def __init__(self, num_classes=10, init_weights=False):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(True)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=512, kernel_size=5, stride=1)  # 修改为 out_channels=512

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        #self.fc1 = nn.Linear(512 * 5 * 5, 120)  # 修改输入维度
        self.fc1 = nn.Linear(512 * 4 * 4, 120)

        self.fc1_to_fc2 = nn.Linear(120, 84)  # 添加一个线性层来匹配修改后的fc1的输出维度

        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        x = torch.flatten(x, start_dim=1)  # 正确展平操作
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)  # 直接使用fc2，假设您移除了fc1_to_fc2
        x = self.relu(x)
        x = self.fc3(x)
        return x
class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(64, 64, 2, stride=1)
        self.layer2 = self.make_layer(64, 128, 2, stride=2)
        self.layer3 = self.make_layer(128, 256, 2, stride=2)
        self.layer4 = self.make_layer(256, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = nn.Sequential()

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class TransformerModel(nn.Module):
    def __init__(self, num_classes=10, d_model=32, nhead=4, num_encoder_layers=2, dim_feedforward=2048, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # 将输入张量调整为 Transformer 需要的形状 (N, C, H, W) -> (N, W, C, H)
        batch_size, _, _, _ = x.size()
        x = x.view(batch_size, -1, x.size(3))  # 将输入张量展平 (N, W, C, H) -> (N * W, C, H)
        x = self.transformer_encoder(x)  # 输入 Transformer 编码器
        x = x.mean(dim=1)  # 沿序列维度求平均，得到整个图像的表示
        x = self.fc(x)  # 全连接层
        return x


from einops.layers.torch import Rearrange
from einops import rearrange

class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3):
        super(VisionTransformer, self).__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim), num_layers=depth)
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        p = self.patch_size
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.transf
        ormer_encoder(x)
        x = x.mean(dim=1)  # 沿序列维度求平均，得到整个图像的表示
        x = self.mlp_head(x)
        return x
