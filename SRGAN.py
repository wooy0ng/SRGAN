# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"{device} is available")

# %% [markdown]
# ### Get Image Path

# %%
import os

PATH = 'Data/'
HR_list = os.listdir(PATH + 'HR/')
print("Image cnt : {}".format(len(HR_list)))

LR_list = os.listdir(PATH + 'LR/')
print("Image cnt : {}".format(len(LR_list)))


# %%
print(HR_list)

# %%
from PIL import Image
import numpy as np

LR_images = []
for path in LR_list:
    img = Image.open(os.path.join(PATH, 'LR', path))
    img = np.asarray(img)
    LR_images.append(img)

HR_images = []
for path in HR_list:
    img = Image.open(os.path.join(PATH, 'HR', path))
    img = np.asarray(img)
    HR_images.append(img)

LR_images = np.asarray(LR_images)
HR_images = np.asarray(HR_images)

# %%
print(f"LR Images shape : {LR_images.shape}")
print(f"HR Images shape : {HR_images.shape}")

# %% [markdown]
# # create B ResidualBlock  
# 

# %%
class B_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(B_ResidualBlock, self).__init__()
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_block = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.PReLU(),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
        )
        if self.stride != 1 or self.in_channels != self.out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.out_channels)
            )
    
    def forward(self, x):
        out = self.conv_block(x)
        if self.stride != 1 or self.in_channels != self.out_channels:
            x = self.downsample(x)
        out = x + out
        return out

# %% [markdown]
# # Generator Network

# %%
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.in_channels = 64
        self.num_bloacks = 5

        self.base = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PReLU()
        )

        self.layer1 = self.make_layer(64, self.num_bloacks, stride=1)    # B_residual blocks
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 256, 3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 256, 3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )
        self.layer5 = nn.Conv2d(64, 3, 9, stride=1, padding=4)


    def make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            block = B_ResidualBlock(self.in_channels, out_channels)
            layers.append(block)
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        base = self.base(x)         # base : used for 'skip connection'
        layer1 = self.layer1(base)  # layer1 : B_residual blocks
        layer2 = self.layer2(layer1)
        layer2 = layer2 + base      # Elementwise Sum
        
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        layer5 = self.layer5(layer4)
        return layer5

# %%
# model
generator = Generator().to(device)

# %%
import torchsummary
torchsummary.summary(generator, (3, 96, 96))

# %% [markdown]
# # Discriminator Network

# %%
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.in_channels = 64

        self.base = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, padding=1),
            nn.LeakyReLU()
        )
        self.conv_layer1 = self.make_layer(64, 64, 2)
        self.conv_layer2 = self.make_layer(64, 128, 1)
        self.conv_layer3 = self.make_layer(128, 128, 2)
        self.conv_layer4 = self.make_layer(128, 256, 1)
        self.conv_layer5 = self.make_layer(256, 256, 2)
        self.conv_layer6 = self.make_layer(256, 512, 1)
        self.conv_layer7 = self.make_layer(512, 512, 2)

        self.fc_layer1 = nn.Sequential(
            nn.Linear(512 * 6 * 6, 1024, bias=False),
            nn.LeakyReLU()
        )
        self.fc_layer2 = nn.Sequential(
            nn.Linear(1024, 1, bias=False),
            nn.Sigmoid()
        )
        
    def make_layer(self, in_channels, out_channels, stride):
        conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
        return conv_block

    def forward(self, x):
        out = self.base(x)
        out = self.conv_layer1(out)
        out = self.conv_layer2(out)
        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.conv_layer5(out)
        out = self.conv_layer6(out)
        out = self.conv_layer7(out)

        out = out.view(out.size(0), -1)
        out = self.fc_layer1(out)
        out = self.fc_layer2(out)
        return out

# %%
discriminator = Discriminator().to(device)
torchsummary.summary(discriminator, (3, 96, 96))

# %%


# %% [markdown]
# ## Generator Test
# - Input 데이터를 넣으면 SR 데이터로 업스케일링이 잘 되는지 확인해보장

# %%
print(f"LR Images shape : {LR_images.shape}")
print(f"HR Images shape : {HR_images.shape}")

# %%
print(LR_images[0].shape)

# %%
from torch.utils.data import DataLoader

# %%
LR_tensor = torch.Tensor(LR_images)
HR_tensor = torch.Tensor(HR_images)

LR_dataloader = DataLoader(LR_tensor)
HR_dataloader = DataLoader(HR_tensor)

# %%
len(LR_dataloader)

# %%
result = []
for data in LR_dataloader:
    inputs = data.to(device)
    outputs = generator(inputs.view(-1, 3, 96, 96))
    result.append(outputs)
    break

print(len(result))

# %%
import numpy as np

# %%
img = np.asarray(result[0].cpu().detach().numpy())


# %%
img.shape

# %%
import matplotlib.pyplot as plt

# %%

plt.imshow(np.reshape(img, (384, 384, 3)))



# %%



