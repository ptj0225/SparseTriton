from sparsetriton.nn.modules.conv import Conv3d
from sparsetriton import SparseTensor
from sparsetriton.tensor import randn
from tqdm import tqdm
from torch import nn
import torch

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv3d(16, 64, (3, 3, 3), 1, 1)
        self.relu1 = nn.ReLU()
        self.conv2 = Conv3d(64, 64, (3, 3, 3), 1, 1)
        self.relu2 = nn.ReLU()
        self.conv3 = Conv3d(64, 64, (3, 3, 3), 1, 1)
        self.relu3 = nn.ReLU()
        self.conv4 = Conv3d(64, 64, (3, 3, 3), 1, 1)
        self.relu4 = nn.ReLU()
        self.conv5 = Conv3d(64, 1, (3, 3, 3), 1, 1) # 변수명 중복 수정

    def forward(self, x):
        # SparseTensor의 특징값(.F)에 대해서만 ReLU를 적용하고 교체(replace)합니다.
        x = self.conv1(x)
        x = x.replace(feats=self.relu1(x.F))
        
        x = self.conv2(x)
        x = x.replace(feats=self.relu2(x.F))
        
        x = self.conv3(x)
        x = x.replace(feats=self.relu3(x.F))
        
        x = self.conv4(x)
        x = x.replace(feats=self.relu4(x.F))
        
        x = self.conv5(x)
        return x
net = Net().to("cuda")


optim = torch.optim.Adam(net.parameters())

# x = randn((512, 512, 512), 10, 16, 512**3 // 100).to("cuda")
y = 10

for _ in tqdm(range(10000)):
    x = randn((512, 512, 512), 1, 16, 512**3 // 20).to("cuda")
    optim.zero_grad()
    out = net(x)
    loss = (out.F - y) ** 2
    loss = loss.mean()
    loss.backward()
    
    optim.step()