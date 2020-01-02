import torch
import torch.nn as nn

class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class DualCNN(nn.Module):
    def __init__(self):
        super(DualCNN, self).__init__()

        self.srcnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=4, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=5, padding=2, bias=False),
            nn.ReLU(inplace=False),
        )

        self.residual_layer = self.make_layer(Conv_ReLU_Block, 16)
        self.input = nn.Conv2d(in_channels=3,  out_channels=64, kernel_size=5, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=1, bias=False)
        self.output1 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=False)

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        Net_S = self.srcnn(x)
        
        out = self.relu(self.input(x))
        out = self.relu(self.residual_layer(out))
        out = self.relu(self.output(out))
        Net_D = self.relu(self.output1(out))
        out = torch.add(Net_D, Net_S)
        return out, Net_S




