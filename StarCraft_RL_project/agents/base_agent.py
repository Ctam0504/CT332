import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class BaseAgent(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(BaseAgent, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Linear(conv_out_size, 512)
        self.head = nn.Linear(512, n_actions)


    def _get_conv_out(self, shape):
        o = torch.zeros(1, *shape)
        o = self.conv1(o)
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return self.head(x)