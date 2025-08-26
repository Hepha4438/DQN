import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """ Deep Q Network (Atari-style) """

    def __init__(self, num_history=4, num_actions=18, input_size=(84, 84)):
        """
        num_history: số frame stack (thường = 4)
        num_actions: số action trong env
        input_size: (H, W) của ảnh đầu vào sau preprocess
        """
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(num_history, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Tính toán kích thước flatten sau conv
        with torch.no_grad():
            dummy = torch.zeros(1, num_history, *input_size)  # [1, C, H, W]
            dummy_out = self._forward_conv(dummy)
            conv_out_size = dummy_out.view(1, -1).size(1)

        self.fc4 = nn.Linear(conv_out_size, 512)
        self.fc5 = nn.Linear(512, num_actions)

    def _forward_conv(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x

    def forward(self, x):
        # x: [B, C=num_history, H, W], dtype float32
        x = x.float() / 255.0  # normalize input ảnh
        x = self._forward_conv(x)
        x = torch.flatten(x, 1)  # [B, -1]
        x = F.relu(self.fc4(x))
        return self.fc5(x)       # [B, num_actions]

