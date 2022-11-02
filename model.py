import torch

from torch import nn
import torch.nn.functional as F

def init_weights(m) :
    if isinstance(m, nn.Linear) :
        torch.nn.init.orthogonal_(m.weight)
        m.bias.data.fill_(0.01)
    if isinstance(m, nn.Conv2d) :
        torch.nn.init.orthogonal_(m.weight)

class Action_Network(nn.Module) :
    def __init__(self, a_dim, SIGMA_FLOOR=0.0) :
        super(Action_Network, self).__init__()
        self.a_dim = a_dim
        self.SIGMA_FLOOR = SIGMA_FLOOR

        self.conv = nn.Sequential(
            #nn.Conv2d(1, 32, (15,15), (1,1), (7,7)),
            #nn.ReLU(inplace=True),

            #nn.Conv2d(32, 32, (7, 7), (1, 1), (3, 3)),
            #nn.ReLU(inplace=True),

            nn.Conv2d(2, 64, (3, 3), (2, 2), (1, 1)),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True),

        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True),

        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True),

        )
        self.out_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(15*15*64, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.a_dim),
            nn.Softmax(dim=1)
        )

        self.conv.apply(init_weights)
        self.out_layer.apply(init_weights)

    def forward(self, x) :
        x = self.conv(x)
        x_init = x
        x = self.conv1(x)
        x = x + x_init
        x_init = x
        x = self.conv2(x)
        x = x + x_init
        x_init = x
        x = self.conv3(x)
        x = x + x_init

        x = self.out_layer(x)

        dist = torch.distributions.Categorical(x)

        return dist

class Critic_Network(nn.Module) :
    def __init__(self) :
        super(Critic_Network, self).__init__()

        self.conv = nn.Sequential(
            #nn.Conv2d(1, 32, (15,15), (1,1), (7,7)),
            #nn.ReLU(inplace=True),

            #nn.Conv2d(32, 32, (7, 7), (1, 1), (3, 3)),
            #nn.ReLU(inplace=True),

            nn.Conv2d(2, 64, (3, 3), (2, 2), (1, 1)),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True),

        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True),

        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(inplace=True),

        )
        self.out_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(15*15*64, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )
        self.conv.apply(init_weights)
        self.out_layer.apply(init_weights)

    def forward(self, x) :
        x = self.conv(x)
        x_init = x
        x = self.conv1(x)
        x = x + x_init
        x_init = x
        x = self.conv2(x)
        x = x + x_init
        x_init = x
        x = self.conv3(x)
        x = x + x_init

        vf = self.out_layer(x)
        return vf.squeeze(1)