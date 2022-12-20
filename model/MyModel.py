from model.mlp_mixer_pytorch.mlp_mixer_pytorch import MLPMixer

import torch.nn as nn
import torch

class MyModel(nn.Module):
    def __init__(self,
        image_size,
        horizon,
        channels = 3,
        patch_size = 10,
        dim = 512,
        depth = 12
        ):

        super().__init__()

        self.bev_fe = MLPMixer((image_size[0], image_size[1]), channels, patch_size, dim, depth, 100)
        self.speed_fe = nn.Sequential(
            nn.Linear(1, 100),
            nn.LayerNorm(100),
            nn.GELU()
        )

        self.regressor = nn.Sequential(
            nn.Linear(200, 50, bias=False),
            nn.LayerNorm(50),
            nn.GELU(),
            nn.Linear(50, horizon * 2)
        )

    def forward(self, bev, speed):
        bev_feature = self.bev_fe(bev)
        speed_feature = self.speed_fe(speed)

        feature = torch.cat([bev_feature, speed_feature], 1)
        out = self.regressor(feature)

        return out

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = MyModel((600,300), 10).to(device)

    bev = torch.ones([5, 3, 600, 300]).to(device)
    speed = torch.ones([5, 1]).to(device)

    model(bev, speed)
