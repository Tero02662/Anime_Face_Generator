import torch.nn as nn
import yaml

with open ("py.yml") as f:
    config = yaml.safe_load(f)

cfg = config["Generator"]


class Generator(nn.Module):
  def __init__(self):
    super().__init__()
    self.main = nn.Sequential(
        nn.ConvTranspose2d(100, cfg['nc']*8, 4, 1, 0, bias=False),
        nn.BatchNorm2d(cfg['nc']*8),
        nn.ReLU(True),

        nn.ConvTranspose2d(cfg['nc']*8, cfg['nc']*4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(cfg['nc']*4),
        nn.ReLU(True),

        nn.ConvTranspose2d(cfg['nc']*4, cfg['nc']*2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(cfg['nc']*2),
        nn.ReLU(True),

        nn.ConvTranspose2d(cfg['nc']*2, cfg['nc'], 4, 2, 1, bias=False),
        nn.BatchNorm2d(cfg['nc']),
        nn.ReLU(True),

        nn.ConvTranspose2d(cfg['nc'], 3, 4, 2, 1, bias=False),
        nn.Tanh()
    )

  def forward(self, x):
    out = self.main(x)
    return out

netG = Generator()