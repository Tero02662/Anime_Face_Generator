import torch.nn as nn
import yaml

with open ("py.yml") as f:
    config = yaml.safe_load(f)

cfg = config["Discriminator"]

class Discriminator(nn.Module):
  def __init__(self):
    super().__init__()
    self.main = nn.Sequential(
        nn.Conv2d(cfg['in_nc'], cfg['nc'], 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, True),

        nn.Conv2d(cfg['nc'], cfg['nc']*2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(cfg['nc']*2),
        nn.LeakyReLU(0.2, True),

        nn.Conv2d(cfg['nc']*2, cfg['nc']*4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(cfg['nc']*4),
        nn.LeakyReLU(0.2, True),

        nn.Conv2d(cfg['nc']*4, cfg['nc']*8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(cfg['nc']*8),
        nn.LeakyReLU(0.2, True),

        nn.Conv2d(cfg['nc']*8, 1, 4, 1, 0, bias=False),
        nn.Sigmoid()
      )

  def forward(self, x):
    out = self.main(x)
    return out

netD = Discriminator()