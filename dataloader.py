import torchvision as tv
import torch
import yaml

with open('py.yml') as f:
    config = yaml.safe_load(f)
cfg = config['Dataset']


dataset = tv.datasets.ImageFolder(
    root=cfg['path'],
    transform=tv.transforms.Compose([
        tv.transforms.Resize(cfg['image_size']),
        tv.transforms.CenterCrop(cfg['image_size']),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
)

dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=2)