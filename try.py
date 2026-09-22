import torch
import torchvision as tv
from Generator import netG
import numpy as np
import matplotlib.pyplot as plt


img_list = []
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
fixed_noise = torch.randn(64, 100, 1, 1, device=device)
with torch.no_grad():
  fake = netG(fixed_noise).detach().cpu()
img_list.append(tv.utils.make_grid(fake, padding=2, normalize=True))
fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)))] for i in img_list]
plt.show()
