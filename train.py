import torch
import torch.nn as nn
from Generator import netG
from Discriminator import netD
from dataloader import dataloader
import yaml
import torchvision as tv

with open ('py.yml') as f:
    config = yaml.safe_load(f)
cfg = config['Optimizer']
img_list = []
G_losses = []
D_losses = []
iters = 0
num_epochs = 5
real_label = 1
fake_label = 0
loss_fn = nn.BCELoss()
netG_optim = torch.optim.Adam(netG.parameters(), lr=cfg['lr'], betas=(cfg['beta1'], cfg['beta2']))
netD_optim = torch.optim.Adam(netD.parameters(), lr=cfg['lr'], betas=(cfg['beta1'], cfg['beta2']))
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
fixed_noise = torch.randn(64, 100, 1, 1, device=device)
netG.to(device)
netD.to(device)
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):


        netD.zero_grad()
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        output = netD(real_cpu).view(-1)
        errD_real = loss_fn(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(b_size, 100, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach()).view(-1)
        errD_fake = loss_fn(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        netD_optim.step()

        netG.zero_grad()
        label.fill_(real_label)
        output = netD(fake).view(-1)
        errG = loss_fn(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        netG_optim.step()

        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        G_losses.append(errG.item())
        D_losses.append(errD.item())

        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(tv.utils.make_grid(fake, padding=2, normalize=True))

        iters += 1