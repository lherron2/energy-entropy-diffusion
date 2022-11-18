#!~/miniconda3/bin/python

import numpy as np
import torch
import torchvision
import einops
import dill as pickle
import os
import sys
sys.path.insert(0,"/fs/classhomes/fall2022/cmsc828w/c828w013/energy-entropy-diffusion/denoising-diffusion-pytorch")
sys.path.append("/fs/classhomes/fall2022/cmsc828w/c828w013/energy-entropy-diffusion/glow")
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import GaussianDiffusion, Unet, Trainer
from  torch.nn.modules.upsampling import Upsample
from glow.model import FlowNet
from glow.modules import SinusoidalPosEmb



print("imports done")

dataset = "mnist"
beta=1e-3
dataset_name = f"mnist_noreg_newbeta"

batch_size = 64
image_size = 32
channels = 3
num_diffusion_timesteps = 1000
beta_start = 0.0001
beta_end = 0.02
device="cuda"

if dataset== "cifar":
  # Load CIFAR10 dataset
  data = torchvision.datasets.CIFAR10(root='.',
      download=True,
      train=True, transform=
          torchvision.transforms.Compose(
              [torchvision.transforms.ToTensor()]
      )
  )
  in_channels = 3

if dataset == "mnist":
  data = torchvision.datasets.MNIST(root='.',
      download=True,
      train=True, transform=
          torchvision.transforms.Compose(
              [torchvision.transforms.ToTensor()]
      )
  )
  in_channels = 3
  x0 = data.data.float()
  mean_diff = (x0 - x0.mean(0))
#  std = x0.std(0)
#  std[std == 0] = 1
#  std_samples = mean_diff/std
  data.data = mean_diff/255.

  d = data.data.view(-1,1,28,28)
  m = Upsample(scale_factor=32/28, mode='nearest')
  data.data = m(d)
  print(data.data.max())
data_loader = torch.utils.data.DataLoader(
    data.data,
    batch_size=batch_size,
    shuffle=True
)

print("data_loaded")

model = FlowNet((image_size,image_size,in_channels),
        128,
        16,
        5,
        1,
        "invconv",
        "affine",
        True)
SinPosEmb = SinusoidalPosEmb(image_size)

#model = Unet(
#    dim=32,
#    dim_mults = (1,2,2,4),
#    channels=3
#).to(device)

model = model.to('cuda')
model.train()
print("model")

def get_beta_schedule(beta_start, beta_end, num_diffusion_timesteps):
    eps = 1e-1
    remaining_betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps-1, dtype=np.float64)
    initial_beta = np.array([(1 - 2*eps) * beta_start**2 + eps])
    betas = np.concatenate([initial_beta, remaining_betas], axis=0)
    return betas

def compute_alphas(alpha_start, alpha_end, num_timesteps):
    s = 1e-2
    t = np.linspace(alpha_start, alpha_end, num_timesteps)
    alphas = np.cos((t + s) / (1+s) * np.pi / 2) ** 2
    return alphas

def retrieve_alphas(alphas, t):
    return alphas.index_select(0,t).view(-1,1,1,1)

def mse(output, e):
    (bs, _, dim, _) = output.shape
    loss = (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)/dim**2
    return loss

def energy(output):
  (bs, _, dim, _) = output.shape
  loss = (output**2).sum(dim=(1,2,3)).mean(dim=0)/dim**2
  return loss

def entropy(ldj):
    return ldj.mean()

alphas = torch.tensor(compute_alphas(0.05, 0.75, 1000)).to(device)
print("alphas_computed")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0,
        betas=(0.9, 0.999), amsgrad=True,
        eps=1e-9)

step = 0
model_idx = 0
entropy_arr, energy_arr, mse_arr, loss_arr = [], [], [], []
os.makedirs(f"/fs/classhomes/fall2022/cmsc828w/c828w013/energy-entropy-diffusion/models/{dataset_name}/metrics", 
            exist_ok=True)
os.makedirs(f"/fs/classhomes/fall2022/cmsc828w/c828w013/energy-entropy-diffusion/models/{dataset_name}/metrics", 
            exist_ok=True)

# for epoch in range(0, 40):
while True:
    for i, x0 in enumerate(data_loader):
    #for i, (x0, _) in enumerate(data_loader):
        x0 = x0.float()
        n = x0.size(0)
        if dataset == "mnist":
          x0 = einops.repeat(x0, 'h i j k ->  h i x j k', x=3).squeeze()

        optimizer.zero_grad()
        
        x0 = x0.to(device)
        e = torch.normal(0, 1, size=x0.size()).to(device)    
        t = torch.randint(low=0, high=1000, size=(n,)).to(device)
        a = retrieve_alphas(alphas, t).to(device).float()
        x = x0 * a.sqrt() + e * (1-a).sqrt()
        temb = SinPosEmb(t.float()).repeat(1,1,image_size,1).reshape(n,1,image_size,image_size)
        output, log_det_jac = model(x, temb)
        x0_preds = (x - output*(1-a).sqrt())/a.sqrt()
        mse_loss = mse(output, e)
        x_t = x0 * a.sqrt() + output * (1-a).sqrt()
        energy_loss = energy(x_t)
        entropy_loss = - beta*entropy(log_det_jac) # check sign
        loss = mse_loss 
        #loss = mse_loss + energy_loss + entropy_loss
        print(loss.detach())
        loss.backward()
        optimizer.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)

        
        if i % 100 == 0:
            print(loss.detach())
            entropy_arr.append(entropy_loss.cpu().detach().numpy())
            energy_arr.append(energy_loss.cpu().detach().numpy())
            mse_arr.append(mse_loss.cpu().detach().numpy())
            loss_arr.append(loss.cpu().detach().numpy())

        if step % 500 == 0:
            print("saved checkpoint")
            x0_preds = (x - output*(1-a).sqrt())/a.sqrt()
            checkpoint = {'state_dict': model.state_dict(),'optimizer' : optimizer.state_dict()}            
            torch.save(checkpoint, os.path.join(f"/fs/classhomes/fall2022/cmsc828w/c828w013/energy-entropy-diffusion/models/{dataset_name}", 
                                    f"model_{step}.pt"))

            np.savez(os.path.join(f"/fs/classhomes/fall2022/cmsc828w/c828w013/energy-entropy-diffusion/models/{dataset_name}/metrics",
                                 f"metrics_model_{step}.npz"), 
                     entropy=np.array(entropy_arr), energy=np.array(energy_arr), mse=np.array(mse_arr), 
                     loss=np.array(loss_arr))

            np.savez(os.path.join(f"/fs/classhomes/fall2022/cmsc828w/c828w013/energy-entropy-diffusion/models/{dataset_name}/metrics",
                                 f"x0_preds_model_{step}.npz"), 
                     x0_preds=x0_preds.cpu().detach().numpy(), x=x.cpu().detach().numpy(), a=a.cpu().detach().numpy(), 
                     e=e.cpu().detach().numpy(), output=output.cpu().detach().numpy())
        step += 1
