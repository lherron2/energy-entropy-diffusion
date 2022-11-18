#!~/miniconda3/bin/python

import numpy as np
import torch
import torchvision
import einops
import os
import sys
sys.path.insert(0,"/fs/classhomes/fall2022/cmsc828w/c828w013/energy-entropy-diffusion/denoising-diffusion-pytorch")
sys.path.append("/fs/classhomes/fall2022/cmsc828w/c828w013/energy-entropy-diffusion/glow")
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import GaussianDiffusion, Unet, Trainer
from glow.model import FlowNet
from glow.modules import SinusoidalPosEmb

def compute_alphas(alpha_start, alpha_end, num_timesteps):
    s = 1e-2
    t = np.linspace(alpha_start, alpha_end, num_timesteps)
    alphas = np.cos((t + s) / (1+s) * np.pi / 2) ** 2
    return alphas

def retrieve_alphas(alphas, t):
    return alphas.index_select(0,t).view(-1,1,1,1)

BP="/fs/classhomes/fall2022/cmsc828w/c828w013/energy-entropy-diffusion/"
dataset = "mnist_noreg_newbeta"
beta=1e-6

model_idx=500
model_path=BP + f"models/{dataset}/model_{model_idx}.pt"
sample_path=BP + f"models/{dataset}/samples"
os.makedirs(sample_path, exist_ok=True)
batch_size = 64
image_size = 32
in_channels = 3
num_diffusion_timesteps = 1000
beta_start = 0.0001
beta_end = 0.02
device="cuda"


model = FlowNet((image_size,image_size,in_channels),
        128,
        12,
        5,
        1,
        "invconv",
        "affine",
        True).to(device)

SinPosEmb = SinusoidalPosEmb(image_size)

model.load_state_dict(torch.load(model_path)['state_dict'])

eta=0.

x = torch.randn(
    batch_size,
    in_channels,
    image_size,
    image_size,
    dtype=torch.float
)

x = x.float()
x = x.to(device)

#betas = torch.from_numpy(
#    get_beta_schedule(beta_start, beta_end, num_diffusion_timesteps)).float().to(device)

alphas = torch.tensor(compute_alphas(0.05, 0.75, 1000)).to(device)

seq = range(1, num_diffusion_timesteps)
noise_arr = []
std_arr = []
with torch.no_grad():
    n = x.size(0)
    seq_next = [0] + list(seq[:-1])
    x0_preds, xs, x_last = [], [x], []
    xt_next = x
    times = list(zip(reversed(seq), reversed(seq_next)))
    for i,j in times:

        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        at = retrieve_alphas(alphas, t.long()).to(device).float()
        at_next = retrieve_alphas(alphas, next_t.long()).to(device).float()
        print(i, j)

        xt = xt_next
        temb = SinPosEmb(t.float()).repeat(1,1,image_size,1).reshape(n,1,image_size,image_size).to(device)
        et, ldj = model(xt, temb)

        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
        x0_preds.append(x0_t.cpu().detach().numpy())
        
        c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
        c2 = ((1 - at_next) - c1**2).sqrt()
        xt_next = at_next.sqrt() * x0_t + c1 * torch.normal(0, 1, size=x.size()).to(device) + c2 * et


np.save(os.path.join(sample_path, f"samples_model_{model_idx}.npy") , x0_t.cpu().detach().numpy())
np.save(os.path.join(sample_path, f"sample_seq_model_{model_idx}.npy") , x0_preds)
