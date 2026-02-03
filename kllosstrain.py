def kl_annealing(epoch, total_epochs):
    return min(1.0, epoch / (0.3 * total_epochs))

import torch

def vae_loss(x, recon, mu, logvar, beta):
    recon_loss = torch.mean((x - recon) ** 2)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl

from src.training.loss import vae_loss
from src.training.annealing import kl_annealing

def train(model, loader, optimizer, epochs, device):
    model.train()

    for epoch in range(epochs):
        beta = kl_annealing(epoch, epochs)

        for (x,) in loader:
            x = x.to(device)

            recon, mu, logvar = model(x)
            loss = vae_loss(x, recon, mu, logvar, beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
