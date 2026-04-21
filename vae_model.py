"""
VAE Model - Clean rewrite, no residual blocks (eliminates inplace bug entirely)
Annapurna Srayashya Iruku (U01150990)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size
        self.register_buffer('window', self._make_window(window_size))

    def _make_window(self, size):
        x = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-x.pow(2) / (2 * 1.5 ** 2))
        g = g / g.sum()
        w = g.unsqueeze(1) @ g.unsqueeze(0)
        return w.unsqueeze(0).unsqueeze(0)

    def forward(self, a, b):
        C1, C2 = 0.0001, 0.0009
        w = self.window.to(a.device).expand(a.size(1), 1, -1, -1)
        p = self.window_size // 2
        mu_a  = F.conv2d(a, w, padding=p, groups=a.size(1))
        mu_b  = F.conv2d(b, w, padding=p, groups=b.size(1))
        mu_aa = F.conv2d(a * a, w, padding=p, groups=a.size(1)) - mu_a * mu_a
        mu_bb = F.conv2d(b * b, w, padding=p, groups=b.size(1)) - mu_b * mu_b
        mu_ab = F.conv2d(a * b, w, padding=p, groups=a.size(1)) - mu_a * mu_b
        ssim  = ((2 * mu_a * mu_b + C1) * (2 * mu_ab + C2)) / \
                ((mu_a**2 + mu_b**2 + C1) * (mu_aa + mu_bb + C2))
        return (1 - ssim).mean()


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1)
        feats = list(vgg.features.children())
        # Three fixed VGG slices at increasing depths
        self.s1 = nn.Sequential(*feats[:4])    # relu1_2
        self.s2 = nn.Sequential(*feats[4:9])   # relu2_2
        self.s3 = nn.Sequential(*feats[9:16])  # relu3_3
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, recon, target):
        # Grayscale -> 3ch
        r = recon.repeat(1, 3, 1, 1)
        t = target.repeat(1, 3, 1, 1).detach()  # no grad through target
        loss = 0.0
        for s in (self.s1, self.s2, self.s3):
            r = s(r)
            t = s(t)
            loss = loss + F.mse_loss(r, t)
        return loss


class Encoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        # Plain conv stack — NO residual connections, NO inplace activations
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1, bias=False), nn.BatchNorm2d(32), nn.LeakyReLU(0.2),   # 112
            nn.Conv2d(32, 64, 4, 2, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),   # 56
            nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2), # 28
            nn.Conv2d(128, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),# 14
            nn.Conv2d(256, 512, 4, 2, 1, bias=False), nn.BatchNorm2d(512), nn.LeakyReLU(0.2),# 7
            nn.Flatten(),
        )
        self.fc_mu     = nn.Linear(512 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(512 * 7 * 7, latent_dim)

    def forward(self, x):
        h = self.net(x)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 512 * 7 * 7)
        # Plain deconv stack — NO residual connections, NO inplace activations
        self.net = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(), # 14
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(), # 28
            nn.ConvTranspose2d(128, 64,  4, 2, 1, bias=False), nn.BatchNorm2d(64),  nn.ReLU(), # 56
            nn.ConvTranspose2d(64,  32,  4, 2, 1, bias=False), nn.BatchNorm2d(32),  nn.ReLU(), # 112
            nn.ConvTranspose2d(32,   1,  4, 2, 1),                                             # 224
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(self.fc(z).view(-1, 512, 7, 7))


class VAE(nn.Module):
    def __init__(self, latent_dim=256, beta=0.1, use_perceptual=True, ssim_weight=0.5):
        super().__init__()
        self.latent_dim     = latent_dim
        self.beta           = beta
        self.ssim_weight    = ssim_weight
        self.use_perceptual = use_perceptual

        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.ssim    = SSIMLoss()
        if use_perceptual:
            self.perceptual = PerceptualLoss()

        self.register_buffer('latent_mean', torch.zeros(latent_dim))
        self.register_buffer('latent_var',  torch.ones(latent_dim))

    def reparameterize(self, mu, logvar):
        if self.training:
            return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return mu

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z    = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar, z

    def update_latent_stats(self, mu):
        with torch.no_grad():
            a = 0.01
            self.latent_mean = (1 - a) * self.latent_mean + a * mu.mean(0)
            self.latent_var  = (1 - a) * self.latent_var  + a * mu.var(0).clamp(min=1e-6)

    def vae_loss(self, recon, x, mu, logvar):
        mse   = F.mse_loss(recon, x)
        ssim  = self.ssim(recon, x)
        recon_loss = (1 - self.ssim_weight) * mse + self.ssim_weight * ssim
        if self.use_perceptual:
            recon_loss = recon_loss + 0.1 * self.perceptual(recon, x)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.beta * kl, recon_loss, kl

    def anomaly_score(self, x):
        with torch.no_grad():
            recon, mu, logvar, z = self.forward(x)
            recon_err = F.mse_loss(recon, x, reduction='none').mean(dim=[1, 2, 3])
            mahal     = ((mu - self.latent_mean) ** 2 / (self.latent_var + 1e-8)).mean(1)
            score     = recon_err + 0.3 * mahal
        return score, recon, z
