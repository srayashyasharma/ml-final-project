"""
VAE Training Script
CS 6840/4840 - Annapurna Srayashya Iruku (U01150990)

Usage:
    python train_vae.py \
        --data_dir /content/drive/MyDrive/NIH_chest \
        --labels_csv /content/drive/MyDrive/NIH_chest/Data_Entry_2017.csv
"""

import argparse, os, math
import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset import get_dataloaders
from vae_model import VAE


def train_vae(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    os.makedirs(args.save_dir, exist_ok=True)

    train_loader, val_loader, _ = get_dataloaders(
        args.data_dir, args.labels_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_per_class=args.max_per_class,
    )

    model = VAE(
        latent_dim=args.latent_dim,
        beta=args.beta,
        use_perceptual=not args.no_perceptual,
        ssim_weight=args.ssim_weight,
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    total_steps  = args.epochs * len(train_loader)
    warmup_steps = total_steps // 10

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.05, 0.5 * (1 + math.cos(math.pi * p)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    # GradScaler only active when CUDA is available
    use_amp = device.type == 'cuda'
    scaler  = GradScaler('cuda', enabled=use_amp)

    history = {'train': [], 'val': []}
    best_val = float('inf')

    for epoch in range(args.epochs):
        # ── train ──
        model.train()
        t_loss = 0.0
        for i, (imgs, _) in enumerate(train_loader):
            imgs = imgs.to(device)
            with autocast(device.type, enabled=use_amp):
                recon, mu, logvar, _ = model(imgs)
                loss, recon_l, kl_l  = model.vae_loss(recon, imgs, mu, logvar)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            model.update_latent_stats(mu)
            t_loss += loss.item()

            if i % 200 == 0:
                print(f"  Ep{epoch+1} [{i}/{len(train_loader)}] "
                      f"loss={loss.item():.4f} recon={recon_l.item():.4f} kl={kl_l.item():.4f}")

        # ── validate ──
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for imgs, _ in val_loader:
                imgs = imgs.to(device)
                recon, mu, logvar, _ = model(imgs)
                loss, _, _ = model.vae_loss(recon, imgs, mu, logvar)
                v_loss += loss.item()

        t_avg = t_loss / len(train_loader)
        v_avg = v_loss / len(val_loader)
        history['train'].append(t_avg)
        history['val'].append(v_avg)
        print(f"Epoch {epoch+1}/{args.epochs}  train={t_avg:.4f}  val={v_avg:.4f}")

        if v_avg < best_val:
            best_val = v_avg
            torch.save({
                'model_state':  model.state_dict(),
                'latent_mean':  model.latent_mean.cpu(),
                'latent_var':   model.latent_var.cpu(),
                'epoch':        epoch,
                'val_loss':     v_avg,
            }, os.path.join(args.save_dir, 'best_vae.pth'))
            print(f"  -> Saved best (val={v_avg:.4f})")

        torch.save(model.state_dict(), os.path.join(args.save_dir, 'latest_vae.pth'))

    # plot
    plt.figure(figsize=(7, 4))
    plt.plot(history['train'], label='Train')
    plt.plot(history['val'],   label='Val')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title('VAE Training Loss'); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'training_curves.png'), dpi=150)
    plt.close()
    print(f"Done. Best val={best_val:.4f}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',      required=True)
    p.add_argument('--labels_csv',    required=True)
    p.add_argument('--save_dir',      default='checkpoints')
    p.add_argument('--latent_dim',    type=int,   default=256)
    p.add_argument('--batch_size',    type=int,   default=32)
    p.add_argument('--epochs',        type=int,   default=40)
    p.add_argument('--lr',            type=float, default=1e-3)
    p.add_argument('--beta',          type=float, default=0.1)
    p.add_argument('--ssim_weight',   type=float, default=0.5)
    p.add_argument('--no_perceptual', action='store_true',
                   help='Disable VGG perceptual loss (use on CPU to save memory)')
    p.add_argument('--num_workers',   type=int,   default=2)
    p.add_argument('--max_per_class', type=int,   default=None)
    args = p.parse_args()
    train_vae(args)
