"""
MLP Classification Head on VAE Latent Space

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.5):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        ce       = F.cross_entropy(logits, targets, reduction='none')
        p_t      = torch.exp(-ce)
        tf       = targets.float()                          # MUST cast Long -> float
        alpha_t  = self.alpha * tf + (1 - self.alpha) * (1 - tf)
        loss     = alpha_t * (1 - p_t) ** self.gamma * ce
        return loss.mean()


class MLPClassifier(nn.Module):
    def __init__(self, latent_dim=256, hidden=512, num_classes=2, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(latent_dim),
            nn.Linear(latent_dim, hidden),
            nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2), nn.GELU(), nn.Dropout(dropout / 2),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, z):
        return self.net(z)


def train_mlp_head(vae, mlp, train_loader, val_loader, device,
                   num_epochs=30, lr=3e-4, unfreeze_encoder=True, patience=7):
    criterion = FocalLoss()
    params    = list(mlp.parameters())

    if unfreeze_encoder:
        # Unfreeze last conv block + projection heads of encoder
        tune = (list(vae.encoder.net[-6:].parameters()) +   # last 2 conv layers + batchnorm + activation
                list(vae.encoder.fc_mu.parameters()) +
                list(vae.encoder.fc_logvar.parameters()))
        params += tune
        print(f"  Unfreezing encoder last block ({sum(p.numel() for p in tune):,} params)")

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    history = {'train_loss': [], 'val_acc': [], 'val_auc': [], 'val_loss': []}
    best_auc, best_mlp_state, best_enc_state, no_imp = 0.0, None, None, 0

    for epoch in range(num_epochs):
        # train
        mlp.train()
        if unfreeze_encoder: vae.train()
        else:                vae.eval()

        t_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if unfreeze_encoder:
                mu, logvar = vae.encoder(imgs)
                z = vae.reparameterize(mu, logvar)
            else:
                with torch.no_grad():
                    _, mu, _, z = vae(imgs)

            loss = criterion(mlp(z), labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            t_loss += loss.item()

        scheduler.step()

        # validate
        vae.eval(); mlp.eval()
        preds, labs, probs, v_loss = [], [], [], 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                _, mu, _, z = vae(imgs)
                logits = mlp(z)
                v_loss += criterion(logits, labels).item()
                probs.extend(torch.softmax(logits, 1)[:, 1].cpu().numpy())
                preds.extend(logits.argmax(1).cpu().numpy())
                labs.extend(labels.cpu().numpy())

        acc = (np.array(preds) == np.array(labs)).mean()
        try:    auc = roc_auc_score(labs, probs)
        except: auc = 0.5

        history['train_loss'].append(t_loss / len(train_loader))
        history['val_loss'].append(v_loss / len(val_loader))
        history['val_acc'].append(acc)
        history['val_auc'].append(auc)

        print(f"  Ep {epoch+1:3d}/{num_epochs}  loss={t_loss/len(train_loader):.4f}"
              f"  val_acc={acc:.4f}  val_auc={auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_mlp_state = {k: v.clone() for k, v in mlp.state_dict().items()}
            best_enc_state = {k: v.clone() for k, v in vae.encoder.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"  Early stop at epoch {epoch+1}  best_auc={best_auc:.4f}")
                break

    if best_mlp_state:
        mlp.load_state_dict(best_mlp_state)
        if unfreeze_encoder:
            vae.encoder.load_state_dict(best_enc_state)

    print(f"  Best val AUC: {best_auc:.4f}")
    return history
