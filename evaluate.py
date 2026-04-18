import os, argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, roc_curve, classification_report,
    adjusted_rand_score, normalized_mutual_info_score, silhouette_score)
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from dataset import get_dataloaders
from vae_model import VAE
from mlp_head import MLPClassifier, train_mlp_head

try:
    import umap
    HAS_UMAP = True
except:
    HAS_UMAP = False

try:
    from skfuzzy.cluster import cmeans as fuzzy_cmeans
    HAS_FUZZY = True
except:
    HAS_FUZZY = False

def extract_all(model, loader, device, max_n=None):
    model.eval()
    mus, labs, scores, imgs16, recons16 = [], [], [], [], []
    n = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            recon, mu, logvar, z = model(x)
            err   = F.mse_loss(recon, x, reduction="none").mean(dim=[1,2,3])
            mahal = ((mu - model.latent_mean)**2 / (model.latent_var + 1e-8)).mean(1)
            score = err + 0.3 * mahal
            mus.append(mu.cpu().numpy())
            labs.append(y.numpy())
            scores.append(score.cpu().numpy())
            if n < 16:
                imgs16.append(x[:4].cpu())
                recons16.append(recon[:4].cpu())
            n += x.size(0)
            if max_n and n >= max_n:
                break
    return (np.concatenate(mus), np.concatenate(labs),
            np.concatenate(scores),
            torch.cat(imgs16)[:16], torch.cat(recons16)[:16])

def eval_anomaly(scores, labels, save_dir):
    auc = roc_auc_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, lw=2, label="AUC=" + str(round(auc,3)))
    plt.plot([0,1],[0,1],"k--", lw=1)
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC Curve - Anomaly Detection")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "roc_curve.png"), dpi=150)
    plt.close()
    print("[1] Anomaly AUC-ROC: " + str(round(auc,4)))
    return auc

def eval_umap(mu, labels, scores, save_dir):
    if not HAS_UMAP:
        print("[2] UMAP skipped")
        return
    print("[2] Running UMAP...")
    X = StandardScaler().fit_transform(mu)
    emb = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1).fit_transform(X)
    fig, axes = plt.subplots(1, 2, figsize=(14,6))
    for cls, col, name in zip([0,1], ["steelblue","crimson"], ["Healthy","Diseased"]):
        m = labels == cls
        axes[0].scatter(emb[m,0], emb[m,1], c=col, s=3, alpha=0.35, label=name+" n="+str(m.sum()))
    axes[0].set_title("UMAP - by True Label")
    axes[0].legend(markerscale=4)
    sc = axes[1].scatter(emb[:,0], emb[:,1], c=scores, cmap="plasma", s=3, alpha=0.4)
    plt.colorbar(sc, ax=axes[1], label="Anomaly Score")
    axes[1].set_title("UMAP - by Anomaly Score")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "umap_latent.png"), dpi=150)
    plt.close()
    print("UMAP saved.")

def eval_clustering(mu, labels, save_dir):
    X = StandardScaler().fit_transform(mu)
    km = KMeans(n_clusters=2, random_state=42, n_init=20).fit_predict(X)
    sil = silhouette_score(X, km, sample_size=min(5000, len(X)))
    ari = adjusted_rand_score(labels, km)
    nmi = normalized_mutual_info_score(labels, km)
    print("[3] K-Means: silhouette=" + str(round(sil,4)) + "  ARI=" + str(round(ari,4)) + "  NMI=" + str(round(nmi,4)))
    results = {"kmeans": {"sil": sil, "ari": ari, "nmi": nmi}}
    if HAS_FUZZY:
        _, u, _, _, _, _, fpc = fuzzy_cmeans(X.T, c=2, m=2, error=0.005, maxiter=500)
        fp = np.argmax(u, axis=0)
        sil2 = silhouette_score(X, fp, sample_size=min(5000, len(X)))
        ari2 = adjusted_rand_score(labels, fp)
        print("Fuzzy CM: silhouette=" + str(round(sil2,4)) + "  ARI=" + str(round(ari2,4)) + "  FPC=" + str(round(fpc,4)))
        results["fuzzy"] = {"sil": sil2, "ari": ari2, "fpc": fpc}
    return results

def eval_heatmaps(imgs, recons, labels_list, save_dir, n=8):
    n = min(n, imgs.size(0))
    def denorm(t): return (t * 0.5 + 0.5).clamp(0, 1)
    fig, axes = plt.subplots(3, n, figsize=(3*n, 9))
    for i in range(n):
        orig = denorm(imgs[i,0]).numpy()
        rec  = denorm(recons[i,0]).numpy()
        diff = np.abs(orig - rec)
        lbl  = "Diseased" if labels_list[i] == 1 else "Healthy"
        axes[0,i].imshow(orig, cmap="gray"); axes[0,i].set_title(lbl, fontsize=8)
        axes[1,i].imshow(rec,  cmap="gray")
        axes[2,i].imshow(diff, cmap="hot")
        for row in range(3): axes[row,i].axis("off")
    plt.suptitle("Anomaly Heatmaps", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "anomaly_heatmaps.png"), dpi=150)
    plt.close()
    print("[4] Heatmaps saved.")

def eval_mlp(vae, train_loader, val_loader, test_loader, device, save_dir, latent_dim, epochs):
    print("[5] Training MLP head...")
    mlp = MLPClassifier(latent_dim=latent_dim).to(device)
    hist = train_mlp_head(vae, mlp, train_loader, val_loader, device, num_epochs=epochs)
    vae.eval(); mlp.eval()
    preds, labs, probs = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            _, mu, _, z = vae(x)
            logits = mlp(z)
            probs.extend(torch.softmax(logits, 1)[:,1].cpu().numpy())
            preds.extend(logits.argmax(1).cpu().numpy())
            labs.extend(y.numpy())
    auc = roc_auc_score(labs, probs)
    acc = (np.array(preds) == np.array(labs)).mean()
    print("Test Report:")
    print(classification_report(labs, preds, target_names=["Healthy","Diseased"]))
    print("MLP AUC-ROC : " + str(round(auc,4)))
    print("MLP Accuracy: " + str(round(acc,4)))
    fig, axes = plt.subplots(1, 3, figsize=(15,4))
    axes[0].plot(hist["train_loss"], label="Train")
    axes[0].plot(hist["val_loss"],   label="Val")
    axes[0].set_title("Loss"); axes[0].legend()
    axes[1].plot(hist["val_acc"],  color="green");  axes[1].set_title("Val Accuracy")
    axes[2].plot(hist["val_auc"],  color="purple"); axes[2].set_title("Val AUC")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "mlp_training.png"), dpi=150)
    plt.close()
    torch.save(mlp.state_dict(), os.path.join(save_dir, "best_mlp.pth"))
    return auc, acc

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    train_loader, val_loader, test_loader = get_dataloaders(
        args.data_dir, args.labels_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_per_class=args.max_per_class)
    ckpt = torch.load(args.vae_checkpoint, map_location=device)
    state = ckpt.get("model_state", ckpt)
    has_perc = any("perceptual" in k for k in state.keys())
    model = VAE(latent_dim=args.latent_dim, beta=args.beta, use_perceptual=has_perc).to(device)
    model.load_state_dict(state)
    if "latent_mean" in ckpt:
        model.latent_mean = ckpt["latent_mean"].to(device)
        model.latent_var  = ckpt["latent_var"].to(device)
    model.eval()
    print("Loaded VAE from " + args.vae_checkpoint)
    print("Extracting latents...")
    mu, labels, scores, imgs, recons = extract_all(model, test_loader, device, max_n=args.max_eval_samples)
    print("Samples: " + str(len(labels)) + "  healthy=" + str((labels==0).sum()) + "  diseased=" + str((labels==1).sum()))
    auc_recon        = eval_anomaly(scores, labels, args.save_dir)
    eval_umap(mu, labels, scores, args.save_dir)
    cluster_res      = eval_clustering(mu, labels, args.save_dir)
    eval_heatmaps(imgs, recons, labels[:16].tolist(), args.save_dir)
    auc_mlp, acc_mlp = eval_mlp(model, train_loader, val_loader, test_loader,
                                 device, args.save_dir, args.latent_dim, args.mlp_epochs)
    print("="*50)
    print("FINAL RESULTS")
    print("="*50)
    print("Anomaly Detection AUC : " + str(round(auc_recon, 4)))
    print("MLP Head AUC          : " + str(round(auc_mlp, 4)))
    print("MLP Head Accuracy     : " + str(round(acc_mlp, 4)))
    print("K-Means ARI           : " + str(round(cluster_res["kmeans"]["ari"], 4)))
    print("K-Means NMI           : " + str(round(cluster_res["kmeans"]["nmi"], 4)))
    if "fuzzy" in cluster_res:
        print("Fuzzy C-Means ARI     : " + str(round(cluster_res["fuzzy"]["ari"], 4)))
    print("="*50)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",         required=True)
    p.add_argument("--labels_csv",       required=True)
    p.add_argument("--vae_checkpoint",   required=True)
    p.add_argument("--save_dir",         default="results")
    p.add_argument("--latent_dim",       type=int,   default=256)
    p.add_argument("--beta",             type=float, default=0.1)
    p.add_argument("--batch_size",       type=int,   default=32)
    p.add_argument("--mlp_epochs",       type=int,   default=30)
    p.add_argument("--num_workers",      type=int,   default=2)
    p.add_argument("--max_eval_samples", type=int,   default=None)
    p.add_argument("--max_per_class",    type=int,   default=None)
    args = p.parse_args()
    main(args)
