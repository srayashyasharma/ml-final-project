# Lung X-Ray Anomaly Detection Using VAE

Author: Annapurna Srayashya Iruku (U01150990)

## Project
Unsupervised anomaly detection for chest X-rays using Variational Autoencoder (VAE).
Dataset: NIH Chest X-Ray14 (112,120 images, 14 disease labels)

## Files
- vae_model.py - VAE architecture (encoder, decoder, SSIM loss, perceptual loss)
- dataset.py - NIH dataset loader with patient-level split
- train_vae.py - Training script
- mlp_head.py - MLP classifier on latent space
- evaluate.py - Full evaluation pipeline
- results/ - All result plots

