# Model Card — SSL ResNet50 (Example)

**Date:** 2025-10-03  
**Task:** Self-supervised representation learning (SimCLR-style), optional linear probe  
**Backbone:** ResNet50 (Keras Applications, pooling='avg')  
**Input size:** 224×224 RGB  
**Augmentations:** RandomResizedCrop, HorizontalFlip, Rotation, Zoom/Contrast  
**Loss:** NT-Xent (InfoNCE), temperature=0.1  
**Optimizer:** Adam, lr=3e-4, batch=64, epochs=100  
**Seed:** 42

**Data:** Unlabeled images under `data/raw/` (not committed).  
**Outputs:** Figures/logs under `outputs/` (ignored).  
**Weights:** Not stored in Git. Upload to Drive/HF and link here when available.

**Results (placeholder):**
- Pretraining loss (final): —
- Linear probe (val): Acc — / F1 —

**Files:**
- `CONFIG.json` — hyperparameters
- `METRICS.json` — final numbers (fill after training)
- `checkpoints/` — many epoch files (ignored)
- `best.keras` — best consolidated model (ignored)
