# Self-Supervised Representation Learning (Images)

Unsupervised pretraining (SimCLR/InfoNCE-style augmentations + contrastive loss) on images, followed by an optional **linear probe** (small classifier on frozen features). The main work lives in a single, cleaned notebook, so the repo stays light and easy to browse.

## Highlights
- **Task:** Learn image representations without labels; optional linear probe for evaluation  
- **Approach:** Strong augmentations → contrastive pretraining → (optional) linear eval  
- **Notebook:** `notebooks/Self_Supervised_Learning_(1).ipynb` (outputs cleared)  
- **Status:** Portfolio-ready structure; datasets & weights are not committed

## Project Structure
```text
.
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ notebooks/
│  └─ Self_Supervised_Learning_(1).ipynb   # main notebook (no outputs saved)
├─ src/
│  ├─ __init__.py
│  ├─ utils.py         # paths, seeding, JSON helpers, timestamp
│  ├─ dataio.py        # tf.data loaders for unlabeled/labeled images
│  ├─ ssl_views.py     # Keras preprocessing pipeline + paired views
│  ├─ models.py        # encoder (ResNet/EfficientNet), projection head, linear probe
│  └─ losses.py        # NT-Xent (InfoNCE) loss in TF
├─ data/
│  ├─ raw/             # place images here (not committed)
│  └─ processed/       # cached features, splits (not committed)
├─ models/             # experiment folders; checkpoints ignored by Git
└─ outputs/
   ├─ figures/         # small curated visuals (optional)
   ├─ metrics/         # tiny JSON/CSV summaries (optional)
   ├─ logs/            # short text logs (optional)
   ├─ reports/         # short md/pdf summaries (optional)
   └─ runs/            # per-experiment bundles (optional)
Note: data/, models/, and outputs/ are ignored by Git; only .gitkeep files are tracked. Keep big files on Drive/local, not in the repo.
```

## Setup

python -m venv .venv

# Windows: .venv\Scripts\activate

# macOS/Linux: bsource .venv/bin/activate

pip install -r requirements.txt

## Data

Images are not committed to this repo.

Unlabeled SSL (recommended):

data/raw/images/*.jpg

Labeled (optional, for linear probe):

data/raw/<class_name>/*.jpg

Colab users: mount Drive and point DATA_ROOT to your Drive folder:

from google.colab import drive

drive.mount('/content/drive')

DATA_ROOT = "/content/drive/MyDrive/Datasets/SSL_Images/images"  # adjust to your path

## How to Use

1. Open notebooks/Self_Supervised_Learning_(1).ipynb.

2. Set DATA_ROOT at the top (Drive or local).

3. Run cells as needed (repo can remain output-free).

4. (Optional) Move reusable code into src/ and import it in the notebook.

## src/ Layout & Quick Imports

```text
src/
├─ __init__.py
├─ utils.py        # paths, seeding, JSON helpers, timestamp
├─ dataio.py       # tf.data loaders for unlabeled/labeled images
├─ ssl_views.py    # Keras preprocessing pipeline + paired views
├─ models.py       # encoder (ResNet/EfficientNet), projection head, linear probe
└─ losses.py       # NT-Xent (InfoNCE) loss

```

from src.utils import project_paths, set_seed

from src.dataio import dataset_unlabeled, dataset_from_directory

from src.ssl_views import make_ssl_augmenter, pair_views

from src.models import build_encoder, build_projection_head, build_linear_probe

from src.losses import ntxent_loss

## Outputs

This repo keeps the notebook output-free for clean diffs. Curated artifacts (small images/JSON) can live under outputs/.


```text
outputs/
├─ figures/   # e.g., ssl_fig_01.png, ssl_fig_02.png
├─ metrics/   # e.g., ssl_placeholder.json (or your real summary)
├─ logs/      # small .txt logs
├─ reports/   # short md/pdf summaries
└─ runs/
```

## Models

Trained checkpoints are stored under models/ but are not committed to Git.

Typical experiment folder contents:

checkpoints/ (ignored by Git)

best.keras / saved_model/ (ignored)

Small text files you can commit: CONFIG.json, METRICS.json, model_card.md

If you host weights elsewhere (Drive / Hugging Face), add the link in the corresponding model_card.md.

