# HOCFormer: Transformer for Higher-Order Co-Fluctuation Brain Networks Classification for Alzheimer's Disease

This is my M.S. thesis project at UNC Greensboro, advised by Dr. Minjeong Kim. The core idea is simple but something that bothered me as I read through the existing literature: almost every machine learning model for Alzheimer's detection from fMRI treats the brain as a collection of pairwise connections. Region A correlates with Region B. That's it. But the brain doesn't actually work that way — regions coordinate in groups of three, four, more — and those group-level dynamics carry information that pairwise models simply can't see.

This project builds a framework called HOCFormer that progressively adds higher-order signals on top of pairwise connectivity and measures how much each one helps.

---

## The Problem

Alzheimer's disease starts damaging the brain years before any symptoms appear. Resting-state fMRI can detect early changes in how brain regions communicate, but standard functional connectivity only looks at pairs of regions. If three regions that normally fire together start decoupling, that won't necessarily show up in any individual pairwise correlation — but it will show up in a three-way co-fluctuation signal.

---

## What HOCFormer Does

It builds a four-level signal hierarchy from resting-state fMRI data:

**Node** → pairwise Pearson functional connectivity (116×116 FC matrix per subject)

**Edge** → instantaneous pairwise co-fluctuation timeseries: `eTS(i,j,t) = z_i(t) × z_j(t)`

**Triangle** → three-way co-fluctuation: `triTS(i,j,k,t) = z_i(t) × z_j(t) × z_k(t)`

**Scaffold** → homological scaffold via persistent homology (GUDHI) — identifies which edges are topologically central to the brain's co-fluctuation loop structure

Each signal type is compressed using RSS binning into 5 Frame Co-Fluctuation (FCC) matrices, then all of them are fed into a Transformer encoder as tokens. The model learns cross-order and cross-bin dependencies through self-attention, with a [CLS] token used for final classification.

The ablation is the point — we add one signal level at a time and measure what changes:

```
Node → Node+Edge → Node+Edge+Triangle → Node+Edge+Triangle+Scaffold
```

---

## Dataset

- **Source:** ADNI (Alzheimer's Disease Neuroimaging Initiative)
- **Subjects:** 250 (Cognitively Normal vs. Alzheimer's Disease, binary classification)
- **Atlas:** AAL-116 (116 brain regions)
- **Timepoints:** 140 per subject
- **Validation:** 10-fold stratified cross-validation

---

## Results

| Configuration | Accuracy | Precision | F1 | AUC-ROC |
|--------------|----------|-----------|-----|---------|
| Node+Edge | 80.14% ± 8.45% | 0.7111 | 0.7458 | 0.8027 |
| Node+Edge+Triangle | 80.19% ± 8.88% | 0.7364 | 0.7290 | 0.7982 |
| Node+Edge+Triangle+Scaffold | **82.19% ± 4.61%** | **0.7956** | 0.7363 | **0.8056** |

The scaffold signal produces the biggest jump — not just in accuracy but in consistency. The standard deviation drops from 0.0845 to 0.0461, meaning the model isn't just getting lucky on certain folds; it's finding patterns that generalize across different patient subsets.

The triangle signal alone adds modest improvement, which makes sense: with 250 subjects and ~250,000 possible triplets to compress, the model doesn't have enough data to reliably extract everything from that space. The scaffold fixes this by using persistent homology to distill the triangle space down to its topologically essential components — edges that participate in the most co-fluctuation loops.

---

## Architecture

- **Token:** each (signal type, RSS bin) pair → one token
- **Projection:** LayerNorm + Linear (input_dim → 256)
- **Embeddings:** learnable region embeddings (positional)
- **Transformer:** 3 pre-norm layers, 4 attention heads, GELU activation, dropout 0.5
- **Classification head:** LayerNorm → Dropout → Linear(256→128) → GELU → Dropout → Linear(128→2)
- **Training:** Adam, lr=1e-4, weight decay=5e-4, ReduceLROnPlateau, early stopping (patience=20)

---

## Code Structure

```
├── dataset.py     # ADNI data loading, graph construction, all signal computation
├── models.py      # AblationTransformer — 4 modes (node / node_edge / node_edge_tri / node_edge_tri_sca)
├── train.py       # 10-fold CV training loop, logging, ablation summary
└── main.py        # Entry point
```

---

## How to Run

```bash
git clone https://github.com/ravyavangaveti/Transformer-for-Higher-Order-Co-Fluctuation-Brain-Networks-Classification-for-Alzheimer-s-Disease.git
cd Transformer-for-Higher-Order-Co-Fluctuation-Brain-Networks-Classification-for-Alzheimer-s-Disease

pip install torch torch-geometric gudhi scikit-learn pandas numpy scipy

python main.py --num_classes 2 --hidden_dim 256 --n_layers 3 --dropout 0.5 --epochs 100 --batch_size 16 --learning_rate 1e-4
```

Update `DATA_DIR` and `LABEL_FILE` in `train.py` to point to your ADNI data directory.

---

## Requirements

- Python 3.8+
- PyTorch
- PyTorch Geometric
- GUDHI (persistent homology)
- scikit-learn
- pandas, numpy, scipy

---

## Key Takeaway

Simpler isn't always better, but focused is. The scaffold signal works so well on a small dataset because persistent homology does the heavy lifting — it throws away 93% of the triangle space and keeps only what's topologically meaningful. That's the kind of signal a model can actually learn from with 250 subjects.

The broader lesson from this project: the brain coordinates in groups, and our models should reflect that. Pairwise connectivity is a strong baseline, but it leaves real diagnostic information on the table.

---

**Ravya Vangaveti**  
M.S. Computer Science — Data Science & Machine Learning  
University of North Carolina at Greensboro  
Advisor: Dr. Minjeong Kim  
[GitHub](https://github.com/ravyavangaveti)
