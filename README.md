# TV-STFN: Tetra-View Spatio-Temporal Fusion Network

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Overview

TV-STFN is a deep learning framework for predicting the membrane permeability of cyclic peptides. It employs a **Tetra-View** multi-modal architecture with four parallel encoder branches:

1. **Dynamic Conformer Encoder** — SchNet-inspired 3D spatial encoding of molecular conformations
2. **Semantic Encoder** — Sequence-based representation learning
3. **Graph Topology Encoder** — 2D molecular graph structure encoding via GCN/GAT
4. **Physicochemical Encoder** — Hand-crafted molecular descriptors

The model integrates **Multi-Instance Learning (MIL)** with attention mechanisms to aggregate information across dynamic conformational ensembles, addressing the challenge of conformational flexibility in cyclic peptides.

## Project Structure

| File | Description |
|------|-------------|
| `model.py` | `TetraViewNet` architecture with residual MLP fusion blocks |
| `encoders.py` | Four encoder branches (DynamicConformer, Semantic, GraphTopology, Physicochemical) |
| `dataset.py` | Data loading, 3D conformer generation, and feature extraction |
| `loss.py` | `CompositeLoss` combining MSE, ranking loss, and classification objectives |
| `train.py` | Main training script with early stopping and evaluation metrics |
| `predict.py` | Standalone prediction on new molecules |
| `preprocess_fast.py` | Fast data preprocessing pipeline |
| `preprocess_smart.py` | Smart preprocessing with caching |
| `preprocess_tetraview.py` | Full TetraView data preparation |
| `requirements.txt` | Python dependencies |

## Installation

```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- PyTorch 2.6.0
- PyTorch Geometric
- RDKit
- pandas, NumPy, SciPy, scikit-learn

## Dataset

The model uses the **CycPeptMPDB** dataset. Place `CycPeptMPDB_Peptide_PAMPA.csv` in the project root directory.

## Usage

### Training

```bash
python train.py
```

This will preprocess the data, train for 100 epochs with early stopping, and save the best model to `best_tetraview_model.pth`.

### Prediction

```bash
python predict.py
```

### Hyperparameter Search

```bash
python auto_search_tvstfn_5fold.py
```

## Model Weights

Pre-trained model weights (`.pth` files) are not included due to size constraints. Generate them by running the training script.

## License

MIT License.
