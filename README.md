# TV-STFN: Tetra-View Spatio-Temporal Fusion Network

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20733474.svg)](https://doi.org/10.5281/zenodo.20733474)

## Overview

TV-STFN is a deep learning framework for predicting the membrane permeability of cyclic peptides. It employs a **Tetra-View** multi-modal architecture with four parallel encoder branches:

1. **Dynamic Conformer Encoder** — SchNet-inspired 3D spatial encoding of molecular conformations
2. **Semantic Encoder** — Sequence-based representation learning
3. **Graph Topology Encoder** — 2D molecular graph structure encoding via GCN/GAT
4. **Physicochemical Encoder** — Hand-crafted molecular descriptors

The model integrates **Multi-Instance Learning (MIL)** with attention mechanisms to aggregate information across dynamic conformational ensembles, addressing the challenge of conformational flexibility in cyclic peptides.

## Project Structure

```
TV-STFN/
├── src/                    # Core model
│   ├── model.py            # TetraViewNet architecture
│   ├── encoders.py         # Four encoder branches
│   ├── loss.py             # CompositeLoss
│   └── dataset.py          # Data loading & feature extraction
├── scripts/                # Run scripts
│   ├── train.py            # Training
│   ├── predict.py          # Prediction
│   └── preprocess.py       # Data preprocessing
├── data/                   # Dataset placeholder
├── requirements.txt        # pip dependencies
├── environment.yml         # Conda environment
└── README.md
```

## Installation

**Using Conda:**
```bash
conda env create -f environment.yml
conda activate tv-stfn
```

**Using pip:**
```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- Python 3.10+
- PyTorch 2.0+, PyTorch Geometric
- RDKit, NumPy, pandas, scikit-learn

## Dataset

The model uses the **CycPeptMPDB** dataset. Place `CycPeptMPDB_Peptide_PAMPA.csv` in the `data/` directory.

> Li J., Yanagisawa K., Sugita M., Fujie T., Ohue M., and Akiyama Y. CycPeptMPDB: A Comprehensive Database of Membrane Permeability of Cyclic Peptides. *J. Chem. Inf. Model.*, 2023, 63(7): 2240–2250. [DOI: 10.1021/acs.jcim.2c01573](https://doi.org/10.1021/acs.jcim.2c01573)

## Usage

### Training

```bash
python scripts/train.py
```

### Prediction

```bash
python scripts/predict.py
```

### Preprocessing

```bash
python scripts/preprocess.py
```

## Data and Software Availability

The dataset supporting the conclusions of this article is available in the CycPeptMPDB repository ([DOI: 10.1021/acs.jcim.3c00110](https://doi.org/10.1021/acs.jcim.3c00110)).

- **Project name:** TV-STFN
- **Project home page:** [https://github.com/Max-0715/TV-STFN](https://github.com/Max-0715/TV-STFN)
- **Archived version:** [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20733474.svg)](https://doi.org/10.5281/zenodo.20733474)
- **Operating system(s):** Platform independent
- **Programming language:** Python
- **Other requirements:** RDKit, PyTorch, PyTorch Geometric
- **License:** MIT
- **Any restrictions to use by non-academics:** None

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
