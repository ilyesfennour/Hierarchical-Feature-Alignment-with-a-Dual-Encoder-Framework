# Hierarchical Feature Alignment with a Dual-Encoder Framework for Unified Image Restoration, Classification, Visualization, and Clustering

Official implementation of the experiments presented in:

> **Multi-task Autoencoders for Image Denoising, Enhancement, Clustering, Classification, and Data Recovery**

This repository provides the training and evaluation code for a family of autoencoder-based models designed to address multiple image-processing tasks within a unified framework.

The implementation includes:

* Image denoising
* Image super-resolution
* Image inpainting / missing-data recovery
* Image classification
* Image clustering
* Latent-space representation learning
* Transfer/alignment of semantic information through the latent representation

---

## 📌 Overview

The proposed framework investigates autoencoder architectures for learning useful image representations while simultaneously supporting different downstream tasks.

The repository contains implementations of three main models:

* **DFA** — baseline dual-feature autoencoder
* **LAE** — latent autoencoder
* **DLAE** — dual latent autoencoder with an additional prediction/classification component

The models can be trained with different datasets and image degradation types.

### Supported degradation types

| Degradation      | Description                                       |
| ---------------- | ------------------------------------------------- |
| Gaussian noise   | Additive Gaussian noise                           |
| Super-resolution | Downsampling followed by reconstruction           |
| Missing data     | Removal of image information followed by recovery |

### Supported datasets

The current implementation includes loaders for:

* MNIST
* CIFAR-10
* SVHN
* CelebA
* ImageNet
* MedMNIST
* DFB-HR

The DFB-HR dataset is additionally supported through a patch-based data-loading pipeline for high-resolution image restoration experiments.

---

## 🏗️ Repository Structure

```text
.
├── README.md
├── requirements.txt
├── LICENSE
│
├── main.py
├── model.py
├── data.py
├── degradation.py
├── get_data_loaders.py
├── get_patched_data_loaders.py
├── training.py
├── training_utilities.py
├── checkpoint_utils.py
│
└── notebooks/
    └── training.ipynb
```

### Main files

| File                          | Description                                               |
| ----------------------------- | --------------------------------------------------------- |
| `main.py`                     | Main entry point and experiment configuration             |
| `model.py`                    | Definitions of DFA, LAE, DLAE, and model components       |
| `data.py`                     | Dataset loading and dataset-specific preprocessing        |
| `degradation.py`              | Image degradation functions                               |
| `get_data_loaders.py`         | Construction of standard training/validation/test loaders |
| `get_patched_data_loaders.py` | Patch-based data loading for DFB-HR                       |
| `training.py`                 | Reconstruction and classification training procedures     |
| `training_utilities.py`       | Loss functions and training utilities                     |
| `checkpoint_utils.py`         | Checkpoint loading, and training continuation             |
| `notebooks/training.ipynb`    | Lightweight notebook interface for running experiments    |

---

# 🚀 Installation

## 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

## 2. Create a virtual environment

We recommend using a dedicated Python environment.

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

For GPU training, install the appropriate PyTorch version for your CUDA configuration if necessary.

---

# 📂 Dataset Preparation

The repository does **not** include the datasets.

**No manual download needed:** MNIST, MedMNIST, CIFAR-10, and SVHN are downloaded automatically by their loaders.

**Manual download (from Kaggle):**

* **CelebA** — [`jessicali9530/celeba-dataset`](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)
* **DFB-HR** — combine three sources into one `DFB-HR/` folder:
  * DIV2K: [`soumikrakshit/div2k-high-resolution-images`](https://www.kaggle.com/datasets/soumikrakshit/div2k-high-resolution-images)
  * Flickr2K: [`daehoyang/flickr2k`](https://www.kaggle.com/datasets/daehoyang/flickr2k)
  * BSD500: [`balraj98/berkeley-segmentation-dataset-500-bsds500`](https://www.kaggle.com/datasets/balraj98/berkeley-segmentation-dataset-500-bsds500)
* **ImageNet** — training/validation: merge [`sautkin/imagenet1k0`](https://www.kaggle.com/datasets/sautkin/imagenet1k0), [`sautkin/imagenet1k1`](https://www.kaggle.com/datasets/sautkin/imagenet1k1), [`sautkin/imagenet1k2`](https://www.kaggle.com/datasets/sautkin/imagenet1k2), [`sautkin/imagenet1k3`](https://www.kaggle.com/datasets/sautkin/imagenet1k3) into `train/`. Testing: [`sautkin/imagenet1kvalid`](https://www.kaggle.com/datasets/sautkin/imagenet1kvalid) into `test/` (sibling of `train/`).

Expected layout:

```text
data/
├── CelebA/
├── DFB-HR/
└── ImageNet/
    ├── train/
    └── test/
```

---

# 💾 Checkpoints

Checkpoint management is implemented in:

```text
checkpoint_utils.py
```

The training framework supports continuing training from an existing checkpoint and loading pretrained models.

---

# 📓 Notebook

A lightweight Jupyter notebook is provided in:

```text
notebooks/training.ipynb
```

The notebook is intended as a convenient interface for launching experiments rather than containing the complete implementation.

The core implementation remains in the Python modules so that experiments can be reproduced from the command line and the code can be reused programmatically.

---

# 📝 Citation

If you use this code or the proposed models in your research, please cite the associated paper:

The citation information will be updated once the paper is published.

---

# 📜 License

This project is released under the license specified in:

---

# 🙏 Acknowledgements

We acknowledge the open-source datasets, libraries, and research projects that made this work possible.

In particular, this project makes use of the Python scientific-computing and deep-learning ecosystem, including PyTorch, torchvision, scikit-learn, and related libraries.

---

# 📬 Contact

For questions regarding the implementation or the research, please open an issue in this repository or contact the authors through their institutional/research profiles.
