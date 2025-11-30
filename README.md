# GINet: Guided Image Network for Sentinel-2 Super-Resolution

[![arXiv](https://img.shields.io/badge/arXiv-2409.02675-B31B1B.svg)](https://arxiv.org/abs/2508.04729)

This repository contains the implementation and additional resources for the following paper:

**Super-Resolution of Sentinel-2 Images Using a Geometry-Guided Back-Projection Network with Self-Attention**  
*Ivan Pereira-Sánchez, Francesc Alcover, Bartomeu Garau, Daniel Torres, Julia Navarro, Catalina Sbert, Joan Duran*  
<!--Submmited to the International Journal of Computer Vision-->

---

## 📄 Abstract
The Sentinel-2 mission provides multispectral imagery with 13 bands at resolutions of 10m, 20m, and 60m. In particular, the 10m bands offer fine structural detail, while the 20m bands capture richer spectral information. In this paper, we propose a geometry-guided super-resolution model for fusing the 10m and 20m bands. Our approach introduces a cluster-based learning procedure to generate a geometry-rich guiding image from the 10m bands. This image is integrated into an unfolded back-projection architecture that leverages image self-similarities through a multi-head attention mechanism, which models nonlocal patch-based interactions across spatial and spectral dimensions. We also generate a dataset for evaluation, comprising three testing sets that include urban, rural, and coastal landscapes. Experimental results demonstrate that our method outperforms both classical and deep learning-based super-resolution and fusion techniques.

<!--
---

## 📚 arXiv Preprint

The paper is currently under revision, and the first preprint is available on [arXiv](https://arxiv.org/abs/2409.02675).


---
-->

## 🛠️ Environment

You can set up the development environment using either **Conda** or **pip**.

#### 📦 Option 1: Using Conda (`environment.yml`)

1. Create the environment:

   ```bash
   conda env create -f environment.yml
   ```

2. Activate the environment:

   ```bash
   conda activate GINet
   ```

---

#### 💡 Option 2: Using pip (`requirements.txt`)

1. (Optional) Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
---

## ⚙️ Setup

To begin, create an .env file in the project root directory and define the `DATASET_PATH` variable, pointing to the directory where your dataset is stored.

The DataModule is built specifically for Sentinel-2 satellite imagery. Functions to read the Sentinel-2 bands as well as a function to generate a PAN-like band are provided.

---
## 🚂 Train

Run the following command:
   ```bash
   python train.py 
   ```
---
## 🗃️ Dataset

The dataset used in our article can be downloaded from [Zenodo](https://zenodo.org/records/17735252).

---
## 🏗️ To-do's:
- Test script
- Upload definitve checkpoints

<!--
---
## 📌 Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{pansharpening2024,
  title={Multi-Head Attention Residual Unfolded Network for Model-Based Pansharpening},
  author={Pereira-S{\'a}nchez, Ivan and Sans, Eloi and Navarro, Julia and Duran, Joan},
  journal={arXiv preprint arXiv:2409.02675},
  year={2024}
}
```
-->
---
## Acknowledgements

This work was funded by MCIN/AEI/10.13039/501100011033/ and by the European Union NextGenerationEU/PRTR via the MaLiSat project TED2021-132644B-I00.
