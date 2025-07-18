# GINet: Guided Image Network for Sentinel-2 Super-Resolution

<!--[![arXiv](https://img.shields.io/badge/arXiv-2409.02675-B31B1B.svg)](https://arxiv.org/abs/2409.02675)-->

This repository contains the implementation and additional resources for the upcoming paper:

**Super-Resolution of Sentinel-2 Images Using a Geometry-Guided Back-Projection Network with Self-Attention**  
*Ivan Pereira-Sánchez, Francesc Alcover, Bartomeu Garau, Daniel Torres, Julia Navarro, Catalina Sbert, Joan Duran*  
<!--Submmited to the International Journal of Computer Vision-->

---

## 📄 Abstract
We propose a deep learning-based method for fusing Sentinel-2 (S2) hyperspectral (20m) and multispectral (10m) data. Our approach introduces a novel cluster-based procedure to generate guiding images that encode the geometric information from the 10m bands. These guiding images are used within a back-projection unfolding architecture that incorporates their spatial detail and exploits image self-similarities. The model is evaluated on three diverse test sets encompassing urban, rural, and coastal regions. Experimental results demonstrate that our method consistently outperforms both classical and deep learning-based super-resolution and fusion techniques, including those specifically designed for S2 data.

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
## Train

Run the following command:
   ```bash
   python train.py 
   ```
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
