
# VAE-Segmentation-Latent-Learning

This repository implements a **Variational Autoencoder (VAE)–based segmentation framework** that learns a compact **latent representation** of input data and uses it to guide segmentation.  
The core idea is to leverage latent structure learning to improve segmentation of **sparse, thin, or topologically constrained objects**, where purely pixel-wise methods often fail.

The model follows an **Encoder–Distribution–Decoder (E-DD)** paradigm, where the latent distribution captures global structure and regularizes the segmentation output.

---

## ✨ Motivation

Standard segmentation networks (e.g., U-Net) rely heavily on local pixel information.  
However, in many domains (medical imaging, scientific imaging, thin structures):

- Target structures are **sparse**
- Connectivity and topology matter
- Supervision may be limited or noisy

By introducing a **VAE latent space**, this framework:
- Encourages **global structural consistency**
- Regularizes segmentation via probabilistic latent variables
- Enables analysis and control of learned shape representations

---

## 🧠 Method Overview

The model consists of:

1. **Encoder**
   - Maps the input image to a latent distribution  
    qφ(z | x) = 𝒩(μ, σ²)


2. **Latent Space**
   - Enforces smoothness and structure via KL divergence
   - Can be sampled, interpolated, or constrained

3. **Decoder / Segmentation Head**
   - Reconstructs segmentation masks from latent variables
   - Optionally conditioned on encoder skip connections

The training objective typically combines:
- Reconstruction or segmentation loss
- KL divergence regularization
- Optional topology-aware or Dice-based losses

---

## 📂 Repository Structure

```

.
├── checkpoints/          # Saved model weights
├── **pycache**/          # Python cache files
├── components.py         # Encoder / decoder building blocks
├── data_loader.py        # Dataset loading & preprocessing
├── losses.py             # Loss function definitions
├── model.py              # VAE segmentation model
├── train.py              # Training script
├── testing.py            # Evaluation / inference script
└── README.md

````

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/gregarious19/VAE-segmentation-latent-learning.git
cd VAE-segmentation-latent-learning
````

### 2. Install dependencies

```bash
pip install torch torchvision numpy tqdm scipy nibabel
```

> **Note:**
> Adjust dependencies depending on your dataset format (e.g., medical images, NumPy arrays).

---

## 📊 Dataset Format

The dataset loader is defined in `data_loader.py`.
You are expected to provide:

* Input images (2D or 3D)
* Corresponding segmentation masks (if supervised)
* Train / validation / test splits

Typical assumptions:

* Images and masks are spatially aligned
* Data is normalized before training
* Masks are binary or multi-class

Modify `data_loader.py` to suit your dataset.

---

## 🚀 Training

Run training using:

```bash
python train.py
```

Typical configurable parameters include:

* Batch size
* Learning rate
* Latent dimensionality
* Number of epochs
* Loss weights (reconstruction vs KL)

Check `train.py` for available arguments and defaults.

---

## 🧪 Testing / Inference

To evaluate a trained model:

```bash
python testing.py \
  --model_path checkpoints/model.pth \
  --data_dir /path/to/test/data
```

The testing script supports:

* Forward inference
* Metric computation (if ground truth is available)
* Saving predicted segmentation masks

---

## 📐 Loss Functions

Defined in `losses.py`.
Common components include:

* **Reconstruction / Segmentation Loss**

  * Binary Cross-Entropy
  * Dice Loss (for sparse structures)

* **KL Divergence**

  * Encourages latent space regularization
  * Prevents overfitting

The total loss is typically:

Loss function:

L = L_seg + β · L_KL



where ( \beta ) controls latent regularization strength.

---

## 🔬 Latent Space Analysis

Because the model is probabilistic:

* Latent variables can be **sampled**
* Interpolations in latent space can reveal learned structure
* Latent dimensionality controls expressiveness vs stability

This makes the framework suitable for:

* Shape analysis
* Uncertainty-aware segmentation
* Downstream geometric or physical modeling

---

## 🧩 Extending the Model

Possible extensions include:

* Topology-preserving losses
* PDE-based constraints on decoded shapes
* Conditional VAEs
* Multi-scale latent representations
* 3D volumetric segmentation

The modular design of `components.py` and `model.py` supports easy experimentation.

---

## 📚 References

* Kingma, D. P., & Welling, M. (2013). *Auto-Encoding Variational Bayes*
* Probabilistic latent variable models for structured prediction
* Latent-space regularization for medical image segmentation

---

## 🧑‍💻 Author

**Pranay Sharma**
PhD Scholar, BITS Pilani – Hyderabad Campus
Research interests: Scientific AI, inverse problems, geometry-aware learning

---

## 📄 License

This project is released under the **MIT License**.
You are free to use, modify, and distribute this code with attribution.

---

## ⭐ Acknowledgements

This project draws inspiration from:

* VAE literature
* U-Net-style segmentation architectures
* Research on latent structure learning for inverse problems

---

