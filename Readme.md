# SafeRoPE

**Risk-specific Head-wise Embedding Rotation for Safe Generation in Rectified Flow Transformers**
 Official Implementation (FLUX/MMDiT-based)

*SafeRoPE: Risk-specific Head-wise Embedding Rotation for Safe Generation in Rectified Flow Transformers*

------

## 📘 Overview

SafeRoPE is a **training-efficient safety alignment framework** designed for modern **Rectified Flow Transformers** such as **FLUX**.
 Instead of retraining the generative model, SafeRoPE identifies **unsafe semantic directions** inside **head-wise Q/K activations** and applies a **low-rank, orthogonal rotation** within **RoPE positional space** to suppress unsafe semantics **without degrading fidelity**.

------

## 🧩 Key Idea

1. Unsafe semantics collection in **low-dimensional subspaces** of specific **attention heads** inside FLUX/MMDiT.

2. For each safety-critical head, SafeRoPE extracts the **head-wise unsafe subspace** using:

   - unsafe Q/K activations
   - SVD to obtain the **principal unsafe directions**

3. SafeRoPE trains a **small skew-symmetric matrix**

   $A_{b,h} \in \mathbb{R}^{r \times r}$

   whose exponential map gives a **rotation on unsafe directions only**.

4. During inference, SafeRoPE rotates only the unsafe components of Q/K, leaving safe semantics intact.

------

## 📂 Repository Structure

```
saferope/
│
├── README.md
├── environment.yaml         # Reproducible Conda environment
│
├── extract_unsafe_subspace.py
│       # Step 1: Collect unsafe Q/K vectors for each head
│       # Step 2: Perform SVD to obtain head-wise unsafe subspaces U_{b,h}
│
├── train_rotary_adapter.py
│       # Step 3: Train skew-symmetric matrices for risk-aware rotations
│
├── inference_saferope.py
│       # Step 4: Inference script with SafeRoPE-enabled FLUX pipeline
├── demo_saferope_inference.ipynb
│       # Visual demonstration: before/after SafeRoPE, unsafe prompt suppression
```

------

## 🚀 Installation

### 1. Create Conda environment

```
conda env create -f environment.yaml
conda activate saferope
```

------

## 🛠 Pipeline Usage

SafeRoPE consists of three major stages:

------

### **Step 1 — Extract Unsafe Subspace**

Run:

```
python extract_unsafe_subspace.py
```

This script:

- Collects Q/K activation vectors from unsafe prompts

- Performs **SVD per head**

- Extracts the top-r singular vectors as **unsafe basis**

- Saves per-head subspaces:

  ```
   head_{b}_{h}.pt
  ```

------

### **Step 2 — Train Rotary Adapter**

```
python train_rotary_adapter.py
```

The training loop:

- Regularizes safe prompts using flow matching
- Penalizes unsafe prompts by reversing unsafe directions
- Updates only:
  - **A_{b,h}**, skew-symmetric matrices
  - **no update to FLUX/MMDiT model weights**

Checkpoint saved to:

```
rotary_adapter_s_param.pth
```

------

### **Step 3 — Safe Inference with SafeRoPE**

Script version:

```
python inference_saferope.py 
```

Notebook version (recommended):

```
demo_saferope_inference.ipynb
```

The notebook includes:

- baseline (vanilla FLUX) generations
- SafeRoPE-applied generations

More visual examples available in:
 `demo_saferope_inference.ipynb`

