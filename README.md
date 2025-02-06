
# **LM-Droid SLAM: Enhancing DROID-SLAM with Levenberg-Marquardt Optimization**

## **Overview**
LM-Droid SLAM is an enhanced version of **DROID-SLAM**, integrating a **Levenberg-Marquardt (LM) solver** for camera pose and inverse depth refinement. The project aims to improve **accuracy and stability** over the original **dense bundle adjustment (DBA) layers** while maintaining the **differentiable** and **end-to-end learnable** architecture of DROID-SLAM.

We evaluate our modifications on the **TartanAir dataset**, analyzing **Absolute Trajectory Error (ATE)** and conducting a **computational efficiency study** to identify bottlenecks in the SLAM pipeline.

---

## **Table of Contents**
- [Introduction](#introduction)
- [Features](#features)
- [Methodology](#methodology)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation & Results](#evaluation--results)
- [Performance Analysis](#performance-analysis)
- [Limitations & Future Work](#limitations--future-work)
- [References](#references)
- [Contributors](#contributors)

---

## **Introduction**
Visual SLAM (Simultaneous Localization and Mapping) is critical for robotics, autonomous vehicles, and AR applications. **DROID-SLAM**, a state-of-the-art learning-based SLAM system, utilizes **DBA layers** for pose and depth refinement but suffers from **high computational cost** and **potential robustness issues**.

Our project modifies **DROID-SLAM** by **replacing DBA layers** with the **Levenberg-Marquardt solver**, known for its effectiveness in handling **non-linear least squares problems**, with the goal of improving:
- Accuracy of camera pose and depth estimation
- Stability across different environments
- Computational efficiency

---

## **Features**
✅ **Integration of LM Solver**: Replaces DBA layers with an LM-based optimization technique.  
✅ **Differentiable Optimization**: Maintains end-to-end backpropagation capability.  
✅ **ATE Comparison**: Evaluates trajectory accuracy improvements.  
✅ **Computational Profiling**: Identifies SLAM pipeline bottlenecks.  
✅ **TartanAir Dataset Benchmarking**: Provides robust evaluation on challenging sequences.  

---

## **Methodology**
### **1. Dense Bundle Adjustment (DBA) in DROID-SLAM**
- Optimizes camera poses and inverse depth using a **Schur Complement and Gauss-Newton update**.
- Implements **learned feature embeddings** for pose refinement.
- Computationally expensive but effective for end-to-end learning.

### **2. Levenberg-Marquardt Solver Implementation**
- Replaces the DBA layer with an **iterative LM solver**.
- Uses **adaptive damping** to switch between **gradient descent** and **Gauss-Newton methods**.
- Aims for **better stability** and **reduced computational cost**.

### **3. Integration with DROID-SLAM Architecture**
- Modified the **update operator** while keeping **Conv-GRU, feature extraction, and context correlation** unchanged.
- LM solver runs **N times per update** to refine depth and pose estimates.

---

## **Installation**
### **Dependencies**
Ensure you have the following dependencies installed:
```bash
pip install torch torchvision numpy opencv-python matplotlib tqdm
```
For CUDA-accelerated computations:
```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
```

### **Clone the Repository**
```bash
git clone https://github.com/your-username/LM-Droid-SLAM.git
cd LM-Droid-SLAM
```

### **Setup Environment**
```bash
conda create --name lmdroidslam python=3.8
conda activate lmdroidslam
pip install -r requirements.txt
```

---

## **Usage**
### **1. Running the SLAM System**
To run the modified SLAM pipeline on the **TartanAir dataset**, use:
```bash
python run_slam.py --config configs/tartan_air.yaml --use-lm
```
For baseline DROID-SLAM:
```bash
python run_slam.py --config configs/tartan_air.yaml --use-dba
```

### **2. Visualizing Trajectory Results**
```bash
python visualize_results.py --dataset tartanair --method LM-SLAM
```

---

**Key Observations:**
- LM solver **without retraining** results in a slight accuracy drop.
- Fully retraining the network **with a differentiable LM solver** is expected to improve performance.
- Current hardware limitations (90GB VRAM needed) prevented complete retraining.

---

## **Performance Analysis**
### **Timing Breakdown (DROID-SLAM vs. LM Solver)**
- **DBA layer contributes ~7.5% of total runtime**.
- **Graph Aggregation (30.7%) and Update Operator (35%)** are the primary bottlenecks.
- LM solver **removes DBA overhead**, but further optimizations are required.

---

## **Limitations & Future Work**
🚧 **Current Limitations:**
- **Accuracy drop** without retraining.
- **Memory-intensive training** prevents full model optimization.
- **Requires high-end GPUs (A100 recommended).**

🚀 **Future Directions:**
- **Integrating differentiable LM solver (e.g., Theseus).**
- **Reducing memory footprint** for LM-based optimization.
- **Optimizing graph aggregation module** for improved runtime.

---

## **References**
[1] Teed, Z., & Deng, J. "DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras." *NeurIPS 2021.*  
[2] Wang, W., Zhu, D., et al. "TartanAir: A Dataset to Push the Limits of Visual SLAM." *arXiv:2003.14338.*  
[3] Mur-Artal, R., & Tardós, J. D. "ORB-SLAM: A Versatile and Accurate Monocular SLAM System." *IEEE Transactions on Robotics, 2015.*

---

## **Contributors**
- **Parth Mahajan** (Northeastern University) - *mahajan.parth@northeastern.edu*
- **Utkarsh Rai** (Northeastern University) - *rai.ut@northeastern.edu*

---
