# ACL_P
Anterior Cruciate Ligament classification with bone morphology
# Point Cloud Classification for Knee Bone Analysis

This repository contains code for training and evaluating a Point Cloud Transformer (PCT) model on knee bone point cloud data. The dataset includes labeled point clouds for classifying knee conditions (e.g., `t1` and `t2`). Below is an overview of the scripts and their functionalities:

---

## Script Descriptions

1. **`kneedata.py`**  
   - **Purpose**: Load and preprocess knee bone point cloud datasets.  
   - **Features**:  
     - Data augmentation (dropout, translation).  
     - Custom PyTorch `Dataset` classes (`BonePointNet`, `FemurTibiaPointNet`).  
     - Handles point cloud sampling and label parsing.  

2. **`main.py`**  
   - **Purpose**: Main training and evaluation script.  
   - **Features**:  
     - Model training with Adam/SGD optimizer and cosine annealing LR scheduler.  
     - Supports both single-bone and fused (femur + tibia) point cloud inputs.  
     - Metrics: Accuracy, AUC, F1-score, Sensitivity, Specificity.  

3. **`model.py`**  
   - **Purpose**: Defines the PCT (Point Cloud Transformer) architecture.  
   - **Key Components**:  
     - `PCT`: Base model for single point cloud input.  
     - `Pct_FT`: Two-stream fusion model for femur and tibia point clouds.  
     - Self-attention layers and downsampling operations.  

4. **`pcd_align.py`**  
   - **Purpose**: Visual alignment and comparison of point clouds.  
   - **Features**:  
     - Generates 2D slices (sagittal/coronal views) for qualitative analysis.  
     - Supports point cloud sampling and visualization.  

5. **`pc_pre.py`**  
   - **Purpose**: Preprocess MRI data to generate point clouds.  
   - **Features**:  
     - Converts medical masks to 3D point clouds.  
     - Normalizes and aligns point clouds.  
     - Samplings to fixed point counts (e.g., 2048 points).  

---

## Setup

### Dependencies  
- Python 3.8+
- PyTorch 1.10+
- Open3D 0.15+
- numpy, sklearn, argparse  
