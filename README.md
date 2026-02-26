# 3D Point Cloud Semantic Segmentation (NPM3D Mini Benchmark)

Professional project focused on semantic segmentation of urban LiDAR point clouds, with two complementary pipelines:
- a high-performance machine learning baseline (`XGBoost`)
- a deep learning approach inspired by `RandLA-Net`

The goal is to classify each point of a test cloud into one of 6 classes and export a leaderboard-ready submission file.

## Dataset Context (Paris-Lille-3D)

The Paris-Lille-3D project is both a dataset and a benchmark for point cloud classification.
The data was produced by a Mobile Laser System (MLS) in two French cities: Paris and Lille.

The original point cloud was labeled entirely by hand with 50 classes to support research on automatic point cloud segmentation and classification algorithms.
In the benchmark workflow, test submissions are evaluated on coarse classes (the original benchmark documentation refers to 10 coarse classes after class regrouping).
The mini benchmark subset used in this repository exposes 6 evaluation classes, as defined in the local `README.txt`.

Reference paper:
- Roynard X., Deschaud J.-E., Goulette F., *Paris-Lille-3D: a large and high-quality ground truth urban point cloud dataset for automatic segmentation and classification*.

BibTeX citation:

```bibtex
@article{roynard2017parislille3d,
  author = {Xavier Roynard and Jean-Emmanuel Deschaud and François Goulette},
  title = {Paris-Lille-3D: A large and high-quality ground-truth urban point cloud dataset for automatic segmentation and classification},
  journal = {The International Journal of Robotics Research},
  volume = {37},
  number = {6},
  pages = {545-557},
  year = {2018},
  doi = {10.1177/0278364918767506}
}
```

License:
- Paris-Lille-3D is distributed under **CC-BY-NC-ND-3.0** (Creative Commons Attribution Non-Commercial No Derivatives).

## Project Highlights

- End-to-end workflow from raw `.ply` point clouds to submission file
- Full-dataset training (`MiniLille1`, `MiniLille2`, `MiniParis1`)
- Strong feature engineering baseline with ensemble modeling
- Deep model notebook for RandLA-Net style training and inference
- Clean inference pipeline preserving test point order (critical for benchmark scoring)

## Visual Results

### Test Point Cloud
![Test Point Cloud](images/test_cloud.png)

### Colored Prediction
![Colored Prediction](images/prediction_colored.png)

> Replace the two images above with your final exported visuals.

## Dataset

- **Training**: `training/*.ply` with fields `x, y, z, class`
- **Test**: `test/MiniDijon9.ply` with fields `x, y, z`
- **Classes (submission labels)**: `1..6` = ground, building, poles, pedestrians, cars, vegetation
- This repository uses a reduced benchmark subset (`README.txt` in this project contains challenge-specific submission details).

## Repository Structure

- `main.py`: initial XGBoost baseline pipeline
- `train_best.py`: improved full-data XGBoost ensemble pipeline
- `randlanet_pipeline.ipynb`: RandLA-Net style deep learning pipeline (train + predict)
- `colorize_ply_from_submission.py`: utility to colorize predictions for visualization
- `training/`, `test/`: benchmark point clouds
- `submission.txt`: example submission output

## Technical Approach

### 1) ML Pipeline (XGBoost, production-ready baseline)

Implemented in `train_best.py`:
- multi-scale geometric feature extraction from raw XYZ
- local height/density statistics at multiple XY grid scales
- class-balanced weighting for rare classes
- ensemble of tuned XGBoost models with GPU-first fallback to CPU
- chunked inference for memory-safe prediction on full test cloud

This pipeline is fast, robust, and a strong baseline for leaderboard performance.

### 2) DL Pipeline (RandLA-Net style)

Implemented in `randlanet_pipeline.ipynb`:
- point chunking/tiling over full scenes
- data augmentation for robust generalization
- local feature aggregation + random downsampling/upsampling blocks
- stage-1 validation training + stage-2 full-data fine-tuning
- test-time voting accumulation and submission export

This pipeline is designed to push beyond classical ML by learning richer local geometric representations.

## Quick Start

### Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy scipy scikit-learn xgboost plyfile open3d tqdm matplotlib pandas
```

### Train + Predict with best ML pipeline

```bash
.venv/bin/python train_best.py
```

Output:
- `submission_best.txt`

### Train + Predict with RandLA-Net notebook

```bash
jupyter notebook randlanet_pipeline.ipynb
```

Notebook output:
- `randlanet_best.pt` (trained weights)
- `submission_randlanet.txt`

## Benchmark Submission Format

Expected format is one integer label per line (`1..6`), in the exact point order of `test/MiniDijon9.ply`.

## Why This Project Matters

This project demonstrates practical ML engineering for 3D data:
- handling large-scale point clouds efficiently
- designing robust feature pipelines and deep models
- balancing experimentation speed with reproducible deployment
- delivering leaderboard-compatible outputs with strict data-format constraints

It demonstrates applied data science and 3D perception capabilities for autonomous systems, mapping, robotics, and geospatial AI applications.
