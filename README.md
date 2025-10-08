# 🧭 Image-Based 3D Modeling Using Structure-from-Motion (SfM)

**Author:** V. V. Naga Satya Sai Abhishek  
**Project Guide:** Dr. Ritwik Kumar Layek  
**Institution:** Indian Institute of Technology, Kharagpur  
**Department:** Electronics and Electrical Communication Engineering  
**Academic Year:** 2024–25  

---

## 🎯 Project Overview
This project implements a **custom Structure-from-Motion (SfM)** pipeline to reconstruct **3D point clouds** from multiple 2D images.  
It integrates the core steps of camera calibration, feature detection, image registration, and bundle adjustment into a **user-friendly GUI**.

The system allows users to:
- Load multiple images of a static object or scene  
- Calibrate the camera using a checkerboard pattern  
- Extract and match visual features between images  
- Verify geometric consistency using RANSAC and Essential/Fundamental matrices  
- Register images incrementally to estimate relative camera poses  
- Perform bundle adjustment to refine 3D structure and camera parameters  
- Visualize intermediate and final **sparse 3D reconstructions**

---

## 🧩 Key Components

| Module | Description |
|--------|-------------|
| **Camera Calibration** | Implements Zhang’s method for intrinsic & extrinsic parameter estimation using checkerboard images. |
| **Feature Extraction** | Supports classical and deep-learning-based feature descriptors (SIFT, ORB, etc.). |
| **Feature Matching** | Uses FLANN (Fast Library for Approximate Nearest Neighbors) for efficient descriptor matching. |
| **Geometric Verification** | Applies RANSAC to filter matches and compute Fundamental/Essential matrices. |
| **Image Registration** | Sequentially adds images and estimates relative poses using PnP + triangulation. |
| **Bundle Adjustment** | Global optimization using Levenberg–Marquardt to minimize reprojection error. |
| **3D Visualization** | Displays reconstructed 3D points and camera poses in the GUI. |

## 🖥️ Graphical User Interface (GUI)

The GUI provides an intuitive way to execute the full SfM pipeline step-by-step.  

### Main Features
- **Add Images:** Import one or more images of the same scene or object.  
- **Calibrate Camera:** Perform intrinsic calibration using checkerboard images.  
- **Feature Extraction:** Detect keypoints and compute feature descriptors.  
- **Feature Matching:** Match features across images using FLANN.  
- **Geometric Verification:** Validate matches via RANSAC and compute two-view geometry.  
- **Register Images:** Incrementally register and triangulate new views.  
- **Bundle Adjustment:** Refine 3D structure and camera poses globally.  
- **View Results:** Visualize 3D point cloud and camera pose projections.
---

## 🚀 Running the Project

### 1. Clone the Repository
### 2. Create a Virtual Environment (recommended)
Using Anaconda:
```bash
conda create -n sfm python=3.11
conda activate sfm
```
or using venv:
```bash
python -m venv sfm
sfm\Scripts\activate     # (Windows)
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the GUI
```bash
python gui.py
```

---

## 📂 Project Structure
```
├── gui.py                     # Entry point: launches the GUI
├── calibration/               # Camera calibration scripts & patterns
├── features/                  # Feature extraction & matching utilities
├── reconstruction/            # Pose estimation, triangulation, bundle adjustment
├── visualization/             # 3D plotting and GUI rendering
├── datasets/                  # Sample image datasets
├── outputs/                   # Reconstructed point clouds and intermediate files
├── requirements.txt
└── README.md
```

---

## 📊 Results
- Sparse 3D point cloud visualization  
- Reconstructed camera trajectories  
- Intermediate matching and geometry verification visualizations  


## 📚 Future Work
- Integrate dense reconstruction using Multi-View Stereo (MVS).  
- Add deep-learning-based feature extractors (SuperPoint, D2-Net).  
- Implement real-time reconstruction pipeline with OpenGL visualization.  

---

## 📜 License
This project is for academic and research use under the IIT Kharagpur MTech program.  
