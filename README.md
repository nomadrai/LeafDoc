# LeafDoc: Plant Leaf Disease Detection Using Image Processing (MATLAB)

A complete **Plant Leaf Disease Detection & Classification System** implemented in **MATLAB** using **CIE $L^*a^*b^*$ K-Means Color Clustering**, **GLCM (Gray-Level Co-occurrence Matrix) Texture & Statistical Feature Extraction (13 Features)**, **Multi-Class Support Vector Machine (SVM)** classification, and a **Universal Chromaticity & Symptom Morphology Analyzer** that works on any leaf image.

---

## Key Features

- **Automatic $L^*a^*b^*$ K-Means Segmentation ($K=3$)**: Deterministically separates any uploaded leaf image into **Cluster 1 (Disease Lesion ROI)**, **Cluster 2 (Healthy Green Tissue)**, and **Cluster 3 (Background)**.
- **13 GLCM & Statistical Features**: Extracts *Contrast, Correlation, Energy, Homogeneity, Mean, Standard Deviation, Entropy, RMS, Variance, Smoothness, Kurtosis, Skewness,* and *IDM (Inverse Difference Moment)*.
- **Hybrid Multi-Class Disease Classifier**: Combines a standardized Multi-Class SVM (`multisvm.m`) with perceptual dataset matching and a universal symptom/chromaticity analyzer so arbitrary leaf photos are accurately diagnosed.
- **Infected Area % & Severity Grading**: Computes the lesion-to-leaf area ratio, highlights infected regions with a red/yellow contour overlay, grades severity (*None/Healthy, Mild, Moderate, Severe*), and provides agricultural treatment recommendations.
- **Dual GUI Interfaces**:
  1. `LeafDoc_App.m` — Modern 6-panel visual pipeline dashboard.
  2. `DetectDisease_GUI.m` (`.fig`) — Classic MATLAB GUIDE interface compatible with modern MATLAB releases (R2018a–R2024b+).

---

## Requirements

- **MATLAB** (R2018a or newer; tested on R2024b)
- **Image Processing Toolbox**
- **Statistics and Machine Learning Toolbox**

---

## Quick Start

### Option 1: Launch from Terminal (Linux / macOS)

```bash
chmod +x run_leafdoc.sh

# Launch the modern 6-panel LeafDoc Dashboard App (Recommended)
./run_leafdoc.sh

# Launch the classic GUIDE GUI (DetectDisease_GUI)
./run_leafdoc.sh classic

# Run the standalone single-file script (Detect.m)
./run_leafdoc.sh script
```

### Option 2: Run Inside MATLAB IDE

Open MATLAB, navigate to the repository folder, and run:

```matlab
LeafDoc_App          % Launches the 6-panel modern dashboard GUI
```

Or run the classic GUIDE GUI or standalone script:

```matlab
DetectDisease_GUI    % Launches the classic GUIDE interface
Detect               % Runs the standalone interactive script
```

---

## Repository Structure

| File / Folder | Description |
| :--- | :--- |
| `LeafDoc_App.m` | Modern 6-panel interactive GUI dashboard with automatic K-Means ROI selection, lesion heatmap overlay, 13-feature table, severity grading, and remedies |
| `DetectDisease_GUI.m` & `.fig` | Classic GUIDE GUI upgraded for modern MATLAB compatibility |
| `EvaluateFeatures.m` | CIE $L^*a^*b^*$ K-Means clustering ($K=3$) + deterministic Disease ROI selection + 13 GLCM & statistical feature extraction |
| `AnalyzeAnyLeaf.m` | Hybrid disease classification engine combining Multi-Class SVM, perceptual dataset matching, and universal symptom/chromaticity morphology analysis |
| `multisvm.m` | Standardized Multi-Class SVM classifier |
| `svmtrain.m` & `svmclassify.m` | Compatibility wrappers for modern MATLAB using `fitcsvm` and `predict` |
| `Detect.m` | Standalone interactive script |
| `Disease Database/` | 5-class leaf disease dataset (*Alternaria Alternata, Anthracnose, Bacterial Blight, Cercospora Leaf Spot, Healthy Leaves*) |
| `Extended_Dataset/` | Extended 13-class multi-crop leaf disease dataset (374 images across Corn, Potato, Tomato, Citrus, Grape, etc.) |
| `Training_Data.mat` & `Accuracy_Data.mat` | Pre-trained feature matrices and class labels |
| `documentation.PDF` | Reference documentation for the image processing and classification methodology |
| `LICENSE` | MIT License |

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

