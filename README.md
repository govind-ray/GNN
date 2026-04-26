# GNN
# 🧠 Hybrid GTN-EEG: Alzheimer's Disease Detection using EEG Signals

## 📌 Overview

This project presents a deep learning-based approach for detecting Alzheimer's Disease (AD) using EEG (Electroencephalography) signals. The model combines **1D Convolutional Neural Networks (CNN)** for temporal feature extraction with **Graph Transformer Networks (GTN)** for modeling relationships between EEG channels.

The system classifies subjects into:

* Normal
* Mild Cognitive Impairment (MCI)
* Alzheimer’s Disease (AD)

---

## 🚀 Key Features

* ✅ EEG signal preprocessing (filtering, normalization, segmentation)
* ✅ 1D CNN for temporal feature extraction
* ✅ Dynamic graph construction using channel correlations
* ✅ GTN for learning complex inter-channel relationships
* ✅ Multi-class classification (Normal / MCI / AD)
* ✅ Visualization of signals, graphs, and predictions
* ✅ Supports both real and synthetic EEG datasets

---

## 🏗️ Architecture

```
EEG Data
   ↓
Preprocessing
   ↓
1D CNN (Feature Extraction)
   ↓
Graph Construction (Correlation-based)
   ↓
GTN Layers (Graph Learning)
   ↓
Fully Connected Layers
   ↓
Classification Output
```

---

## 📂 Project Structure

```
project/
├── train.py                  # Training script
├── hybrid_gtn_model.py      # Model architecture (CNN + GTN)
├── data_loader.py           # Data preprocessing & loading
├── run_pipeline.py          # Full pipeline (train + evaluate)
├── run_full_evaluation.py   # Evaluation & visualization
├── visualization.py         # Visualization tools
├── test_fixes.py            # Testing script
├── data/                    # EEG dataset (.mat files)
└── checkpoints/             # Saved models & outputs
```

---

## ⚙️ Installation

```bash
pip install torch torchvision torchaudio
pip install numpy scipy scikit-learn matplotlib seaborn networkx tqdm
```

---

## ▶️ Usage

### 🔹 Run with Synthetic Data (Quick Test)

```bash
python run_pipeline.py --synthetic --epochs 5
```

### 🔹 Run with Real EEG Data

1. Download dataset from:
   https://data.mendeley.com/datasets/sgzbgwjfkr/5

2. Place files in `data/`:

```
Normal.mat
MCI.mat
AD.mat
```

3. Run:

```bash
python run_pipeline.py --data_dir ./data --epochs 100
```

---

## 📊 Output

After training, results are saved in:

```
checkpoints/run_YYYYMMDD_HHMMSS/
```

Includes:

* best_model.pth
* training_history.png
* confusion_matrix.png
* visualization outputs

---

## 🧠 Model Details

### 🔹 1D CNN

* Extracts temporal features from EEG signals
* Uses multiple Conv1d layers with pooling

### 🔹 GTN (Graph Transformer Network)

* Builds graph based on EEG channel correlations
* Learns complex relationships between brain regions

---

## 📈 Results

* Improved classification performance compared to traditional methods
* Robust performance on both synthetic and real EEG data
* Supports visualization for interpretability

---

## 🔮 Future Work

* Integration with real-time EEG systems
* Deployment as a clinical decision support tool
* Incorporation of GCN/GAT for comparison
* Larger dataset evaluation

---

## 👨‍💻 Authors

* Govind Ray
* Vikas Kumar Mishra

---

## 📜 License

This project is for academic and research purposes.

---

## ⭐ Acknowledgements

* Based on Graph Transformer Networks (NeurIPS 2019)
* EEG dataset from Mendeley Data

---
