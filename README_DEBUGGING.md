# Hybrid GTN-EEG Alzheimer's Detection - DEBUGGED VERSION

## 🔧 Issues Fixed

### 1. **Training Script Issues**
- ✅ **Missing train_epoch loop code** - Complete training loop now implemented
- ✅ **Config path bug** - Now saves absolute paths instead of relative
- ✅ **Better error handling** for data loading

### 2. **Model Architecture Issues**
- ✅ **Missing GTN layer initialization** - Complete FastGTNLayer initialization added
- ✅ **Proper tuple handling** in forward pass outputs

### 3. **Data Loader Issues**
- ✅ **Robust .mat file parsing** - Handles multiple data structures
- ✅ **Better field extraction** - Checks multiple field names
- ✅ **Proper transpose handling** for different array shapes

### 4. **Evaluation Issues**
- ✅ **Fixed synthetic data fallback** in evaluation
- ✅ **Proper config loading** with absolute paths
- ✅ **Unicode character fixes** for cross-platform compatibility

## 📦 Installation

```bash
# Install dependencies
pip install torch torchvision torchaudio --break-system-packages
pip install numpy scipy scikit-learn matplotlib seaborn networkx tqdm --break-system-packages
```

## 🚀 Quick Start

### Option 1: Run with Synthetic Data (Quick Test)
```bash
# Test the pipeline with synthetic data
python run_pipeline.py --synthetic --epochs 5

# This will:
# 1. Generate synthetic EEG-like data
# 2. Train for 5 epochs (quick test)
# 3. Run evaluation and visualization
# 4. Save results to checkpoints/run_YYYYMMDD_HHMMSS/
```

### Option 2: Run with Real Data
```bash
# First, download the dataset from:
# https://data.mendeley.com/datasets/sgzbgwjfkr/5
# Place Normal.mat, MCI.mat, AD.mat in ./data/ folder

# Then run:
python run_pipeline.py --data_dir ./data --epochs 100

# For a longer training run:
python run_pipeline.py --data_dir ./data --epochs 200 --batch_size 32
```

## 📁 File Structure

```
project/
├── train.py                    # Fixed training script
├── hybrid_gtn_model.py        # Fixed model architecture  
├── data_loader.py             # Data loading and preprocessing
├── run_pipeline.py            # Main runner script
├── run_full_evaluation.py     # Evaluation and visualization
├── visualization.py           # Visualization utilities
├── data/                      # Place .mat files here
│   ├── Normal.mat
│   ├── MCI.mat
│   └── AD.mat
└── checkpoints/               # Training outputs saved here
    └── run_YYYYMMDD_HHMMSS/
        ├── config.json
        ├── best_model.pth
        ├── training_history.png
        ├── test_results.json
        └── viz_*.png (visualizations)
```

## 🐛 Key Bugs Fixed

### Bug #1: Incomplete Training Loop
**Original Issue:**
```python
# Lines were truncated in train.py
pbar = tqdm(self.train_loader, desc='Training', leave=False)
for batch_x, batch_y in pbar:
    # MISSING CODE HERE
```

**Fix:**
```python
for batch_x, batch_y in pbar:
    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
    self.optimizer.zero_grad()
    outputs = self.model(batch_x)
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    loss = self.criterion(outputs, batch_y)
    loss.backward()
    self.optimizer.step()
    # ... rest of loop
```

### Bug #2: Missing GTN Layer Initialization
**Original Issue:**
```python
# GTN layers initialization was truncated
self.gtn_layers = nn.ModuleList()
in_dim = feature_dim
for i in range(num_gtn_layers):
    self.gtn_layers.append(
        FastGTNLayer(
            # MISSING INITIALIZATION PARAMETERS
```

**Fix:**
```python
self.gtn_layers = nn.ModuleList()
in_dim = feature_dim
for i in range(num_gtn_layers):
    self.gtn_layers.append(
        FastGTNLayer(
            in_channels=in_dim,
            out_channels=gtn_hidden_dim,
            num_edges=3,
            num_nodes=num_eeg_channels,
            num_channels=num_graph_channels
        )
    )
    in_dim = gtn_hidden_dim
```

### Bug #3: Relative Path Issues
**Original Issue:**
```python
# Config saved relative path which breaks on evaluation
config['data_dir'] = args.data_dir  # Could be './data'
```

**Fix:**
```python
# Save absolute path or synthetic marker
if using_synthetic:
    config['data_dir'] = 'synthetic'
else:
    config['data_dir'] = os.path.abspath(args.data_dir)
```

### Bug #4: .mat File Field Extraction
**Original Issue:**
- Code didn't handle all possible .mat file structures
- Would fail on structured arrays with different field names

**Fix:**
```python
# Check multiple possible field names
for field_name in ['data', 'EEG', 'eeg', 'signal', 'X']:
    if field_name in element.dtype.names:
        eeg_data = element[field_name]
        found_data = True
        break
```

## 🎯 Usage Examples

### Example 1: Quick 5-Epoch Test
```bash
python run_pipeline.py --synthetic --epochs 5
```

**Expected Output:**
```
==========================================================
STARTING HYBRID GTN-EEG PIPELINE
==========================================================
Using SYNTHETIC data as requested.

>>> PHASE 1: TRAINING MODEL
Creating synthetic EEG dataset...
Starting training on cuda
Epoch [1/5] Train Loss: 1.0234, Acc: 0.4567 | Val Loss: 0.9876, Acc: 0.5123
...
✓ New best model saved!

>>> PHASE 2: EVALUATION & VISUALIZATION
Running Model Evaluation and Visualization
Visualizations saved to checkpoints/run_20240215_143022
```

### Example 2: Full Training Run
```bash
python run_pipeline.py --data_dir ./data --epochs 100 --batch_size 32 --lr 0.0005
```

### Example 3: Manual Steps
```bash
# Step 1: Train only
python train.py --data_dir ./data --epochs 50 --output_dir my_checkpoints

# Step 2: Evaluate specific checkpoint
python run_full_evaluation.py --checkpoint_dir my_checkpoints/run_20240215_143022
```

## 📊 Expected Outputs

After successful run, you'll find in `checkpoints/run_YYYYMMDD_HHMMSS/`:

1. **config.json** - Model and training configuration
2. **best_model.pth** - Best model checkpoint
3. **training_history.png** - Loss and accuracy curves
4. **test_results.json** - Test set metrics
5. **test_confusion_matrix_full.png** - Confusion matrix
6. **viz_channel_graph.png** - Channel correlation graph
7. **viz_signals.png** - Sample EEG signals
8. **viz_features.png** - Extracted features heatmap
9. **viz_prediction.png** - Sample prediction with confidence

## 🔍 Troubleshooting

### Issue: "No data found to prepare dataloaders"
**Solution:** Make sure .mat files are in the correct directory:
```bash
ls -l ./data/
# Should show: Normal.mat, MCI.mat, AD.mat
```

### Issue: "CUDA out of memory"
**Solution:** Reduce batch size:
```bash
python run_pipeline.py --batch_size 8 --epochs 50
```

### Issue: "Error loading .mat file structure"
**Solution:** Check the structure:
```python
from scipy.io import loadmat
mat = loadmat('./data/Normal.mat')
print(mat.keys())  # See available keys
```

The data loader will try multiple field names automatically.

### Issue: Training is very slow
**Solution:**
- Use GPU if available (automatically detected)
- Reduce number of channels or sequence length
- Use synthetic data for quick testing

## 📈 Performance Expectations

### With Synthetic Data (5 epochs):
- Training time: ~2-5 minutes (CPU)
- Expected accuracy: 60-80% (synthetic is easier)
- Purpose: Code testing and validation

### With Real Data (100 epochs):
- Training time: ~30-60 minutes (CPU), ~5-10 minutes (GPU)
- Expected accuracy: 70-85% (depends on data quality)
- Purpose: Actual research results

## 🎓 Model Architecture

```
Input: [batch, 32 channels, 1000 timesteps]
    ↓
CNN Feature Extractor (per channel)
    ↓
Features: [batch, 32 channels, 128 features]
    ↓
Channel Correlation Graph Construction
    ↓
GTN Layers (2 layers)
    ↓
Graph Embeddings: [batch, 32 channels, 64 hidden]
    ↓
Global Pooling & Classification
    ↓
Output: [batch, 3 classes] (Normal, MCI, AD)
```

## 📝 Citation

If you use this code, please cite the original GTN paper:
```
Yun, S., Jeong, M., Kim, R., Kang, J., & Kim, H. J. (2019).
Graph transformer networks. 
Advances in Neural Information Processing Systems, 32.
```

## 🆘 Getting Help

If issues persist:
1. Check all files are using the FIXED versions
2. Verify Python and package versions
3. Try with synthetic data first
4. Check the error traceback carefully

## ✅ Verification Checklist

- [ ] All dependencies installed
- [ ] Files replaced with fixed versions
- [ ] Can run with synthetic data (`--synthetic`)
- [ ] Can train for 5 epochs without errors
- [ ] Outputs are saved to checkpoints directory
- [ ] Can load and evaluate saved model

---

**Last Updated:** 2024
**Status:** ✅ All major bugs fixed and tested
