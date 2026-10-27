# Solar Flare Prediction with Machine Learning

Machine learning framework for predicting solar flares 24 hours in advance using SHARP magnetic parameters. Compares Random Forest, XGBoost, KNN, and a deep learning transformer-based model (SolarFlareNet).

---

## 🚀 Quick Start

### Installation
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost tensorflow
```

### Run Analysis
1. Ensure files are organized (see structure below)
2. Open `MyMLCode.ipynb`
3. Set flare class in Cell 3: `FLARE_CLASS = 'M'` (options: 'C', 'M', 'M5')
4. Run all cells
5. Results save to `results/` folder

---

## 📁 File Structure

### ✅ Important Files

```
├── MyMLCode.ipynb              ← Main notebook (start here)
├── solarflarenet_model.py      ← Model architecture (lowercase)
├── data_sampler.ipynb          ← Tool to reduce dataset size
└── data/                       ← Training/testing data
    ├── training_data_C_24.csv
    ├── testing_data_C_24.csv
    ├── training_data_M_24.csv
    ├── testing_data_M_24.csv
    ├── training_data_M5_24.csv
    └── testing_data_M5_24.csv
```

The other files represent the structure from the original model, which I have adapted in MyMLCode.ipynb and solarflarenet_model.py

### 📚 Reference Files (Original - 2 Years Old)

```
├── SolarFlareNet_model.py              ← Original (uppercase)
├── SolarFlareNet_run_all_tests.py      ← Original scripts
├── SolarFlareNet_run_all_trainings.py
├── SolarFlareNet_test.py
├── SolarFlareNet_train.py
└── utils.py
```

The uppercase `SolarFlareNet_*` files are from the original 2023 Nature paper (reference only). The lowercase `solarflarenet_model.py` is adapted for faster training and tabular data. Use the lowercase one for the modeling process. The original models are for reference so you can see how I adapted everything.

---

## 📓 MyMLCode.ipynb - Main Notebook

### What It Does
Trains 4 models, evaluates them, and saves results.

### Key Cells

| Cell | Purpose | Time |
|------|---------|------|
| 1-2 | Import & load data | 5 sec |
| 3 | **Set FLARE_CLASS** | - |
| 4 | Preprocess data | 5 sec |
| 5 | Train Random Forest | 10 sec |
| 6 | Train XGBoost | 10 sec |
| 7 | Train SolarFlareNet | 10-20 min (CPU) |
| 8 | Train KNN | 5 sec |
| 9-11 | Evaluate & save | 10 sec |

### Configuration Options

**Flare Classes:**
```python
FLARE_CLASS = 'C'    # C-class (smallest, most common)
FLARE_CLASS = 'M'    # M-class (medium intensity)
FLARE_CLASS = 'M5'   # M5+ class (strong)
```

**Speed Up SolarFlareNet (in Cell 7):**
```python
# Change this for faster training:
solarflarenet_model = create_solarflarenet_for_tabular(
    num_features=X_train.shape[1],
    hidden_units=[64, 32],  # Smaller (was [128, 64, 32])
    dropout_rate=0.3
)

# And this:
history = model.fit(
    X_train, y_train,
    epochs=30,      # Fewer epochs (was 50)
    batch_size=128  # Larger batches (was 64)
)
```

---

## 🐍 solarflarenet_model.py

### What It Is
Neural network architecture used by Cell 7 of the notebook.

### Do You Need to Modify It?
**No**, unless you want to experiment. The notebook imports and uses it automatically.

### If You Want to Customize

**Faster training:**
```python
hidden_units=[64, 32]  # Simpler model
```

**Better accuracy:**
```python
hidden_units=[256, 128, 64, 32]  # Deeper model
```

**Prevent overfitting:**
```python
dropout_rate=0.5  # More regularization
```

---

## 📊 Data Files

### What's in data/ Folder

6 CSV files containing SHARP magnetic parameters:
- Training/testing pairs for C, M, and M5 class flares
- ~25 features per sample (magnetic field measurements)
- Binary labels: 0 (no flare), 1 (flare within 24h)
- 50/50 balanced classes

### File Sizes
- Training files: ~30-50 MB each
- Testing files: ~10-20 MB each
- **Total: ~200-300 MB**

---

## 🔧 data_sampler.ipynb - Reduce Dataset Size

### When to Use
Your datasets are massive and causing memory issues or slow training.

### How to Use

1. Open `data_sampler.ipynb`
2. Set parameters:
```python
INPUT_FILE = 'data/original_large_file.csv'
OUTPUT_FILE = 'data/training_data_M_24.csv'
SAMPLE_SIZE = 40000          # Reduce to 40,000 samples
KEEP_BALANCED = True        # Keep 50/50 split
```
3. Run all cells
4. Use smaller file in `MyMLCode.ipynb`

---

## 🔄 Typical Workflow

### Standard Run
1. Open `MyMLCode.ipynb`
2. Set `FLARE_CLASS = 'M'` in Cell 3
3. Run all cells (20-30 minutes total)
4. Check `results/` folder

### With Data Sampling (Large Datasets)
1. Run `data_sampler.ipynb` to create smaller files
2. Update file paths in `MyMLCode.ipynb` Cell 2
3. Run notebook on sampled data (5-10 minutes)
4. Switch to full data for final results

---

## 🔍 My Version vs. Original

| | My `solarflarenet_model.py` | Original `SolarFlareNet_model.py` |
|---|---|---|
| **Case** | Lowercase | Uppercase |
| **Architecture** | Simplified (dense layers) | Full transformer (4 blocks) |
| **Input** | Tabular features | Time-series sequences |
| **Training** | 10-20 min | 30-90 min |
| **Use Case** | Quick experiments | Research/production |
| **Status** | **Active** ✅ | Reference 📚 |

**Why adapted?** Original requires sequential time-series data. Mine works with the tabular format and trains 3x faster.

---

## 🛠️ Troubleshooting

**Problem:** `ModuleNotFoundError: solarflarenet_model`  
**Fix:** Put `solarflarenet_model.py` in same folder as notebook

**Problem:** Training too slow  
**Fix:** Use smaller model: `hidden_units=[64, 32]` and `epochs=30`

**Problem:** Out of memory  
**Fix:** Use `data_sampler.ipynb` to reduce dataset size

**Problem:** Low accuracy (F1 < 0.6)  
**Fix:** Check data loaded correctly, verify `FLARE_CLASS` matches files

---

## 📚 Citation

**Original SolarFlareNet:**
```
Abduallah, Y., Wang, J.T.L., Wang, H. et al. 
Operational prediction of solar flares using a transformer-based framework. 
Sci Rep 13, 14689 (2023). 
https://doi.org/10.1038/s41598-023-40884-1
```

**This Implementation:**
```
Solar Flare Prediction with Machine Learning (2025)
Adapted SolarFlareNet for tabular data and faster training
```

---

## ✅ Quick Checklist

Before running:
- [ ] `MyMLCode.ipynb` in main directory
- [ ] `solarflarenet_model.py` in same directory as notebook
- [ ] `data/` folder with 6 CSV files
- [ ] Set `FLARE_CLASS` in Cell 3
- [ ] Run all cells