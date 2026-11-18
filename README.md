# PathSnap: Machine Learning Map-Matching & Trajectory Reconstruction

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.5+-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**PathSnap** is a machine learning and sequential map-matching system developed for the PES University Machine Learning Hackathon (UE24CS352A). PathSnap accurately matches noisy GPS observations to complex road networks and reconstructs true vehicle trajectories using spatial candidate generation, probabilistic machine learning across 5 distinct algorithm families, and Bayesian hyperparameter optimization.

---

## 📁 Repository Structure (Evaluation 1)

```
PathSnap/
├── data/                               # Preprocessed candidate feature datasets
│   ├── train_features.parquet          # Compressed training features (508,340 candidate rows)
│   ├── train_features.csv.gz           # Gzipped training features
│   ├── test_features.parquet           # Compressed testing features (191,600 candidate rows)
│   └── test_features.csv.gz            # Gzipped testing features
├── notebooks/                          # Interactive analysis & evaluation notebooks
│   ├── 01_EDA_and_Visualization.ipynb  # Comprehensive EDA, trajectory plots & Folium maps
│   └── 02_Model_Training_and_HMM_Reconstruction.ipynb # Model benchmarking & evaluation
├── results/
│   ├── figures/                        # Diagnostic plots & figures
│   │   ├── eda_sampling_intervals.png  # GPS sampling interval distribution
│   │   ├── eda_trip_lengths_and_points.png # Trip trajectory characteristics
│   │   ├── model_comparison_metrics.png# 5-family CV and testing metric comparison
│   │   ├── confusion_matrices.png      # Confusion matrices across candidate classification
│   │   └── bayesian_opt_history.png    # Optuna Bayesian hyperparameter optimization trace
│   ├── maps/                           # Standalone interactive Folium maps
│   │   ├── interactive_map_trip_00000001.html      # Training trip 1 ground truth vs GPS
│   │   └── interactive_map_trip_00000000_test.html # Testing trip 0 GPS trajectory
│   ├── metrics/                        # Benchmark summaries & experiment logs
│   │   ├── dataset_inventory.csv       # Summary of all 90 train & 10 test trips
│   │   ├── trip_summary_train.csv      # Per-trip statistics (train)
│   │   ├── trip_summary_test.csv       # Per-trip statistics (test)
│   │   ├── model_comparison_table.csv  # 5-family comparison across all evaluation criteria
│   │   ├── bayesian_opt_trials.csv     # 12 Optuna optimization trial logs
│   │   ├── tuned_vs_untuned_comparison.csv # Performance delta before & after tuning
│   │   ├── algorithm_exploration_tournament.csv # Broad 10-algorithm exploration
│   │   └── hmm_vs_classifier_comparison.csv    # Sequence modeling performance logs
│   └── models/                         # Serialized trained model weights (.joblib)
│       ├── scaler.joblib               # StandardScaler for continuous spatial features
│       ├── logistic_regression.joblib  # Linear / Probabilistic baseline
│       ├── distance_weighted_knn.joblib# Instance-based spatial model
│       ├── random_forest.joblib        # Ensemble bagging model
│       ├── lightgbm_classifier.joblib  # Gradient boosting baseline
│       ├── multi_layer_perceptron_mlp.joblib # Neural network architecture
│       └── best_tuned_model.joblib     # Bayesian-tuned LightGBM champion model
├── src/
│   ├── api/
│   │   └── main.py                     # FastAPI backend & distributed inference server
│   ├── ui/
│   │   └── app.py                      # Evaluator Streamlit interface with Folium maps
│   ├── eda/
│   │   └── explore.py                  # Exploratory Data Analysis & Folium map generator
│   ├── preprocessing/
│   │   ├── pipeline.py                 # Metric projection, KDTree & candidate feature extraction
│   │   └── build_dataset.py            # Batch dataset extraction builder
│   ├── models/
│   │   ├── train_models.py             # 5-family training & GroupKFold benchmarking suite
│   │   └── bayesian_optimization.py    # Optuna Bayesian hyperparameter search
│   └── hmm/
│       └── decoder.py                  # Probabilistic sequential transition & Viterbi recurrence
├── requirements.txt                    # Project dependencies
└── README.md                           # Documentation
```

---

## 🔍 Exploratory Data Analysis (EDA) Highlights

Our analysis of all 90 training trips and 10 held-out student testing trips revealed:
1. **Sampling Interval Distribution**:
   - The median GPS sampling interval is 1.00s.
   - 98.4% of consecutive observations arrive within $[0.8\text{s}, 1.2\text{s}]$, confirming regular sampling with occasional sensor dropouts (up to 4.5s).
2. **Spatial Coordinate Handling**:
   - Raw coordinates are in WGS84 geographic degrees (latitude $\approx 55.4^\circ$, longitude $\approx 37.2^\circ$).
   - Standard Euclidean distance in degrees causes severe ellipsoidal distortion ($\cos(55.4^\circ) \approx 0.5678$).
   - **Solution**: We implement an exact local transverse Equirectangular metric projection centered at the dataset centroid:
     $$x = R \cdot (\lambda - \lambda_0) \cos\left(\frac{\phi + \phi_0}{2}\right), \quad y = R \cdot (\phi - \phi_0)$$
     where $R = 6,371,000 \text{ m}$.
3. **Road Network Graph & Sparsity**:
   - Graph components consist of directed arcs connecting topological nodes.
   - For every GPS observation, candidate arcs are indexed using a SciPy `cKDTree` in projected metric coordinates.
   - Choosing candidate size $K=5$ achieves **98.86% ground-truth candidate recall** while keeping feature matrix size tractable (508,340 candidate rows).

---

## 🔬 Multi-Model Comparison (5 Algorithm Families)

All models were evaluated under **Group-Aware 3-Fold Cross-Validation** (grouped by `trip_id`) to prevent data leakage between overlapping GPS points from the same trip, and evaluated on all 10 held-out Student Testing trips:

| Algorithm Family | Model | Key Configuration | CV Macro F1 | Observation Top-1 Acc | Route Continuity | Test Macro F1 | Log Loss | Inference Speed |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear / Probabilistic** | Logistic Regression | $C=1.0$, L-BFGS, Scaled | 0.7992 | 75.18% | 91.72% | 0.8667 | 0.2014 | **0.69 ms/1k** |
| **Instance-Based** | KNN (Distance-Weighted) | $k=9$, Distance Weights | 0.7917 | 73.13% | 91.03% | 0.8494 | 0.3821 | 172.73 ms/1k |
| **Ensemble (Bagging)** | Random Forest | 100 Trees, Max Depth 12 | **0.8200** | **76.15%** | **92.86%** | **0.8766** | 0.1842 | 38.69 ms/1k |
| **Gradient Boosting** | LightGBM Classifier | 120 Trees, Leaves 31, LR 0.08 | 0.8177 | 75.53% | 91.73% | 0.8738 | **0.1795** | **3.32 ms/1k** |
| **Neural Network** | Multi-Layer Perceptron | Hidden (64, 32), ReLU, Adam | 0.8085 | 74.92% | 91.50% | 0.8681 | 0.1980 | 1.85 ms/1k |

### Broad Exploration Tournament (10 Distinct Algorithms)
In addition to the core 5 models, a broad tournament evaluated:
- **Linear Models**: Logistic Regression, Ridge Classifier, SGD Classifier
- **Tree Ensembles**: Random Forest, Extra Trees, Gradient Boosting, LightGBM, AdaBoost
- **Instance & Margin**: Distance-Weighted KNN, Linear SVC
- **Neural Networks**: Multi-Layer Perceptron (MLP)

Complete results are recorded in `results/metrics/algorithm_exploration_tournament.csv`.

---

## ⚙️ Bayesian Hyperparameter Optimization (Optuna)

Hyperparameter optimization was performed on the champion gradient boosting architecture using **Tree-structured Parzen Estimator (TPE)** over 12 trials:

* **Tuned Parameters**:
  - `num_leaves`: $[15, 127]$
  - `max_depth`: $[4, 14]$
  - `learning_rate`: $[0.01, 0.20]$ (log scale)
  - `min_child_samples`: $[10, 100]$
  - `subsample`: $[0.6, 1.0]$
  - `colsample_bytree`: $[0.6, 1.0]$
  - `reg_alpha`: $[10^{-8}, 10.0]$ (log scale)
  - `reg_lambda`: $[10^{-8}, 10.0]$ (log scale)

* **Optimization Results**:
  - **Best Trial**: Trial 10
  - **Macro F1**: Improved from $0.8738 \to \mathbf{0.8769}$
  - **Observation Top-1 Accuracy**: Improved from $75.53\% \to \mathbf{76.20\%}$
  - **Binary Log Loss**: Reduced from $0.1795 \to \mathbf{0.1042}$ (41.9% reduction in uncertainty)

Saved model: `results/models/best_tuned_model.joblib`.

---

## 🗺️ Interactive Folium Visualizations

Interactive HTML maps are generated with dual high-contrast tile providers (**CartoDB Voyager** and **Esri WorldStreetMap**) to ensure offline and local-file compatibility without 403 access blocks:
- `results/maps/interactive_map_trip_00000001.html`: Visualizes raw GPS points, ground-truth traversed arcs, and candidate segments.
- `results/maps/interactive_map_trip_00000000_test.html`: Visualizes held-out evaluation trajectory with color-coded directional polylines and start/end markers.

---

## 🛠️ Execution & Reproduction Instructions

### 1. Environment Setup
```bash
git clone https://github.com/abhiram289/PathSnap.git
cd PathSnap
pip install -r requirements.txt
```

### 2. Run Exploratory Data Analysis & Map Generation
```bash
python src/eda/explore.py
```
Outputs dataset inventory, trip summaries, sampling interval distributions, and interactive Folium maps to `results/`.

### 3. Run Preprocessing & Feature Extraction
```bash
python src/preprocessing/build_dataset.py
```
Generates candidate feature matrices with metric projection and cKDTree spatial indexing.

### 4. Train & Benchmark All 5 Model Families
```bash
python src/models/train_models.py
```
Performs 3-fold GroupKFold cross-validation, saves diagnostic plots, confusion matrices, and serialized model weights in `results/models/`.

### 5. Run Bayesian Optimization
```bash
python src/models/bayesian_optimization.py
```
Executes Optuna TPE search and saves the tuned champion model.

### 6. Run HMM & Viterbi Sequence Reconstruction Benchmark
```bash
python src/hmm/decoder.py
```
Evaluates classifier-only vs. HMM Viterbi route decoding and route continuity.

---

## 💻 Two-Laptop Distributed Deployment

The system is architected for distributed evaluation across two machines connected on the same local network:

### Laptop 1: Model & API Server
Hosts the trained machine learning models, road network topologies, and the FastAPI inference engine.
```bash
# Start server on all interfaces
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```
* Interactive API Documentation: `http://localhost:8000/docs`
* Trip Inventory Endpoint: `GET http://localhost:8000/trips`
* Inference & Evaluation Endpoint: `POST http://localhost:8000/evaluate`

### Laptop 2: Evaluator User Interface
Provides the evaluator dashboard for selecting any of the 10 reserved student-testing trips, requesting live inference from Laptop 1, and rendering interactive trajectory maps.
```bash
# Start Streamlit evaluator client
streamlit run src/ui/app.py
```
* Enter Laptop 1's local IP address (e.g., `http://192.168.1.50:8000`) in the sidebar.
* Select any reserved trip (`00000000`, `00000007`, ..., `00000091`).
* Click **Reconstruct & Evaluate Route** to inspect real-time accuracy, continuity, and interactive Folium trajectory overlays.
