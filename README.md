# PathSnap: Map-Matching & Trajectory Reconstruction

GPS map-matching and route reconstruction project for the PES University Machine Learning Hackathon (UE24CS352A). PathSnap takes noisy GPS traces and matches them to a road network using spatial candidate generation, multi-model classification, Optuna hyperparameter tuning, and an HMM/Viterbi decoder.

## Repository Structure

```
PathSnap/
├── data/
│   ├── train_features.parquet          # Training candidate features (508,340 rows)
│   ├── train_features.csv.gz
│   ├── test_features.parquet           # Testing candidate features (191,600 rows)
│   └── test_features.csv.gz
├── notebooks/
│   ├── 01_EDA_and_Visualization.ipynb  # EDA, trajectory plots, Folium maps
│   └── 02_Model_Training_and_HMM_Reconstruction.ipynb
├── results/
│   ├── figures/                        # EDA, model comparison, confusion matrices, Optuna plots
│   ├── maps/                           # Interactive Folium HTML maps
│   ├── metrics/                        # Dataset inventory, model benchmarks, Optuna & HMM logs
│   └── models/                         # Saved scalers and trained .joblib models
├── src/
│   ├── api/main.py                     # FastAPI inference & evaluation backend
│   ├── ui/app.py                       # Streamlit evaluator UI
│   ├── eda/explore.py                  # EDA summary & Folium map generation
│   ├── preprocessing/
│   │   ├── pipeline.py                 # Metric projection, KDTree, feature extraction
│   │   └── build_dataset.py            # Batch dataset builder
│   ├── models/
│   │   ├── train_models.py             # 5-family training & GroupKFold evaluation
│   │   └── bayesian_optimization.py    # Optuna TPE tuning
│   └── hmm/decoder.py                  # HMM transitions & Viterbi decoding
├── requirements.txt
└── README.md
```

## EDA Summary

From inspecting all 90 training trips and 10 student-testing trips:
- **Sampling intervals**: Median interval is 1.00s, with 98.4% of steps falling in [0.8s, 1.2s] and occasional dropouts up to 4.5s.
- **Coordinate projection**: Raw coordinates are in WGS84 degrees (~55.4 deg lat, ~37.2 deg lon), where degree Euclidean distance is distorted ($\cos(55.4^\circ) \approx 0.5678$). Coordinates are projected to local meters using an equirectangular projection centered at the dataset centroid ($R = 6,371,000\text{ m}$).
- **Candidate generation**: Road arcs are indexed with `scipy.spatial.cKDTree` in projected meters. Using $K=5$ candidates gives 98.86% candidate recall while keeping the training table at 508,340 rows.

## Model Comparison

Models were evaluated using 3-fold `GroupKFold` (grouped by `trip_id` to avoid trip leakage) and tested on the 10 held-out student-testing trips:

| Algorithm Family | Model | Key Configuration | CV Macro F1 | Obs Top-1 Acc | Route Continuity | Test Macro F1 | Log Loss | Inference Speed |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Linear / Probabilistic | Logistic Regression | C=1.0, L-BFGS, Scaled | 0.7992 | 75.18% | 91.72% | 0.8667 | 0.2014 | 0.69 ms/1k |
| Instance-Based | KNN (Distance-Weighted) | k=9, Distance Weights | 0.7917 | 73.13% | 91.03% | 0.8494 | 0.3821 | 172.73 ms/1k |
| Ensemble (Bagging) | Random Forest | 100 Trees, Max Depth 12 | 0.8200 | 76.15% | 92.86% | 0.8766 | 0.1842 | 38.69 ms/1k |
| Gradient Boosting | LightGBM Classifier | 120 Trees, Leaves 31, LR 0.08 | 0.8177 | 75.53% | 91.73% | 0.8738 | 0.1795 | 3.32 ms/1k |
| Neural Network | Multi-Layer Perceptron | Hidden (64, 32), ReLU, Adam | 0.8085 | 74.92% | 91.50% | 0.8681 | 0.1980 | 1.85 ms/1k |

An additional 10-algorithm comparison (including Ridge, SGD, Extra Trees, Gradient Boosting, AdaBoost, and Linear SVC) is logged in `results/metrics/algorithm_exploration_tournament.csv`.

## Bayesian Optimization (Optuna)

LightGBM was tuned with Optuna's TPE sampler over 12 trials across `num_leaves` [15, 127], `max_depth` [4, 14], `learning_rate` [0.01, 0.20], `min_child_samples` [10, 100], `subsample` [0.6, 1.0], `colsample_bytree` [0.6, 1.0], `reg_alpha` [1e-8, 10.0], and `reg_lambda` [1e-8, 10.0]:

- Best trial: Trial 10
- Test Macro F1: `0.8738` -> `0.8769`
- Observation Top-1 Accuracy: `75.53%` -> `76.20%`
- Binary Log Loss: `0.1795` -> `0.1042`

The tuned model is saved at `results/models/best_tuned_model.joblib`.

## Folium Maps

Interactive HTML maps in `results/maps/` (`interactive_map_trip_00000001.html` and `interactive_map_trip_00000000_test.html`) plot raw GPS points, candidate segments, and ground-truth/predicted routes using CartoDB Voyager and Esri WorldStreetMap tiles.

## Setup and Running

```bash
git clone https://github.com/abhiram289/PathSnap.git
cd PathSnap
pip install -r requirements.txt
```

Run the pipeline steps:

```bash
# 1. EDA and Folium maps
python src/eda/explore.py

# 2. Candidate dataset generation
python src/preprocessing/build_dataset.py

# 3. Train and evaluate all 5 model families
python src/models/train_models.py

# 4. Run Optuna hyperparameter tuning
python src/models/bayesian_optimization.py

# 5. Evaluate HMM + Viterbi route decoding
python src/hmm/decoder.py
```

## Two-Laptop Deployment

**Laptop 1 (Model & FastAPI Server):**

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

- Docs: `http://localhost:8000/docs`
- List trips: `GET /trips`
- Evaluate trip: `POST /evaluate`

**Laptop 2 (Streamlit UI):**

```bash
streamlit run src/ui/app.py
```

Enter Laptop 1's LAN IP (e.g., `http://192.168.1.50:8000`) in the sidebar, pick a trip (`00000000` through `00000091`), and click **Reconstruct & Evaluate Route** to view accuracy, continuity, and the Folium map.
