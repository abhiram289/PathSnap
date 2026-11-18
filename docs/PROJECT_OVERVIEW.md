# PathSnap – Project Overview

## 1. Introduction

PathSnap is a machine learning-based map-matching and trajectory reconstruction system developed for the PES University Machine Learning Hackathon (UE24CS352A).

The project aims to match noisy GPS observations to the correct road segments and reconstruct vehicle routes using spatial processing, machine learning classification and Hidden Markov Model (HMM) sequence decoding.

## 2. Problem Statement

GPS devices frequently produce noisy location observations. A GPS coordinate may not lie exactly on the road being travelled because of measurement errors, signal interference and environmental conditions.

Map matching addresses this problem by identifying the most likely road segment corresponding to each GPS observation.

PathSnap combines spatial candidate generation and machine learning to improve the identification of road segments and reconstruct continuous routes.

## 3. Project Objectives

- Analyse GPS trajectory and road network datasets.
- Perform exploratory data analysis.
- Generate candidate road segments for GPS observations.
- Extract spatial and movement-related features.
- Train and compare multiple machine learning algorithms.
- Optimize model hyperparameters using Bayesian optimization.
- Apply HMM and Viterbi decoding for sequential route reconstruction.
- Provide route evaluation through an API and interactive interface.

## 4. Major Features

### 4.1 Exploratory Data Analysis

Analyses GPS sampling intervals, trip lengths, route characteristics, missing values and coordinate validity.

### 4.2 Spatial Candidate Generation

Uses a KD-tree to efficiently identify nearby road segment candidates for each GPS observation.

### 4.3 Feature Engineering

Extracts distance, segment geometry, heading, GPS speed and time-related features.

### 4.4 Machine Learning

Compares five algorithm families:

- Logistic Regression
- Distance-Weighted K-Nearest Neighbours
- Random Forest
- LightGBM
- Multi-Layer Perceptron

### 4.5 Bayesian Optimization

Uses Optuna to optimize LightGBM hyperparameters.

### 4.6 Sequential Route Reconstruction

Uses Hidden Markov Model concepts and Viterbi decoding to identify a more continuous sequence of road segments.

### 4.7 Interactive Visualization

Uses Folium maps and a Streamlit interface to display GPS trajectories and reconstructed routes.

## 5. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data manipulation |
| NumPy | Numerical computation |
| Scikit-learn | Machine learning |
| LightGBM | Gradient boosting |
| Optuna | Hyperparameter optimization |
| SciPy | KD-tree spatial search |
| NetworkX | Graph-related analysis |
| Matplotlib | Data visualization |
| Folium | Interactive maps |
| FastAPI | Backend API |
| Streamlit | User interface |
| Joblib | Model serialization |

## 6. Project Workflow

1. Load GPS trajectory and road network data.
2. Perform exploratory data analysis.
3. Project geographic coordinates into metric coordinates.
4. Generate nearby road segment candidates using KD-tree.
5. Extract candidate-level features.
6. Train multiple classification models.
7. Evaluate models using cross-validation and held-out testing data.
8. Optimize LightGBM using Optuna.
9. Apply HMM and Viterbi route reconstruction.
10. Serve predictions through FastAPI.
11. Visualize results using Streamlit and Folium.

## 7. Repository Structure

- `src/api/` – FastAPI application.
- `src/eda/` – Exploratory data analysis.
- `src/hmm/` – Sequential route decoder.
- `src/models/` – Model training and optimization.
- `src/preprocessing/` – Spatial preprocessing and feature extraction.
- `src/ui/` – Streamlit interface.
- `notebooks/` – Experimentation notebooks.
- `data/` – Preprocessed feature datasets.
- `results/figures/` – Generated visualizations.
- `results/maps/` – Interactive map outputs.
- `results/metrics/` – Evaluation metrics.
- `results/models/` – Serialized trained models.

## 8. Expected Outcome

The project provides an integrated map-matching system that uses machine learning and sequential decoding to identify road segments corresponding to noisy GPS observations and reconstruct vehicle trajectories.