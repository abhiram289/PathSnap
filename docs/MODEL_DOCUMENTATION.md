# PathSnap – Machine Learning Model Documentation

## 1. Introduction

PathSnap uses supervised machine learning to identify the most likely road segment corresponding to each GPS observation.

The model training implementation is located in:

`src/models/train_models.py`

Five machine learning algorithm families are compared.

## 2. Problem Formulation

For each GPS observation, multiple nearby road segment candidates are generated.

Each candidate is represented using 13 engineered features.

The model predicts the probability that a candidate belongs to the ground-truth route.

This is formulated as a binary classification problem:

- Label 1: Correct road segment.
- Label 0: Incorrect road segment.

The candidate with the highest predicted probability is selected as the Top-1 prediction for each observation.

## 3. Features Used

The model uses the following 13 features:

1. dist_to_segment
2. dist_to_u
3. dist_to_v
4. dist_diff_min
5. dist_rank
6. dist_rank_norm
7. segment_length
8. proj_ratio
9. heading_diff
10. cos_heading_diff
11. sin_heading_diff
12. gps_speed
13. gps_dt

## 4. Machine Learning Models

### 4.1 Logistic Regression

Algorithm family: Linear and probabilistic.

Logistic Regression estimates the probability of a candidate being the correct road segment.

It is used as a computationally efficient baseline.

Configuration:
- C = 1.0
- max_iter = 500
- solver = lbfgs

### 4.2 Distance-Weighted K-Nearest Neighbours

Algorithm family: Instance-based learning.

KNN classifies candidates using nearby training observations.

Distance weighting gives closer neighbours more influence.

Configuration:
- n_neighbors = 9
- weights = distance
- algorithm = kd_tree

The training implementation uses a representative subsample of up to 35,000 training points.

### 4.3 Random Forest

Algorithm family: Ensemble learning / Bagging.

Random Forest combines predictions from multiple decision trees.

It can capture nonlinear relationships between spatial and movement features.

Configuration:
- n_estimators = 100
- max_depth = 12
- min_samples_split = 5

### 4.4 LightGBM

Algorithm family: Gradient Boosting.

LightGBM builds decision trees sequentially, where each new tree improves the predictions of the previous ensemble.

It is efficient for structured tabular datasets.

Configuration:
- num_leaves = 31
- learning_rate = 0.08
- n_estimators = 120

### 4.5 Multi-Layer Perceptron

Algorithm family: Neural Networks.

MLP learns nonlinear relationships using interconnected neural network layers.

Configuration:
- hidden_layer_sizes = (64, 32)
- activation = relu
- max_iter = 40
- early_stopping = True

## 5. Feature Scaling

StandardScaler is applied to models sensitive to feature magnitudes:

- Logistic Regression
- KNN
- MLP

Random Forest and LightGBM use the unscaled feature values in the training implementation.

## 6. Model Evaluation

The project uses GroupKFold with three splits.

Grouping is performed using trip IDs to reduce leakage between observations from the same trip.

Models are evaluated on held-out student-testing data.

The evaluation metrics include:

### Macro F1-score

Measures the harmonic mean of precision and recall for each class, giving equal importance to both classes.

### Top-1 Accuracy

Measures the percentage of GPS observations for which the highest-scoring candidate is the correct road segment.

### Route Continuity

Measures the percentage of consecutive predicted road segments that are identical or share a connected road network node.

### Log Loss

Measures the quality of predicted probabilities. Lower values indicate better probabilistic predictions.

### Precision and Recall

Precision measures how many predicted positive candidates are correct.

Recall measures how many actual positive candidates are identified.

### Inference Time

Measures prediction speed and helps compare computational efficiency.

## 7. Model Comparison

Results from:

`results/metrics/model_comparison_table.csv`

| Model | Test Macro F1 | Top-1 Accuracy | Route Continuity |
|---|---:|---:|---:|
| Logistic Regression | 0.8667 | 75.18% | 91.72% |
| Distance-Weighted KNN | 0.8494 | 73.13% | 91.03% |
| Random Forest | 0.8766 | 76.15% | 92.86% |
| LightGBM | 0.8738 | 75.53% | 91.73% |
| MLP | 0.8754 | 75.94% | 92.63% |

## 8. Best Performing Model

Random Forest achieved the highest student-test Macro F1-score of 0.8766.

It also achieved:

- Top-1 Accuracy: 76.15%
- Route Continuity: 92.86%

Therefore, Random Forest performed best on these two principal classification and continuity metrics in the reported comparison.

LightGBM, however, offers a competitive alternative with substantially faster inference than Random Forest in the reported benchmark.

The best model depends on whether predictive performance or computational efficiency is prioritized.

## 9. Bayesian Hyperparameter Optimization

The project uses Optuna to optimize LightGBM.

The optimization implementation is located in:

`src/models/bayesian_optimization.py`

The search uses the Tree-structured Parzen Estimator (TPE) approach.

The documented optimization uses 12 trials.

Parameters explored include:

- num_leaves
- max_depth
- learning_rate
- min_child_samples
- subsample
- colsample_bytree
- reg_alpha
- reg_lambda

The best tuned LightGBM results are:

| Metric | Untuned | Tuned |
|---|---:|---:|
| Student-Test Macro F1 | 0.8738 | 0.8769 |
| Top-1 Accuracy | 75.53% | 76.20% |
| Log Loss | 0.1236 | 0.1042 |

The tuned model improves the reported Macro F1-score, Top-1 Accuracy and Log Loss.

## 10. HMM and Viterbi Decoding

After candidate classification, the project applies sequential route reconstruction.

The HMM decoder is implemented in:

`src/hmm/decoder.py`

The Hidden Markov Model considers relationships between consecutive road segment candidates.

The Viterbi algorithm finds a high-probability sequence of hidden states.

In this application:

- Observations correspond to GPS measurements.
- Hidden states correspond to road segment candidates.
- Emission probabilities represent how well a candidate explains a GPS observation.
- Transition probabilities represent the plausibility of moving between consecutive road segments.

This helps improve route consistency compared with selecting the highest-scoring candidate independently at every observation.

## 11. Saved Model Artifacts

The trained models are stored in:

`results/models/`

Important files include:

- logistic_regression.joblib
- distance_weighted_knn.joblib
- random_forest.joblib
- lightgbm_classifier.joblib
- multi_layer_perceptron_mlp.joblib
- best_tuned_model.joblib
- scaler.joblib

## 12. Conclusion

PathSnap compares multiple machine learning approaches for GPS map matching.

Random Forest achieves the highest Macro F1-score in the reported baseline comparison, while Bayesian optimization improves the LightGBM model.

HMM and Viterbi decoding add sequential reasoning to improve route reconstruction.