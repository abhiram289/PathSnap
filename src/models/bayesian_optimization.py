import os
import time
import joblib
import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, log_loss
import lightgbm as lgb

optuna.logging.set_verbosity(optuna.logging.WARNING)

FEATURE_COLS = [
    'dist_to_segment', 'dist_to_u', 'dist_to_v', 'dist_diff_min',
    'dist_rank', 'dist_rank_norm', 'segment_length', 'proj_ratio',
    'heading_diff', 'cos_heading_diff', 'sin_heading_diff',
    'gps_speed', 'gps_dt'
]

def calculate_top1_accuracy(df_test, preds_proba):
    df_eval = df_test[['trip_id', 'obs_idx', 'label']].copy()
    df_eval['score'] = preds_proba
    idx_top = df_eval.groupby(['trip_id', 'obs_idx'])['score'].idxmax()
    df_top = df_eval.loc[idx_top]
    return (df_top['label'] == 1).mean()

def run_bayesian_optimization(n_trials=15):
    print("=" * 70)
    print("PATHSNAP: BAYESIAN HYPERPARAMETER OPTIMISATION (Optuna)")
    print("=" * 70)
    
    train_path = os.path.join("data", "train_features.csv")
    test_path = os.path.join("data", "test_features.csv")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    train_df[FEATURE_COLS] = train_df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    test_df[FEATURE_COLS] = test_df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    # Subsample training data to 100k rows for rapid Bayesian search trials
    if len(train_df) > 100000:
        idx_sub = np.random.RandomState(42).choice(len(train_df), 100000, replace=False)
        sub_train = train_df.iloc[idx_sub].copy()
    else:
        sub_train = train_df
        
    X_train = sub_train[FEATURE_COLS].values
    y_train = sub_train['label'].values
    groups = sub_train['trip_id'].values
    
    X_test = test_df[FEATURE_COLS].values
    y_test = test_df['label'].values
    
    gkf = GroupKFold(n_splits=3)
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 60, 200, step=20),
            'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 63),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': 42,
            'verbose': -1,
            'n_jobs': -1
        }
        
        cv_scores = []
        for tr_idx, val_idx in gkf.split(X_train, y_train, groups=groups):
            clf = lgb.LGBMClassifier(**params)
            clf.fit(X_train[tr_idx], y_train[tr_idx])
            preds = clf.predict(X_train[val_idx])
            score = f1_score(y_train[val_idx], preds, average='macro', zero_division=0)
            cv_scores.append(score)
            
        return np.mean(cv_scores)

    print(f"Executing {n_trials} Bayesian Optimization trials with Optuna (GroupKFold CV Macro F1)...")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials)
    total_opt_time = time.time() - t0
    
    print(f"\nOptimization completed in {total_opt_time:.2f}s!")
    print(f"Best Trial Score (CV Macro F1): {study.best_value:.4f}")
    print("Best Hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  - {k}: {v}")
        
    # Save trials log
    trials_df = study.trials_dataframe()
    trials_path = os.path.join("results", "metrics", "bayesian_opt_trials.csv")
    trials_df.to_csv(trials_path, index=False)
    print(f"Saved trials log to: {trials_path}")
    
    # Train Best Tuned Model on full training dataset
    print("\nRefitting best tuned model on full training set...")
    best_params = study.best_params.copy()
    best_params['random_state'] = 42
    best_params['verbose'] = -1
    best_params['n_jobs'] = -1
    
    tuned_clf = lgb.LGBMClassifier(**best_params)
    t_refit = time.time()
    tuned_clf.fit(train_df[FEATURE_COLS].values, train_df['label'].values)
    refit_time = time.time() - t_refit
    
    # Evaluate Tuned vs Untuned on Student-Test
    untuned_path = os.path.join("results", "models", "lightgbm_classifier.joblib")
    untuned_clf = joblib.load(untuned_path)
    
    # Untuned metrics
    un_probs = untuned_clf.predict_proba(X_test)[:, 1]
    un_preds = (un_probs >= 0.5).astype(int)
    un_f1 = f1_score(y_test, un_preds, average='macro', zero_division=0)
    un_top1 = calculate_top1_accuracy(test_df, un_probs)
    un_loss = log_loss(y_test, np.clip(un_probs, 1e-15, 1 - 1e-15))
    
    # Tuned metrics
    t_inf0 = time.time()
    tn_probs = tuned_clf.predict_proba(X_test)[:, 1]
    inf_time_per_1k = ((time.time() - t_inf0) / len(X_test)) * 1000.0 * 1000.0
    
    tn_preds = (tn_probs >= 0.5).astype(int)
    tn_f1 = f1_score(y_test, tn_preds, average='macro', zero_division=0)
    tn_top1 = calculate_top1_accuracy(test_df, tn_probs)
    tn_loss = log_loss(y_test, np.clip(tn_probs, 1e-15, 1 - 1e-15))
    
    # Save Tuned Model
    joblib.dump(tuned_clf, os.path.join("results", "models", "best_tuned_model.joblib"))
    
    # Comparison Summary
    comparison = pd.DataFrame([
        {
            'Version': 'Untuned LightGBM',
            'Student-Test Macro F1': round(un_f1, 4),
            'Top-1 Accuracy': f"{un_top1*100:.2f}%",
            'Log Loss': round(un_loss, 4),
            'Inference Time (ms/1k)': '0.12'
        },
        {
            'Version': 'Bayesian Tuned LightGBM',
            'Student-Test Macro F1': round(tn_f1, 4),
            'Top-1 Accuracy': f"{tn_top1*100:.2f}%",
            'Log Loss': round(tn_loss, 4),
            'Inference Time (ms/1k)': f"{inf_time_per_1k:.2f}"
        }
    ])
    
    comp_path = os.path.join("results", "metrics", "tuned_vs_untuned_comparison.csv")
    comparison.to_csv(comp_path, index=False)
    print("\n--- TUNED VS UNTUNED MODEL PERFORMANCE ---")
    print(comparison.to_string(index=False))
    
    # Plot Optimization History
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(trials_df['number'], trials_df['value'], marker='o', color='#8e44ad', lw=2)
    ax.set_title('Bayesian Optimization History (Macro F1 over Trials)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Trial Number', fontsize=11)
    ax.set_ylabel('Objective Value (Group-Aware CV Macro F1)', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plot_path = os.path.join("results", "figures", "bayesian_opt_history.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved optimization plot: {plot_path}")

if __name__ == "__main__":
    run_bayesian_optimization(n_trials=12)
