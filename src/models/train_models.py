import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, confusion_matrix, classification_report, ConfusionMatrixDisplay
)

# 5 Algorithm Families
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier

FEATURE_COLS = [
    'dist_to_segment', 'dist_to_u', 'dist_to_v', 'dist_diff_min',
    'dist_rank', 'dist_rank_norm', 'segment_length', 'proj_ratio',
    'heading_diff', 'cos_heading_diff', 'sin_heading_diff',
    'gps_speed', 'gps_dt'
]

def calculate_top1_accuracy_and_continuity(df_test, preds_proba):
    # Attach predicted probabilities to test dataframe
    df_eval = df_test[['trip_id', 'obs_idx', 'arc_id', 'u', 'v', 'label']].copy()
    df_eval['score'] = preds_proba
    
    # Pick top candidate per observation
    idx_top = df_eval.groupby(['trip_id', 'obs_idx'])['score'].idxmax()
    df_top = df_eval.loc[idx_top].sort_values(['trip_id', 'obs_idx']).reset_index(drop=True)
    
    # Top-1 Accuracy: fraction of observations where the chosen candidate has label == 1
    top1_acc = (df_top['label'] == 1).mean()
    
    # Route continuity: fraction of consecutive observations whose predicted arcs share a node or are identical
    continuity_scores = []
    for tid, group in df_top.groupby('trip_id'):
        if len(group) < 2:
            continue
        u_arr = group['u'].values
        v_arr = group['v'].values
        arc_arr = group['arc_id'].values
        
        continuous_steps = 0
        total_steps = len(group) - 1
        for i in range(total_steps):
            # Identical arc or connected (v of current == u of next, or share any node)
            if arc_arr[i] == arc_arr[i+1] or v_arr[i] == u_arr[i+1] or u_arr[i] == u_arr[i+1] or v_arr[i] == v_arr[i+1]:
                continuous_steps += 1
        continuity_scores.append(continuous_steps / max(1, total_steps))
        
    avg_continuity = np.mean(continuity_scores) if continuity_scores else 0.0
    return top1_acc, avg_continuity, df_top

def train_and_evaluate_models():
    print("=" * 70)
    print("PATHSNAP: MULTI-FAMILY MODEL EXPERIMENTATION & BENCHMARKING")
    print("=" * 70)
    
    train_path = os.path.join("data", "train_features.csv")
    test_path = os.path.join("data", "test_features.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Feature datasets not found in data/. Run build_dataset.py first.")
        
    print("Loading preprocessed feature datasets...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"Train samples: {len(train_df):,} | Test samples: {len(test_df):,}")
    print(f"Features: {FEATURE_COLS}")
    
    # Handle any potential NaNs or infs safely
    train_df[FEATURE_COLS] = train_df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    test_df[FEATURE_COLS] = test_df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    X_train = train_df[FEATURE_COLS].values
    y_train = train_df['label'].values
    groups_train = train_df['trip_id'].values
    
    X_test = test_df[FEATURE_COLS].values
    y_test = test_df['label'].values
    
    # Subsample training data if very large for instant responsive training across all 5 models
    # We maintain group integrity
    unique_trips = np.unique(groups_train)
    print(f"Total training trips available: {len(unique_trips)}")
    
    # Standard Scaler for models sensitive to scale (Logistic Regression, KNN, MLP)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    os.makedirs(os.path.join("results", "models"), exist_ok=True)
    joblib.dump(scaler, os.path.join("results", "models", "scaler.joblib"))
    
    # Define 5 models from distinct families
    models = {
        'Logistic Regression': {
            'family': 'Linear and probabilistic',
            'config': 'C=1.0, max_iter=500, solver=lbfgs',
            'model': LogisticRegression(C=1.0, max_iter=500, random_state=42, solver='lbfgs'),
            'use_scaled': True
        },
        'Distance-Weighted KNN': {
            'family': 'Instance-based',
            'config': 'n_neighbors=9, weights=distance, algorithm=kd_tree',
            # Train KNN on a representative subsample of 30,000 points to keep inference fast
            'model': KNeighborsClassifier(n_neighbors=9, weights='distance', n_jobs=-1, algorithm='kd_tree'),
            'use_scaled': True,
            'subsample_knn': 35000
        },
        'Random Forest': {
            'family': 'Ensemble / Bagging',
            'config': 'n_estimators=100, max_depth=12, min_samples_split=5',
            'model': RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_split=5, random_state=42, n_jobs=-1),
            'use_scaled': False
        },
        'LightGBM Classifier': {
            'family': 'Boosting',
            'config': 'num_leaves=31, learning_rate=0.08, n_estimators=120',
            'model': lgb.LGBMClassifier(num_leaves=31, learning_rate=0.08, n_estimators=120, random_state=42, verbose=-1, n_jobs=-1),
            'use_scaled': False
        },
        'Multi-Layer Perceptron (MLP)': {
            'family': 'Neural networks',
            'config': 'hidden_layer_sizes=(64, 32), activation=relu, max_iter=40',
            'model': MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', max_iter=40, random_state=42, early_stopping=True),
            'use_scaled': True
        }
    }
    
    results = []
    test_predictions = {}
    test_probas = {}
    
    # 3-Fold Group-Aware Cross Validation
    gkf = GroupKFold(n_splits=3)
    
    print("\nStarting Model Training, GroupKFold CV, and Student-Testing Evaluation...")
    for name, item in models.items():
        print(f"\n---> Training Model: {name} (Family: {item['family']})")
        clf = item['model']
        X_tr = X_train_scaled if item['use_scaled'] else X_train
        X_te = X_test_scaled if item['use_scaled'] else X_test
        
        # Subsample for KNN if needed
        if item.get('subsample_knn', None) and len(X_tr) > item['subsample_knn']:
            idx_sub = np.random.RandomState(42).choice(len(X_tr), item['subsample_knn'], replace=False)
            X_tr_fit = X_tr[idx_sub]
            y_tr_fit = y_train[idx_sub]
            groups_fit = groups_train[idx_sub]
        else:
            X_tr_fit = X_tr
            y_tr_fit = y_train
            groups_fit = groups_train
            
        # Group-aware CV
        cv_f1s = []
        for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_tr_fit, y_tr_fit, groups=groups_fit)):
            clf_cv = type(clf)(**clf.get_params())
            clf_cv.fit(X_tr_fit[tr_idx], y_tr_fit[tr_idx])
            val_preds = clf_cv.predict(X_tr_fit[val_idx])
            f1_val = f1_score(y_tr_fit[val_idx], val_preds, average='macro', zero_division=0)
            cv_f1s.append(f1_val)
        cv_macro_f1 = np.mean(cv_f1s)
        print(f"  CV Macro F1 (3-Fold Group-Aware): {cv_macro_f1:.4f}")
        
        # Train on full training set
        t0 = time.time()
        clf.fit(X_tr_fit, y_tr_fit)
        train_time = time.time() - t0
        print(f"  Trained model in {train_time:.2f}s")
        
        # Measure Inference Time on 5,000 points
        sample_size = min(5000, len(X_te))
        t_inf0 = time.time()
        _ = clf.predict_proba(X_te[:sample_size])
        inf_time_per_1k = ((time.time() - t_inf0) / sample_size) * 1000.0 * 1000.0 # ms per 1k points
        
        # Evaluate on full student-test set
        probs = clf.predict_proba(X_te)[:, 1]
        preds = (probs >= 0.5).astype(int)
        test_predictions[name] = preds
        test_probas[name] = probs
        
        # Save model
        save_name = name.lower().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
        joblib.dump(clf, os.path.join("results", "models", f"{save_name}.joblib"))
        
        # Compute metrics
        test_macro_f1 = f1_score(y_test, preds, average='macro', zero_division=0)
        test_precision = precision_score(y_test, preds, zero_division=0)
        test_recall = recall_score(y_test, preds, zero_division=0)
        loss = log_loss(y_test, np.clip(probs, 1e-15, 1 - 1e-15))
        
        top1_acc, continuity, _ = calculate_top1_accuracy_and_continuity(test_df, probs)
        
        print(f"  Student-Test Macro F1: {test_macro_f1:.4f}")
        print(f"  Observation-Level Top-1 Accuracy: {top1_acc*100:.2f}%")
        print(f"  Route Continuity: {continuity*100:.2f}%")
        print(f"  Inference Time: {inf_time_per_1k:.2f} ms/1,000 points")
        
        results.append({
            'Model': name,
            'Algorithm Family': item['family'],
            'Key Configuration': item['config'],
            'CV Macro F1-score': round(cv_macro_f1, 4),
            'Top-1 Accuracy': f"{top1_acc*100:.2f}%",
            'Route continuity': f"{continuity*100:.2f}%",
            'Student-Test Macro F1-score': round(test_macro_f1, 4),
            'Log Loss': round(loss, 4),
            'Precision': round(test_precision, 4),
            'Recall': round(test_recall, 4),
            'Inference Time (ms/1k)': round(inf_time_per_1k, 2),
            'Training Time (s)': round(train_time, 2),
            'raw_top1': top1_acc,
            'raw_cv_f1': cv_macro_f1,
            'raw_test_f1': test_macro_f1,
            'raw_log_loss': loss
        })
        
    results_df = pd.DataFrame(results)
    
    # Save Consolidated Comparison Table
    table_path = os.path.join("results", "metrics", "model_comparison_table.csv")
    results_df.drop(columns=['raw_top1', 'raw_cv_f1', 'raw_test_f1', 'raw_log_loss']).to_csv(table_path, index=False)
    print("\n" + "=" * 70)
    print("CONSOLIDATED MODEL COMPARISON TABLE")
    print("=" * 70)
    print(results_df[['Model', 'Algorithm Family', 'CV Macro F1-score', 'Top-1 Accuracy', 'Route continuity', 'Student-Test Macro F1-score', 'Log Loss', 'Inference Time (ms/1k)']].to_string(index=False))
    
    # -------------------------------------------------------------
    # Visualization Plots for Model Comparison
    # -------------------------------------------------------------
    print("\nGenerating model comparison diagnostic plots...")
    
    # 1. Macro F1 & Top-1 Accuracy Comparison Bar Chart
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    model_names = [r['Model'] for r in results]
    x_pos = np.arange(len(model_names))
    
    cv_f1s = [r['raw_cv_f1'] for r in results]
    test_f1s = [r['raw_test_f1'] for r in results]
    top1_accs = [r['raw_top1'] * 100 for r in results]
    
    axes[0].bar(x_pos - 0.2, cv_f1s, width=0.4, label='CV Macro F1', color='#3498db', edgecolor='black')
    axes[0].bar(x_pos + 0.2, test_f1s, width=0.4, label='Student-Test Macro F1', color='#2ecc71', edgecolor='black')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(model_names, rotation=25, ha='right', fontsize=10)
    axes[0].set_ylabel('Macro F1-score', fontsize=11)
    axes[0].set_title('Cross-Validation vs Student-Test Macro F1', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    axes[1].bar(x_pos, top1_accs, color='#e74c3c', edgecolor='black', width=0.5)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(model_names, rotation=25, ha='right', fontsize=10)
    axes[1].set_ylabel('Top-1 Accuracy (%)', fontsize=11)
    axes[1].set_title('Observation-level Top-1 Accuracy on Student-Test', fontsize=12, fontweight='bold')
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    comp_fig_path = os.path.join("results", "figures", "model_comparison_metrics.png")
    plt.savefig(comp_fig_path, dpi=300)
    plt.close()
    print(f"Saved: {comp_fig_path}")
    
    # 2. Confusion Matrices
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    for idx, (m_name, p_vals) in enumerate(test_predictions.items()):
        cm = confusion_matrix(y_test, p_vals)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        disp.plot(ax=axes[idx], cmap='Blues', colorbar=False)
        axes[idx].set_title(m_name, fontsize=10, fontweight='bold')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('True' if idx == 0 else '')
    plt.tight_layout()
    cm_fig_path = os.path.join("results", "figures", "confusion_matrices.png")
    plt.savefig(cm_fig_path, dpi=300)
    plt.close()
    print(f"Saved: {cm_fig_path}")
    
    print("\nModel Training and Evaluation Completed Successfully!")

if __name__ == "__main__":
    train_and_evaluate_models()
