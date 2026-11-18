"""
PathSnap: Generate additional result graphs from EXISTING project outputs.

This script does NOT modify existing project code, retrain models, or overwrite
existing figures/CSVs. New files are prefixed with "generated_".

Run from the PathSnap project root:
    python generate_all_result_graphs.py
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
METRICS = ROOT / "results" / "metrics"
FIGURES = ROOT / "results" / "figures"
MAPS = ROOT / "results" / "maps"
FIGURES.mkdir(parents=True, exist_ok=True)
METRICS.mkdir(parents=True, exist_ok=True)

generated = []
skipped = []

def locate_csv(name):
    p = METRICS / name
    return p if p.exists() else None

def read_csv(name):
    p = locate_csv(name)
    if p is None:
        skipped.append(f"{name}: file not found")
        return None
    try:
        df = pd.read_csv(p)
        print(f"Loaded {name}: {len(df)} rows")
        return df
    except Exception as e:
        skipped.append(f"{name}: could not read ({e})")
        return None

def find_col(df, candidates):
    """Case-insensitive exact column matching, then partial matching."""
    if df is None:
        return None
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    for candidate in candidates:
        for low, original in lookup.items():
            if candidate.lower() in low:
                return original
    return None

def save_plot(fig, filename):
    path = FIGURES / filename
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    generated.append(path)
    print(f"Saved graph: {path.relative_to(ROOT)}")

def save_table(df, filename):
    path = METRICS / filename
    df.to_csv(path, index=False)
    generated.append(path)
    print(f"Saved table: {path.relative_to(ROOT)}")

# ---------------------------------------------------------------------
# 1. Five-model Log Loss comparison (uses existing model comparison CSV)
# ---------------------------------------------------------------------
models = read_csv("model_comparison_table.csv")
if models is not None:
    model_col = find_col(models, ["Model", "Algorithm", "Classifier", "Algorithm Family"])
    loss_col = find_col(models, ["Log Loss", "Test Log Loss", "Student-Test Log Loss"])
    if model_col and loss_col:
        vals = pd.to_numeric(models[loss_col], errors="coerce")
        valid = models.loc[vals.notna(), [model_col]].copy()
        valid["Log Loss"] = vals[vals.notna()].values
        if not valid.empty:
            fig, ax = plt.subplots(figsize=(9, 5))
            bars = ax.bar(valid[model_col].astype(str), valid["Log Loss"])
            ax.set_title("Log Loss Comparison Across Models")
            ax.set_ylabel("Log Loss (lower is better)")
            ax.tick_params(axis="x", rotation=25)
            for bar, value in zip(bars, valid["Log Loss"]):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                        f"{value:.4f}", ha="center", va="bottom", fontsize=8)
            save_plot(fig, "generated_log_loss_comparison.png")
        else:
            skipped.append("Log Loss graph: no numeric Log Loss values")
    else:
        skipped.append(f"Log Loss graph: expected model/loss columns not found. Columns: {list(models.columns)}")

# Additional core model metric chart: Macro F1
if models is not None:
    model_col = find_col(models, ["Model", "Algorithm", "Classifier"])
    f1_col = find_col(models, ["Student-Test Macro F1-score", "Student-Test Macro F1", "Test Macro F1", "Macro F1"])
    if model_col and f1_col:
        vals = pd.to_numeric(models[f1_col], errors="coerce")
        if vals.notna().any():
            fig, ax = plt.subplots(figsize=(9, 5))
            bars = ax.bar(models.loc[vals.notna(), model_col].astype(str), vals.dropna())
            ax.set_title("Student-Test Macro F1 Across Models")
            ax.set_ylabel("Macro F1 (higher is better)")
            ax.set_ylim(0, 1)
            ax.tick_params(axis="x", rotation=25)
            for b, v in zip(bars, vals.dropna()):
                ax.text(b.get_x()+b.get_width()/2, b.get_height(), f"{v:.4f}",
                        ha="center", va="bottom", fontsize=8)
            save_plot(fig, "generated_test_macro_f1.png")

# ---------------------------------------------------------------------
# 2. HMM aggregate comparison, per-trip deltas, and graph
# ---------------------------------------------------------------------
hmm = read_csv("hmm_vs_classifier_comparison.csv")
if hmm is not None:
    acc_base = find_col(hmm, ["Classifier_Top1_Acc"])
    acc_hmm = find_col(hmm, ["HMM_Viterbi_Acc"])
    cont_base = find_col(hmm, ["Classifier_Continuity"])
    cont_hmm = find_col(hmm, ["HMM_Continuity"])
    trip_col = find_col(hmm, ["trip_id", "trip"])
    if all([acc_base, acc_hmm, cont_base, cont_hmm]):
        for c in [acc_base, acc_hmm, cont_base, cont_hmm]:
            hmm[c] = pd.to_numeric(hmm[c], errors="coerce")
        summary = pd.DataFrame({
            "Metric": ["Top-1 Accuracy (%)", "Route Continuity (%)"],
            "Classifier": [hmm[acc_base].mean(), hmm[cont_base].mean()],
            "HMM + Viterbi": [hmm[acc_hmm].mean(), hmm[cont_hmm].mean()]
        })
        save_table(summary, "generated_hmm_aggregate_summary.csv")

        x = np.arange(len(summary))
        width = 0.35
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x-width/2, summary["Classifier"], width, label="Classifier only")
        ax.bar(x+width/2, summary["HMM + Viterbi"], width, label="HMM + Viterbi")
        ax.set_xticks(x, summary["Metric"])
        ax.set_ylabel("Percentage")
        ax.set_title("Classifier-Only vs HMM + Viterbi")
        ax.legend()
        save_plot(fig, "generated_hmm_aggregate_comparison.png")

        detail = hmm.copy()
        detail["Accuracy change (percentage points)"] = detail[acc_hmm] - detail[acc_base]
        detail["Continuity change (percentage points)"] = detail[cont_hmm] - detail[cont_base]
        save_table(detail, "generated_hmm_per_trip_analysis.csv")

        if trip_col:
            fig, ax = plt.subplots(figsize=(11, 5))
            xx = np.arange(len(detail))
            ax.plot(xx, detail["Accuracy change (percentage points)"], marker="o",
                    label="Accuracy change")
            ax.plot(xx, detail["Continuity change (percentage points)"], marker="s",
                    label="Continuity change")
            ax.axhline(0, linewidth=1)
            ax.set_xticks(xx, detail[trip_col].astype(str), rotation=45)
            ax.set_xlabel("Trip")
            ax.set_ylabel("Change (percentage points)")
            ax.set_title("Per-Trip Change After HMM Decoding")
            ax.legend()
            save_plot(fig, "generated_hmm_per_trip_changes.png")
    else:
        skipped.append(f"HMM analysis: expected columns missing. Columns: {list(hmm.columns)}")

# ---------------------------------------------------------------------
# 3. Bayesian optimisation trial history and best-trial table
# ---------------------------------------------------------------------
trials = read_csv("bayesian_opt_trials.csv")
if trials is not None:
    value_col = find_col(trials, ["value", "Value", "objective"])
    number_col = find_col(trials, ["number", "trial", "Trial"])
    if value_col:
        trials[value_col] = pd.to_numeric(trials[value_col], errors="coerce")
        usable = trials.dropna(subset=[value_col]).copy()
        if not number_col:
            usable["_trial_number"] = np.arange(len(usable))
            number_col = "_trial_number"
        usable[number_col] = pd.to_numeric(usable[number_col], errors="coerce")
        usable = usable.dropna(subset=[number_col]).sort_values(number_col)
        if not usable.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(usable[number_col], usable[value_col], marker="o", label="Trial CV Macro F1")
            ax.plot(usable[number_col], usable[value_col].cummax(), linestyle="--",
                    label="Best-so-far CV Macro F1")
            ax.set_xlabel("Trial")
            ax.set_ylabel("Objective value")
            ax.set_title("Bayesian Optimisation Trial History")
            ax.legend()
            save_plot(fig, "generated_bayesian_trials.png")
            best = usable.loc[[usable[value_col].idxmax()]]
            save_table(best, "generated_bayesian_best_trial.csv")
    else:
        skipped.append(f"Bayesian analysis: no objective 'value' column. Columns: {list(trials.columns)}")

# ---------------------------------------------------------------------
# 4. Tuned vs untuned table and available metric charts
# ---------------------------------------------------------------------
tuned = read_csv("tuned_vs_untuned_comparison.csv")
if tuned is not None:
    version_col = find_col(tuned, ["Version", "Model", "Variant"])
    metric_candidates = [
        ("Student-Test Macro F1", ["Student-Test Macro F1", "Test Macro F1", "Macro F1"]),
        ("Top-1 Accuracy", ["Top-1 Accuracy", "Top1 Accuracy"]),
        ("Log Loss", ["Log Loss"]),
        ("Inference Time", ["Inference Time (ms/1k)", "Inference Time"])
    ]
    available = []
    for label, candidates in metric_candidates:
        col = find_col(tuned, candidates)
        if col:
            numeric = pd.to_numeric(tuned[col].astype(str).str.replace("%", "", regex=False),
                                    errors="coerce")
            if numeric.notna().any():
                available.append((label, col, numeric))
    if version_col and available:
        for label, col, numeric in available:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            valid = numeric.notna()
            ax.bar(tuned.loc[valid, version_col].astype(str), numeric[valid])
            ax.set_title(f"Tuned vs Untuned: {label}")
            ax.set_ylabel(label)
            ax.tick_params(axis="x", rotation=15)
            save_plot(fig, "generated_tuned_untuned_" +
                      "".join(ch.lower() if ch.isalnum() else "_" for ch in label).strip("_") + ".png")
    else:
        skipped.append(f"Tuned comparison: required columns unavailable. Columns: {list(tuned.columns)}")

# ---------------------------------------------------------------------
# 5. Trip-level dataset summary from existing EDA summaries
# ---------------------------------------------------------------------
for filename, partition in [("trip_summary_train.csv", "train"),
                            ("trip_summary_test.csv", "test")]:
    df = read_csv(filename)
    if df is None:
        continue
    missing_col = find_col(df, ["missing_pts", "missing_points", "missing"])
    invalid_col = find_col(df, ["invalid_coords", "invalid_coordinates", "invalid"])
    rows = []
    for label, col in [("Missing GPS points", missing_col),
                       ("Invalid coordinates", invalid_col)]:
        if col:
            values = pd.to_numeric(df[col], errors="coerce")
            rows.append({"Partition": partition, "Measure": label,
                         "Total": values.sum(), "Trips": len(df)})
    if rows:
        save_table(pd.DataFrame(rows), f"generated_data_quality_{partition}.csv")
    else:
        skipped.append(f"{filename}: missing/invalid columns not found; columns: {list(df.columns)}")

# ---------------------------------------------------------------------
# 6. Dataset inventory summary (preserve source values)
# ---------------------------------------------------------------------
inventory = read_csv("dataset_inventory.csv")
if inventory is not None:
    save_table(inventory, "generated_dataset_inventory_copy.csv")

# ---------------------------------------------------------------------
# 7. Report existing maps and image assets; no fake screenshot generation
# ---------------------------------------------------------------------
existing_maps = sorted(MAPS.glob("*.html")) if MAPS.exists() else []
existing_figures = sorted(FIGURES.glob("*.png"))
asset_rows = ([{"Type": "Folium HTML map", "Path": str(p.relative_to(ROOT))}
               for p in existing_maps] +
              [{"Type": "PNG figure", "Path": str(p.relative_to(ROOT))}
               for p in existing_figures if not p.name.startswith("generated_")])
if asset_rows:
    save_table(pd.DataFrame(asset_rows), "generated_existing_visual_assets.csv")

print("\n" + "="*68)
print("GENERATION COMPLETE")
print("="*68)
print(f"New files generated: {len(generated)}")
for p in generated:
    print("  +", p.relative_to(ROOT))
if skipped:
    print("\nNot generated (missing source data/columns):")
    for item in skipped:
        print("  -", item)
print("\nNo existing source code or original result files were modified.")
print("Note: CSV-based summaries cannot create real UI/API screenshots or training-loss curves.")
