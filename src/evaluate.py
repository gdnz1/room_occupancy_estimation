"""
Model training, evaluation, ablation experiments, and metrics export module.
"""
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src import config
from src.data_loader import load_raw_data, temporal_train_test_split, stratified_random_split
from src.feature_engineering import build_full_feature_dataset, get_feature_sets
from src.models import get_model_zoo, get_decision_tree_pipeline, get_logistic_regression_pipeline
from src.visualization import (
    plot_confusion_matrices,
    plot_feature_importance,
    plot_feature_set_ablation,
    plot_temporal_vs_random_split
)


def evaluate_model_pipeline(
    pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Any]:
    """
    Fit pipeline on training data, predict on test data, and compute comprehensive multiclass metrics.
    """
    # Fit pipeline (scaling is fit only on train)
    pipeline.fit(X_train, y_train)
    
    # Predict
    y_pred = pipeline.predict(X_test)
    y_train_pred = pipeline.predict(X_train)
    
    acc = accuracy_score(y_test, y_pred)
    train_acc = accuracy_score(y_train, y_train_pred)
    
    macro_prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    
    weighted_prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    weighted_rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    
    cm = confusion_matrix(y_test, y_pred, labels=config.TARGET_CLASSES)
    
    return {
        "train_accuracy": train_acc,
        "accuracy": acc,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_prec,
        "weighted_recall": weighted_rec,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm,
        "fitted_pipeline": pipeline,
        "y_pred": y_pred
    }


def run_experiment_algorithm_comparison(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Experiment 1: Benchmark baseline algorithms on Feature Set A (All Sensors) under temporal split.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: BASELINE ALGORITHMS BENCHMARK (FEATURE SET A - TEMPORAL SPLIT)")
    print("=" * 70)
    
    features = config.FEATURE_SET_A
    X_train = train_df[features]
    y_train = train_df[config.TARGET_COL]
    X_test = test_df[features]
    y_test = test_df[config.TARGET_COL]
    
    model_zoo = get_model_zoo()
    results_list = []
    cm_dict = {}
    fitted_models = {}
    
    for name, pipeline in model_zoo.items():
        eval_res = evaluate_model_pipeline(pipeline, X_train, y_train, X_test, y_test)
        
        results_list.append({
            "Model": name,
            "Feature Set": "Feature Set A (All Sensors)",
            "Train Acc": round(eval_res["train_accuracy"], 4),
            "Accuracy": round(eval_res["accuracy"], 4),
            "Macro Prec": round(eval_res["macro_precision"], 4),
            "Macro Recall": round(eval_res["macro_recall"], 4),
            "Macro F1": round(eval_res["macro_f1"], 4),
            "Weighted F1": round(eval_res["weighted_f1"], 4)
        })
        
        cm_dict[name] = eval_res["confusion_matrix"]
        fitted_models[name] = eval_res["fitted_pipeline"]
        
        print(f"[{name:<24}] Test Acc: {eval_res['accuracy']*100:6.2f}% | Macro F1: {eval_res['macro_f1']*100:6.2f}% | Weighted F1: {eval_res['weighted_f1']*100:6.2f}%")
        
    results_df = pd.DataFrame(results_list).sort_values("Macro F1", ascending=False).reset_index(drop=True)
    return results_df, cm_dict, fitted_models


def run_experiment_feature_set_ablation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Experiment 2: Feature Set Ablation using Decision Tree (depth=5) across Sets A, B, C, D, E.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: FEATURE SET ABLATION STUDY (DECISION TREE DEPTH=5)")
    print("=" * 70)
    
    feature_sets = get_feature_sets()
    ablation_list = []
    fitted_ablation_models = {}
    
    for set_name, feature_cols in feature_sets.items():
        X_train = train_df[feature_cols]
        y_train = train_df[config.TARGET_COL]
        X_test = test_df[feature_cols]
        y_test = test_df[config.TARGET_COL]
        
        pipeline = get_decision_tree_pipeline(max_depth=5)
        eval_res = evaluate_model_pipeline(pipeline, X_train, y_train, X_test, y_test)
        
        ablation_list.append({
            "Feature Set": set_name,
            "Num Features": len(feature_cols),
            "Accuracy": round(eval_res["accuracy"], 4),
            "Macro Precision": round(eval_res["macro_precision"], 4),
            "Macro Recall": round(eval_res["macro_recall"], 4),
            "Macro F1": round(eval_res["macro_f1"], 4),
            "Weighted F1": round(eval_res["weighted_f1"], 4)
        })
        fitted_ablation_models[set_name] = eval_res["fitted_pipeline"]
        print(f"[{set_name:<36} ({len(feature_cols):2d} feats)] Acc: {eval_res['accuracy']*100:6.2f}% | Macro F1: {eval_res['macro_f1']*100:6.2f}%")
        
    ablation_df = pd.DataFrame(ablation_list)
    return ablation_df, fitted_ablation_models


def run_experiment_split_comparison(
    full_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Experiment 3: Compare Chronological (Temporal) Split vs. Stratified Random Split to evaluate leakage impact.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: TEMPORAL SPLIT VS. RANDOM STRATIFIED SPLIT (LEAKAGE DEMO)")
    print("=" * 70)
    
    features = config.FEATURE_SET_A
    target = config.TARGET_COL
    
    # 1. Temporal Split
    train_t, test_t = temporal_train_test_split(full_df, save_to_disk=False)
    pipe_t = get_decision_tree_pipeline(max_depth=5)
    res_t = evaluate_model_pipeline(pipe_t, train_t[features], train_t[target], test_t[features], test_t[target])
    
    # 2. Stratified Random Split
    train_r, test_r = stratified_random_split(full_df)
    pipe_r = get_decision_tree_pipeline(max_depth=5)
    res_r = evaluate_model_pipeline(pipe_r, train_r[features], train_r[target], test_r[features], test_r[target])
    
    split_df = pd.DataFrame([
        {
            "Split Strategy": "Chronological (Temporal 80/20)",
            "Accuracy": round(res_t["accuracy"], 4),
            "Macro F1": round(res_t["macro_f1"], 4),
            "Weighted F1": round(res_t["weighted_f1"], 4),
            "Notes": "Realistic temporal generalization (past -> future)"
        },
        {
            "Split Strategy": "Stratified Random (Random 80/20)",
            "Accuracy": round(res_r["accuracy"], 4),
            "Macro F1": round(res_r["macro_f1"], 4),
            "Weighted F1": round(res_r["weighted_f1"], 4),
            "Notes": "Artificially inflated due to temporal leakage between 30s steps"
        }
    ])
    
    for _, row in split_df.iterrows():
        print(f"[{row['Split Strategy']:<32}] Acc: {row['Accuracy']*100:6.2f}% | Macro F1: {row['Macro F1']*100:6.2f}%")
        
    return split_df


def execute_full_evaluation() -> None:
    """
    Execute all experiments, produce figures, metrics tables, and serialize trained models.
    """
    # 1. Load and build processed dataset
    print("Loading dataset and building engineered features...")
    raw = load_raw_data()
    full_df = build_full_feature_dataset(raw)
    train_df, test_df = temporal_train_test_split(full_df)
    
    # 2. Experiment 1: Algorithm Comparison
    model_results_df, cm_dict, fitted_models = run_experiment_algorithm_comparison(train_df, test_df)
    model_results_df.to_csv(config.METRICS_DIR / "model_comparison_table.csv", index=False)
    print(f"\nSaved model comparison table to: {config.METRICS_DIR / 'model_comparison_table.csv'}")
    
    # 3. Experiment 2: Feature Set Ablation
    ablation_df, fitted_ablation_models = run_experiment_feature_set_ablation(train_df, test_df)
    ablation_df.to_csv(config.METRICS_DIR / "ablation_results.csv", index=False)
    print(f"Saved ablation results to: {config.METRICS_DIR / 'ablation_results.csv'}")
    
    # 4. Experiment 3: Split Strategy Comparison
    split_df = run_experiment_split_comparison(full_df)
    split_df.to_csv(config.METRICS_DIR / "split_comparison.csv", index=False)
    print(f"Saved split comparison to: {config.METRICS_DIR / 'split_comparison.csv'}")
    
    # 5. Export Key Figures
    print("\nGenerating evaluation and comparison figures...")
    
    # Confusion Matrices for representative models
    selected_cms = {
        "Logistic Regression": cm_dict["Logistic Regression"],
        "KNN (k=5)": cm_dict["KNN (k=5)"],
        "Decision Tree (depth=5)": cm_dict["Decision Tree (depth=5)"],
        "Decision Tree (depth=7)": cm_dict["Decision Tree (depth=7)"]
    }
    plot_confusion_matrices(selected_cms)
    
    # Feature Importance from Decision Tree (depth=5) on Set A
    best_dt_pipe = fitted_models["Decision Tree (depth=5)"]
    dt_classifier = best_dt_pipe.named_steps["classifier"]
    plot_feature_importance(config.FEATURE_SET_A, dt_classifier.feature_importances_)
    
    # Feature Set Ablation Figure
    plot_feature_set_ablation(ablation_df)
    
    # Temporal vs Random Split Figure
    plot_temporal_vs_random_split(split_df)
    
    # 6. Save Best Models to outputs/models/
    print("\nSaving trained models to outputs/models/ ...")
    for name, pipe in fitted_models.items():
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("=", "_")
        model_file = config.MODELS_DIR / f"{safe_name}.joblib"
        joblib.dump(pipe, model_file)
        
    print("All Phase 5 experiments and artifacts successfully completed!")


if __name__ == "__main__":
    execute_full_evaluation()
