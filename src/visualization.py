"""
Visualization module for generating publication-quality figures and plots for Room Occupancy Estimation.
"""
from typing import Optional, List, Dict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from src import config

# Set global publication styling
plt.rcParams.update({
    "font.sans-serif": "DejaVu Sans",
    "font.family": "sans-serif",
    "figure.titlesize": 14,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.autolayout": True,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--"
})

PALETTE = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77"]


def plot_class_distribution(df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """
    Plot target class distribution (Room_Occupancy_Count) with counts and percentages.
    """
    counts = df[config.TARGET_COL].value_counts().sort_index()
    total = len(df)
    
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    bars = ax.bar(counts.index.astype(str), counts.values, color=PALETTE, edgecolor="black", alpha=0.85, width=0.55)
    
    for bar in bars:
        height = bar.get_height()
        pct = (height / total) * 100
        ax.annotate(f"{height:,}\n({pct:.1f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
        
    ax.set_title("Room Occupancy Target Class Distribution (0–3 Persons)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Occupants (Room_Occupancy_Count)", fontsize=11, labelpad=8)
    ax.set_ylabel("Observation Count", fontsize=11, labelpad=8)
    ax.set_ylim(0, max(counts.values) * 1.18)
    
    out_file = save_path or (config.FIGURES_DIR / "01_class_distribution.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def plot_sensor_distributions(df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """
    Plot multi-panel distributions (histograms + KDE) for key sensor modalities.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=300)
    
    modality_specs = [
        ("S1_Temp", "Temperature (°C) - Sensor 1", "#e74c3c"),
        ("S1_Light", "Light (Lux) - Sensor 1", "#f39c12"),
        ("S1_Sound", "Sound (Volts) - Sensor 1", "#2980b9"),
        ("S5_CO2", "CO2 Concentration (ppm)", "#27ae60"),
        ("S5_CO2_Slope", "CO2 Slope (Rate of Change)", "#8e44ad"),
        ("S6_PIR", "PIR Motion Sensor 6 (Binary Activity)", "#34495e")
    ]
    
    for ax, (col, title, color) in zip(axes.flatten(), modality_specs):
        if col == "S6_PIR":
            val_counts = df[col].value_counts().sort_index()
            ax.bar(val_counts.index.astype(str), val_counts.values, color=color, edgecolor="black", alpha=0.8, width=0.4)
            for x, y in zip(val_counts.index.astype(str), val_counts.values):
                ax.text(x, y + len(df)*0.02, f"{y:,}", ha="center", fontweight="bold", fontsize=9)
            ax.set_xlabel("PIR State (0: No Motion, 1: Motion)")
            ax.set_ylabel("Count")
        else:
            sns.histplot(df[col], kde=True, ax=ax, color=color, alpha=0.6, edgecolor="black", bins=30)
            ax.set_xlabel(title)
            ax.set_ylabel("Density / Count")
        ax.set_title(title, fontweight="bold", fontsize=11)
        
    plt.suptitle("Distribution of Environmental & Motion Sensor Channels", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    out_file = save_path or (config.FIGURES_DIR / "02_sensor_distributions.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def plot_sensor_vs_occupancy_boxplots(df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """
    Plot boxplots of primary sensor readings grouped by room occupancy count.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    
    pairs = [
        ("S1_Temp", "Temperature (°C) vs. Occupancy", "Temperature (°C)"),
        ("S1_Light", "Light (Lux) vs. Occupancy", "Light Intensity (Lux)"),
        ("S1_Sound", "Sound (Volts) vs. Occupancy", "Sound (Volts)"),
        ("S5_CO2", "CO2 (ppm) vs. Occupancy", "CO2 Concentration (ppm)")
    ]
    
    for ax, (col, title, ylabel) in zip(axes.flatten(), pairs):
        sns.boxplot(
            data=df,
            x=config.TARGET_COL,
            y=col,
            hue=config.TARGET_COL,
            legend=False,
            palette=PALETTE,
            ax=ax,
            boxprops=dict(alpha=0.85, edgecolor="black"),
            flierprops=dict(marker='o', markersize=3, alpha=0.4)
        )
        ax.set_title(title, fontweight="bold", fontsize=12)
        ax.set_xlabel("Room Occupancy Count (Persons)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        
    plt.suptitle("Sensor Response Across Different Room Occupancy Levels", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    
    out_file = save_path or (config.FIGURES_DIR / "03_sensor_vs_occupancy_boxplots.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def plot_correlation_heatmap(df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """
    Plot correlation matrix heatmap for numeric features and target variable.
    """
    numeric_cols = config.FEATURE_SET_A + [config.TARGET_COL]
    corr = df[numeric_cols].corr()
    
    plt.figure(figsize=(14, 11), dpi=300)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient"}
    )
    plt.title("Correlation Matrix of Environmental Features & Room Occupancy", fontsize=14, fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    
    out_file = save_path or (config.FIGURES_DIR / "04_correlation_heatmap.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def plot_temporal_trend(df: pd.DataFrame, sample_days: int = 2, save_path: Optional[str] = None) -> None:
    """
    Plot time series of CO2, Light, and Occupancy over a representative time window.
    """
    df_temp = df.copy()
    df_temp["Datetime"] = pd.to_datetime(df_temp["Date"] + " " + df_temp["Time"], format="%Y/%m/%d %H:%M:%S")
    df_temp = df_temp.sort_values("Datetime").reset_index(drop=True)
    subset = df_temp.iloc[:3500].copy()
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 9), dpi=300, sharex=True)
    
    axes[0].plot(subset["Datetime"], subset[config.TARGET_COL], color="#2b5c8f", lw=1.8, label="Occupancy Count")
    axes[0].set_ylabel("Occupancy (Persons)", fontweight="bold", fontsize=10)
    axes[0].set_title("Temporal Dynamics: Occupancy vs. Environmental Sensors", fontweight="bold", fontsize=13, pad=10)
    axes[0].set_yticks([0, 1, 2, 3])
    axes[0].legend(loc="upper right")
    
    axes[1].plot(subset["Datetime"], subset["S1_Light"], color="#d95f02", lw=1.5, label="Light Intensity (S1_Light)")
    axes[1].set_ylabel("Light (Lux)", fontweight="bold", fontsize=10)
    axes[1].legend(loc="upper right")
    
    axes[2].plot(subset["Datetime"], subset["S5_CO2"], color="#27ae60", lw=1.8, label="CO2 Concentration (ppm)")
    axes[2].set_ylabel("CO2 (ppm)", fontweight="bold", fontsize=10)
    axes[2].set_xlabel("Timestamp", fontweight="bold", fontsize=10)
    axes[2].legend(loc="upper right")
    
    for ax in axes:
        ax.grid(True, alpha=0.3, linestyle="--")
        
    plt.tight_layout()
    out_file = save_path or (config.FIGURES_DIR / "05_temporal_trend.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def plot_confusion_matrices(cm_dict: Dict[str, np.ndarray], save_path: Optional[str] = None) -> None:
    """
    Plot 2x2 multi-panel confusion matrices for key benchmark models.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 11), dpi=300)
    model_keys = list(cm_dict.keys())[:4]
    
    for ax, name in zip(axes.flatten(), model_keys):
        cm = cm_dict[name]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["0", "1", "2", "3"],
            yticklabels=["0", "1", "2", "3"],
            annot_kws={"size": 11, "weight": "bold"}
        )
        ax.set_title(f"Confusion Matrix: {name}", fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel("Predicted Label (Occupancy)", fontsize=10)
        ax.set_ylabel("True Label (Occupancy)", fontsize=10)
        
    plt.suptitle("Model Prediction Behavior Across Occupancy Classes (Test Set)", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    
    out_file = save_path or (config.FIGURES_DIR / "06_confusion_matrices.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    model_name: str = "Decision Tree (All Sensors)",
    save_path: Optional[str] = None
) -> None:
    """
    Plot horizontal bar chart of feature importances sorted descending.
    """
    feat_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    bars = ax.barh(feat_df["Feature"], feat_df["Importance"], color="#2b5c8f", edgecolor="black", alpha=0.85)
    
    for bar in bars:
        width = bar.get_width()
        if width > 0.005:
            ax.annotate(f"{width:.3f}",
                        xy=(width, bar.get_y() + bar.get_height() / 2),
                        xytext=(4, 0),
                        textcoords="offset points",
                        va="center", ha="left", fontsize=9, fontweight="bold")
            
    ax.set_title(f"Feature Importance Ranking: {model_name}", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Relative Importance (Gini / MDI)", fontsize=11, labelpad=8)
    ax.set_xlim(0, max(feat_df["Importance"]) * 1.15)
    plt.tight_layout()
    
    out_file = save_path or (config.FIGURES_DIR / "07_feature_importance.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def plot_feature_set_ablation(ablation_df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """
    Plot bar comparison of Accuracy and Macro F1 score across Feature Sets.
    """
    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    x = np.arange(len(ablation_df))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, ablation_df["Accuracy"] * 100, width, label="Accuracy (%)", color="#2b5c8f", edgecolor="black", alpha=0.85)
    rects2 = ax.bar(x + width/2, ablation_df["Macro F1"] * 100, width, label="Macro F1-Score (%)", color="#d95f02", edgecolor="black", alpha=0.85)
    
    for rect in list(rects1) + list(rects2):
        height = rect.get_height()
        ax.annotate(f"{height:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
        
    ax.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
    ax.set_title("Ablation Study: Model Performance Across Different Feature Sets", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(ablation_df["Feature Set"], rotation=15, ha="right", fontsize=10)
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right")
    plt.tight_layout()
    
    out_file = save_path or (config.FIGURES_DIR / "08_feature_set_ablation.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def plot_temporal_vs_random_split(comparison_df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """
    Plot bar chart comparing Temporal Split vs. Stratified Random Split to illustrate leakage.
    """
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    x = np.arange(len(comparison_df))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, comparison_df["Accuracy"] * 100, width, label="Accuracy (%)", color="#1b9e77", edgecolor="black", alpha=0.85)
    rects2 = ax.bar(x + width/2, comparison_df["Macro F1"] * 100, width, label="Macro F1-Score (%)", color="#7570b3", edgecolor="black", alpha=0.85)
    
    for rect in list(rects1) + list(rects2):
        height = rect.get_height()
        ax.annotate(f"{height:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
        
    ax.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
    ax.set_title("Data Splitting Impact: Temporal Split vs. Random Stratified Split", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df["Split Strategy"], fontsize=10)
    ax.set_ylim(0, 115)
    ax.legend(loc="lower right")
    plt.tight_layout()
    
    out_file = save_path or (config.FIGURES_DIR / "09_temporal_vs_random_split.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")


def generate_all_eda_figures(df: Optional[pd.DataFrame] = None) -> None:
    """
    Generate and save all 5 EDA figures into outputs/figures/.
    """
    if df is None:
        from src.data_loader import load_raw_data
        df = load_raw_data()
        
    print("Generating EDA Figures...")
    plot_class_distribution(df)
    plot_sensor_distributions(df)
    plot_sensor_vs_occupancy_boxplots(df)
    plot_correlation_heatmap(df)
    plot_temporal_trend(df)
    print("All 5 EDA figures successfully generated!")
