"""
Generate high-resolution PNG graphic tables for Medium publication.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

OUTPUT_DIR = Path("c:/Users/gokde/Desktop/bootcamp/project_medium/outputs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def render_table_image(
    title: str,
    headers: list,
    data: list,
    col_widths: list,
    filename: str,
    highlight_row: int = 0
):
    fig, ax = plt.subplots(figsize=(12, len(data) * 0.65 + 1.6), dpi=300)
    ax.axis("off")
    ax.axis("tight")
    
    # Title
    plt.title(title, fontsize=14, fontweight="bold", pad=16, color="#1a252f", loc="center")
    
    table = ax.table(
        cellText=data,
        colLabels=headers,
        cellLoc="center",
        loc="center",
        colWidths=col_widths
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    
    # Header styling
    for col_idx in range(len(headers)):
        cell = table[(0, col_idx)]
        cell.set_facecolor("#2b5c8f")
        cell.set_text_props(color="white", fontweight="bold", fontsize=10.5)
        cell.set_edgecolor("#1a365d")
        cell.set_linewidth(1.2)
        
    # Row styling
    for row_idx in range(len(data)):
        for col_idx in range(len(headers)):
            cell = table[(row_idx + 1, col_idx)]
            cell.set_edgecolor("#e2e8f0")
            cell.set_linewidth(0.8)
            
            # Align first column to left
            if col_idx == 0 or (len(headers) == 6 and col_idx == 5):
                cell.set_text_props(ha="left" if col_idx == 5 else "center")
                
            if row_idx == highlight_row:
                cell.set_facecolor("#e6f4ea")  # Light green highlight for best
                cell.set_text_props(fontweight="bold", color="#137333")
            elif row_idx % 2 == 1:
                cell.set_facecolor("#f8fafc")  # Subtle zebra stripe
            else:
                cell.set_facecolor("#ffffff")
                
    out_path = OUTPUT_DIR / filename
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close()
    print(f"Generated table image: {out_path}")


def generate_all_tables():
    # Table 1: Model Comparison
    title_1 = "Model Performans Karşılaştırması (Feature Set A - 16 Sensör | Temporal Split)"
    headers_1 = ["Model", "Test Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Weighted F1"]
    data_1 = [
        ["Decision Tree (depth=5)", "95.06%", "88.42%", "73.19%", "76.85%", "94.66%"],
        ["Decision Tree (depth=7)", "95.06%", "88.42%", "73.19%", "76.85%", "94.66%"],
        ["Logistic Regression", "95.26%", "88.59%", "66.81%", "61.58%", "93.63%"],
        ["Decision Tree (depth=3)", "92.05%", "83.24%", "61.04%", "67.85%", "90.94%"],
        ["KNN (k=3)", "93.48%", "55.90%", "46.34%", "46.85%", "93.08%"],
        ["KNN (k=5)", "92.65%", "51.49%", "43.61%", "44.18%", "92.44%"],
        ["KNN (k=7)", "91.41%", "44.77%", "40.43%", "41.83%", "91.52%"]
    ]
    col_widths_1 = [0.26, 0.14, 0.15, 0.14, 0.14, 0.14]
    render_table_image(title_1, headers_1, data_1, col_widths_1, "table_01_model_comparison.png", highlight_row=0)

    # Table 2: Feature Set Ablation
    title_2 = "Feature Set Ablation Çalışması (Decision Tree depth=5 | Temporal Split)"
    headers_2 = ["Feature Set", "Değişken", "Accuracy", "Macro F1", "Weighted F1", "Temel Çıkarım"]
    data_2 = [
        ["Set E (Sensors + Time)", "19", "95.95%", "78.98%", "95.62%", "Zaman bilgisi en yüksek skoru sağlıyor"],
        ["Set A (All Sensors)", "16", "95.06%", "76.85%", "94.66%", "Tüm 16 sensör baseline performansı"],
        ["Set B (No PIR)", "14", "92.05%", "67.85%", "90.94%", "PIR çıkarılınca Macro F1'de %9.0 net düşüş"],
        ["Set C (Environmental Only)", "14", "92.05%", "67.85%", "90.94%", "Yalnızca pasif çevresel ölçümler"],
        ["Set D (Engineered Summary)", "10", "91.61%", "52.34%", "89.91%", "16 sensör -> 10 özet değişkene iniyor"]
    ]
    col_widths_2 = [0.24, 0.10, 0.12, 0.12, 0.13, 0.35]
    render_table_image(title_2, headers_2, data_2, col_widths_2, "table_02_feature_ablation.png", highlight_row=0)

if __name__ == "__main__":
    generate_all_tables()
