import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    csv_path = os.path.join(repo_root, "outputs", "csv", "comparison_models_results.csv")
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    # Format NaN to '-' and float values to 4 decimals
    for col in df.columns:
        if col not in ["Model", "Dataset", "Type"]:
            df[col] = df[col].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "-")
            
    # Shorten names to fit table nicely
    df["Dataset"] = df["Dataset"].replace({
        "Energy Efficiency (ENB2012)": "Energy Efficiency",
        "Concrete Compressive Strength": "Concrete Strength",
        "Real Estate Valuation": "Real Estate",
        "Breast Cancer Wisconsin": "Breast Cancer",
        "Data Banknote Authentication": "Banknote Auth",
        "Tic-Tac-Toe Endgame": "Tic-Tac-Toe"
    })
    
    # Configure font and style to avoid querying system fonts (triggers sandbox block)
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    # Filter datasets to keep table clean
    df_reg = df[df["Type"] == "regression"].drop(columns=["Type", "Accuracy", "Precision", "Recall", "F1-Score"])
    df_clf = df[df["Type"] == "classification"].drop(columns=["Type", "RMSE", "MAE", "R2"])
    
    # Render regression table
    fig, ax = plt.subplots(figsize=(10, 11))
    ax.axis("tight")
    ax.axis("off")
    
    table_data = df_reg.values
    col_labels = df_reg.columns
    
    table = ax.table(cellText=table_data, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.4)
    
    # Style table headers and cells
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#1f77b4")
        elif row % 2 == 0:
            cell.set_facecolor("#f2f2f2")
            
    plt.title("MLP Optimizer Comparison (Regression)", fontsize=14, weight="bold", pad=20)
    
    reg_plot_path = os.path.join(repo_root, "outputs", "plots", "comparison_results_table_regression.png")
    plt.savefig(reg_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    # Render classification table
    fig, ax = plt.subplots(figsize=(10, 11))
    ax.axis("tight")
    ax.axis("off")
    
    table_data = df_clf.values
    col_labels = df_clf.columns
    
    table = ax.table(cellText=table_data, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.4)
    
    # Style table headers and cells
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#2ca02c")
        elif row % 2 == 0:
            cell.set_facecolor("#f2f2f2")
            
    plt.title("MLP Optimizer Comparison (Classification)", fontsize=14, weight="bold", pad=20)
    
    clf_plot_path = os.path.join(repo_root, "outputs", "plots", "comparison_results_table_classification.png")
    plt.savefig(clf_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Saved regression table image to: {reg_plot_path}")
    print(f"Saved classification table image to: {clf_plot_path}")

if __name__ == "__main__":
    main()
