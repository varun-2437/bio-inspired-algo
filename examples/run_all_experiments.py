import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from mlp_sboa.optimizers import SBOA_nn
from mlp_sboa.model import predict
from mlp_sboa.metrics import rmse, mae

# Define the datasets list and their configurations
DATASETS_CONFIG = {
    "Auto-MPG": {
        "type": "regression",
        "features": ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin"],
        "target": "mpg",
        "hidden_dim": 8
    },
    "Energy Efficiency (ENB2012)": {
        "type": "regression",
        "features": ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"],
        "target": "Y1",
        "hidden_dim": 10
    },
    "Concrete Compressive Strength": {
        "type": "regression",
        "target_col_index": 8, # 9th column (concrete compressive strength)
        "feature_col_indices": list(range(8)),
        "hidden_dim": 10
    },
    "Real Estate Valuation": {
        "type": "regression",
        "features": [
            "X2 house age", 
            "X3 distance to the nearest MRT station", 
            "X4 number of convenience stores", 
            "X5 latitude", 
            "X6 longitude"
        ],
        "target": "Y house price of unit area",
        "hidden_dim": 6
    },
    "Breast Cancer Wisconsin": {
        "type": "classification",
        "hidden_dim": 9
    },
    "Blood Transfusion": {
        "type": "classification",
        "features": ["Recency (months)", "Frequency (times)", "Monetary (c.c. blood)", "Time (months)"],
        "target": "whether he/she donated blood in March 2007",
        "hidden_dim": 6
    },
    "Data Banknote Authentication": {
        "type": "classification",
        "hidden_dim": 6
    },
    "Tic-Tac-Toe Endgame": {
        "type": "classification",
        "hidden_dim": 9
    }
}

def load_dataset(name, repo_root):
    data_dir = os.path.join(repo_root, "data")
    data_all_dir = os.path.join(repo_root, "data all")

    if name == "Auto-MPG":
        # check raw first
        csv_path = os.path.join(data_dir, "cleaned", "auto-mpg.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # If it's the small sample (only 5 rows), fallback to data all
            if len(df) < 10:
                dat_path = os.path.join(data_all_dir, "auto+mpg", "auto-mpg.data")
                df = parse_auto_mpg_data(dat_path)
        else:
            dat_path = os.path.join(data_all_dir, "auto+mpg", "auto-mpg.data")
            df = parse_auto_mpg_data(dat_path)
        
        X = df[DATASETS_CONFIG[name]["features"]].astype(float).values
        y = df[[DATASETS_CONFIG[name]["target"]]].astype(float).values
        return X, y

    elif name == "Energy Efficiency (ENB2012)":
        excel_path = os.path.join(data_all_dir, "ENB2012_data.xlsx")
        df = pd.read_excel(excel_path)
        X = df[DATASETS_CONFIG[name]["features"]].astype(float).values
        y = df[[DATASETS_CONFIG[name]["target"]]].astype(float).values
        return X, y

    elif name == "Concrete Compressive Strength":
        excel_path = os.path.join(data_all_dir, "concrete+compressive+strength", "Concrete_Data.xls")
        df = pd.read_excel(excel_path)
        X = df.iloc[:, DATASETS_CONFIG[name]["feature_col_indices"]].astype(float).values
        y = df.iloc[:, [DATASETS_CONFIG[name]["target_col_index"]]].astype(float).values
        return X, y

    elif name == "Real Estate Valuation":
        excel_path = os.path.join(data_all_dir, "Real estate valuation data set.xlsx")
        df = pd.read_excel(excel_path)
        X = df[DATASETS_CONFIG[name]["features"]].astype(float).values
        y = df[[DATASETS_CONFIG[name]["target"]]].astype(float).values
        return X, y

    elif name == "Breast Cancer Wisconsin":
        dat_path = os.path.join(data_all_dir, "breast+cancer+wisconsin+original", "breast-cancer-wisconsin.data")
        df = pd.read_csv(dat_path, header=None)
        # column 0 is ID, columns 1-9 are features, column 10 is class (2=benign, 4=malignant)
        # handle missing values marked as '?'
        df = df.replace('?', np.nan)
        df = df.apply(pd.to_numeric, errors='coerce')
        # fill nan with mean
        df = df.fillna(df.mean())
        
        X = df.iloc[:, 1:10].astype(float).values
        # map 2->0, 4->1
        y = df.iloc[:, [10]].values
        y = np.where(y == 4, 1.0, 0.0)
        return X, y

    elif name == "Blood Transfusion":
        dat_path = os.path.join(data_all_dir, "blood+transfusion+service+center", "transfusion.data")
        df = pd.read_csv(dat_path)
        X = df[DATASETS_CONFIG[name]["features"]].astype(float).values
        y = df[[DATASETS_CONFIG[name]["target"]]].astype(float).values
        return X, y

    elif name == "Data Banknote Authentication":
        dat_path = os.path.join(data_all_dir, "data_banknote_authentication.txt")
        df = pd.read_csv(dat_path, header=None)
        X = df.iloc[:, 0:4].astype(float).values
        y = df.iloc[:, [4]].astype(float).values
        return X, y

    elif name == "Tic-Tac-Toe Endgame":
        dat_path = os.path.join(data_all_dir, "tic+tac+toe+endgame", "tic-tac-toe.data")
        df = pd.read_csv(dat_path, header=None)
        # Encode categorical features: x->1, o->-1, b->0
        mapping = {'x': 1.0, 'o': -1.0, 'b': 0.0}
        for col in range(9):
            df[col] = df[col].map(mapping)
        X = df.iloc[:, 0:9].astype(float).values
        # target is column 9: positive->1, negative->0
        y = df.iloc[:, [9]].apply(lambda val: 1.0 if val[9] == 'positive' else 0.0, axis=1).values.reshape(-1, 1)
        return X, y

    else:
        raise ValueError(f"Unknown dataset name: {name}")

def parse_auto_mpg_data(dat_path):
    rows = []
    import re
    with open(dat_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = re.split(r"\s+", line, maxsplit=8)
            if len(parts) < 9:
                continue
            rows.append(parts)
    colnames = ["mpg","cylinders","displacement","horsepower","weight","acceleration","model_year","origin","car_name"]
    df = pd.DataFrame(rows, columns=colnames)
    df["horsepower"] = df["horsepower"].replace("?", np.nan)
    df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")
    # convert other columns to numeric
    for col in colnames[:-1]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["horsepower"] = df["horsepower"].fillna(df["horsepower"].mean())
    return df

def run_experiment(name, X, y, search_agents, max_iterations, repo_root):
    print(f"\n================ Running Experiment: {name} ================")
    print(f"Dataset type: {DATASETS_CONFIG[name]['type']}, Shape: {X.shape}")
    
    # Train/test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    hidden_dim = DATASETS_CONFIG[name]["hidden_dim"]
    
    # Run SBOA
    res = SBOA_nn(X_train, y_train, hidden_dim, search_agents, max_iterations, -1, 1)
    
    weights = res["weights"]
    biases = res["biases"]
    norm_X = res["normalization_X"]
    norm_y = res["normalization_y"]
    curve = res["sboa_curve"]
    
    # Prediction on normalized inputs
    min_X, max_X = norm_X
    X_test_norm = (X_test - min_X) / (max_X - min_X + 1e-12)
    
    # Predict and un-normalize target predictions
    y_pred_norm = predict(weights, biases, X_test_norm, data_y=None)
    y_pred = y_pred_norm * (norm_y[1] - norm_y[0] + 1e-12) + norm_y[0]
    
    # Compute metrics
    metrics_dict = {}
    is_classification = (DATASETS_CONFIG[name]["type"] == "classification")
    
    if is_classification:
        # threshold predictions at 0.5
        y_pred_binary = (y_pred_norm >= 0.5).astype(int)
        y_test_binary = (y_test >= 0.5).astype(int)
        
        acc = accuracy_score(y_test_binary, y_pred_binary)
        prec = precision_score(y_test_binary, y_pred_binary, zero_division=0)
        rec = recall_score(y_test_binary, y_pred_binary, zero_division=0)
        f1 = f1_score(y_test_binary, y_pred_binary, zero_division=0)
        
        metrics_dict["Accuracy"] = acc
        metrics_dict["Precision"] = prec
        metrics_dict["Recall"] = rec
        metrics_dict["F1-Score"] = f1
        print(f"Results: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1-Score={f1:.4f}")
    else:
        err_rmse = rmse(y_test, y_pred)
        err_mae = mae(y_test, y_pred)
        
        # Calculate R-squared
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1.0 - (ss_res / (ss_tot + 1e-12))
        
        metrics_dict["RMSE"] = err_rmse
        metrics_dict["MAE"] = err_mae
        metrics_dict["R2"] = r2
        print(f"Results: RMSE={err_rmse:.4f}, MAE={err_mae:.4f}, R2={r2:.4f}")
        
    return curve, metrics_dict

def main():
    parser = argparse.ArgumentParser(description="Run SBOA experiments across all datasets.")
    parser.add_argument("--agents", type=int, default=30, help="Number of search agents (population size)")
    parser.add_argument("--iterations", type=int, default=50, help="Number of maximum SBOA iterations")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    results = []
    curves = {}
    
    for name in DATASETS_CONFIG.keys():
        try:
            X, y = load_dataset(name, repo_root)
            curve, metrics = run_experiment(name, X, y, args.agents, args.iterations, repo_root)
            curves[name] = curve
            
            row = {"Dataset": name, "Type": DATASETS_CONFIG[name]["type"]}
            row.update({k: f"{v:.4f}" for k, v in metrics.items()})
            results.append(row)
        except Exception as e:
            print(f"Failed running experiment on {name}: {e}")
            import traceback
            traceback.print_exc()

    # Save results table
    out_dir = os.path.join(repo_root, "outputs")
    os.makedirs(os.path.join(out_dir, "csv"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)
    
    df_results = pd.DataFrame(results)
    csv_path = os.path.join(out_dir, "csv", "comparison_results.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\nSaved comparison summary table to: {csv_path}")
    print("\nComparison Results:")
    print(df_results.to_string(index=False))

    # Plot convergence curves
    plt.figure(figsize=(12, 8))
    for name, curve in curves.items():
        plt.plot(curve, label=name, marker="o", markersize=3)
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness (Normalized MSE)")
    plt.title(f"SBOA MLP Convergence comparison ({args.agents} agents, {args.iterations} iterations)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plot_path = os.path.join(out_dir, "plots", "comparison_convergence.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved convergence plot to: {plot_path}")

if __name__ == "__main__":
    main()
