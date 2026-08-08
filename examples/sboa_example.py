import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mlp_sboa.optimizers import SBOA_nn
from mlp_sboa.metrics import rmse
from mlp_sboa.utils import hd


def ensure_data_unzipped(repo_root):
    zpath = os.path.join(repo_root, "data.zip")
    outdir = os.path.join(repo_root, "data")
    if os.path.exists(outdir) and os.listdir(outdir):
        return outdir
    if os.path.exists(zpath):
        with zipfile.ZipFile(zpath, "r") as z:
            z.extractall(outdir)
        print("Extracted data.zip to data/")
    return outdir


def load_auto_mpg(data_dir):
    # handles either auto-mpg.csv or auto-mpg.data
    csv_path = os.path.join(data_dir, "auto-mpg", "auto-mpg.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        dat_path = os.path.join(data_dir, "auto-mpg", "auto-mpg.data")
        # parse UCI auto-mpg.data format
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
        df["horsepower"].replace("?", np.nan, inplace=True)
        df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")
        df["horsepower"].fillna(df["horsepower"].mean(), inplace=True)
        df.to_csv(os.path.join(data_dir, "auto-mpg", "auto-mpg.csv"), index=False)
    return df


if __name__ == "__main__":
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = ensure_data_unzipped(repo_root)

    # load a dataset we uploaded: auto-mpg
    df = load_auto_mpg(data_dir)
    # prepare X, y like the R script: y=mpg, X=columns 2:8
    features = ["cylinders","displacement","horsepower","weight","acceleration","model_year","origin"]
    X = df[features].astype(float).values
    y = df[["mpg"]].astype(float).values

    # quick experiment with small population/iterations to test pipeline
    hidden_dim = 8
    SearchAgents = 20
    Max_iterations = 10
    lowerbound = -1
    upperbound = 1

    print("Starting SBOA (short run) on auto-mpg (this may take some time)...")
    result = SBOA_nn(X, y, hidden_dim, SearchAgents, Max_iterations, lowerbound, upperbound)

    # save convergence
    out_dir = os.path.join(repo_root, "outputs")
    os.makedirs(os.path.join(out_dir, "csv"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)

    curve = result["sboa_curve"]
    np.savetxt(os.path.join(out_dir, "csv", "sboa_convergence_auto_mpg.csv"), curve, delimiter=",")

    # plot
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(curve, marker=".")
    plt.xlabel("Iteration")
    plt.ylabel("Best cost")
    plt.title("SBOA convergence (auto-mpg)")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "plots", "sboa_convergence_auto_mpg.png"))

    print("Saved convergence CSV and PNG to outputs/")
