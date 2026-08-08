"""
Generate visualizations with REAL trained weights and biases from our SBOA-MLP model.
Trains SBOA on the Auto-MPG dataset and produces:
  1. MLP architecture diagram with real weight values on each connection
  2. Flat vector mapping showing actual parameter values
  3. Convergence curve from real training
  4. Weight heatmaps for W1 and W2
  5. Optimization flowchart with real dataset dimensions
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from matplotlib import cm
import pandas as pd
from sklearn.model_selection import train_test_split

# resolve package imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mlp_sboa.optimizers import SBOA_nn
from mlp_sboa.model import predict

plt.rcParams['font.family'] = 'DejaVu Sans'

# ─── Configuration ───
DATASET = "Auto-MPG"
FEATURES = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin"]
TARGET = "mpg"
HIDDEN_DIM = 8
SEARCH_AGENTS = 30
MAX_ITERATIONS = 50
VIS_DIR = os.path.join(os.path.dirname(__file__), "..", "visualization")

# ─── Load data ───
def load_auto_mpg():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    csv_path = os.path.join(repo_root, "data", "auto-mpg.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        import re
        dat_path = os.path.join(repo_root, "data all", "auto+mpg", "auto-mpg.data")
        rows = []
        with open(dat_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = re.split(r"\s+", line, maxsplit=8)
                if len(parts) < 9: continue
                rows.append(parts)
        colnames = ["mpg","cylinders","displacement","horsepower","weight","acceleration","model_year","origin","car_name"]
        df = pd.DataFrame(rows, columns=colnames)
        df["horsepower"] = df["horsepower"].replace("?", np.nan)
        for col in colnames[:-1]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["horsepower"] = df["horsepower"].fillna(df["horsepower"].mean())
    X = df[FEATURES].astype(float).values
    y = df[[TARGET]].astype(float).values
    return X, y, FEATURES, TARGET


# ─── 1. MLP Architecture with Real Weights ───
def draw_mlp_real(W1, b1, W2, b2, feature_names, target_name):
    """Draw MLP network diagram with real weight values on connections."""
    input_dim = W1.shape[0]   # 7
    hidden_dim = W1.shape[1]  # 8
    output_dim = W2.shape[1]  # 1

    fig, ax = plt.subplots(figsize=(18, 12))
    ax.axis("off")

    x_in, x_hid, x_out = 1.5, 5.0, 8.5
    
    # vertical positions (centered)
    y_inputs = np.linspace(0.5, input_dim * 1.2, input_dim)
    y_hiddens = np.linspace(0.5, hidden_dim * 1.0, hidden_dim)
    y_outputs = [np.mean(y_hiddens)]

    # Normalize weight magnitudes for coloring
    all_w1 = np.abs(W1).flatten()
    all_w2 = np.abs(W2).flatten()
    w1_max = max(all_w1.max(), 0.01)
    w2_max = max(all_w2.max(), 0.01)

    # Draw W1 connections
    for i in range(input_dim):
        for j in range(hidden_dim):
            w_val = W1[i, j]
            intensity = min(abs(w_val) / w1_max, 1.0)
            color = (0.8, 0.2, 0.2, 0.15 + 0.7 * intensity) if w_val < 0 else (0.2, 0.2, 0.8, 0.15 + 0.7 * intensity)
            lw = 0.5 + 2.5 * intensity
            ax.plot([x_in, x_hid], [y_inputs[i], y_hiddens[j]], color=color, linewidth=lw, zorder=1)

    # Draw W2 connections
    for j in range(hidden_dim):
        for k in range(output_dim):
            w_val = W2[j, k]
            intensity = min(abs(w_val) / w2_max, 1.0)
            color = (0.8, 0.2, 0.2, 0.2 + 0.7 * intensity) if w_val < 0 else (0.2, 0.2, 0.8, 0.2 + 0.7 * intensity)
            lw = 1.0 + 3.0 * intensity
            ax.plot([x_hid, x_out], [y_hiddens[j], y_outputs[k]], color=color, linewidth=lw, zorder=1)
            # label W2 values (fewer connections so we can show them all)
            mid_x = (x_hid + x_out) / 2 + 0.1
            mid_y = (y_hiddens[j] + y_outputs[k]) / 2
            ax.text(mid_x, mid_y, f"{w_val:.2f}", fontsize=6, color="black", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="none", alpha=0.8), zorder=5)

    # Draw input nodes
    node_r = 0.3
    for i in range(input_dim):
        circle = patches.Circle((x_in, y_inputs[i]), node_r, edgecolor="#1565C0", facecolor="#BBDEFB", lw=2, zorder=3)
        ax.add_patch(circle)
        ax.text(x_in, y_inputs[i], feature_names[i][:6], ha="center", va="center", weight="bold", fontsize=7, zorder=4)

    # Draw hidden nodes with bias values
    for j in range(hidden_dim):
        circle = patches.Circle((x_hid, y_hiddens[j]), node_r, edgecolor="#2E7D32", facecolor="#C8E6C9", lw=2, zorder=3)
        ax.add_patch(circle)
        ax.text(x_hid, y_hiddens[j] + 0.05, f"H{j+1}", ha="center", va="center", weight="bold", fontsize=8, zorder=4)
        ax.text(x_hid, y_hiddens[j] - 0.12, f"b={b1[j]:.2f}", ha="center", va="center", fontsize=5, color="dimgray", zorder=4)

    # Draw output node
    for k in range(output_dim):
        circle = patches.Circle((x_out, y_outputs[k]), node_r, edgecolor="#C62828", facecolor="#FFCDD2", lw=2, zorder=3)
        ax.add_patch(circle)
        ax.text(x_out, y_outputs[k] + 0.05, target_name, ha="center", va="center", weight="bold", fontsize=8, zorder=4)
        ax.text(x_out, y_outputs[k] - 0.12, f"b={b2[k]:.3f}", ha="center", va="center", fontsize=5, color="dimgray", zorder=4)

    # Layer labels
    ax.text(x_in, max(y_inputs) + 1.0, f"Input Layer\n({input_dim} features)", ha="center", va="center", weight="bold", color="#1565C0", fontsize=13)
    ax.text(x_hid, max(y_hiddens) + 1.0, f"Hidden Layer\n({hidden_dim} neurons, Sigmoid)", ha="center", va="center", weight="bold", color="#2E7D32", fontsize=13)
    ax.text(x_out, max(y_hiddens) + 1.0, f"Output Layer\n({output_dim} neuron, Linear)", ha="center", va="center", weight="bold", color="#C62828", fontsize=13)

    # Weight matrix shape annotations
    ax.text((x_in + x_hid) / 2, -0.3, f"W1: {input_dim}×{hidden_dim} = {input_dim*hidden_dim} weights", ha="center", fontsize=10, style="italic", color="dimgray")
    ax.text((x_hid + x_out) / 2, -0.3, f"W2: {hidden_dim}×{output_dim} = {hidden_dim*output_dim} weights", ha="center", fontsize=10, style="italic", color="dimgray")

    # Legend for line coloring
    ax.plot([], [], color=(0.2, 0.2, 0.8, 0.8), linewidth=2, label="Positive weight")
    ax.plot([], [], color=(0.8, 0.2, 0.2, 0.8), linewidth=2, label="Negative weight")
    ax.text(0.5, -0.7, "Line thickness ∝ |weight magnitude|   •   Blue = positive   •   Red = negative", ha="left", fontsize=9, style="italic", color="gray")

    plt.title("SBOA-Optimized MLP Architecture — Auto-MPG Dataset (Real Trained Weights)", fontsize=15, weight="bold", pad=20)
    ax.set_xlim(0.5, 9.5)
    ax.set_ylim(-1.2, max(max(y_inputs), max(y_hiddens)) + 1.5)

    plt.savefig(os.path.join(VIS_DIR, "mlp_real_weights.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  ✅ mlp_real_weights.png")


# ─── 2. Flat Vector Mapping with Real Values ───
def draw_vector_mapping_real(W1, b1, W2, b2):
    input_dim, hidden_dim = W1.shape
    output_dim = W2.shape[1]
    total = input_dim * hidden_dim + hidden_dim + hidden_dim * output_dim + output_dim

    flat_vec = np.concatenate([W1.flatten(), b1.flatten(), W2.flatten(), b2.flatten()])

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), gridspec_kw={"height_ratios": [1, 2]})

    # Top: schematic segments
    ax = axes[0]
    ax.axis("off")
    segments = [
        {"name": f"W1\n({input_dim}×{hidden_dim}={input_dim*hidden_dim})", "count": input_dim * hidden_dim, "color": "#BBDEFB"},
        {"name": f"b1\n({hidden_dim})", "count": hidden_dim, "color": "#C8E6C9"},
        {"name": f"W2\n({hidden_dim}×{output_dim}={hidden_dim*output_dim})", "count": hidden_dim * output_dim, "color": "#FFCDD2"},
        {"name": f"b2\n({output_dim})", "count": output_dim, "color": "#F8BBD0"},
    ]
    x_start = 0.5
    total_width = 14.0
    for seg in segments:
        w = total_width * (seg["count"] / total)
        rect = patches.FancyBboxPatch((x_start, 0.5), w, 1.5, boxstyle="round,pad=0.05", edgecolor="black", facecolor=seg["color"], lw=2)
        ax.add_patch(rect)
        ax.text(x_start + w / 2, 1.25, seg["name"], ha="center", va="center", fontsize=9, weight="bold")
        x_start += w
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 2.5)
    ax.set_title(f"Search Agent Position Vector — {total} Parameters Total", fontsize=13, weight="bold", pad=10)

    # Bottom: actual parameter values bar chart
    ax2 = axes[1]
    colors = []
    w1_n = input_dim * hidden_dim
    b1_n = hidden_dim
    w2_n = hidden_dim * output_dim
    b2_n = output_dim
    colors.extend(["#64B5F6"] * w1_n)
    colors.extend(["#66BB6A"] * b1_n)
    colors.extend(["#EF5350"] * w2_n)
    colors.extend(["#EC407A"] * b2_n)

    ax2.bar(range(total), flat_vec, color=colors, width=1.0, edgecolor="none")
    ax2.set_xlabel("Parameter Index", fontsize=11)
    ax2.set_ylabel("Value", fontsize=11)
    ax2.set_title("Actual Optimized Parameter Values (SBOA Best Position Vector)", fontsize=12, weight="bold")
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.set_xlim(-0.5, total - 0.5)

    # Add segment boundary lines
    boundaries = [w1_n, w1_n + b1_n, w1_n + b1_n + w2_n]
    for bnd in boundaries:
        ax2.axvline(x=bnd - 0.5, color="black", linewidth=1.5, linestyle="--", alpha=0.5)

    # Labels
    labels_x = [w1_n / 2, w1_n + b1_n / 2, w1_n + b1_n + w2_n / 2, w1_n + b1_n + w2_n + b2_n / 2]
    labels_t = ["W1", "b1", "W2", "b2"]
    y_top = max(flat_vec.max(), abs(flat_vec.min())) * 0.9
    for lx, lt in zip(labels_x, labels_t):
        ax2.text(lx, y_top, lt, ha="center", va="bottom", fontsize=11, weight="bold", color="dimgray")

    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "flat_vector_real_values.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  ✅ flat_vector_real_values.png")


# ─── 3. Convergence Curve ───
def draw_convergence(curve):
    fig, ax = plt.subplots(figsize=(10, 6))
    iterations = np.arange(1, len(curve) + 1)
    ax.plot(iterations, curve, color="#1565C0", linewidth=2.5, marker="o", markersize=4, markerfacecolor="#BBDEFB", markeredgecolor="#1565C0")
    ax.fill_between(iterations, curve, alpha=0.1, color="#1565C0")
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Best Fitness (MSE)", fontsize=12)
    ax.set_title(f"SBOA Convergence Curve — Auto-MPG Dataset\n(Agents={SEARCH_AGENTS}, Iterations={MAX_ITERATIONS})", fontsize=13, weight="bold")
    ax.grid(True, alpha=0.3)

    # Annotate start and end
    ax.annotate(f"Start: {curve[0]:.4f}", xy=(1, curve[0]), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="gray"), xytext=(5, curve[0] * 1.05))
    ax.annotate(f"Final: {curve[-1]:.4f}", xy=(len(curve), curve[-1]), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="gray"), xytext=(len(curve) - 15, curve[-1] * 1.3))

    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "convergence_real.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  ✅ convergence_real.png")


# ─── 4. Weight Heatmaps ───
def draw_weight_heatmaps(W1, b1, W2, b2, feature_names, target_name):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), gridspec_kw={"width_ratios": [3, 1, 1]})

    # W1 heatmap
    ax1 = axes[0]
    im1 = ax1.imshow(W1, cmap="RdBu_r", aspect="auto", interpolation="nearest")
    ax1.set_title(f"W1 (Input → Hidden)\nShape: {W1.shape[0]}×{W1.shape[1]}", fontsize=12, weight="bold")
    ax1.set_ylabel("Input Features")
    ax1.set_xlabel("Hidden Neurons")
    ax1.set_yticks(range(W1.shape[0]))
    ax1.set_yticklabels([f[:8] for f in feature_names], fontsize=8)
    ax1.set_xticks(range(W1.shape[1]))
    ax1.set_xticklabels([f"H{j+1}" for j in range(W1.shape[1])], fontsize=8)
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    # Show values in cells
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            ax1.text(j, i, f"{W1[i,j]:.2f}", ha="center", va="center", fontsize=6,
                     color="white" if abs(W1[i,j]) > W1.max()*0.6 else "black")

    # W2 heatmap
    ax2 = axes[1]
    im2 = ax2.imshow(W2, cmap="RdBu_r", aspect="auto", interpolation="nearest")
    ax2.set_title(f"W2 (Hidden → Output)\nShape: {W2.shape[0]}×{W2.shape[1]}", fontsize=12, weight="bold")
    ax2.set_ylabel("Hidden Neurons")
    ax2.set_xlabel("Output")
    ax2.set_yticks(range(W2.shape[0]))
    ax2.set_yticklabels([f"H{j+1}" for j in range(W2.shape[0])], fontsize=8)
    ax2.set_xticks(range(W2.shape[1]))
    ax2.set_xticklabels([target_name], fontsize=8)
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    for i in range(W2.shape[0]):
        for j in range(W2.shape[1]):
            ax2.text(j, i, f"{W2[i,j]:.3f}", ha="center", va="center", fontsize=8,
                     color="white" if abs(W2[i,j]) > W2.max()*0.6 else "black")

    # Bias bar chart
    ax3 = axes[2]
    all_biases = np.concatenate([b1, b2])
    labels = [f"b1[{i}]" for i in range(len(b1))] + [f"b2[{i}]" for i in range(len(b2))]
    colors_b = ["#66BB6A"] * len(b1) + ["#EC407A"] * len(b2)
    ax3.barh(range(len(all_biases)), all_biases, color=colors_b, edgecolor="black", linewidth=0.5)
    ax3.set_yticks(range(len(all_biases)))
    ax3.set_yticklabels(labels, fontsize=8)
    ax3.set_xlabel("Bias Value")
    ax3.set_title("Biases (b1 & b2)", fontsize=12, weight="bold")
    ax3.axvline(x=0, color="black", linewidth=0.5)

    plt.suptitle("SBOA-Optimized Weight & Bias Analysis — Auto-MPG", fontsize=14, weight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "weight_heatmaps_real.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  ✅ weight_heatmaps_real.png")


# ─── 5. Optimization Flowchart with real dimensions ───
def draw_flowchart_real(input_dim, hidden_dim, output_dim, total_params):
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.axis("off")

    def draw_box(x, y, w, h, text, color, edge="black"):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15", edgecolor=edge, facecolor=color, lw=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=9, weight="bold", wrap=True)

    def arrow(x1, y1, x2, y2, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", lw=2, color="#333"))
        if label:
            ax.text((x1+x2)/2 + 0.15, (y1+y2)/2, label, fontsize=8, style="italic", color="dimgray")

    # Step 1
    draw_box(3.0, 7.0, 4.0, 0.9, f"Initialize {SEARCH_AGENTS} Search Agents\n(Random vectors of {total_params} dimensions, range [-1,1])", "#E0E0E0")
    arrow(5.0, 7.0, 5.0, 6.3)

    # Step 2
    draw_box(3.0, 5.4, 4.0, 0.9, f"Unpack 1D Vector → MLP Params\nW1: {input_dim}×{hidden_dim}  b1: {hidden_dim}\nW2: {hidden_dim}×{output_dim}  b2: {output_dim}", "#BBDEFB")
    arrow(5.0, 5.4, 5.0, 4.7)

    # Step 3
    draw_box(3.0, 3.8, 4.0, 0.9, f"Forward Pass through MLP\nInput({input_dim}) → Sigmoid({hidden_dim}) → Linear({output_dim})\nCompute MSE Loss = Fitness", "#FFCDD2")
    arrow(5.0, 3.8, 5.0, 3.1)

    # Step 4
    draw_box(3.0, 2.2, 4.0, 0.9, f"SBOA Update Rules\nPhase 1 (t<T/3): Random walk exploration\nPhase 2 (T/3<t<2T/3): Best-guided exploitation\nPhase 3 (t>2T/3): Lévy flight refinement", "#C8E6C9")

    # Loop arrow
    ax.annotate("", xy=(2.8, 5.85), xytext=(2.8, 2.65),
                arrowprops=dict(arrowstyle="-|>", lw=2.5, color="#FF6F00", connectionstyle="bar,fraction=-0.25"))
    ax.text(1.2, 4.3, f"Repeat\n{MAX_ITERATIONS}\nIterations", ha="center", va="center", fontsize=10, weight="bold", color="#FF6F00")

    # Exit arrow
    arrow(7.0, 2.65, 8.5, 2.65)
    draw_box(8.5, 2.2, 2.5, 0.9, f"Best Agent Found!\nOptimal {total_params} params\n→ Trained MLP Weights", "#FFF9C4", edge="#F9A825")

    plt.title("SBOA-MLP Training Pipeline — Auto-MPG (Real Dimensions)", fontsize=14, weight="bold", pad=20)
    ax.set_xlim(0.5, 11.5)
    ax.set_ylim(1.5, 8.3)

    plt.savefig(os.path.join(VIS_DIR, "optimization_flowchart_real.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  ✅ optimization_flowchart_real.png")


# ─── Main ───
def main():
    os.makedirs(VIS_DIR, exist_ok=True)
    print("=" * 60)
    print("GENERATING VISUALIZATIONS WITH REAL TRAINED WEIGHTS")
    print("=" * 60)

    # 1. Load dataset
    print(f"\n📊 Loading {DATASET} dataset...")
    X, y, feature_names, target_name = load_auto_mpg()
    print(f"   Shape: X={X.shape}, y={y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Train SBOA
    print(f"\n🦅 Training SBOA on {DATASET} (agents={SEARCH_AGENTS}, iterations={MAX_ITERATIONS})...")
    res = SBOA_nn(X_train, y_train, HIDDEN_DIM, SEARCH_AGENTS, MAX_ITERATIONS, -1, 1)

    W1 = res["weights"]["W1"]
    W2 = res["weights"]["W2"]
    b1 = res["biases"]["b1"]
    b2 = res["biases"]["b2"]
    curve = res["curve"]

    input_dim = W1.shape[0]
    hidden_dim = W1.shape[1]
    output_dim = W2.shape[1]
    total_params = input_dim * hidden_dim + hidden_dim + hidden_dim * output_dim + output_dim

    print(f"\n   W1 shape: {W1.shape}")
    print(f"   b1 shape: {b1.shape}")
    print(f"   W2 shape: {W2.shape}")
    print(f"   b2 shape: {b2.shape}")
    print(f"   Total parameters: {total_params}")
    print(f"   Final best fitness: {curve[-1]:.6f}")

    # 3. Generate visualizations
    print(f"\n🎨 Generating visualizations to: {VIS_DIR}/")

    draw_mlp_real(W1, b1, W2, b2, feature_names, target_name)
    draw_vector_mapping_real(W1, b1, W2, b2)
    draw_convergence(curve)
    draw_weight_heatmaps(W1, b1, W2, b2, feature_names, target_name)
    draw_flowchart_real(input_dim, hidden_dim, output_dim, total_params)

    print(f"\n✅ All 5 visualizations saved to: {VIS_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
