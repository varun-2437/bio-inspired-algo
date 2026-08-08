import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_mlp():
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis("off")
    plt.rcParams['font.family'] = 'DejaVu Sans'

    # Layer coordinates
    inputs = [1, 2, 3]
    hiddens = [0.5, 1.5, 2.5, 3.5]
    outputs = [2]
    
    x_in = 1
    x_hid = 3
    x_out = 5
    
    # Draw connections (Weights)
    for y_in in inputs:
        for y_hid in hiddens:
            ax.plot([x_in, x_hid], [y_in, y_hid], color="gray", alpha=0.5, linestyle="-", linewidth=1)
            
    for y_hid in hiddens:
        for y_out in outputs:
            ax.plot([x_hid, x_out], [y_hid, y_out], color="gray", alpha=0.6, linestyle="-", linewidth=1.5)
            
    # Draw Nodes
    node_rad = 0.25
    # Inputs
    for y_in in inputs:
        circle = patches.Circle((x_in, y_in), node_rad, edgecolor="#1f77b4", facecolor="#aec7e8", lw=2, zorder=3)
        ax.add_patch(circle)
        ax.text(x_in, y_in, f"X{y_in}", ha="center", va="center", weight="bold", fontsize=10, zorder=4)
        
    # Hiddens
    for i, y_hid in enumerate(hiddens, 1):
        circle = patches.Circle((x_hid, y_hid), node_rad, edgecolor="#2ca02c", facecolor="#98df8a", lw=2, zorder=3)
        ax.add_patch(circle)
        ax.text(x_hid, y_hid, f"H{i}", ha="center", va="center", weight="bold", fontsize=10, zorder=4)
        
    # Output
    for y_out in outputs:
        circle = patches.Circle((x_out, y_out), node_rad, edgecolor="#d62728", facecolor="#ff9896", lw=2, zorder=3)
        ax.add_patch(circle)
        ax.text(x_out, y_out, "Y", ha="center", va="center", weight="bold", fontsize=10, zorder=4)
        
    # Labels
    ax.text(x_in, 4.2, "Input Layer\n(Features)", ha="center", va="center", weight="bold", color="#1f77b4", fontsize=12)
    ax.text(x_hid, 4.2, "Hidden Layer\n(Sigmoid Activation)", ha="center", va="center", weight="bold", color="#2ca02c", fontsize=12)
    ax.text(x_out, 4.2, "Output Layer\n(Linear Output)", ha="center", va="center", weight="bold", color="#d62728", fontsize=12)
    
    # Weight matrices labels
    ax.text((x_in + x_hid)/2, 0.2, "Weights W1 & Biases b1", ha="center", va="center", style="italic", fontsize=10, color="dimgray")
    ax.text((x_hid + x_out)/2, 0.2, "Weights W2 & Biases b2", ha="center", va="center", style="italic", fontsize=10, color="dimgray")
    
    plt.title("Multi-Layer Perceptron (MLP) Neural Network Architecture", fontsize=14, weight="bold", pad=20)
    ax.set_xlim(0.3, 5.7)
    ax.set_ylim(-0.3, 4.6)
    
    vis_dir = "visualization"
    os.makedirs(vis_dir, exist_ok=True)
    plt.savefig(os.path.join(vis_dir, "mlp_structure.png"), dpi=300, bbox_inches="tight")
    plt.close()

def draw_vector_mapping():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    # Vector box represent
    box_y = 1.5
    box_h = 1.0
    
    # Draw flat vector representation
    rect = patches.Rectangle((0.5, box_y), 9.0, box_h, edgecolor="black", facecolor="#eaeaea", lw=2)
    ax.add_patch(rect)
    
    # Segments
    # Total dim = W1 + b1 + W2 + b2
    # e.g. 3*4 + 4 + 4*1 + 1 = 12 + 4 + 4 + 1 = 21 parameters
    sections = [
        {"name": "W1 (Weights Input-Hidden)", "range": "0 to 11", "width": 4.5, "start": 0.5, "color": "#aec7e8"},
        {"name": "b1 (Biases Hidden)", "range": "12 to 15", "width": 1.5, "start": 5.0, "color": "#98df8a"},
        {"name": "W2 (Weights Hidden-Output)", "range": "16 to 19", "width": 2.0, "start": 6.5, "color": "#ff9896"},
        {"name": "b2 (Bias Output)", "range": "20", "width": 1.0, "start": 8.5, "color": "#f7b6d2"}
    ]
    
    for sec in sections:
        s_rect = patches.Rectangle((sec["start"], box_y), sec["width"], box_h, edgecolor="black", facecolor=sec["color"], lw=1.5)
        ax.add_patch(s_rect)
        # Text labels
        ax.text(sec["start"] + sec["width"]/2, box_y + box_h/2, f"{sec['name']}\nIndices: {sec['range']}", 
                ha="center", va="center", fontsize=8, weight="bold")
        
    ax.text(5.0, 3.2, "Search Agent Representation (1D Flat Position Vector)", ha="center", va="center", weight="bold", fontsize=13)
    ax.text(5.0, 0.8, "The bio-inspired optimizer optimizes this flat array.\nDuring fitness evaluation, the vector is unpacked back into the 2D weight matrices and biases.", 
            ha="center", va="center", style="italic", fontsize=10, color="dimgray")
            
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.8)
    
    vis_dir = "visualization"
    plt.savefig(os.path.join(vis_dir, "flat_vector_mapping.png"), dpi=300, bbox_inches="tight")
    plt.close()

def draw_optimization_flowchart():
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis("off")
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    # helper for boxes
    def draw_box(x, y, w, h, text, color):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", edgecolor="black", facecolor=color, lw=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", weight="bold", fontsize=9)
        
    def draw_arrow(x_start, y_start, x_end, y_end):
        ax.annotate("", xy=(x_end, y_end), xytext=(x_start, y_start),
                    arrowprops=dict(arrowstyle="->", lw=2, color="black"))
                    
    # Flow steps
    draw_box(3.5, 6.5, 3.0, 0.8, "Start Optimization\nInitialize Population (Random Vectors)", "#d3d3d3")
    draw_arrow(5.0, 6.5, 5.0, 5.5)
    
    draw_box(3.5, 4.7, 3.0, 0.8, "Unpack Vector into MLP\nWeights and Biases", "#aec7e8")
    draw_arrow(5.0, 4.7, 5.0, 3.7)
    
    draw_box(3.5, 2.9, 3.0, 0.8, "Compute Neural Network Loss\n(MSE / Cross-Entropy Loss)\n= Fitness Score", "#ff9896")
    draw_arrow(5.0, 2.9, 5.0, 1.9)
    
    draw_box(3.5, 1.1, 3.0, 0.8, "Update Vector Positions\n(SBOA, PSO, GTO formulas)\nbased on Fitness", "#98df8a")
    
    # Loop back arrow
    ax.annotate("", xy=(3.5, 5.1), xytext=(3.5, 1.5),
                arrowprops=dict(arrowstyle="->", lw=2, color="dimgray", connectionstyle="bar,fraction=-0.3"))
    ax.text(1.8, 3.3, "Loop Iterations\n(Max Iterations)", ha="center", va="center", style="italic", color="dimgray", fontsize=9)
    
    # End arrow
    draw_arrow(6.5, 1.5, 8.0, 1.5)
    draw_box(8.0, 1.1, 1.8, 0.8, "Optimal Weights Found\nSave Best Model", "#ffbb78")
    
    plt.title("MLP Training Loop using Bio-Inspired Metaheuristics", fontsize=14, weight="bold", pad=20)
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0.5, 7.8)
    
    vis_dir = "visualization"
    plt.savefig(os.path.join(vis_dir, "optimization_flowchart.png"), dpi=300, bbox_inches="tight")
    plt.close()

def main():
    print("Generating visualizations...")
    draw_mlp()
    draw_vector_mapping()
    draw_optimization_flowchart()
    print("Done! Visualizations saved to the 'visualization' directory.")

if __name__ == "__main__":
    main()
