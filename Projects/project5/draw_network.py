# Shriman Raghav Srinivasan
# CS 5330: Pattern Recognition & Computer Vision
# Project 5: Recognition using Deep Networks - Network Diagram Generator

import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_network_architecture():
    """
    Draws a clean, professional schematic diagram of MyNetwork architecture
    and saves it to network_diagram.png.
    """
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 6)
    ax.axis('off')
    
    # Define layers as (x, y_center, width, height, label, color, description)
    layers = [
        (0.0, 2.5, 0.8, 2.0, "Input\nImage", "#E0E0E0", "28x28x1"),
        
        # Conv1
        (1.8, 2.5, 1.0, 3.0, "Conv 1\n(5x5)", "#FFF2CC", "24x24x10"),
        (3.0, 2.5, 0.8, 2.2, "MaxPool 1\n+ ReLU", "#FFE699", "12x12x10"),
        
        # Conv2
        (4.5, 2.5, 1.0, 2.5, "Conv 2\n(5x5)", "#F8CBAD", "8x8x20\n(50% Drop)"),
        (5.7, 2.5, 0.8, 1.8, "MaxPool 2\n+ ReLU", "#F4B183", "4x4x20"),
        
        # Flatten
        (7.0, 2.5, 0.5, 3.8, "Flatten", "#BDD7EE", "320"),
        
        # FC Layers
        (8.3, 2.5, 0.5, 2.5, "FC 1\n+ ReLU", "#9BC2E6", "50"),
        (9.6, 2.5, 0.5, 1.5, "FC 2\n+ Softmax", "#A9D08E", "10")
    ]
    
    # Draw boxes
    for idx, (x, y, w, h, name, color, desc) in enumerate(layers):
        # Draw box
        rect = patches.FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.03",
            linewidth=1.5, edgecolor="#2B2B2B", facecolor=color,
            zorder=3
        )
        ax.add_patch(rect)
        
        # Text labels
        ax.text(x, y, name, ha='center', va='center', fontsize=9.5, fontweight='bold', color='#1A1A1A', zorder=4)
        ax.text(x, y - h/2 - 0.35, desc, ha='center', va='top', fontsize=9, fontweight='semibold', color='#404040', zorder=4)
        
        # Draw connection arrows (except for the last layer)
        if idx < len(layers) - 1:
            next_x = layers[idx+1][0]
            next_w = layers[idx+1][2]
            arrow_start = x + w/2 + 0.05
            arrow_end = next_x - next_w/2 - 0.08
            ax.annotate("",
                        xy=(arrow_end, y), xytext=(arrow_start, y),
                        arrowprops=dict(arrowstyle="-|>", color="#404040", lw=2, mutation_scale=15),
                        zorder=2)
            
    # Add title and annotations
    ax.text(5.0, 5.2, "MyNetwork CNN Architecture", ha='center', va='center', fontsize=15, fontweight='bold', color='#1A1A1A')
    
    plt.tight_layout()
    plt.savefig("network_diagram.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("Successfully generated network_diagram.png")

def main(argv):
    draw_network_architecture()

if __name__ == "__main__":
    main(sys.argv)
