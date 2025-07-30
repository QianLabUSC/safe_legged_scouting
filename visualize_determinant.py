#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def visualize_determinant_csv():
    """
    Read determinant data from CSV file and create visualization
    """
    # Path to CSV file in home directory
    home_dir = os.path.expanduser("~")
    csv_path = os.path.join(home_dir, "determinant_visualization.csv")
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        print("Make sure to run the C++ node to generate the data first.")
        return
    
    # Read CSV data
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} data points from {csv_path}")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # Extract data
    x = df['x'].values
    y = df['y'].values
    determinant = df['determinant'].values
    
    # Determine grid resolution (assuming square grid)
    unique_x = np.unique(x)
    unique_y = np.unique(y)
    resolution = len(unique_x)
    
    print(f"Grid resolution: {resolution}x{len(unique_y)}")
    print(f"X range: [{x.min():.3f}, {x.max():.3f}]")
    print(f"Y range: [{y.min():.3f}, {y.max():.3f}]")
    print(f"Determinant range: [{determinant.min():.6f}, {determinant.max():.6f}]")
    
    # Reshape data into grid format
    X = x.reshape(resolution, -1)
    Y = y.reshape(resolution, -1)
    Z = determinant.reshape(resolution, -1)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Heatmap visualization
    im1 = ax1.imshow(Z, extent=[x.min(), x.max(), y.min(), y.max()], 
                     origin='lower', cmap='RdYlBu_r', aspect='auto')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_title('Jacobian Determinant Heatmap')
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Determinant Value')
    
    # Contour plot
    contour = ax2.contourf(X, Y, Z, levels=20, cmap='RdYlBu_r')
    ax2.contour(X, Y, Z, levels=20, colors='black', alpha=0.3, linewidths=0.5)
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    ax2.set_title('Jacobian Determinant Contours')
    cbar2 = plt.colorbar(contour, ax=ax2)
    cbar2.set_label('Determinant Value')
    
    # Add grid
    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(home_dir, "determinant_visualization.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    
    # Show statistics
    print(f"\nStatistics:")
    print(f"Mean determinant: {determinant.mean():.6f}")
    print(f"Std determinant: {determinant.std():.6f}")
    print(f"Min determinant: {determinant.min():.6f}")
    print(f"Max determinant: {determinant.max():.6f}")
    
    # Check for near-singular regions (determinant close to 0)
    singular_threshold = 1e-6
    singular_points = np.abs(determinant) < singular_threshold
    if np.any(singular_points):
        print(f"Warning: {np.sum(singular_points)} points have near-singular determinants (|det| < {singular_threshold})")
        singular_coords = list(zip(x[singular_points], y[singular_points]))
        print(f"Singular point coordinates: {singular_coords}")
    
    plt.show()

if __name__ == "__main__":
    visualize_determinant_csv()