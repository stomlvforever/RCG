import torch
import matplotlib
matplotlib.use('Agg')     # 或者 'Qt5Agg'，取决于你的系统上装了哪个 GUI 库
import matplotlib.pyplot as plt
import numpy as np
plt.rc("font",family='Nimbus Sans')

def visualize_node_label_distribution(g, name, class_boundaries):
    """
    Visualize the distribution of processed node labels.
    
    Args:
        g: Graph object containing tar_node_y (node labels)
        name: Name for saving the output image
        class_boundaries: Bucket boundaries for classification
    """
    # Process node labels
    processed_node_labels = torch.log10(g.tar_node_y * 1e21) / 6

    # Identify artificially added 1e-30 values
    artificial_mask = torch.isclose(g.tar_node_y, torch.tensor(1e-30), atol=1e-32)
    print(f"Artificially added nodes count: {artificial_mask.sum().item()}")

    # Apply clipping to all data
    processed_node_labels[processed_node_labels < 0] = 0.0
    processed_node_labels[processed_node_labels > 1] = 1.0

    # Perform bucketing (using all data)
    node_label_c = torch.bucketize(processed_node_labels.squeeze(), class_boundaries)

    # Get valid labels (non-artificial) for visualization
    valid_labels_np = processed_node_labels[~artificial_mask].cpu().numpy()

    # Create figure with specified style
    plt.figure()
    ax = plt.gca()

    # Set gray background inside plot only
    ax.set_facecolor('lightgray')

    # Set white background for figure (outer area)
    plt.gcf().set_facecolor('white')

    # Plot histogram
    plt.hist(valid_labels_np, 
            bins=50,
            density=True,      # Y-axis as density
            color='orange',    # Orange bars
            edgecolor='white') # White edges

    # Customize labels
    plt.xlabel('normalized label', fontsize=22)  # Larger x-axis label
    plt.ylabel('density', fontsize=22)         # Larger y-axis label

    # Add a white grid for better visibility
    ax.grid(True, color='white', linestyle='-', linewidth=0.5)

    # Save figure
    plt.savefig(f'imgs/node_label_dist_{name}.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()
    
def visualize_edge_label_distribution(g, name, class_boundaries):
    """
    Process and visualize edge label distribution with specified styling.
    
    Args:
        g: Graph object containing edge labels (tar_edge_y)
        name: Identifier for naming the output file
        class_boundaries: Tensor of boundaries for bucketing
        save_dir: Directory to save visualization (default: "imgs")
    """
    # Process edge labels
    edge_labels = torch.log10(g.tar_edge_y * 1e21) / 6

    # Identify artificially added 1e-30 values
    artificial_mask = torch.isclose(g.tar_edge_y, torch.tensor(1e-30, device=g.tar_edge_y.device), 
                      atol=1e-32)
    print(f"Artificially added edge count: {artificial_mask.sum().item()}")

    # Apply clipping
    edge_labels = torch.clamp(edge_labels, 0.0, 1.0)

    # Bucketize labels
    edge_label_c = torch.bucketize(edge_labels.squeeze(), class_boundaries)

    # Prepare valid labels for visualization
    valid_labels_np = edge_labels[~artificial_mask].cpu().numpy()

    # Create styled visualization
    plt.figure()
    ax = plt.gca()
    
    # Set backgrounds
    ax.set_facecolor('lightgray')          # Plot area
    plt.gcf().set_facecolor('white')       # Figure background
    
    # Create histogram
    plt.hist(valid_labels_np,
             bins=50,
             density=True,
             color='orange',
             edgecolor='white')
    
    # Label styling
    plt.xlabel('normalized label', fontsize=22)
    plt.ylabel('density', fontsize=22)
    
    # Grid styling
    ax.grid(True, color='white', linestyle='-', linewidth=0.5)
    
    # Save and close
    plt.savefig(f'imgs/edge_label_dist_{name}.png', 
               bbox_inches='tight', 
               pad_inches=0.1, 
               dpi=300)
    plt.close()
    
    return edge_label_c  # Return bucketed labels for further use if needed

def plot_edge_label_distribution(edge_labels: np.ndarray, class_boundaries: np.ndarray):
    """
    Plots the distribution of edge label classes.
    
    Args:
        edge_labels: 1D numpy array of normalized edge label values (floats in [0, 1]).
        class_boundaries: 1D numpy array of boundaries, e.g. [0.2, 0.4, 0.6, 0.8].
    """
    # Bucketize into class indices
    edge_label_c = np.digitize(edge_labels, class_boundaries, right=False)
    
    # Count occurrences per class
    num_classes = len(class_boundaries) + 1
    counts = np.bincount(edge_label_c, minlength=num_classes)
    
    # Prepare labels
    class_labels = []
    prev = 0.0
    for b in class_boundaries:
        class_labels.append(f"[{prev:.2f}, {b:.2f})")
        prev = b
    class_labels.append(f"[{prev:.2f}, 1.00]")
    
    # Plot
    plt.figure()
    plt.bar(range(num_classes), counts)
    plt.xticks(range(num_classes), class_labels, rotation=45, ha='right')
    plt.xlabel("Edge Label Class Range")
    plt.ylabel("Number of Samples")
    plt.title("Edge Label Class Distribution")
    plt.tight_layout()
    plt.show()
