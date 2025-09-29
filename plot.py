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
    
def visualize_tar_edge_y_distribution(tar_edge_y, name, save_dir='imgs'):
    """
    可视化tar_edge_y数据的分布情况，包括直方图、箱线图、密度图和累积分布图。
        
    Args:
        tar_edge_y: torch.Tensor, 边的电容值数据
        name: str, 数据集名称，用于文件命名
        save_dir: str, 保存目录，默认为'imgs'
        
    Returns:
        dict: 包含统计信息和过滤分析的字典
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    norm = 0
    
    if norm == 1:
        '''归一化方式 1 '''
        # tar_edge_y /= 300

        '''归一化方式 2 '''
        edge_min = tar_edge_y.min()
        edge_max = tar_edge_y.max()
        # 最大值减最小值的归一化
        if edge_max > edge_min:  # 避免除零错误
            tar_edge_y = (tar_edge_y - edge_min) / (edge_max - edge_min)      
                
        tar_edge_y[tar_edge_y < 0] = 0.0
        tar_edge_y[tar_edge_y > 1] = 1.0    
    # 将tensor转换为numpy数组
    
    tar_edge_y_np = tar_edge_y.detach().cpu().numpy()
    
    # 创建分布图
    plt.figure(figsize=(12, 8))
    
    # 子图1: 直方图 - 应用新的颜色配置
    plt.subplot(2, 2, 1)
    ax1 = plt.gca()
    ax1.set_facecolor('lightgray')          # Plot area
    plt.gcf().set_facecolor('white')       # Figure background
    
    plt.hist(tar_edge_y_np, bins=50, density=True, color='orange', edgecolor='white')
    plt.title('tar_edge_y Distribution (Histogram)')
    plt.xlabel('Capacitance Value')
    plt.ylabel('Density')
    ax1.grid(True, color='white', linestyle='-', linewidth=0.5)
    
    # 子图2: 箱线图
    plt.subplot(2, 2, 2)
    plt.boxplot(tar_edge_y_np)
    plt.title('tar_edge_y Distribution (Boxplot)')
    plt.ylabel('Capacitance Value')
    plt.grid(True, alpha=0.3)
    
    # 子图3: 密度图
    plt.subplot(2, 2, 3)
    plt.hist(tar_edge_y_np, bins=50, density=True, alpha=0.7, color='green', edgecolor='black')
    plt.title('tar_edge_y Density Distribution')
    plt.xlabel('Capacitance Value')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)
    
    # 子图4: 累积分布
    plt.subplot(2, 2, 4)
    sorted_data = np.sort(tar_edge_y_np)
    y_vals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    plt.plot(sorted_data, y_vals, color='red', linewidth=2)
    plt.title('tar_edge_y Cumulative Distribution')
    plt.xlabel('Capacitance Value')
    plt.ylabel('Cumulative Probability')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 使用数据集名称作为文件名，保存到imgs目录
    if norm == 1:
        save_path = f'{save_dir}/after_norm_tar_edge_y_dist_{name}.png'
    else:
        save_path = f'{save_dir}/tar_edge_y_dist_{name}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # 使用close()而不是show()，避免在服务器环境中显示图形
    
    # 计算统计信息
    # stats = {
    #     'count': len(tar_edge_y_np),
    #     'min': tar_edge_y_np.min(),
    #     'max': tar_edge_y_np.max(),
    #     'mean': tar_edge_y_np.mean(),
    #     'median': np.median(tar_edge_y_np),
    #     'std': tar_edge_y_np.std(),
    #     'q25': np.percentile(tar_edge_y_np, 25),
    #     'q75': np.percentile(tar_edge_y_np, 75)
    # }
    
    # # 检查当前过滤条件的影响
    # current_filter = (tar_edge_y < 1e-15) & (tar_edge_y > 1e-21)
    # filtered_count = current_filter.sum().item()
    
    # filter_analysis = {
    #     'current_condition': '1e-21 < value < 1e-15',
    #     'filtered_count': filtered_count,
    #     'total_count': len(tar_edge_y_np),
    #     'retention_rate': filtered_count / len(tar_edge_y_np) * 100,
    #     'suggested_min': tar_edge_y_np.min() * 0.9,
    #     'suggested_max': tar_edge_y_np.max() * 1.1
    # }
    
    # 打印统计信息
    # print(f"\n=== tar_edge_y 统计信息 ===")
    # print(f"数据点数量: {stats['count']}")
    # print(f"最小值: {stats['min']:.4f}")
    # print(f"最大值: {stats['max']:.4f}")
    # print(f"平均值: {stats['mean']:.4f}")
    # print(f"中位数: {stats['median']:.4f}")
    # print(f"标准差: {stats['std']:.4f}")
    # print(f"25%分位数: {stats['q25']:.4f}")
    # print(f"75%分位数: {stats['q75']:.4f}")
    
    # print(f"\n=== 过滤条件分析 ===")
    # print(f"当前过滤条件: {filter_analysis['current_condition']}")
    # print(f"满足条件的数据点: {filter_analysis['filtered_count']} / {filter_analysis['total_count']}")
    # print(f"过滤后保留比例: {filter_analysis['retention_rate']:.2f}%")
    
    # print(f"\n=== 建议的过滤条件 ===")
    # print(f"建议范围: {filter_analysis['suggested_min']:.1f} < value < {filter_analysis['suggested_max']:.1f}")
    
    # print(f"\n图片已保存到: {save_path}")
    
    # return {
    #     'statistics': stats,
    #     'filter_analysis': filter_analysis,
    #     'save_path': save_path
    # }