import torch
import os
import logging
import time
from pathlib import Path
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import to_undirected

class SealSramDataset(InMemoryDataset):
    def __init__(
        self,
        name,
        root,
        neg_edge_ratio=1.0,
        to_undirected=True,
        task_level='node',
        task_type='classification',  # 新增：'classification' 或 'regression'
        transform=None, 
        pre_transform=None
    ) -> None:
        
        self.name = 'sram'
        self.task_type = task_type
        
        # 定义分类和回归数据集名称
        self.classification_datasets = [
            # 暂时没有分类数据集
        ]
        
        self.regression_datasets = [
            # 所有数据集都用于回归
            'integrated_position_prediction_graph',
            'integrated_position_prediction_with_global_features',
            'integrated_power_density_prediction_graph',
            'integrated_power_density_prediction_with_global_features'
        ]
        
        # 解析数据集名称并根据任务类型过滤
        if '+' in name:
            all_names = name.split('+')
        else:
            all_names = [name]
        
        # 根据任务类型过滤数据集
        if task_type == 'classification':
            self.names = [n for n in all_names if n in self.classification_datasets]
        elif task_type == 'regression':
            self.names = [n for n in all_names if n in self.regression_datasets]
        else:
            raise ValueError(f"Invalid task_type: {task_type}. Must be 'classification' or 'regression'")
        
        # 如果没有对应任务类型的数据集，设置为空
        if not self.names:
            print(f"No {task_type} datasets found in {all_names}")
            self.empty_dataset = True
            return
        else:
            self.empty_dataset = False
            print(f"SealSramDataset ({task_type}) includes {self.names} circuits")
        
        self.folder = os.path.join(root, self.name)
        self.neg_edge_ratio = neg_edge_ratio
        self.to_undirected = to_undirected
        self.data_lengths = {}
        self.data_offsets = {}
        self.task_level = task_level
    
        # 统一10维
        self.max_net_node_feat = torch.ones((1, 10))
        self.max_dev_node_feat = torch.ones((1, 10))
        
        # 初始化父类
        super().__init__(self.folder, transform, pre_transform)
        
        # 如果是空数据集，直接返回
        if self.empty_dataset:
            return
        
        # 加载所有数据
        data_list = []
        for i, name in enumerate(self.names):
            if os.path.exists(self.processed_paths[i]):
                # 修复PyTorch 2.6安全加载问题
                try:
                    loaded_data, loaded_slices = torch.load(self.processed_paths[i], weights_only=False)
                except Exception:
                    # 如果上面失败，尝试添加安全全局变量
                    try:
                        from torch_geometric.data.data import DataEdgeAttr
                        torch.serialization.add_safe_globals([DataEdgeAttr])
                        loaded_data, loaded_slices = torch.load(self.processed_paths[i])
                    except Exception:
                        # 最后尝试上下文管理器
                        from torch_geometric.data.data import DataEdgeAttr
                        with torch.serialization.safe_globals([DataEdgeAttr]):
                            loaded_data, loaded_slices = torch.load(self.processed_paths[i])
                
                self.data_offsets[name] = len(data_list)
                
                if self.task_level == 'node':
                    self.data_lengths[name] = loaded_data.y.size(0) if hasattr(loaded_data, 'y') else loaded_data.num_nodes
                elif self.task_level == 'edge':
                    self.data_lengths[name] = loaded_data.edge_label.size(0) if hasattr(loaded_data, 'edge_label') else loaded_data.num_edges
                else:
                    raise ValueError(f"Invalid task level: {self.task_level}")

                if loaded_slices is not None:
                    # 简化处理，如果没有utils模块
                    data_list.append(loaded_data)
                else:
                    data_list.append(loaded_data)
                
                print(f"load processed {name}, "+
                      f"len(data_list)={self.data_lengths[name]}, "+
                      f"data_offset={self.data_offsets[name]} ")
        
        # 合并多个图到数据列表
        if data_list:
            self.data, self.slices = self.collate(data_list)

    def sram_graph_load(self, name, raw_path):
        logging.info(f"raw_path: {raw_path}")
        g = torch.load(raw_path, weights_only=False, map_location='cpu')
        if isinstance(g, list):
            g = g[0]
        
        print(f"Loaded graph: {g}")
        
        g.name = name
        
        if hasattr(g, 'node_type'):
            g.x = g.node_type.view(-1, 1)
        
        # 处理标签
        if self.task_level == 'node':
            if not hasattr(g, 'y') or g.y is None:
                g.y = torch.zeros(g.num_nodes, 1)
            else:
                g.y = normalize_y(g.y)
        
        elif self.task_level == 'edge':
            if hasattr(g, 'edge_index'):
                g.edge_label_index = g.edge_index
                g.edge_label = torch.zeros(g.num_edges, 1)
        
        # 转为无向图
        if self.to_undirected and hasattr(g, 'edge_index'):
            if hasattr(g, 'edge_type'):
                g.edge_index, g.edge_type = to_undirected(
                    g.edge_index, g.edge_type, g.num_nodes, reduce='mean'
                )
            else:
                g.edge_index = to_undirected(g.edge_index, num_nodes=g.num_nodes)
        
        print(f"Processed graph - Nodes: {g.num_nodes}, Edges: {g.num_edges}")
        return g

    def single_g_process(self, idx: int):
        if self.empty_dataset:
            return 0
            
        print(f"processing dataset {self.names[idx]}...")
        
        # 加载图
        graph = self.sram_graph_load(self.names[idx], self.raw_paths[idx])
        
        if self.task_level == 'node':
            torch.save((graph, None), self.processed_paths[idx])
            return graph.y.size(0)
        elif self.task_level == 'edge':
            torch.save((graph, None), self.processed_paths[idx])
            return graph.edge_label.size(0)
        else:
            raise ValueError(f"Invalid task level: {self.task_level}")

    def process(self):
        if self.empty_dataset:
            return
            
        for i, name in enumerate(self.names):
            if os.path.exists(self.processed_paths[i]):
                logging.info(f"Found process file of {name} in {self.processed_paths[i]}, skipping process()")
                continue 
            
            self.single_g_process(i)

    @property
    def raw_file_names(self):
        if self.empty_dataset:
            return []
            
        raw_file_names = []
        for name in self.names:
            raw_file_names.append(name+'.pt')
        return raw_file_names
    
    @property
    def processed_dir(self) -> str:
        if self.task_level == 'edge':
            base_dir = 'processed_for_edges'
        elif self.task_level == 'node':
            base_dir = 'processed_for_nodes'
        else:
            raise ValueError(f"No defination of task {self.task_level}!")
        
        # 为不同任务类型创建不同的目录
        return os.path.join(self.root, f"{base_dir}_{self.task_type}")

    @property
    def processed_file_names(self):
        if self.empty_dataset:
            return []
            
        processed_names = []
        for i, name in enumerate(self.names):
            if self.neg_edge_ratio < 1.0:
                name += f"_nr{self.neg_edge_ratio:.1f}"
            processed_names.append(name+"_processed.pt")
        return processed_names

def performat_SramDataset(dataset_dir, name, 
                          neg_edge_ratio, to_undirected, 
                          task_level):
    """
    返回分类和回归两个数据集对象
    """
    start = time.perf_counter()

    # 创建分类数据集
    try:
        classification_dataset = SealSramDataset(
            name=name, root=dataset_dir,
            neg_edge_ratio=neg_edge_ratio,
            to_undirected=to_undirected,
            task_level=task_level,
            task_type='classification'
        )
        if classification_dataset.empty_dataset:
            classification_dataset = None
    except Exception as e:
        print(f"Error creating classification dataset: {e}")
        classification_dataset = None
    
    # 创建回归数据集
    try:
        regression_dataset = SealSramDataset(
            name=name, root=dataset_dir,
            neg_edge_ratio=neg_edge_ratio,
            to_undirected=to_undirected,
            task_level=task_level,
            task_type='regression'
        )
        if regression_dataset.empty_dataset:
            regression_dataset = None
    except Exception as e:
        print(f"Error creating regression dataset: {e}")
        regression_dataset = None

    elapsed = time.perf_counter() - start
    timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed)) \
            + f'{elapsed:.2f}'[-3:]
    print(f"PID = {os.getpid()}")
    print(f"Building datasets from {name} took {timestr}")
    
    if classification_dataset is not None:
        print('Classification Dataloader: Loading success.')
    else:
        print('No classification datasets found.')
        
    if regression_dataset is not None:
        print('Regression Dataloader: Loading success.')
    else:
        print('No regression datasets found.')

    return classification_dataset, regression_dataset

def normalize_y(y):
    """
    对y进行归一化处理，将每个真实值减去最大值并除以（最大值减去最小值）。

    Args:
        y (torch.Tensor): 原始的标签张量。

    Returns:
        torch.Tensor: 归一化后的标签张量。
    """
    max_y = y.max()
    min_y = y.min()

    # 使用公式进行归一化
    normalized_y = (y - min_y) / (max_y - min_y)

    return normalized_y

# 便利函数：只获取回归数据集
def get_regression_dataset(dataset_dir, name, 
                          neg_edge_ratio, to_undirected, 
                          task_level):
    """
    只返回回归数据集
    """
    _, regression_dataset = performat_SramDataset(
        dataset_dir, name, neg_edge_ratio, to_undirected, task_level
    )
    return regression_dataset

# 便利函数：只获取分类数据集
def get_classification_dataset(dataset_dir, name, 
                              neg_edge_ratio, to_undirected, 
                              task_level):
    """
    只返回分类数据集
    """
    classification_dataset, _ = performat_SramDataset(
        dataset_dir, name, neg_edge_ratio, to_undirected, task_level
    )
    return classification_dataset