import torch
from sklearn.model_selection import train_test_split
import numpy as np
import os
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import to_undirected
import logging
import time
from pathlib import Path
from utils import (
    get_pos_neg_edges, add_tar_edges_to_g, get_balanced_edges, 
    collated_data_separate)
from torch.utils.data import Dataset
from collections import Counter
from torch_geometric.data import Batch
from torch_geometric.utils import subgraph
import matplotlib
matplotlib.use('Agg')     # 或者 'Qt5Agg'，取决于你的系统上装了哪个 GUI 库
import matplotlib.pyplot as plt
import numpy as np
from plot import visualize_node_label_distribution, visualize_edge_label_distribution, plot_edge_label_distribution, visualize_tar_edge_y_distribution

# 指定中文字体（以黑体 SimHei 为例，Windows 下通常就有；Linux 则需先安装）
plt.rc("font",family='Nimbus Sans')

class SealSramDataset(InMemoryDataset):
    def __init__(
        self,
        name, #add
        root, #add
        neg_edge_ratio=1.0,
        to_undirected=True,
        sample_rates=[1.0], 
        task_level='node',
        task_type='resister',
        net_only=False,
        transform=None, 
        pre_transform=None,
        class_boundaries=[0.2, 0.4, 0.6, 0.8]
    ) -> None:
        """ The SRAM dataset. 
        It can be a combination of several large circuit graphs or millions of sampled subgraphs.
        Args:
            name (str): The name of the dataset.
            root (str): The root directory of the dataset.
            neg_edge_ratio (float): The ratio of negative edges to positive edges.
            to_undirected (bool): Whether to convert the graph to an undirected graph.
            sample_rates (list): The sampling rates of target edges for each dataset.
            task_type (str): The task type. It can be 'classification', 'regression', 'node_classification', 'node_regression'.
            num_classes (int): The number of classes.
            class_boundaries (list): The boundaries of the classes.
        """
        self.name = 'sram'
        self.class_boundaries = torch.tensor(class_boundaries)
        print("self.class_boundaries", self.class_boundaries)
        # assert 0
        ## split the dataset according to '+' in the name
        if '+' in name:
            self.names = name.split('+')
        else:
            self.names = [name]
            
        print(f"SealSramDataset includes {self.names} circuits")

        self.sample_rates = sample_rates
        assert len(self.names) == len(self.sample_rates), \
            f"len of dataset:{len(self.names)}, len of sample_rate: {len(self.sample_rates)}"
        
        self.folder = os.path.join(root, self.name)
        self.neg_edge_ratio = neg_edge_ratio
        self.to_undirected = to_undirected
        ## data_lengths can be the number of total datasets or the number of subgraphs
        self.data_lengths = {}
        ## offset index for each graph
        self.data_offsets = {}

        self.task_level = task_level
        self.net_only = net_only
        self.task_type = task_type
    
        self.max_net_node_feat = torch.ones((1, 17)) # the max feature dimension of net and dev nodes
        self.max_dev_node_feat = torch.ones((1, 17))

        super().__init__(self.folder, transform, pre_transform)
        data_list = []

        for i, name in enumerate(self.names):
            ## If a processed data file exsit, we load it directly
            loaded_data, loaded_slices = torch.load(self.processed_paths[i])
              
            self.data_offsets[name] = len(data_list)
            if self.task_level == 'node':
                self.data_lengths[name] = loaded_data.y.size(0)
            elif self.task_level == 'edge':
                self.data_lengths[name] = loaded_data.edge_label.size(0)
            else:
                raise ValueError(f"Invalid task level: {self.task_level}")

            if loaded_slices is not None:
                data_list += collated_data_separate(loaded_data, loaded_slices)#[:data_len]
            else:
                data_list.append(loaded_data)
            
            print(f"load processed {name}, "+
                  f"len(data_list)={self.data_lengths[name]}, "+
                  f"data_offset={self.data_offsets[name]} ")
            
        ## combine multiple graphs into data list
        self.data, self.slices = self.collate(data_list)

    def norm_nfeat(self, ntypes,task_type):
        # if self._data is None or self.slices is None:
        #     self.data, self.slices = self.collate(self._data_list)
        #     self._data_list = None
        # 首先检查哪些节点类型实际存在
        if self.task_type == 'capacitance':
            existing_types = torch.unique(self._data.node_type)
            print(f"实际存在的节点类型: {existing_types}")
            
            for ntype in ntypes:
                if ntype not in existing_types:
                    print(f"节点类型 {ntype} 不存在，跳过")
                    continue
                    
                node_mask = self._data.node_type == ntype
                num_nodes_of_type = node_mask.sum().item()
                
                if num_nodes_of_type == 0:
                    print(f"节点类型 {ntype} 数量为0，跳过")
                    continue
                    
                print(f"node_mask.sum():{node_mask.sum()}")
                
                # 如果是 DEV 类型（device 节点），跳过标准化
                if ntype == 1:
                    print(f"跳过 DEV 类型节点标准化（数量: {num_nodes_of_type}）")
                    continue
                    
                # 标准化其他类型节点
                max_node_feat, _ = self._data.node_attr[node_mask].max(dim=0, keepdim=True)
                max_node_feat[max_node_feat == 0.0] = 1.0
                self._data.node_attr[node_mask] /= max_node_feat
                print(f"标准化类型 {ntype} 节点（数量: {num_nodes_of_type}）")

            if self.task_level == 'edge':
                ## normalize edge_label i.e., coupling capacitance
                self._data.edge_label = torch.log10(self._data.edge_label * 1e21)

                self._data.edge_label /= 6

                self._data.edge_label[self._data.edge_label < 0] = 0.0
                
                self._data.edge_label[self._data.edge_label > 1] = 1.0
                edge_label_c = torch.bucketize(self._data.edge_label, self.class_boundaries)
                
                num_classes = len(self.class_boundaries) + 1
                edge_labels_np = self._data.edge_label.cpu().numpy().flatten()
                class_bounds_np = self.class_boundaries.cpu().numpy()
                plot_edge_label_distribution(edge_labels_np, class_bounds_np)
                plt.savefig(f'imgs/edge_label_dist_{num_classes}.png')
                self._data.edge_label = torch.stack(
                    [self._data.edge_label, edge_label_c], dim=1
                )
                self._data_list = None

            elif self.task_level == 'node':
                self._data.y = torch.log10(self._data.y * 1e20) / 6

                self._data.y[self._data.y < 0] = 0.0
                self._data.y[self._data.y > 1] = 1.0
                node_label_c = torch.bucketize(self._data.y.squeeze(), self.class_boundaries)
                
                artificial_mask = torch.log10(self._data.y * 1e20) / 6 < -1.5  # 识别阈值
        
                ## 绘图时只使用真实数据
                if artificial_mask.any():
                    valid_labels = self._data.y[~artificial_mask].cpu().numpy().flatten()
                    print(f"[绘图过滤] 已排除{artificial_mask.sum().item()}个人工添加的节点")
                else:
                    valid_labels = self._data.y.cpu().numpy().flatten()
                
                num_classes = len(self.class_boundaries) + 1
                plot_edge_label_distribution(valid_labels, self.class_boundaries.cpu().numpy())
                plt.savefig(f'imgs/node_label_dist_{num_classes}.png')
                plt.close()

                self._data.y = torch.stack(
                    [self._data.y.squeeze(), node_label_c], dim=1
                )
                print("self._data.y", self._data.y)
                self._data_list = None

        elif self.task_type == 'resister':

            if self.task_level == 'edge':
                
                current_offset = 0
                normalized_labels = []
                
                for dataset_name in self.names:
                    dataset_length = self.data_lengths[dataset_name]
                    # 获取当前数据集的标签
                    dataset_labels = self._data.edge_label[current_offset:current_offset + dataset_length]
                    
                    # 对当前数据集单独归一化
                    edge_min = dataset_labels.min()
                    edge_max = dataset_labels.max()
                    
                    if edge_max > edge_min:
                        normalized_dataset_labels = (dataset_labels - edge_min) / (edge_max - edge_min)
                    else:
                        normalized_dataset_labels = dataset_labels  # 如果最大最小值相等，保持原值
                    
                    # 限制范围到 [0, 1]
                    normalized_dataset_labels[normalized_dataset_labels < 0] = 0.0
                    normalized_dataset_labels[normalized_dataset_labels > 1] = 1.0
                    
                    normalized_labels.append(normalized_dataset_labels)
                    
                    # print(f"数据集 {dataset_name}: 原始范围 [{edge_min:.6f}, {edge_max:.6f}] -> 归一化范围 [0, 1]")
                    current_offset += dataset_length

                    ''' 画出整个数据集的标签分布图 '''  
                    tar_edge_y_np = normalized_dataset_labels.detach().cpu().numpy()
                    # 创建单张直方图
                    plt.figure(figsize=(10, 6))
                    ax = plt.gca()
                    ax.set_facecolor('white')          # 白底
                    plt.gcf().set_facecolor('white')   # 白底

                    plt.hist(tar_edge_y_np, bins=50, density=True, color='orange', edgecolor='white', alpha=0.8)  # 橙色柱状图
                    #plt.title(f'Edge Label Distribution ', fontsize=14, fontweight='bold')
                    plt.xlabel('normalized label', fontsize=12)
                    plt.ylabel('density', fontsize=12)
                    ax.grid(True, color='lightgray', linestyle='-', linewidth=0.5)  # 灰网格

                    # 保存图片
                    plt.tight_layout()
                    plt.savefig(f'imgs/{dataset_name}_tar_edge_y_dist.png', dpi=300, bbox_inches='tight', facecolor='white')
                    plt.close()  # 释放内存
                # 将所有归一化后的标签合并
                self._data.edge_label = torch.cat(normalized_labels, dim=0)            

                edge_label_c = torch.bucketize(self._data.edge_label.squeeze(), self.class_boundaries)
        
                '''添加：显示每个数据集的分布统计''' 
                print("\n=== 归一化后各数据集的label分布 ===")
                current_offset = 0
                for dataset_name in self.names:
                    dataset_length = self.data_lengths[dataset_name]
                    dataset_labels = self._data.edge_label[current_offset:current_offset + dataset_length]
                    dataset_class_labels = edge_label_c[current_offset:current_offset + dataset_length]
                    
                    print(f"\n数据集: {dataset_name}")
                    print(f"样本数量: {dataset_length}")
                    print(f"归一化值范围: [{dataset_labels.min():.4f}, {dataset_labels.max():.4f}]")
                    
                    # 统计各类别分布
                    unique_classes, class_counts = torch.unique(dataset_class_labels, return_counts=True)
                    print(f"类别分布:")
                    for cls, count in zip(unique_classes, class_counts):
                        percentage = count.item() / dataset_length * 100
                        print(f"  类别 {cls.item()}: {count.item()} 样本 ({percentage:.2f}%)")
                    
                    current_offset += dataset_length
                print("=" * 50)

                self._data.edge_label = torch.stack(
                    [self._data.edge_label, edge_label_c], dim=1
                )
                # assert 0
                self._data_list = None

            elif self.task_level == 'node':

                self._data.y = torch.log10(self._data.y * 1e20) / 6

                self._data.y[self._data.y < 0] = 0.0
                self._data.y[self._data.y > 1] = 1.0
                node_label_c = torch.bucketize(self._data.y.squeeze(), self.class_boundaries)
                
                artificial_mask = torch.log10(self._data.y * 1e20) / 6 < -1.5  # 识别阈值
        
                ## 绘图时只使用真实数据
                if artificial_mask.any():
                    valid_labels = self._data.y[~artificial_mask].cpu().numpy().flatten()
                    print(f"[绘图过滤] 已排除{artificial_mask.sum().item()}个人工添加的节点")
                else:
                    valid_labels = self._data.y.cpu().numpy().flatten()
                
                ## 保持原有的存储方式不变
                
                ## 使用过滤后的数据绘图
                num_classes = len(self.class_boundaries) + 1
                plot_edge_label_distribution(valid_labels, self.class_boundaries.cpu().numpy())
                plt.savefig(f'imgs/node_label_dist_{num_classes}.png')
                plt.close()
                

                self._data.y = torch.stack(
                    [self._data.y.squeeze(), node_label_c], dim=1
                )
                print("self._data.y", self._data.y)
                self._data_list = None

    def set_cl_embeds(self, embeds):
        """
        Set dataset attribute `x` as the embeddings learned by SGRL.
        Args:
            embeds (torch.Tensor): The embeddings [N, cl_hid_dim] learned by SGRL.
        """
        if self._data is None or self.slices is None:
            self.data, self.slices = self.collate(self._data_list)
            self._data_list = None

        self._data.x = embeds
        self._data_list = None
        print("Setting CL embeddings to x...")
        print('self._data', self._data)
        # print('self.slices', self.slices)

    def sram_graph_load(self, name, raw_path):
        """
        In the loaded circuit graph, ground capacitance values are stored in "tar_node_y' attribute of the node. 
        There are two edge sets. 
        The first is the connections existing in circuit topology, attribute names start with "edge". 
        For example, "g.edge_index" and "g.edge_type".
        The second is edges to be predicted, which are parasitic coupling edges. Their attribute names start with "tar_edge". 
        For example, "g.tar_edge_index" and "g.tar_edge_type".
        Coupling capacitance values are stored in the 'tar_edge_y' attribute of the edge.
        Args:
            name (str): The name of the dataset.
            raw_path (str): The path of the raw data file.
        Returns:
            g (torch_geometric.data.Data): The processed homo graph data.
            处理后的图包含：

        节点特征：g.x（节点类型）和g.node_attr（电路统计特征）
        边连接：g.edge_index（结构边）
        目标数据：

        g.tar_node_y：节点接地电容（节点级任务）
        g.tar_edge_y：边耦合电容（边级任务）
        g.tar_edge_index：目标边的连接关系
        """
        # print(f"name:{name}")
        # assert 0
        logging.info(f"raw_path: {raw_path}")
        hg = torch.load(raw_path, weights_only=False)
        if isinstance(hg, list):
            hg = hg[0]

        power_net_ids = torch.tensor([0, 1])
        
        if name == "sandwich":
            # VDD VSS TTVDD
            power_net_ids = torch.tensor([0, 1, 1422])
        elif name == "ultra8t":
            # VDD VSS SRMVDD
            power_net_ids = torch.tensor([0, 1, 377])
        elif name == "sram_sp_8192w":
            # VSSE VDDCE VDDPE
            power_net_ids = torch.tensor([0, 1, 2])
        elif name == "ssram":
            # VDD VSS VVDD
            power_net_ids = torch.tensor([0, 1, 352])
        elif name == "digtime":
            power_net_ids = torch.tensor([0, 1])
        elif name == "timing_ctrl":
            power_net_ids = torch.tensor([0, 1])
        elif name == "array_128_32_8t":
            power_net_ids = torch.tensor([0, 1])
        
        """ graph transform """ 
        ### remove the power pins
        subset_dict = {}
        for ntype in hg.node_types:
            subset_dict[ntype] = torch.ones(hg[ntype].num_nodes, dtype=torch.bool)
            if ntype == 'net':
                subset_dict[ntype][power_net_ids] = False

        hg = hg.subgraph(subset_dict)
        if self.task_type == 'resister':
            hg = hg.edge_type_subgraph([
                ('device', 'device-pin', 'pin'),  # 器件到引脚的连接
                ('pin', 'pin-net', 'net'),        # 引脚到网络的连接
                ('pin', 'r_p2p', 'pin'),       
            ])
        else:
            hg = hg.edge_type_subgraph([
                ('device', 'device-pin', 'pin'),
                ('pin', 'pin-net', 'net'),
                ('pin', 'cc_p2n', 'net'),
                ('pin', 'cc_p2p', 'pin'),
                ('net', 'cc_n2n', 'net'),
            ])
        print(f"hg:{hg}")

        g = hg.to_homogeneous() # 异构图转同构图 Data(edge_index=[2, 931250], x=[249570, 17], y=[931250], node_type=[249570], edge_type=[931250])
        g.name = name
        assert hasattr(g, 'node_type')
        assert hasattr(g, 'edge_type')
        edge_offset = 0
        tar_edge_y = []
        tar_node_y = []
        g._n2type = {}
        node_feat = []
        max_feat_dim = 17 # the max feature dimension of nodes
        print(hg.node_types)

        hg['device'].y = torch.ones((hg['device'].x.shape[0], 1)) * 1e-30   # 1e-30 is the minimum ground capacitance value

        for n, ntype in enumerate(hg.node_types):
            g._n2type[ntype] = n # 节点类型到整数的映射
            feat = hg[ntype].x
            feat = torch.nn.functional.pad(feat, (0, max_feat_dim-feat.size(1)))
            node_feat.append(feat)
            tar_node_y.append(hg[ntype].y)
        
        ## There is 'node_type' attribute after transforming hg to g.
        ## The 'node_type' is used as default node feature, g.x.
        g.x = g.node_type.view(-1, 1)
        ## circuit statistic features
        g.node_attr = torch.cat(node_feat, dim=0)

        if self.task_level == 'node' :
            if self.net_only:
                net_mask = g.node_type == g._n2type['net'] 
                g.tar_node_y = torch.zeros((g.num_nodes, 1)) 
                net_nodes = torch.where(net_mask)[0]
                g.tar_node_y[net_nodes] = hg['net'].y  # assign the ground capacitance value to the net node
            else:
                ## lumped ground capacitance on net/pin nodes
                g.tar_node_y = torch.cat(tar_node_y, dim=0)
            g.y = g.tar_node_y

        g._e2type = {}

        for e, (edge, store) in enumerate(hg.edge_items()):
            g._e2type[edge] = e
            if self.task_type == 'resister':
                if 'r' in edge[1]: # 寄生耦合边
                    tar_edge_y.append(store['y']) 
                else:
                    edge_offset += store['edge_index'].shape[1]
            else:
                if 'cc' in edge[1]:
                    tar_edge_y.append(store['y']) 
                else:
                    edge_offset += store['edge_index'].shape[1]

        g._num_ntypes = len(g._n2type)
        g._num_etypes = len(g._e2type)

        logging.info(f"g._n2type {g._n2type}")
        logging.info(f"g._e2type {g._e2type}")

        tar_edge_index = g.edge_index[:, edge_offset:]
        tar_edge_type = g.edge_type[edge_offset:]
        tar_edge_y = torch.cat(tar_edge_y)

        ## restrict the capcitance value range 
        if self.task_type == 'capacitance' :
            legel_edge_mask = (tar_edge_y < 1e-15) & (tar_edge_y > 1e-21)

        if self.task_level == 'node' :
            legal_node_mask = (g.tar_node_y < 1e-15) & (g.tar_node_y > 1e-21)
            legal_node_mask = legal_node_mask.squeeze()
            print(f"(~legal_node_mask).sum().item():{(~legal_node_mask).sum().item()},legal_node_mask:{legal_node_mask.size()}")

            g.tar_node_y[~legal_node_mask] = 1e-30
            visualize_node_label_distribution(g, name, self.class_boundaries)

        if self.task_type == 'resister':
            ''' 过滤方式 1 '''
            lower_percentile = 50  # 下界阈值       
            upper_percentile = 400
            
            ''' 过滤方式 2 '''
            # lower_percentile = torch.quantile(tar_edge_y, 0.3)
            # upper_percentile = torch.quantile(tar_edge_y, 0.9)  # 上界使用90百分位数
            
            legel_edge_mask = (tar_edge_y >= lower_percentile) & (tar_edge_y <= upper_percentile)
            
            print(f"Original tar_edge_y count: {len(tar_edge_y)}")
            print(f"After filtering (removing top/bottom 5%): {legel_edge_mask.sum().item()}")
            print(f"Lower bound: {lower_percentile:.2e}, Upper bound: {upper_percentile:.2e}")

            print(f"\nProcessing dataset {name}, node types: {torch.unique(g.node_type)}")  # 显示过滤后的节点类型
            print(f"Node type counts: {torch.bincount(g.node_type)}")  # 各类型节点数量

        g.tar_edge_y = tar_edge_y[legel_edge_mask]
        g.tar_edge_index = tar_edge_index[:, legel_edge_mask]  
        g.tar_edge_type = tar_edge_type[legel_edge_mask]

        print(f"g.tar_edge_y.size:{g.tar_edge_y.size()},g.name:{g.name}")
        visualize_tar_edge_y_distribution(g.tar_edge_y, g.name)        

        _, g.tar_edge_dist = g.tar_edge_type.unique(return_counts=True)

        g.edge_type = g.edge_type[0:edge_offset]
        g.edge_index = g.edge_index[:, 0:edge_offset]
        
        if self.task_type == 'capacitance' :
            # Visualization for both node and edge tasks
            if self.task_level == 'node':
                processed_labels = torch.log10(g.tar_node_y * 1e21) / 6
                artificial_mask = torch.isclose(g.tar_node_y, torch.tensor(1e-30), atol=1e-32)
                task_name = "node"
            else:
                processed_labels = torch.log10(g.tar_edge_y * 1e21) / 6
                artificial_mask = torch.zeros_like(processed_labels, dtype=torch.bool)
                task_name = "edge"

            processed_labels[processed_labels < 0] = 0.0
            processed_labels[processed_labels > 1] = 1.0

            valid_labels_np = processed_labels[~artificial_mask].cpu().numpy()

            # 强制设置图像大小为1750*1350像素
            plt.figure(figsize=(17.5, 13.5))  # 17.5英寸 * 100dpi = 1750像素，13.5英寸 * 100dpi = 1350像素
            ax = plt.gca()
            ax.set_facecolor('white')
            plt.gcf().set_facecolor('white')

            plt.hist(valid_labels_np, 
                    bins=50,
                    density=True,
                    color='orange',
                    edgecolor='white')

            plt.xlabel('normalized label',fontsize=70)
            plt.ylabel('density',fontsize=70)
            plt.xticks(fontsize=50)  # 调整x轴刻度值字体大小
            plt.yticks(fontsize=50)  # 调整y轴刻度值字体大小
            ax.grid(True, color='gray', linestyle='-', linewidth=0.5)

            plt.savefig(f'imgs/{task_name}_label_dist_{name}.png', bbox_inches='tight', pad_inches=0.1, dpi=100)
            plt.close()

        ## convert to undirected edges
        if self.to_undirected:
                g.edge_index, g.edge_type = to_undirected(
                    g.edge_index, g.edge_type, g.num_nodes, reduce='mean'
                )

        return g

    def single_g_process(self, idx: int):
        print(f"processing dataset {self.names[idx]} "+ 
                    f"with sample_rate {self.sample_rates[idx]}...")
        ## we can load multiple graphs
        graph = self.sram_graph_load(self.names[idx], self.raw_paths[idx])
        print(f"loaded graph {graph}")
        if self.task_type == 'resister':
            if self.task_level == 'edge':
                # 直接使用原始的正样本边和标签
                links = graph.tar_edge_index
                labels = graph.tar_edge_y  # 直接使用原有的分桶标签

            elif self.task_level == 'node':
                # 节点分类任务：只设置 y，不触碰边逻辑
                graph.y = graph.tar_node_y.squeeze()

            else:
                raise ValueError(f"No definition of task {self.task_level} in this version!")
        elif self.task_type == 'capacitance':
            if self.task_level == 'edge':
                # 1) 生成负样本
                neg_edge_index, neg_edge_type = get_pos_neg_edges(
                    graph, neg_ratio=self.neg_edge_ratio
                )
                # 2) 平衡正负样本
                (
                    pos_edge_index, pos_edge_type, pos_edge_y,
                    neg_edge_index, neg_edge_type
                ) = get_balanced_edges(
                    graph,
                    neg_edge_index,
                    neg_edge_type,
                    self.neg_edge_ratio,
                    self.sample_rates[idx]
                )
                # 用于 LinkNeighborLoader 的字段
                links = pos_edge_index
                labels = pos_edge_y

            elif self.task_level == 'node':
                # 节点分类任务：只设置 y，不触碰边逻辑
                graph.y = graph.tar_node_y.squeeze()

            else:
                raise ValueError(f"No definition of task {self.task_level} in this version!")
        ## remove the redundant attributes in this version
        del graph.tar_node_y
        del graph.tar_edge_index
        del graph.tar_edge_type
        del graph.tar_edge_y

                
        if self.task_level == 'node':
            torch.save((graph, None), self.processed_paths[idx])
            return graph.y.size(0)
        elif self.task_level == 'edge':
            ## To use LinkNeighborLoader, the target links rename to edge_label_index
            ## target edge labels rename to edge_label
            graph.edge_label_index = links
            graph.edge_label = labels
            torch.save((graph, None), self.processed_paths[idx])
            return graph.edge_label.size(0)
        else:
            raise ValueError(f"No defination of task {self.task_type} in this version!")

    def process(self):
        data_lens_for_split = []
        p = Path(self.processed_dir)
        ## we can have multiple graphs
        for i, name in enumerate(self.names):
            ## if there is a processed file, we skip the self.single_g_process()
            if os.path.exists(self.processed_paths[i]):
                logging.info(f"Found process file of {name} in {self.processed_paths[i]}, skipping process()")
                continue 
                        
            data_lens_for_split.append(
                self.single_g_process(i)
            )

    @property
    def raw_file_names(self):
        raw_file_names = []
        for name in self.names:
            raw_file_names.append(name+'.pt')
        
        return raw_file_names
    
    @property
    def processed_dir(self) -> str:
        if self.task_level == 'edge':
            return os.path.join(self.root, 'processed_for_edges')
        elif self.task_level == 'node':
            return os.path.join(self.root, 'processed_for_nodes')
        else:
            raise ValueError(f"No defination of task {self.task_level}!")

    @property
    def processed_file_names(self):
        processed_names = []
        for i, name in enumerate(self.names):
            if self.sample_rates[i] < 1.0:
                name += f"_s{self.sample_rates[i]}"

            if self.neg_edge_ratio < 1.0:
                name += f"_nr{self.neg_edge_ratio:.1f}"
            processed_names.append(name+"_processed.pt")
        return processed_names

def adaption_for_sgrl(dataset):
    """
    It is only used for contrastive learning (SGRL).
    """
    data_list = []

    for i, name in enumerate(dataset.names):
        single_graph = Data(
            x=dataset[i].node_type, 
            edge_index=dataset[i].edge_index, 
            edge_attr=dataset[i].edge_type
        )
        single_graph.node_attr = dataset[i].node_attr
        data_list.append(single_graph)

    # Create a big graph concate all graphs from the `data_list`
    batch = Batch.from_data_list(data_list)

    ## Change the size of x
    batch.x = batch.x.view(-1, 1)
    ## Rename the `edge_attr` to `edge_type`
    batch.edge_type = batch.edge_attr

    ## remove the redundant attributes
    del batch.edge_attr

    print("attributes in big batch", batch)
    print("batch.ptr", batch.ptr)

    return batch

def performat_SramDataset(dataset_dir, name, 
                          neg_edge_ratio, to_undirected, 
                          small_dataset_sample_rates, large_dataset_sample_rates,
                          task_level,task_type,
                          net_only,
                          class_boundaries
                          ):
    start = time.perf_counter()
    names = name.split('+')
    sr_list = [
        large_dataset_sample_rates if gname == 'ultra8t' or gname == 'sandwich' else small_dataset_sample_rates 
        for gname in names
    ]

    dataset = SealSramDataset(
            name=name, root=dataset_dir,
            neg_edge_ratio=neg_edge_ratio,
            to_undirected=to_undirected,
            sample_rates=sr_list,
            task_level=task_level,
            task_type=task_type,              
            net_only=net_only,
            class_boundaries=class_boundaries
        )

    elapsed = time.perf_counter() - start
    timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed)) \
            + f'{elapsed:.2f}'[-3:]
    print(f"PID = {os.getpid()}")
    print(f"Building dataset {name} took {timestr}")
    print('Dataloader: Loading success.')

    return dataset
