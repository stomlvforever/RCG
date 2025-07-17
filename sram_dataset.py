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
from torch_geometric.data import Batch
from torch_geometric.utils import subgraph
import matplotlib
matplotlib.use('Agg')     # 或者 'Qt5Agg'，取决于你的系统上装了哪个 GUI 库
import matplotlib.pyplot as plt
import numpy as np
# 指定中文字体（以黑体 SimHei 为例，Windows 下通常就有；Linux 则需先安装）
plt.rc("font",family='Nimbus Sans')

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

class SealSramDataset(InMemoryDataset):
    def __init__(
        self,
        name, #add
        root, #add
        neg_edge_ratio=1.0,
        to_undirected=True,
        sample_rates=[1.0], 
        task_level='node',
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
    
        self.max_net_node_feat = torch.ones((1, 17)) # the max feature dimension of net and dev nodes
        self.max_dev_node_feat = torch.ones((1, 17))

        super().__init__(self.folder, transform, pre_transform)
        data_list = []

        for i, name in enumerate(self.names):
            ## If a processed data file exsit, we load it directly
            loaded_data, loaded_slices = torch.load(self.processed_paths[i])
            # print("loaded_data", loaded_data)
            # print("loaded_slices", loaded_slices)
            
            
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

    def norm_nfeat(self, ntypes):
        if self._data is None or self.slices is None:
            self.data, self.slices = self.collate(self._data_list)
            self._data_list = None
        # 首先检查哪些节点类型实际存在
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

            # print("="*50)
            # print("边标签分布分析:")
            # print(f"总边标签数量: {len(self._data.edge_label)}")
            
            # # Count values in each range
            # range_0_02 = ((self._data.edge_label >= 0.0) & (self._data.edge_label < 0.2)).sum().item()
            # range_02_04 = ((self._data.edge_label >= 0.2) & (self._data.edge_label < 0.4)).sum().item()
            # range_04_06 = ((self._data.edge_label >= 0.4) & (self._data.edge_label < 0.6)).sum().item()
            # range_06_08 = ((self._data.edge_label >= 0.6) & (self._data.edge_label < 0.8)).sum().item()
            # range_08_1 = ((self._data.edge_label >= 0.8) & (self._data.edge_label <= 1.0)).sum().item()
                    
            # # Verify total count
            # total_counted = range_0_02 + range_02_04 + range_04_06 + range_06_08 + range_08_1
            # print(f"统计总数: {total_counted} (应该匹配总数: {len(self._data.edge_label)})")
            # print("="*50)
            
            
            # # Also print the class distribution after bucketizing
            # print("分桶后的类别分布:")
            # unique_classes, class_counts = torch.unique(edge_label_c, return_counts=True)
            # for i, (cls, count) in enumerate(zip(unique_classes, class_counts)):
            #     percentage = count.item() / len(edge_label_c) * 100
            #     print(f"类别 {cls.item()}: {count.item()} 样本 ({percentage:.2f}%)")
            # print("="*50)
            # # assert 0
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
            ## normalize the node label i.e., lumped ground capacitance
            # print(self._data.y.max())
            # print(self._data.y.min())
            # print(self._data.y) 
            # print(self._data) 
            # assert 0
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
            # print("="*50)
            # print("节点标签分布分析:")
            # print(f"总节点标签数量: {len(self._data.y)}")
            
            # # 定义节点类型名称映射
            # node_type_names = {0: 'NET', 1: 'DEV', 2: 'PIN'}
            # # types, counts = torch.unique(self._data.node_type, return_counts=True)
            # # for t, cnt in zip(types.tolist(), counts.tolist()):
            # #     name = node_type_names.get(t, f'未知类型{t}')
            # #     print(f"node_type = {t}: 名称 = {name}, 数量 = {cnt}")
            # # assert 0
            # # """
            # # 节点标签分布分析:
            # # 总节点标签数量: 22335575
            # # node_type = 0: 名称 = NET, 数量 = 2050724
            # # node_type = 1: 名称 = DEV, 数量 = 5073987
            # # node_type = 2: 名称 = PIN, 数量 = 15210864
            # # """
            # # 获取所有节点类型
            # unique_node_types = torch.unique(self._data.node_type) #['net', 'device', 'pin']
            # # print(f"unique_node_types{unique_node_types}")
            # # assert 0
            # # 分析每种节点类型的标签分布
            # for node_type in unique_node_types:
            #     node_type_mask = self._data.node_type == node_type
            #     node_type_name = node_type_names.get(node_type.item(), f'类型{node_type.item()}')
                
            #     print(f"\n{node_type_name}节点 (类型 {node_type.item()}):")
            #     print(f"  节点数量: {node_type_mask.sum().item()}")
                
            #     # 当前节点类型的标签
            #     node_type_labels = self._data.y[node_type_mask]
                
            #     if len(node_type_labels) > 0:
            #         # 计算每个范围内的样本数
            #         range_0_02 = ((node_type_labels >= 0.0) & (node_type_labels < 0.2)).sum().item()
            #         range_02_04 = ((node_type_labels >= 0.2) & (node_type_labels < 0.4)).sum().item()
            #         range_04_06 = ((node_type_labels >= 0.4) & (node_type_labels < 0.6)).sum().item()
            #         range_06_08 = ((node_type_labels >= 0.6) & (node_type_labels < 0.8)).sum().item()
            #         range_08_1 = ((node_type_labels >= 0.8) & (node_type_labels <= 1.0)).sum().item()
                    
            #         total_node_type = len(node_type_labels)
            #         print(f"  范围 [0.0, 0.2): {range_0_02} 样本 ({range_0_02/total_node_type*100:.2f}%)")
            #         print(f"  范围 [0.2, 0.4): {range_02_04} 样本 ({range_02_04/total_node_type*100:.2f}%)")
            #         print(f"  范围 [0.4, 0.6): {range_04_06} 样本 ({range_04_06/total_node_type*100:.2f}%)")
            #         print(f"  范围 [0.6, 0.8): {range_06_08} 样本 ({range_06_08/total_node_type*100:.2f}%)")
            #         print(f"  范围 [0.8, 1.0]: {range_08_1} 样本 ({range_08_1/total_node_type*100:.2f}%)")
                    
            #         # 验证总数
            #         total_counted = range_0_02 + range_02_04 + range_04_06 + range_06_08 + range_08_1
            #         print(f"  统计总数: {total_counted} (应该匹配: {total_node_type})")
                    
            #         # 分桶后的类别分布
            #         node_type_classes = torch.bucketize(node_type_labels, self.class_boundaries)
            #         unique_classes, class_counts = torch.unique(node_type_classes, return_counts=True)
            #         print(f"  分桶后的类别分布:")
            #         for cls, count in zip(unique_classes, class_counts):
            #             percentage = count.item() / len(node_type_classes) * 100
            #             print(f"    类别 {cls.item()}: {count.item()} 样本 ({percentage:.2f}%)")
            
            # # 总体分布
            # print(f"\n总体分布:")
            # range_0_02 = ((self._data.y >= 0.0) & (self._data.y < 0.2)).sum().item()
            # range_02_04 = ((self._data.y >= 0.2) & (self._data.y < 0.4)).sum().item()
            # range_04_06 = ((self._data.y >= 0.4) & (self._data.y < 0.6)).sum().item()
            # range_06_08 = ((self._data.y >= 0.6) & (self._data.y < 0.8)).sum().item()
            # range_08_1 = ((self._data.y >= 0.8) & (self._data.y <= 1.0)).sum().item()
            
            # total_nodes = len(self._data.y)
            # print(f"范围 [0.0, 0.2): {range_0_02} 样本 ({range_0_02/total_nodes*100:.2f}%)")
            # print(f"范围 [0.2, 0.4): {range_02_04} 样本 ({range_02_04/total_nodes*100:.2f}%)")
            # print(f"范围 [0.4, 0.6): {range_04_06} 样本 ({range_04_06/total_nodes*100:.2f}%)")
            # print(f"范围 [0.6, 0.8): {range_06_08} 样本 ({range_06_08/total_nodes*100:.2f}%)")
            # print(f"范围 [0.8, 1.0]: {range_08_1} 样本 ({range_08_1/total_nodes*100:.2f}%)")
            
            # # 验证总数
            # total_counted = range_0_02 + range_02_04 + range_04_06 + range_06_08 + range_08_1
            # print(f"统计总数: {total_counted} (应该匹配总数: {total_nodes})")
            
            # 总体分桶后的类别分布
            # print("总体分桶后的类别分布:")
            # unique_classes, class_counts = torch.unique(node_label_c, return_counts=True)
            # for i, (cls, count) in enumerate(zip(unique_classes, class_counts)):
            #     percentage = count.item() / len(node_label_c) * 100
            #     print(f"类别 {cls.item()}: {count.item()} 样本 ({percentage:.2f}%)")
            # print("="*50)
            # assert 0
            

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
        hg = torch.load(raw_path)
        print(f"hg:{hg}")
        if isinstance(hg, list):
            hg = hg[0]
        # print("hg", hg)
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
        
        # ===== 添加调试打印：查看原始异构图的节点类型 =====
        # print(f"原始异构图的节点类型: {hg.node_types}")
        # for ntype in hg.node_types:
        #     print(f"节点类型 '{ntype}': 数量 = {hg[ntype].num_nodes}")
        #     if hasattr(hg[ntype], 'x'):
        #         print(f"  特征维度: {hg[ntype].x.shape}")
        #     if hasattr(hg[ntype], 'y'):
        #         print(f"  标签维度: {hg[ntype].y.shape}")
        #         print(f"  标签范围: [{hg[ntype].y.min():.2e}, {hg[ntype].y.max():.2e}]")
        # print("=" * 50)
        
        """ graph transform """ 
        ### remove the power pins
        subset_dict = {}
        for ntype in hg.node_types:
            subset_dict[ntype] = torch.ones(hg[ntype].num_nodes, dtype=torch.bool)
            if ntype == 'net':
                subset_dict[ntype][power_net_ids] = False

        hg = hg.subgraph(subset_dict)
        hg = hg.edge_type_subgraph([
            ('device', 'device-pin', 'pin'),  # 器件到引脚的连接
            ('pin', 'pin-net', 'net'),        # 引脚到网络的连接
            ('pin', 'cc_p2n', 'net'),        # 引脚到网络的寄生耦合
            ('pin', 'cc_p2p', 'pin'),        # 引脚到引脚的寄生耦合
            ('net', 'cc_n2n', 'net'),        # 网络到网络的寄生耦合
        ])
        # print(f"hg:{hg}")
        # assert 0
        ### transform hetero g into homo g
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

        hg['device'].y = torch.ones((hg['device'].x.shape[0], 1)) * 1e-30   # 1e-30 is the minimum ground capacitance value
        # print(hg.node_types)
        # assert 0
        for n, ntype in enumerate(hg.node_types):
            g._n2type[ntype] = n # 节点类型到整数的映射
            feat = hg[ntype].x
            feat = torch.nn.functional.pad(feat, (0, max_feat_dim-feat.size(1)))
            node_feat.append(feat)
            tar_node_y.append(hg[ntype].y)
        
        # print("映射关系 g._n2type =", g._n2type)
        # # 然后再统计每个 node_type 的数量
        # types, counts = torch.unique(g.node_type, return_counts=True)
        # for t, cnt in zip(types.tolist(), counts.tolist()):
        #     name = [k for k,v in g._n2type.items() if v == t][0]
        #     print(f"{name}({t}) 节点数 = {cnt}")        
        # assert 0
        # ===== 添加调试打印：查看节点类型映射 =====
        # print(f"节点类型到整数的映射 g._n2type: {g._n2type}")
        # print(f"同构图总节点数: {g.num_nodes}")
        # print(f"同构图节点类型分布:")
        # for ntype, type_id in g._n2type.items():
        #     mask = g.node_type == type_id
        #     count = mask.sum().item()
        #     print(f"  {ntype} (type_id={type_id}): {count} 个节点")
        # print("=" * 50)
        
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
                g.tar_node_y[net_nodes] = hg['net'].y  # assign the ground capacitance value to the net nodes
                
                # # ===== 添加调试打印：仅net节点的标签 =====
                # print(f"仅处理net节点模式:")

                # print(f"  net节点标签范围: [{g.tar_node_y[net_nodes].min():.2e}, {g.tar_node_y[net_nodes].max():.2e}]")
                
            else:
                ## lumped ground capacitance on net/pin nodes
                g.tar_node_y = torch.cat(tar_node_y, dim=0)
            # print(f"g.tar_node_y.shape[0]:{g.tar_node_y.shape[0]}")

                # # ===== 添加调试打印：所有节点的标签 =====
                # print(f"处理所有节点模式:")
                # print(f"  总标签数量: {g.tar_node_y.shape[0]}")
                # print(f"  总标签范围: [{g.tar_node_y.min():.2e}, {g.tar_node_y.max():.2e}]")
                
                # 分别打印每种节点类型的标签统计
            #     start_idx = 0
            #     for ntype in hg.node_types:
            #         type_id = g._n2type[ntype]
            #         mask = g.node_type == type_id
            #         nodes_of_type = torch.where(mask)[0]
            #         if len(nodes_of_type) > 0:
            #             labels_of_type = g.tar_node_y[nodes_of_type]
            #             print(f"  {ntype} 节点标签:")
            #             print(f"    数量: {len(nodes_of_type)}")
            #             print(f"    范围: [{labels_of_type.min():.2e}, {labels_of_type.max():.2e}]")
            #             print(f"    均值: {labels_of_type.mean():.2e}")
            #             print(f"    标准差: {labels_of_type.std():.2e}")
                        
            #             # 显示标签分布的直方图信息
            #             unique_labels, counts = labels_of_type.unique(return_counts=True)
            #             print(f"    唯一标签数: {len(unique_labels)}")
            #             if len(unique_labels) <= 10:
            #                 print(f"    标签分布: {dict(zip(unique_labels.tolist(), counts.tolist()))}")
            #             else:
            #                 print(f"    前5个最常见标签: {dict(zip(unique_labels[:5].tolist(), counts[:5].tolist()))}")
            
            # print("=" * 50)
            g.y = g.tar_node_y
        # print(f"g.y.max:{g.y.max()}")
        """
        tensor([[7.1959e-18],
        [5.6060e-15],
        [1.0105e-16],
        ...,
        [0.0000e+00],
        [0.0000e+00],
        [0.0000e+00]])
        """
        g._e2type = {}

        for e, (edge, store) in enumerate(hg.edge_items()):
            g._e2type[edge] = e
            if 'cc' in edge[1]: # 寄生耦合边
                tar_edge_y.append(store['y']) 
            else:
                # edge_index's shape [2, num_edges]
                edge_offset += store['edge_index'].shape[1]
        # 分离结构边（电路拓扑）和目标边（寄生耦合
        g._num_ntypes = len(g._n2type)
        # print(f"g._n2type:{g._n2type}") #g._n2type:{'net': 0, 'device': 1, 'pin': 2}
        g._num_etypes = len(g._e2type)
        # print(f"g._e2type:{g._e2type}") #g._e2type:{('device', 'device-pin', 'pin'): 0, ('pin', 'pin-net', 'net'): 1, ('pin', 'cc_p2n', 'net'): 2, ('pin', 'cc_p2p', 'pin'): 3, ('net', 'cc_n2n', 'net'): 4}

        logging.info(f"g._n2type {g._n2type}")
        logging.info(f"g._e2type {g._e2type}")
        # print(f"edge_offset:{edge_offset}") #306385
        # assert 0
        tar_edge_index = g.edge_index[:, edge_offset:]
        tar_edge_type = g.edge_type[edge_offset:]
        tar_edge_y = torch.cat(tar_edge_y)
        # print(f"tar_edge_y:{tar_edge_y}") #tar_edge_y:tensor([3.4233e-18, 3.8098e-19, 3.0819e-19,  ..., 1.5311e-17, 2.0015e-17,6.0056e-18])

        # testing
        # for i in range(tar_edge_type.min(), tar_edge_type.max()+1):
        #     mask = tar_edge_type == i
        #     print("tar_edge_type", tar_edge_type[mask][0], "tar_edge_y", tar_edge_y[mask][0])
        # assert 0

        ## restrict the capcitance value range 
        legel_edge_mask = (tar_edge_y < 1e-15) & (tar_edge_y > 1e-21)
        """
        # legal_node_mask = (g.tar_node_y < 1e-15) & (g.tar_node_y > 1e-21)
        # legal_node_mask = legal_node_mask.squeeze()
        # g.tar_node_y = g.tar_node_y[legal_node_mask]
        # g.x = g.x[legal_node_mask]
        # g.node_attr = g.node_attr[legal_node_mask]  
        # g.node_type = g.node_type[legal_node_mask]
        # print(f"node_attr:{g.node_attr.size()},legal_node_mask：{legal_node_mask.size()}，tar_edge_index：{tar_edge_index.size()}，legel_edge_mask:{legel_edge_mask.size()}")
        # print(f"tar_edge_type:{tar_edge_type},g.x:{g.x},g.y:{g.y}")
        # assert 0

        node_attr:torch.Size([249570, 17]),legal_node_mask：torch.Size([249570])，tar_edge_index：torch.Size([2, 624865])，legel_edge_mask:torch.Size([624865])
        tar_edge_type:tensor([2, 2, 2,  ..., 4, 4, 4]),node_type:tensor([0, 0, 0,  ..., 2, 2, 2]),g.x:tensor([[0],
        [0],
        [0],
        ...,
        [2],
        [2],
        [2]]),g.y:tensor([[7.1959e-18],
        [5.6060e-15],
        [1.0105e-16],
        ...,
        [2.2452e-19],
        [2.3456e-18],
        [2.0348e-20]])
        """
        if self.task_level == 'node' :
            legal_node_mask = (g.tar_node_y < 1e-15) & (g.tar_node_y > 1e-21)
            legal_node_mask = legal_node_mask.squeeze()
            print(f"(~legal_node_mask).sum().item():{(~legal_node_mask).sum().item()},legal_node_mask:{legal_node_mask.size()}") #(~legal_node_mask).sum().item():126659,legal_node_mask:torch.Size([249570])
        
        # 替换非法值（此处用 1e-30 保证对数变换有效）
        # 假设：
        # - g.tar_node_y 是原始数据（可能包含 1e-30）
        # - legal_node_mask 是合法掩码（False 表示非法节点）
        # 获取非法节点的值
        # invalid_node_values = g.tar_node_y[~legal_node_mask]

        # # 过滤掉 0（或 1e-30），并找最小值
        # if len(invalid_node_values) > 0:
        #     non_zero_values = invalid_node_values[invalid_node_values > 1e-30]  # 或 != 0.0
        #     if len(non_zero_values) > 0:
        #         min_non_zero_in_invalid = non_zero_values.min()
        #         print(f"非法节点中非零的最小值: {min_non_zero_in_invalid}") 非法节点中非零的最小值: 1.0339799565148436e-25
        #     else:
        #         print("所有非法节点的值都是 0（或 1e-30）")
        # else:
        #     print("没有非法节点")

            g.tar_node_y[~legal_node_mask] = 1e-30
            #================================  画图:node level =======================================
            processed_node_labels = torch.log10(g.tar_node_y * 1e21) / 6

            # Identify artificially added 1e-30 values
            artificial_mask = torch.isclose(g.tar_node_y, torch.tensor(1e-30), atol=1e-32)
            print(f"Artificially added nodes count: {artificial_mask.sum().item()}")

            # Apply clipping to all data
            processed_node_labels[processed_node_labels < 0] = 0.0
            processed_node_labels[processed_node_labels > 1] = 1.0

            # Perform bucketing (using all data)
            node_label_c = torch.bucketize(processed_node_labels.squeeze(), self.class_boundaries)

            # [Keep all your existing analysis code here...]

            # Visualization with your requested style
            # Visualization with your requested style
            valid_labels_np = processed_node_labels[~artificial_mask].cpu().numpy()

            plt.figure()
            ax = plt.gca()

            # Set gray background inside plot only
            ax.set_facecolor('lightgray')

            # Set white background for figure (outer area)
            plt.gcf().set_facecolor('white')

            plt.hist(valid_labels_np, 
                    bins=50,
                    density=True,      # Y-axis as density
                    color='orange',    # Orange bars
                    edgecolor='white') # White edges

            plt.xlabel('normalized label', fontsize=22)  # 增大x轴标签字体
            plt.ylabel('density', fontsize=22)         # 增大y轴标签字体

            # Add a white grid for better visibility
            ax.grid(True, color='white', linestyle='-', linewidth=0.5)

            plt.savefig(f'imgs/node_label_dist_{name}.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
            plt.close()
            
        """
        node_attr:torch.Size([122911, 17]),legal_node_mask：torch.Size([249570])，tar_edge_index：torch.Size([2, 624865])，legel_edge_mask:torch.Size([624865])
        tar_edge_type:tensor([2, 2, 2,  ..., 4, 4, 4]),g.x:tensor([[0],
        [0],
        [0],
        ...,
        [2],
        [2],
        [2]]),g.y:tensor([[7.1959e-18],
        [5.6060e-15],
        [1.0105e-16],
        ...,
        [2.2452e-19],
        [2.3456e-18],
        [2.0348e-20]])
        """

        # tar_edge_src_y = g.tar_node_y[tar_edge_index[0, :]].squeeze()
        # tar_edge_dst_y = g.tar_node_y[tar_edge_index[1, :]].squeeze()
        # legel_node_mask = (tar_edge_src_y < 1e-13) & (tar_edge_src_y > 1e-23)
        # legel_node_mask &= (tar_edge_dst_y < 1e-13) & (tar_edge_dst_y > 1e-23)
        # g.legal_node_mask = legal_node_mask.squeeze()


        # print(f"\nProcessing dataset {name}, node types: {torch.unique(g.node_type)}")  # 显示过滤后的节点类型
        # print(f"Node type counts: {torch.bincount(g.node_type)}")  # 各类型节点数量

        g.tar_edge_y = tar_edge_y[legel_edge_mask]# & legel_node_mask]
        g.tar_edge_index = tar_edge_index[:, legel_edge_mask]# & legel_node_mask]
        g.tar_edge_type = tar_edge_type[legel_edge_mask]# & legel_node_mask]
        
        # logging.info(f"we filter out the edges with Cc > 1e-15 and Cc < 1e-21 " + 
        #              f"{legel_edge_mask.size(0)-legel_edge_mask.sum()}")
        # logging.info(f"we filter out the edges with src/dst Cg > 1e-13 and Cg < 1e-23 " +
        #              f"{legel_node_mask.size(0)-legel_node_mask.sum()}")

        ## Calculate target edge type distributions (Cc_p2n : Cc_p2p : Cc_n2n)
        _, g.tar_edge_dist = g.tar_edge_type.unique(return_counts=True)
        # print(f"g.tar_edge_dist:{g.tar_edge_dist}") #tensor([261027, 285103,  58461])
        ## remove target edges from the original g
        g.edge_type = g.edge_type[0:edge_offset]
        g.edge_index = g.edge_index[:, 0:edge_offset]
        
        
        #================================  画图:egde level =======================================
        edge_labels = torch.log10(g.tar_edge_y * 1e21) / 6

        # Identify artificially added 1e-30 values
        artificial_mask = torch.isclose(g.tar_edge_y, torch.tensor(1e-30), atol=1e-32)
        print(f"Artificially added edge count: {artificial_mask.sum().item()}")

        # Apply clipping to all data
        edge_labels[edge_labels < 0] = 0.0
        edge_labels[edge_labels > 1] = 1.0

        # Perform bucketing (using all data)
        edge_label_c = torch.bucketize(edge_labels.squeeze(), self.class_boundaries)

        # [Keep all your existing analysis code here...]

        # Visualization with your requested style
        # Visualization with your requested style
        valid_labels_np = edge_labels[~artificial_mask].cpu().numpy()

        plt.figure()
        ax = plt.gca()

        # Set gray background inside plot only
        ax.set_facecolor('lightgray')

        # Set white background for figure (outer area)
        plt.gcf().set_facecolor('white')

        plt.hist(valid_labels_np, 
                bins=50,
                density=True,      # Y-axis as density
                color='orange',    # Orange bars
                edgecolor='white') # White edges

        plt.xlabel('normalized label', fontsize=22)  # 增大x轴标签字体
        plt.ylabel('density', fontsize=22)         # 增大y轴标签字体

        # Add a white grid for better visibility
        ax.grid(True, color='white', linestyle='-', linewidth=0.5)

        plt.savefig(f'imgs/edge_label_dist_{name}.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()
        
        ## convert to undirected edges
        if self.to_undirected:
                g.edge_index, g.edge_type = to_undirected(
                    g.edge_index, g.edge_type, g.num_nodes, reduce='mean'
                )
        
        # print("\n=== 节点类型分布验证 ===")
        # unique_types, type_counts = torch.unique(g.node_type, return_counts=True)
        # print("存在的节点类型及数量:")
        # for typ, cnt in zip(unique_types.tolist(), type_counts.tolist()):
        #     print(f"类型 {typ}: {cnt} 个节点")

        # print("节点类型映射字典:", g._n2type)
        
        return g

    def single_g_process(self, idx: int):
        print(f"processing dataset {self.names[idx]} "+ 
                     f"with sample_rate {self.sample_rates[idx]}...")
        ## we can load multiple graphs
        graph = self.sram_graph_load(self.names[idx], self.raw_paths[idx])
        print(f"loaded graph {graph}")
        # print(f"self.raw_paths:{self.raw_paths}")
        # assert 0
        ## generate negative edges for the loaded graph
        neg_edge_index, neg_edge_type = get_pos_neg_edges(
            graph, neg_ratio=self.neg_edge_ratio)  # 为边级任务生成负样本
        
        
        ## sample a portion of pos/neg edges
        (
            pos_edge_index, pos_edge_type, pos_edge_y,
            neg_edge_index, neg_edge_type
        ) = get_balanced_edges(
            graph, neg_edge_index, neg_edge_type, 
            self.neg_edge_ratio, self.sample_rates[idx]
        )


        if self.task_level == 'edge' :
            ## We only consider the positive edges in the regression task.
            links = pos_edge_index  # [2, Np]
            labels = pos_edge_y
        elif self.task_level == 'node':
            # node classification
            graph.y = graph.tar_node_y.squeeze()  # assume shape is [num_nodes]
        else:
            raise ValueError(f"No defination of task {self.task_level} in this version!")
        
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
                          task_level,
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