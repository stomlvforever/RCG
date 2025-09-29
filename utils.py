"""
RCG项目工具函数模块

本模块提供了图神经网络训练中所需的核心工具函数，主要包括：
1. 正负边采样：为边级任务生成平衡的正负样本
2. 结构化负采样：基于图结构的负边生成策略
3. 边类型平衡：确保不同边类型的样本平衡
4. 图数据增强：将负边添加到原始图中
5. 数据分离：从批处理数据中分离单个样本

主要用于SRAM电路图的电阻预测任务，支持目标边类型：
- r_p2p: PIN到PIN的电阻

Author: RCG Project Team
Date: 2024
"""

import torch
from torch_geometric.data import Data
from torch_geometric.utils import (to_undirected, 
                                   structured_negative_sampling, 
                                   negative_sampling, 
                                   dropout_edge, dropout_node, subgraph)
import logging
from torch_geometric.data.separate import separate
import copy


def get_pos_neg_edges(
        g, sample_type='structured', task_type='resister',
        force_undirected=False, 
        neg_ratio=1.0,
    ):
    """
    为边级预测任务生成正负样本边
    
    该函数支持两种负采样策略：
    1. 全局采样(global)：从整个图中随机采样负边
    2. 结构化采样(structured)：基于正边的结构化负采样
    
    Args:
        g (Data): PyTorch Geometric图对象，包含目标边信息
        sample_type (str): 采样类型，'global'或'structured'
        force_undirected (bool): 是否强制负边为无向边
        neg_ratio (float): 负边与正边的比例
        
    Returns:
        tuple: (正边索引, 负边索引, 负边类型)
            - pos_edge_index (Tensor[2, N_pos]): 正边的源节点和目标节点索引
            - neg_edge_index (Tensor[2, N_neg]): 负边的源节点和目标节点索引  
            - neg_edge_type (Tensor[N_neg]): 负边对应的边类型
            
    Note:
        支持三种边类型的负采样：
        - 类型2: PIN-NET耦合电容负样本
        - 类型3: PIN-PIN耦合电容负样本
        - 类型4: NET-NET耦合电容负样本
    """
    if neg_ratio > 1.0 or sample_type == 'global':
        neg_edge_index = negative_sampling(
            g.tar_edge_index, g.num_nodes,
            force_undirected=force_undirected,
            num_neg_samples=int(g.tar_edge_index.size(1)*neg_ratio),
        )
        print(f"neg_edge_index:{neg_edge_index}")
        neg_edge_type = torch.zeros(neg_edge_index.size(1), dtype=torch.long)
        for i in range(neg_edge_index.size(1)):
            node_pair = neg_edge_index[:, i]
            ntypes = set(g.node_type[node_pair].tolist())
            # for neg edge types are related to target edge types
            if task_type == 'capacitance':
                # weight for neg Cc_p2n
                if ntypes == {0, 2}: 
                    neg_edge_type[i] = 2
                # weight for Cc_p2p
                elif ntypes == {2}: 
                    neg_edge_type[i] = 3
                # weight for Cc_n2n
                elif ntypes == {0}:
                    neg_edge_type[i] = 4
                legal_mask = neg_edge_type > 0
            elif task_type =='resister':
                if ntypes == {2}:  # 两个pin节点 -> r_p2p电阻
                    neg_edge_type[i] = 5  # 使用新的边类型编号        
                legal_mask = neg_edge_type == 5
        print(
            f"Using global negtive sampling, #pos={g.tar_edge_index.size(1)}, " + 
            f"#neg={neg_edge_index[:, legal_mask].size(1)}")
        return g.tar_edge_index, neg_edge_index[:,legal_mask], neg_edge_type[legal_mask]
    
    if neg_ratio==0.0:
        return  torch.zeros(2, 0), torch.zeros(0) 
    
    
    neg_edge_index = []
    neg_edge_type  = []
    tar_edge_offset = 0
    for i in range(g.tar_edge_dist.size(0)):
        pos_edges = g.tar_edge_index[:, tar_edge_offset:g.tar_edge_dist[i]+tar_edge_offset]
        pos_edge_src, pos_edge_dst, neg_edge_dst = structured_negative_sampling(
            pos_edges, g.num_nodes, contains_neg_self_loops=False,
        )
        tar_edge_offset += g.tar_edge_dist[i]
        # neg edge sampling
        indices = torch.randperm(neg_edge_dst.size(0))[
            :int(neg_edge_dst.size(0) * neg_ratio), 
        ]
        neg_edge_index.append(
            torch.stack((pos_edge_src[indices], neg_edge_dst[indices]), dim=0)
        )
        neg_edge_type += [i + g.tar_edge_type[0]] * indices.size(0)

        print(
            f"Using structured negtive sampling for target etype {i}, " + 
            f"pos={pos_edges.size(1)}, #neg={neg_edge_index[-1].size(1)}")
    
    return torch.cat(neg_edge_index, 1), torch.tensor(neg_edge_type)


def get_balanced_edges(
    g, neg_edge_index, neg_edge_type,
    neg_edge_ratio, sample_ratio = 1.0,
):
    """
    获取类型平衡的正负边样本
    
    该函数确保不同边类型的样本数量平衡，避免类别不平衡问题。
    通过找到最少的边类型数量作为基准，对其他类型进行下采样。
    
    Args:
        g (Data): 包含正边信息的图对象
        neg_edge_index (Tensor[2, N]): 负边索引
        neg_edge_type (Tensor[N]): 负边类型
        neg_edge_ratio (float): 负边与正边的比例
        sample_ratio (float): 采样比例，用于控制总体样本数量
        
    Returns:
        tuple: 平衡后的正负边数据
            - pos_edge_index (Tensor[2, N_pos]): 平衡后的正边索引
            - pos_edge_type (Tensor[N_pos]): 正边类型
            - pos_edge_y (Tensor[N_pos]): 正边的目标值(电阻值)
            - neg_edge_index (Tensor[2, N_neg]): 平衡后的负边索引
            - neg_edge_type (Tensor[N_neg]): 负边类型
            
    Note:
        使用最小边类型数量作为平衡基准，确保训练数据的类别平衡
    """
    tar_edge_offset = 0
    min_edge_num = g.tar_edge_dist.min()
    neg_edge_index_list = []
    neg_edge_type_list  = []
    pos_edge_index_list = []
    pos_edge_type_list  = []
    pos_edge_y_list  = []
    for i in range(g.tar_edge_dist.size(0)):
        # for pos edges
        pos_edges = g.tar_edge_index[
            :, tar_edge_offset:g.tar_edge_dist[i]+tar_edge_offset
        ]
        pos_etypes = g.tar_edge_type[
            tar_edge_offset:g.tar_edge_dist[i]+tar_edge_offset
        ]
        pos_edge_y = g.tar_edge_y[
            tar_edge_offset:g.tar_edge_dist[i]+tar_edge_offset
        ]
        tar_edge_offset += g.tar_edge_dist[i]

        indices = torch.randperm(pos_edges.size(1))[
            :int(min_edge_num * sample_ratio), 
        ]
        pos_edge_index_list.append(pos_edges[:, indices])
        pos_edge_type_list.append(pos_etypes[indices])
        pos_edge_y_list.append(pos_edge_y[indices])
        
        logging.info(f"Edge type {i}, balanced pos edge num: {pos_edge_type_list[-1].size(0)}")

        if neg_edge_ratio == 0.0:
            neg_edge_index_list.append(torch.zeros(2, 0))
            neg_edge_type_list.append(torch.zeros(0))
        else:
            # for neg edges
            neg_edge_mask = (neg_edge_type - neg_edge_type.min()) == i
            assert neg_edge_index.size(1) == neg_edge_type.size(0)
            neg_edges = neg_edge_index[:, neg_edge_mask]
            neg_etypes = neg_edge_type[neg_edge_mask]
            indices = torch.randperm(neg_edges.size(1))[
                :int(min_edge_num * neg_edge_ratio * sample_ratio), 
            ]
            neg_edge_index_list.append(neg_edges[:, indices])
            neg_edge_type_list.append(neg_etypes[indices])

        logging.info(f"Edge type {i}, balanced neg edge num: {neg_edge_type_list[-1].size(0)}")

    return (torch.cat(pos_edge_index_list, 1), torch.cat(pos_edge_type_list), 
            torch.cat(pos_edge_y_list),  # these are target Cc values
            torch.cat(neg_edge_index_list, 1), torch.cat(neg_edge_type_list),)


def add_tar_edges_to_g(g, neg_edge_index, neg_edge_type):
    """
    将目标边(正边+负边)添加到原始图中，创建增强图
    
    该函数将生成的负边与原有正边合并，添加到原始图结构中，
    用于图神经网络的训练。支持有向图和无向图。
    
    Args:
        g (Data): 原始图对象
        neg_edge_index (Tensor[2, N]): 负边索引
        neg_edge_type (Tensor[N]): 负边类型
        
    Returns:
        Data: 增强后的图对象，包含原始边和目标边
        
    Note:
        - 如果原图是无向图，会自动处理边的双向性
        - 保留原图的所有节点特征和属性
        - 添加边类型偏移量等元信息
    """
    added_edges_index = torch.cat((g.tar_edge_index, neg_edge_index), 1)
    added_edge_type = torch.cat((g.tar_edge_type, neg_edge_type))
    if g.is_undirected():
        added_edges_index, added_edge_type = to_undirected(
            added_edges_index, added_edge_type, g.num_nodes, reduce='mean'
        )
    logging.info(f"#added edges={g.tar_edge_index.size(1)+neg_edge_index.size(1)} "+
                    f"#undirected added edges={added_edges_index.size(1)}")
    aug_g = Data()
    aug_g.edge_index = torch.cat((g.edge_index, added_edges_index), dim=1)
    aug_g.edge_type = torch.cat((g.edge_type, added_edge_type)).long()
    # aug_g.neg_edge_mask = aug_g.edge_type >= g._num_etypes
    aug_g.x = g.x
    aug_g.node_type = g.node_type.long()
    aug_g.node_attr = g.node_attr
    aug_g.num_pos_etype = g._num_etypes
    aug_g.num_ntypes = g._num_ntypes
    aug_g.tar_edge_type_offset = g.tar_edge_type.min()
    # aug_g.tar_edge_dist = g.tar_edge_dist
    # del g.tar_edge_index
    # del g.tar_edge_type
    aug_g.name = g.name
    print("DEBUG: got aug_g", aug_g)
    return aug_g


def collated_data_separate(data: Data, slices, idx: int=None):
    """
    从批处理数据中分离出单个图样本
    
    这是InMemoryDataset.collate()的逆操作，用于从合并的批处理数据中
    提取单个图样本，常用于调试和数据分析。
    
    Args:
        data (Data): 批处理后的图数据
        slices (dict): 数据切片信息，记录每个样本的边界
        idx (int, optional): 要提取的样本索引。如果为None，提取所有样本
        
    Returns:
        Data or List[Data]: 
            - 如果指定idx，返回单个Data对象
            - 如果idx为None，返回Data对象列表
            
    Note:
        使用深拷贝确保返回的数据独立于原始批处理数据
    """
    if idx is not None:
        separated_data = separate(
            cls=data.__class__,
            batch=data,
            idx=idx,
            slice_dict=slices,
            decrement=False,
        )
        return copy.copy(separated_data)
    
    data_list = []
    for i in range(data.y.size(0)):
        separated_data = separate(
            cls=data.__class__,
            batch=data,
            idx=i,
            slice_dict=slices,
            decrement=False,
        )
        data_list.append(copy.copy(separated_data))
    return data_list

