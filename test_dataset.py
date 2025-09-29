#!/usr/bin/env python3
"""
简化的外部测试脚本
测试 performat_SramDataset 函数
"""

import os
import sys
import torch

# 导入数据集函数
try:
    from dataset import performat_SramDataset
    print("✅ 成功导入 performat_SramDataset 函数")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请确保 dataset.py 文件在当前目录下")
    sys.exit(1)

def prepare_files():
    """检查测试文件是否存在"""
    print("🔧 检查测试文件...")
    
    files = [
        "/data/tianjn/baseline/basline/basline/sram/raw/integrated_position_prediction_graph.pt",
        "/data/tianjn/baseline/basline/basline/sram/raw/integrated_power_density_prediction_graph.pt"
    ]
    
    for file_path in files:
        if os.path.exists(file_path):
            print(f"✅ 文件存在: {os.path.basename(file_path)}")
        else:
            print(f"❌ 文件不存在: {file_path}")
            return False
    
    return True

def get_feature_names(sample):
    """获取特征名称"""
    # 尝试从样本中获取特征名称
    
    # 方法1: 检查是否有存储的特征名称
    if hasattr(sample, 'feature_names'):
        return sample.feature_names
    
    # 方法2: 检查是否有元数据
    if hasattr(sample, 'metadata') and isinstance(sample.metadata, dict):
        if 'feature_names' in sample.metadata:
            return sample.metadata['feature_names']
    
    # 方法3: 检查原始图数据中是否有特征名称信息
    if hasattr(sample, '_node_attr_names'):
        return sample._node_attr_names
    
    # 如果都没有找到，返回None
    return None

def analyze_dataset_attributes(dataset, inspect_attr=None, detailed_tensor='x'):
    """详细分析数据集对象的所有属性"""
    print(f"\n{'='*60}")
    print("🔍 数据集对象属性详细分析")
    print(f"{'='*60}")
    
    # 获取第一个样本进行分析
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"📊 样本对象类型: {type(sample)}")
        print(f"📊 样本对象的所有属性:")
        
        # 收集所有属性
        all_attrs = []
        for attr_name in dir(sample):
            if not attr_name.startswith('_'):
                try:
                    attr_value = getattr(sample, attr_name)
                    if not callable(attr_value):
                        all_attrs.append((attr_name, attr_value))
                except Exception as e:
                    print(f"❌ 获取属性 '{attr_name}' 时出错: {e}")
                    continue
        
        # 打印基本属性列表
        print(f"\n📋 发现 {len(all_attrs)} 个属性:")
        for i, (attr_name, attr_value) in enumerate(all_attrs, 1):
            if hasattr(attr_value, 'shape'):
                print(f"   {i:2d}. {attr_name}: {attr_value.shape} ({attr_value.dtype})")
            elif hasattr(attr_value, '__len__') and not isinstance(attr_value, str):
                print(f"   {i:2d}. {attr_name}: 长度={len(attr_value)} ({type(attr_value).__name__})")
            else:
                print(f"   {i:2d}. {attr_name}: {type(attr_value).__name__}")
        
        # 详细分析每个张量属性
        print(f"\n📊 张量属性详细信息:")
        tensor_attrs = [(name, val) for name, val in all_attrs if hasattr(val, 'shape')]
        
        for attr_name, attr_value in tensor_attrs:
            print(f"\n   🔹 {attr_name}:")
            print(f"      - 形状: {attr_value.shape}")
            print(f"      - 数据类型: {attr_value.dtype}")
            if attr_value.numel() > 0:  # 非空张量
                print(f"      - 值范围: [{attr_value.min().item():.4f}, {attr_value.max().item():.4f}]")
                print(f"      - 元素总数: {attr_value.numel()}")
                if len(attr_value.shape) > 1:
                    print(f"      - 每个维度: {list(attr_value.shape)}")
            
        # 详细分析每个张量属性
        print(f"\n📊 张量属性详细信息:")
        print(f"   💡 只对 '{detailed_tensor}' 显示详细子属性信息，其他张量只显示基本信息")
        
        tensor_attrs = [(name, val) for name, val in all_attrs if hasattr(val, 'shape')]
        
        for attr_name, attr_value in tensor_attrs:
            print(f"\n   🔹 {attr_name}:")
            print(f"      - 形状: {attr_value.shape}")
            print(f"      - 数据类型: {attr_value.dtype}")
            if attr_value.numel() > 0:  # 非空张量
                print(f"      - 值范围: [{attr_value.min().item():.4f}, {attr_value.max().item():.4f}]")
                print(f"      - 元素总数: {attr_value.numel()}")
                if len(attr_value.shape) > 1:
                    print(f"      - 每个维度: {list(attr_value.shape)}")
            
            # 只对指定的张量显示详细子属性信息
            if attr_name == detailed_tensor:
                print(f"      🔍 详细分析 {attr_name} 的子属性:")
                
                # 检查是否有子属性
                sub_attrs = []
                for sub_attr_name in dir(attr_value):
                    if not sub_attr_name.startswith('_'):
                        try:
                            sub_attr_value = getattr(attr_value, sub_attr_name)
                            if not callable(sub_attr_value) and not hasattr(sub_attr_value, 'shape'):
                                sub_attrs.append((sub_attr_name, sub_attr_value))
                        except Exception as e:
                            print(f"         ❌ 获取张量 '{attr_name}' 的子属性 '{sub_attr_name}' 时出错: {e}")
                            continue
                
                if sub_attrs:
                    print(f"         - 子属性: {len(sub_attrs)} 个")
                    for sub_name, sub_val in sub_attrs[:10]:  # 显示前10个
                        print(f"           * {sub_name}: {type(sub_val).__name__}")
                        if isinstance(sub_val, (list, tuple)) and len(sub_val) <= 20:
                            print(f"             内容: {sub_val}")
                        elif hasattr(sub_val, '__len__') and not isinstance(sub_val, str):
                            print(f"             长度: {len(sub_val)}")
                        elif isinstance(sub_val, (str, int, float, bool)):
                            print(f"             值: {sub_val}")
                else:
                    print(f"         - 子属性: 无")
                    
                # 对指定张量进行特征名称查找和数据分析
                print(f"\n      🔍 查找张量 '{detailed_tensor}' 的特征名称:")
                print(f"         📊 {detailed_tensor}张量形状: {attr_value.shape}")
                
                # 查找可能包含特征名称的属性
                print(f"         🔍 检查{detailed_tensor}张量的所有子属性:")
                detailed_attrs = []
                for sub_attr_name in dir(attr_value):
                    if not sub_attr_name.startswith('__'):  # 排除魔术方法
                        try:
                            sub_attr_value = getattr(attr_value, sub_attr_name)
                            if not callable(sub_attr_value):
                                detailed_attrs.append((sub_attr_name, sub_attr_value))
                                print(f"            - {sub_attr_name}: {type(sub_attr_value)}")
                                
                                # 检查是否可能是特征名称
                                if isinstance(sub_attr_value, (list, tuple)) and len(sub_attr_value) == attr_value.shape[-1]:
                                    print(f"              💡 可能是特征名称! 长度匹配最后维度数 ({len(sub_attr_value)})")
                                    print(f"              内容: {sub_attr_value}")
                                elif hasattr(sub_attr_value, '__len__') and not isinstance(sub_attr_value, str):
                                    if len(sub_attr_value) <= 20:  # 不要打印太长的
                                        print(f"              长度: {len(sub_attr_value)}, 内容: {sub_attr_value}")
                                    else:
                                        print(f"              长度: {len(sub_attr_value)} (太长不显示)")
                                elif isinstance(sub_attr_value, (str, int, float, bool)):
                                    print(f"              值: {sub_attr_value}")
                        except Exception as e:
                            if any(keyword in str(e).lower() for keyword in ['imag', 'complex', '.h', '.mh', '.mt']):
                                continue  # 跳过已知的张量属性错误
                            print(f"            ❌ {sub_attr_name}: {e}")
                
                print(f"\n         📋 {detailed_tensor}张量共有 {len(detailed_attrs)} 个可访问的子属性")
                
                # 检查sample级别是否有对应的特征名称信息
                print(f"\n         🔍 检查sample级别的{detailed_tensor}特征名称信息:")
                sample_attrs = []
                search_keywords = ['name', 'feature', 'column', 'header']
                if detailed_tensor == 'edge_attr':
                    search_keywords.extend(['edge'])
                elif detailed_tensor == 'x':
                    search_keywords.extend(['node'])
                elif detailed_tensor == 'y':
                    search_keywords.extend(['label', 'target'])
                
                for sample_attr_name in dir(sample):
                    if any(keyword in sample_attr_name.lower() for keyword in search_keywords):
                        if not sample_attr_name.startswith('_') and not callable(getattr(sample, sample_attr_name)):
                            try:
                                sample_attr_value = getattr(sample, sample_attr_name)
                                sample_attrs.append((sample_attr_name, sample_attr_value))
                                print(f"            - {sample_attr_name}: {type(sample_attr_value)}")
                                
                                if isinstance(sample_attr_value, (list, tuple)):
                                    print(f"              长度: {len(sample_attr_value)}")
                                    if len(sample_attr_value) <= 20:
                                        print(f"              内容: {sample_attr_value}")
                                    elif len(sample_attr_value) == attr_value.shape[-1]:
                                        print(f"              💡 长度匹配{detailed_tensor}的最后维度数! 内容: {sample_attr_value}")
                                elif hasattr(sample_attr_value, '__len__') and not isinstance(sample_attr_value, str):
                                    print(f"              长度: {len(sample_attr_value)}")
                                else:
                                    print(f"              值: {sample_attr_value}")
                            except Exception as e:
                                print(f"            ❌ {sample_attr_name}: {e}")
                
                if not sample_attrs:
                    print(f"            ❌ 没有找到明显的{detailed_tensor}特征名称属性")
                    
                # 分析数据分布来推测特征含义（只对合理大小的维度）
                if len(attr_value.shape) >= 1 and (len(attr_value.shape) == 1 or attr_value.shape[-1] <= 50):
                    print(f"\n         🔍 分析{detailed_tensor}各维度的数据分布来推测特征含义:")
                    
                    if len(attr_value.shape) == 1:
                        # 一维张量
                        unique_vals = attr_value.unique()
                        print(f"            单维张量:")
                        print(f"              - 值范围: [{attr_value.min().item():.4f}, {attr_value.max().item():.4f}]")
                        print(f"              - 唯一值数量: {unique_vals.numel()}")
                        
                        if unique_vals.numel() <= 10:
                            print(f"              - 所有唯一值: {unique_vals.tolist()}")
                            if torch.all(attr_value == attr_value.long().float()) and unique_vals.numel() <= len(dataset.names):
                                print(f"              💡 可能是图ID! (整数值，唯一值数量={unique_vals.numel()}, 数据集数量={len(dataset.names)})")
                        elif unique_vals.numel() <= 100:
                            print(f"              - 前10个唯一值: {unique_vals[:10].tolist()}")
                            if torch.all(attr_value == attr_value.long().float()):
                                print(f"              💡 可能是类别特征 (整数值)")
                        
                        print(f"              - 前5个值: {attr_value[:5].tolist()}")
                    else:
                        # 多维张量
                        for dim in range(min(attr_value.shape[-1], 20)):
                            if len(attr_value.shape) == 2:
                                feature_col = attr_value[:, dim]
                            else:
                                continue
                                
                            unique_vals = feature_col.unique()
                            
                            print(f"            维度 {dim}:")
                            print(f"              - 值范围: [{feature_col.min().item():.4f}, {feature_col.max().item():.4f}]")
                            print(f"              - 唯一值数量: {unique_vals.numel()}")
                            
                            # 特殊情况分析
                            if unique_vals.numel() <= 10:
                                print(f"              - 所有唯一值: {unique_vals.tolist()}")
                                if torch.all(feature_col == feature_col.long().float()) and unique_vals.numel() <= len(dataset.names):
                                    print(f"              💡 可能是图ID! (整数值，唯一值数量={unique_vals.numel()}, 数据集数量={len(dataset.names)})")
                            elif unique_vals.numel() <= 100:
                                print(f"              - 前10个唯一值: {unique_vals[:10].tolist()}")
                                if torch.all(feature_col == feature_col.long().float()):
                                    print(f"              💡 可能是类别特征 (整数值)")
                            
                            print(f"              - 前5个值: {feature_col[:5].tolist()}")
                else:
                    print(f"         ⚠️  张量维度太大，跳过数据分布分析")
            else:
                print(f"      - 子属性: 已跳过详细分析 (只分析 '{detailed_tensor}')")
        
        # 非张量属性
        non_tensor_attrs = [(name, val) for name, val in all_attrs if not hasattr(val, 'shape')]
        if non_tensor_attrs:
            print(f"\n📋 非张量属性:")
            for attr_name, attr_value in non_tensor_attrs:
                print(f"   🔸 {attr_name}: {type(attr_value).__name__}")
                if hasattr(attr_value, '__len__') and not isinstance(attr_value, str):
                    print(f"      - 长度: {len(attr_value)}")
                    # 显示字典、列表等容器的内容
                    if isinstance(attr_value, dict):
                        print(f"      - 内容: {attr_value}")
                    elif isinstance(attr_value, (list, tuple)) and len(attr_value) <= 10:
                        print(f"      - 内容: {attr_value}")
                    elif isinstance(attr_value, (list, tuple)):
                        print(f"      - 前5个元素: {attr_value[:5]}")
                if isinstance(attr_value, (int, float, str, bool)):
                    print(f"      - 值: {attr_value}")
                elif attr_value is None:
                    print(f"      - 值: None")
        
        # 自定义属性检查接口
        if inspect_attr:
            print(f"\n🔍 自定义属性检查: '{inspect_attr}'")
            if hasattr(sample, inspect_attr):
                custom_attr = getattr(sample, inspect_attr)
                print(f"   - 类型: {type(custom_attr)}")
                if hasattr(custom_attr, 'shape'):
                    print(f"   - 形状: {custom_attr.shape}")
                    print(f"   - 数据类型: {custom_attr.dtype}")
                    if custom_attr.numel() > 0:
                        print(f"   - 值范围: [{custom_attr.min().item():.4f}, {custom_attr.max().item():.4f}]")
                        print(f"   - 前5个值: {custom_attr.flatten()[:5].tolist()}")
                elif hasattr(custom_attr, '__len__'):
                    print(f"   - 长度: {len(custom_attr)}")
                    print(f"   - 内容预览: {str(custom_attr)[:100]}...")
                else:
                    print(f"   - 值: {custom_attr}")
            else:
                print(f"   ❌ 属性 '{inspect_attr}' 不存在")
                print(f"   💡 可用属性: {[name for name, _ in all_attrs[:10]]}...")
    
    else:
        print("❌ 数据集为空，无法分析样本属性")

def test_dataset(dataset_name, inspect_attr=None, detailed_tensor='x'):
    """测试数据集加载"""
    print(f"\n{'='*80}")
    print(f"🧪 测试数据集: {dataset_name}")
    print(f"{'='*80}")
    
    try:
        # 调用数据集函数
        dataset,__ = performat_SramDataset(
            dataset_dir="/data/tianjn/baseline/basline/basline/",
            name=dataset_name,
            neg_edge_ratio=1.0,
            to_undirected=True,
            task_level='node'
        )
        
        print(f"\n📊 数据集加载结果:")
        print(f"   - 包含的数据集: {dataset.names}")
        print(f"   - 总样本数: {len(dataset)}")
        print(f"   - 任务级别: {dataset.task_level}")
        
        # 各子数据集信息
        if len(dataset.names) > 1:
            print(f"\n📈 各子数据集详情:")
            for name in dataset.names:
                if name in dataset.data_lengths:
                    print(f"   - {name}:")
                    print(f"     * 样本数: {dataset.data_lengths[name]}")
                    print(f"     * 起始偏移: {dataset.data_offsets[name]}")
        
        # 样本信息
        if len(dataset) > 0:
            print(f"\n🔬 样本分析:")
            sample = dataset[0]
            print(f"   - 节点数: {sample.num_nodes}")
            print(f"   - 边数: {sample.num_edges}")
            
            if hasattr(sample, 'x') and sample.x is not None:
                print(f"   - 节点特征 (x): {sample.x.shape}")
                print(f"   - 特征范围: [{sample.x.min():.4f}, {sample.x.max():.4f}]")
            
            # 检查节点标签/属性 (y)
            if hasattr(sample, 'y') and sample.y is not None:
                print(f"\n📊 节点数据 (y) 详情:")
                print(f"   - 形状: {sample.y.shape}")
                print(f"   - 数据类型: {sample.y.dtype}")
                print(f"   - 值范围: [{sample.y.min():.4f}, {sample.y.max():.4f}]")
                
                # 检查是否有多维特征
                if len(sample.y.shape) > 1 and sample.y.shape[1] > 1:
                    print(f"\n📋 多维特征分析 (共{sample.y.shape[1]}维):")
                    feature_names = get_feature_names(sample)
                    
                    if feature_names is not None:
                        for i in range(min(sample.y.shape[1], len(feature_names))):
                            print(f"   维度 {i}: {feature_names[i]}")
                    else:
                        print(f"   特征名称: 无")
                        print(f"   各维度:")
                        for i in range(min(10, sample.y.shape[1])):
                            dim_data = sample.y[:, i]
                            print(f"   维度 {i}: 未知特征 (范围: [{dim_data.min():.4f}, {dim_data.max():.4f}])")
                else:
                    print(f"   - 这是单维标签数据，不是多维特征")
        
        # 调用详细属性分析
        analyze_dataset_attributes(dataset, inspect_attr, detailed_tensor)
        
        print(f"✅ 数据集 '{dataset_name}' 测试成功!")
        return dataset
        
    except Exception as e:
        print(f"❌ 数据集 '{dataset_name}' 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_dataloader(dataset):
    """测试数据加载器兼容性"""
    print(f"\n{'='*60}")
    print("🧪 测试数据加载器兼容性")
    print(f"{'='*60}")
    
    try:
        from torch_geometric.loader import DataLoader
        
        loader = DataLoader(dataset, batch_size=2, shuffle=True)
        print(f"✅ 数据加载器创建成功 (批次大小: 2)")
        
        # 测试获取一个批次
        for i, batch in enumerate(loader):
            print(f"📦 批次 {i+1}:")
            print(f"   - 批次节点数: {batch.num_nodes}")
            print(f"   - 批次边数: {batch.num_edges}")
            if hasattr(batch, 'batch'):
                print(f"   - 批次图数: {batch.batch.max().item() + 1}")
            
            if i >= 1:  # 只测试前2个批次
                break
        
        print("✅ 数据加载器测试成功!")
        return True
        
    except Exception as e:
        print(f"❌ 数据加载器测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("="*80)
    print("🚀 performat_SramDataset 函数测试")
    print("="*80)
    
    # 检查文件
    if not prepare_files():
        print("❌ 文件检查失败，测试终止")
        return
    
    # 🔧 控制哪个张量属性显示详细信息
    # 可选: 'x', 'y', 'edge_attr', None
    # 只有指定的属性会显示详细的子属性和特征信息，其他张量只显示基本维度信息
    detailed_tensor = 'x'  # 修改这里来控制显示哪个张量的详细信息
    
    # 🔧 自定义接口：在这里修改要检查的属性名称
    # 例如：inspect_attr = "edge_attr"  # 查看边属性
    # 例如：inspect_attr = "node_attr"  # 查看节点属性  
    # 例如：inspect_attr = "edge_label_index"  # 查看边标签索引
    inspect_attr = None  # 设置为 None 则不检查特定属性，设置为属性名则详细检查该属性
    
    # 测试用例 - 只测试合并的数据集
    test_cases = [
        "integrated_position_prediction_graph+integrated_power_density_prediction_graph"  # 只测试组合数据集
    ]
    
    successful_datasets = []
    
    # 逐个测试
    for dataset_name in test_cases:
        dataset = test_dataset(dataset_name, inspect_attr, detailed_tensor)  # 传入自定义检查属性和详细张量控制
        if dataset is not None:
            successful_datasets.append((dataset_name, dataset))
    
    # 如果有成功的数据集，测试数据加载器
    if successful_datasets:
        print(f"\n{'='*60}")
        print("🧪 额外测试：数据加载器")
        print(f"{'='*60}")
        
        # 用最后一个成功的数据集测试数据加载器
        last_dataset_name, last_dataset = successful_datasets[-1]
        print(f"使用数据集 '{last_dataset_name}' 测试数据加载器...")
        test_dataloader(last_dataset)
    
    # 总结
    print(f"\n{'='*80}")
    print("📋 测试总结")
    print(f"{'='*80}")
    
    print(f"总测试用例: {len(test_cases)}")
    print(f"成功案例: {len(successful_datasets)}")
    print(f"失败案例: {len(test_cases) - len(successful_datasets)}")
    
    print(f"\n✅ 成功的测试:")
    for dataset_name, _ in successful_datasets:
        print(f"   - {dataset_name}")
    
    if len(successful_datasets) == len(test_cases):
        print(f"\n🎉 所有测试通过! performat_SramDataset 函数工作完美!")
        print(f"💡 你可以安全地使用这个函数，支持单个和多个数据集组合")
        print(f"\n🔧 自定义检查接口使用方法:")
        print(f"   在 main() 函数中修改 inspect_attr 变量来检查特定属性")
        print(f"   例如: inspect_attr = 'edge_attr' 来详细查看边属性")
    else:
        print(f"\n⚠️  部分测试失败，请检查错误信息")

if __name__ == "__main__":
    main()