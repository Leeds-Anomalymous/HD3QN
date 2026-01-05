import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from datasets import ImbalancedDataset
from Model import BiLSTM
def load_model(model_path, model_type, input_shape, num_classes=9):
    """
    加载训练好的模型
    
    Args:
        model_path: 模型文件路径
        model_type: 模型类型
        input_shape: 输入形状
        num_classes: 类别数量
        
    Returns:
        loaded_model: 加载好的模型
    """
    # 根据模型类型创建相应的模型
    if model_type == 'BiLSTM':
        model = BiLSTM(input_shape, output_dim=num_classes)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 加载模型权重
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model

def get_class_samples(dataset_name, rho=0.01):
    """
    从数据集中获取每个类别的样本
    
    Args:
        dataset_name: 数据集名称
        rho: 不平衡率
        
    Returns:
        class_samples: 每个类别的样本字典
        test_dataset: 测试数据集对象
    """
    # 创建数据集
    dataset = ImbalancedDataset(dataset_name=dataset_name, rho=rho, batch_size=64)
    
    # 获取测试数据
    _, _, test_data, test_labels = dataset.get_full_dataset()
    
    # 获取所有可能的类别
    all_classes = sorted(np.unique(test_labels.numpy()))
    
    # 为每个类别找到一个样本
    class_samples = {}
    
    for cls in all_classes:
        # 找出该类别的所有样本索引
        indices = np.where(test_labels.numpy() == cls)[0]
        
        if len(indices) > 0:
            # 从该类别中选择第一个样本
            samples = test_data[indices]
            class_samples[cls] = samples
    
    return class_samples, dataset

def visualize_q_values(model, class_samples, model_type, save_path=None):
    """
    可视化每个类别样本的Q值
    
    Args:
        model: 训练好的Q网络模型
        class_samples: 每个类别的样本字典
        model_type: 模型类型
        save_path: 保存图表的路径
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # 存储每个类别的Q值
    q_values_dict = {}
    
    with torch.no_grad():
        for cls, samples in class_samples.items():
            # 将样本移到设备上
            samples = samples.float().to(device)
            
            # 根据模型类型进行数据预处理
            if model_type.startswith('TBM_conv1d'):
                if samples.dim() == 3 and samples.shape[1] == 1024 and samples.shape[2] == 3:
                    samples = samples.transpose(1, 2)
            
            # 分批推理每个类别全部样本并对 Q 值求平均
            q_batches = []
            for batch in torch.split(samples, 64):
                q_batches.append(model(batch))
            q_values = torch.cat(q_batches, dim=0).mean(dim=0).cpu().numpy()
            q_values_dict[cls] = q_values
    
    # 创建Q值表格
    classes = list(q_values_dict.keys())
    q_values_array = np.array([q_values_dict[cls] for cls in classes])
    
    # 创建DataFrame
    df = pd.DataFrame(q_values_array, index=[f'Class {cls}' for cls in classes],
                     columns=[f'Action {i}' for i in range(q_values_array.shape[1])])
    
    # 打印Q值表格
    print("\n每个类别样本的Q值:")
    print(df)
    
    # 绘制热图
    plt.figure(figsize=(12, 10))
    sns.heatmap(df, annot=True, fmt='.4f', cmap='coolwarm',
               xticklabels=df.columns, yticklabels=df.index,
               annot_kws={'fontsize':12})
    plt.xlabel('Qvalue', fontsize=16)
    plt.ylabel('Sample Class', fontsize=16)
    plt.title('Q-value Distribution for Each Class Sample', fontsize=18)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Q值热图已保存到 {save_path}")
    
    plt.show()
    
    return df

def main():
    # 设置参数
    model_path = '/workspace/RL/DRLimb-Multi/multi_class_results_dueling/TBM_0.01_rho0.01_BiLSTM_reward1_gamma0.1_训练完成比1_第1次.pth'  # 修改为你的模型路径
    model_type = 'BiLSTM'  # 模型类型
    dataset_name = 'TBM_0.01'  # 数据集名称
    rho = 0.01  # 不平衡率
    input_shape = (1024, 3)  # 输入形状
    save_dir = '/workspace/RL/DRLimb-Multi/multi_class_results_dueling'  # 保存目录
    
    # 创建保存目录（如果不存在）
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载模型
    model = load_model(model_path, model_type, input_shape)
    print(f"成功加载模型: {model_path}")
    
    # 获取每个类别的样本
    class_samples, dataset = get_class_samples(dataset_name, rho)
    print(f"已获取 {len(class_samples)} 个类别的样本")
    
    # 设置保存路径
    save_path = os.path.join(save_dir, f'{dataset_name}_{model_type}_q_values.png')
    
    # 可视化Q值
    q_values_df = visualize_q_values(model, class_samples, model_type, save_path)
    
    # 保存Q值表格为Excel
    excel_path = os.path.join(save_dir, f'{dataset_name}_{model_type}_q_values.xlsx')
    q_values_df.to_excel(excel_path)
    print(f"Q值表格已保存到 {excel_path}")

if __name__ == "__main__":
    main()
