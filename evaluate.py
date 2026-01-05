import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
import pandas as pd
from datetime import datetime
import os
import torch.nn.functional as F

def compute_group_accuracy(y_true, y_pred, group_indices):
    """
    计算特定组的准确率
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        group_indices: 该组包含的类别索引
    
    Returns:
        float: 该组的准确率
    """
    # 找出属于该组的样本
    group_mask = np.isin(y_true, group_indices)
    
    if not np.any(group_mask):
        return 0.0  # 如果没有该组的样本，返回0
        
    # 计算该组的准确率
    group_true = y_true[group_mask]
    group_pred = y_pred[group_mask]
    
    return accuracy_score(group_true, group_pred)

def compute_metrics(y_true, y_pred):
    """计算各类评估指标"""
    # 总体准确率
    accuracy = accuracy_score(y_true, y_pred)
    
    # 头部类别[0,1,2]的准确率
    head_acc = compute_group_accuracy(y_true, y_pred, [0, 1, 2])
    
    # 中部类别[3,4,5]的准确率
    mid_acc = compute_group_accuracy(y_true, y_pred, [3, 4, 5])
    
    # 尾部类别[6,7,8]的准确率
    tail_acc = compute_group_accuracy(y_true, y_pred, [6, 7, 8])
    
    return {
        'accuracy': accuracy,
        'head_acc': head_acc,
        'mid_acc': mid_acc,
        'tail_acc': tail_acc
    }
def balance_classes_for_confusion(y_true, y_pred, seed=42):
    """均衡各类别样本数量以绘制公平的混淆矩阵"""
    unique_classes, counts = np.unique(y_true, return_counts=True)
    min_count = counts.min()
    if min_count == 0:
        print("高层某些类别在测试集中不存在，无法进行数量平衡，将使用原始样本。")
        return y_true, y_pred, False
    rng = np.random.default_rng(seed)
    balanced_indices = []
    for cls in unique_classes:
        cls_indices = np.where(y_true == cls)[0]
        if len(cls_indices) > min_count:
            cls_indices = rng.choice(cls_indices, size=min_count, replace=False)
        balanced_indices.extend(cls_indices.tolist())
    balanced_indices = np.array(balanced_indices)
    return y_true[balanced_indices], y_pred[balanced_indices], True

def plot_confusion_matrix(y_true, y_pred, save_path=None, model_type=None, dataset_name=None, 
                         training_ratio=None, rho=None):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    
    # 增大方格内数字字体并保持其它参数
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(9),  # 9个类别
                yticklabels=range(9),  # 9个类别
                annot_kws={'fontsize':16})
    
    # 放大坐标轴标签字体和刻度字体
    plt.xlabel('Predicted Label', fontsize=18)
    plt.ylabel('True Label', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存到 {save_path}")
    
    # 关闭图形以释放内存，不显示窗口
    plt.close()

def evaluate_model_hierarchical(hierarchical_dqn, test_loader, save_dir='/root/autodl-tmp/checkpoints', 
                                dataset_name=None, training_ratio=None, rho=None, dataset_obj=None, 
                                run_number=None, model_type=None, reward_multiplier=1.0, 
                                high_discount_factor=0.1, low_discount_factor=0.01, 
                                high_eta=0.01, low_eta=0.05, ood_reward_scale=0.01):
    """
    评估分层DQN模型性能
    
    Args:
        hierarchical_dqn: 训练好的分层DQN模型
        test_loader: 测试数据加载器
        save_dir: 保存结果的目录
        dataset_name: 数据集名称
        training_ratio: 训练完成比例
        rho: 不平衡率
        dataset_obj: 数据集对象，用于获取样本数量统计
        run_number: 运行次数编号,用于文件命名
        model_type: 模型类型名称
        reward_multiplier: 奖励倍数
        high_discount_factor: 高层策略折扣因子
        low_discount_factor: 低层策略折扣因子
        high_eta: 高层软更新系数
        low_eta: 低层软更新系数
        ood_reward_scale: OOD奖励缩放系数
    """
    device = hierarchical_dqn.device
    hierarchical_dqn.high_q_net.eval()
    hierarchical_dqn.low_q_net.eval()
    
    all_preds = []
    all_labels = []
    all_high_preds = []  # 存储高层预测
    
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.float().to(device)
            batch_size = data.size(0)
            
            for i in range(batch_size):
                current_state = data[i:i+1]
                current_label = labels[i].item()
                
                # 高层策略选择目标(不使用epsilon贪婪)
                high_q_values = hierarchical_dqn.high_q_net(current_state)
                
                # 获取高层Q值排序索引（从大到小）
                high_q_sorted_indices = high_q_values[0].argsort(descending=True)
                
                # 尝试每个高层动作，直到低层不选择OOD
                final_low_action = None
                final_high_action = None
                
                for high_action in high_q_sorted_indices:
                    high_action = high_action.item()
                    
                    # 低层策略选择具体动作(应用掩码)
                    goal_onehot = F.one_hot(torch.tensor([high_action]), num_classes=3).float().to(device)
                    low_q_values = hierarchical_dqn.low_q_net(current_state, goal_onehot)
                    
                    # 应用掩码（只有goal=0时对OOD掩码，其他goal不掩码OOD）
                    if high_action == 0:
                        # 正常类：掩码OOD
                        mask = hierarchical_dqn.get_action_mask(high_action, allow_ood=False)
                    else:
                        # K类和M类：不掩码OOD
                        mask = hierarchical_dqn.get_action_mask(high_action, allow_ood=True)
                    
                    masked_q = low_q_values.clone()
                    masked_q[0, mask == 0] = -float('inf')
                    low_action = masked_q.argmax().item()
                    
                    # 如果低层没有选择OOD，则接受这个决策
                    if low_action != hierarchical_dqn.ood_action:
                        final_low_action = low_action
                        final_high_action = high_action
                        break
                    # 如果选择了OOD，继续尝试下一个高层动作
                
                # 如果所有高层动作都导致选择OOD（理论上不会发生，因为goal=0会掩码OOD）
                # 则使用最后一次的决策
                # if final_low_action is None:
                #     final_high_action = high_q_sorted_indices[0].item()
                #     goal_onehot = F.one_hot(torch.tensor([final_high_action]), num_classes=3).float().to(device)
                #     low_q_values = hierarchical_dqn.low_q_net(current_state, goal_onehot)
                    
                #     if final_high_action == 0:
                #         mask = hierarchical_dqn.get_action_mask(final_high_action, allow_ood=False)
                #     else:
                #         mask = hierarchical_dqn.get_action_mask(final_high_action, allow_ood=True)
                    
                #     masked_q = low_q_values.clone()
                #     masked_q[0, mask == 0] = -float('inf')
                #     final_low_action = masked_q.argmax().item()
                
                all_high_preds.append(final_high_action)
                all_preds.append(final_low_action)
                all_labels.append(current_label)
    
    # 转换为numpy数组
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_high_preds = np.array(all_high_preds)
    
    # 计算低层策略指标
    metrics = compute_metrics(all_labels, all_preds)
    
    # 计算高层策略准确率
    all_true_goals = np.array([hierarchical_dqn.get_goal_from_label(label) for label in all_labels])
    high_accuracy = accuracy_score(all_true_goals, all_high_preds)
    
    # 打印结果
    print("\n===== 分层模型评估结果 =====")
    print(f"高层策略准确率: {high_accuracy:.4f}")
    print(f"低层策略总体准确率: {metrics['accuracy']:.4f}")
    print(f"头部准确率 [0,1,2]: {metrics['head_acc']:.4f}")
    print(f"中部准确率 [3,4,5]: {metrics['mid_acc']:.4f}")
    print(f"尾部准确率 [6,7,8]: {metrics['tail_acc']:.4f}")
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取数据集统计信息
    max_class_count = 0
    min_class_count = float('inf')
    test_samples_count = 0
    
    if dataset_obj is not None:
        try:
            # 获取类别分布
            class_distribution = dataset_obj.get_class_distribution()
            train_counts = class_distribution['train']
            
            # 获取最多样本的类别（通常是正常样本，即类别0）
            max_class_count = train_counts[0]  # 假设类别0有最多样本
            
            # 获取最少样本的类别（通常是类别8）
            min_class_count = min(train_counts)
            
            # 测试集总样本数
            test_samples_count = sum(class_distribution['test'])
            
            print(f"最多样本类别数量: {max_class_count}")
            print(f"最少样本类别数量: {min_class_count}")
            print(f"测试集样本总数: {test_samples_count}")
        except Exception as e:
            print(f"获取数据集统计信息时出错: {e}")
    
    # 准备DataFrame数据
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_data = {
        '评估时间': [current_time],
        '数据集名称': [dataset_name if dataset_name else 'Unknown'],
        '模型类型': [model_type if model_type else 'Hierarchical'],
        '训练完成比例': [training_ratio if training_ratio is not None else 'Unknown'],
        '不平衡率rho': [rho if rho is not None else 'Unknown'],
        '奖励倍数': [reward_multiplier if reward_multiplier is not None else 1.0],
        '高层折扣因子': [high_discount_factor if high_discount_factor is not None else 0.1],
        '低层折扣因子': [low_discount_factor if low_discount_factor is not None else 0.01],
        '高层eta': [high_eta if high_eta is not None else 0.01],
        '低层eta': [low_eta if low_eta is not None else 0.05],
        'OOD奖励缩放': [ood_reward_scale if ood_reward_scale is not None else 0.01],
        '最多(正常)样本数': [max_class_count],
        '最少样本数': [min_class_count],
        '测试集样本数': [test_samples_count],
        '高层策略准确率': [high_accuracy],
        '低层总体准确率': [metrics['accuracy']],
        '头部准确率': [metrics['head_acc']],
        '中部准确率': [metrics['mid_acc']],
        '尾部准确率': [metrics['tail_acc']]
    }
    
    new_df = pd.DataFrame(new_data)
    
    # Excel文件路径
    excel_path = os.path.join(save_dir, 'hierarchical_evaluation_results.xlsx')
    
    # 检查是否存在现有文件
    if os.path.exists(excel_path):
        try:
            # 读取现有数据（包含标题行）
            existing_df = pd.read_excel(excel_path, header=0)
            # 合并数据
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        except Exception as e:
            print(f"读取现有Excel文件时出错: {e}")
            print("将创建新文件")
            combined_df = new_df
    else:
        combined_df = new_df
    
    # 保存到Excel文件（包含标题行）
    try:
        combined_df.to_excel(excel_path, index=False, header=True)
        print(f"评估结果已保存到 {excel_path}")
        print(f"当前文件包含 {len(combined_df)} 条评估记录")
    except Exception as e:
        print(f"保存Excel文件时出错: {e}")
    
    # 绘制低层策略混淆矩阵
    dataset_str = dataset_name if dataset_name else 'Unknown'
    model_str = model_type if model_type else 'Hierarchical'
    rho_str = f"rho{rho}" if rho is not None else 'rhoUnknown'
    reward_str = f"reward{reward_multiplier}" if reward_multiplier is not None else 'reward1.0'
    high_gamma_str = f"highGamma{high_discount_factor}" if high_discount_factor is not None else 'highGamma0.1'
    low_gamma_str = f"lowGamma{low_discount_factor}" if low_discount_factor is not None else 'lowGamma0.01'
    high_eta_str = f"highEta{high_eta}" if high_eta is not None else 'highEta0.01'
    low_eta_str = f"lowEta{low_eta}" if low_eta is not None else 'lowEta0.05'
    ood_scale_str = f"oodScale{ood_reward_scale}" if ood_reward_scale is not None else 'oodScale0.01'
    ratio_str = f"{training_ratio}" if training_ratio is not None else 'Unknown'
    
    # 低层策略混淆矩阵
    cm_filename = f'{dataset_str}_{model_str}_{rho_str}_{reward_str}_{high_gamma_str}_{low_gamma_str}_{high_eta_str}_{low_eta_str}_{ood_scale_str}_训练完成比{ratio_str}_第{run_number}次_low_cm.png'
    cm_path = os.path.join(save_dir, cm_filename)
    plot_confusion_matrix(all_labels, all_preds, save_path=cm_path, model_type=model_type,
                         dataset_name=dataset_name, training_ratio=training_ratio, rho=rho)
    
    # 绘制高层策略混淆矩阵
    high_cm_filename = f'{dataset_str}_{model_str}_{rho_str}_{reward_str}_{high_gamma_str}_{low_gamma_str}_{high_eta_str}_{low_eta_str}_{ood_scale_str}_训练完成比{ratio_str}_第{run_number}次_high_cm.png'
    high_cm_path = os.path.join(save_dir, high_cm_filename)

    balanced_true_goals, balanced_high_preds, balanced_flag = balance_classes_for_confusion(
        all_true_goals, all_high_preds)
    if balanced_flag:
        print("高层混淆矩阵已对各类别样本数进行均衡处理。")

    high_cm = confusion_matrix(balanced_true_goals, balanced_high_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(high_cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Healthy', 'Scratch', 'Scuffing'],
                yticklabels=['Healthy', 'Scratch', 'Scuffing'],
                annot_kws={'fontsize':16})
    plt.xlabel('Predicted Goal', fontsize=18)
    plt.ylabel('True Goal', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(high_cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"高层策略混淆矩阵已保存到 {high_cm_path}")
    
    return metrics, high_accuracy

def evaluate_model(model, test_loader, save_dir='/root/autodl-tmp/checkpoints', dataset_name=None, training_ratio=None, rho=None, dataset_obj=None, run_number=None, model_type=None, reward_multiplier=1.0, discount_factor=0.1):
    """
    评估模型性能并计算相关指标
    
    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        save_dir: 保存结果的目录
        dataset_name: 数据集名称
        training_ratio: 训练完成比例
        rho: 不平衡率
        dataset_obj: 数据集对象，用于获取样本数量统计
        run_number: 运行次数编号，用于文件命名
        model_type: 模型类型名称
        reward_multiplier: 奖励倍数
        discount_factor: 折扣因子
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.float().to(device)
            
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 转换为numpy数组
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 计算指标
    metrics = compute_metrics(all_labels, all_preds)
    
    # 打印结果
    print("\n===== 模型评估结果 =====")
    print(f"总体准确率: {metrics['accuracy']:.4f}")
    print(f"头部准确率 [0,1,2]: {metrics['head_acc']:.4f}")
    print(f"中部准确率 [3,4,5]: {metrics['mid_acc']:.4f}")
    print(f"尾部准确率 [6,7,8]: {metrics['tail_acc']:.4f}")
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取数据集统计信息
    max_class_count = 0
    min_class_count = float('inf')
    test_samples_count = 0
    
    if dataset_obj is not None:
        try:
            # 获取类别分布
            class_distribution = dataset_obj.get_class_distribution()
            train_counts = class_distribution['train']
            
            # 获取最多样本的类别（通常是正常样本，即类别0）
            max_class_count = train_counts[0]  # 假设类别0有最多样本
            
            # 获取最少样本的类别（通常是类别8）
            min_class_count = min(train_counts)
            
            # 测试集总样本数
            test_samples_count = sum(class_distribution['test'])
            
            print(f"最多样本类别数量: {max_class_count}")
            print(f"最少样本类别数量: {min_class_count}")
            print(f"测试集样本总数: {test_samples_count}")
        except Exception as e:
            print(f"获取数据集统计信息时出错: {e}")
    
    # 准备DataFrame数据
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_data = {
        '评估时间': [current_time],
        '数据集名称': [dataset_name if dataset_name else 'Unknown'],
        '模型类型': [model_type if model_type else 'Unknown'],
        '训练完成比例': [training_ratio if training_ratio is not None else 'Unknown'],
        '不平衡率rho': [rho if rho is not None else 'Unknown'],
        '奖励倍数': [reward_multiplier if reward_multiplier is not None else 1.0],
        '折扣因子': [discount_factor if discount_factor is not None else 0.1],
        '最多(正常)样本数': [max_class_count],
        '最少样本数': [min_class_count],
        '测试集样本数': [test_samples_count],
        '总体准确率': [metrics['accuracy']],
        '头部准确率': [metrics['head_acc']],
        '中部准确率': [metrics['mid_acc']],
        '尾部准确率': [metrics['tail_acc']]
    }
    
    new_df = pd.DataFrame(new_data)
    
    # Excel文件路径
    excel_path = os.path.join(save_dir, 'evaluation_results.xlsx')
    
    # 检查是否存在现有文件
    if os.path.exists(excel_path):
        try:
            # 读取现有数据（包含标题行）
            existing_df = pd.read_excel(excel_path, header=0)
            # 合并数据
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        except Exception as e:
            print(f"读取现有Excel文件时出错: {e}")
            print("将创建新文件")
            combined_df = new_df
    else:
        combined_df = new_df
    
    # 保存到Excel文件（包含标题行）
    try:
        combined_df.to_excel(excel_path, index=False, header=True)
        print(f"评估结果已保存到 {excel_path}")
        print(f"当前文件包含 {len(combined_df)} 条评估记录")
    except Exception as e:
        print(f"保存Excel文件时出错: {e}")
    
    # 绘制混淆矩阵
    # 生成带数据集名称、模型类型、不平衡率、奖励倍数、折扣因子、训练完成比例和序号的文件名
    dataset_str = dataset_name if dataset_name else 'Unknown'
    model_str = model_type if model_type else 'Unknown'
    rho_str = f"rho{rho}" if rho is not None else 'rhoUnknown'
    reward_str = f"reward{reward_multiplier}" if reward_multiplier is not None else 'reward1.0'
    gamma_str = f"gamma{discount_factor}" if discount_factor is not None else 'gamma0.1'
    ratio_str = f"{training_ratio}" if training_ratio is not None else 'Unknown'
    
    cm_filename = f'{dataset_str}_{model_str}_{rho_str}_{reward_str}_{gamma_str}_训练完成比{ratio_str}_第{run_number}次_cm.png'
    cm_path = os.path.join(save_dir, cm_filename)
    
    plot_confusion_matrix(all_labels, all_preds, save_path=cm_path, model_type=model_type,
                         dataset_name=dataset_name, training_ratio=training_ratio, rho=rho)
    
    return metrics