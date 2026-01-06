TEST_ONLY = False  # 设置为 True 时只进行评估,不进行训练

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, defaultdict
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from datasets import ImbalancedDataset
from Model import Transformer
from evaluate import evaluate_model_hierarchical  # 需要新的评估函数
import pandas as pd

def set_random_seed(seed):
    """设置所有随机数种子以确保实验的可重现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"已设置随机数种子: {seed}")

class HierarchicalTransformer(nn.Module):
    """低层策略的Transformer，使用 cross attention 融合 goal(one-hot)"""
    def __init__(self, input_shape, goal_dim=3, output_dim=9):
        super(HierarchicalTransformer, self).__init__()
        # 引入 extra_dim=goal_dim，启用 Transformer 的 cross attention
        self.transformer = Transformer(input_shape, output_dim=output_dim, extra_dim=goal_dim)
        
    def forward(self, state, goal):
        # goal 为 [B, 3] 的 one-hot，作为 extra_features 传入 cross attention
        goal = goal.float()
        return self.transformer(state, extra_features=goal)

# class ReplayMemory:
#     """普通经验回放池（用于高层策略）"""
#     def __init__(self, capacity):
#         self.memory = deque(maxlen=capacity)
        
#     def push(self, data):
#         """存储经验"""
#         self.memory.append(data)
        
#     def sample(self, batch_size):
#         """随机采样"""
#         return random.sample(self.memory, min(batch_size, len(self.memory)))
    
#     def __len__(self):
#         return len(self.memory)

class PrioritizedReplayMemory:
    """
    优先级经验回放池 (Prioritized Experience Replay)
    使用 TD-error 作为优先级，配合 SumTree 数据结构实现高效采样
    """
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.capacity = capacity
        self.alpha = alpha  # 优先级指数
        self.beta_start = beta_start  # 重要性采样权重初始值
        self.beta_frames = beta_frames
        self.frame = 1
        
        # 使用 deque 简化实现，存储 (priority, data)
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0
        
    def push(self, data, error=None):
        """
        data: 经验元组
        error: TD-error，如果为 None 则使用最大优先级
        """
        priority = self.max_priority if error is None else (abs(error) + 1e-6) ** self.alpha
        self.buffer.append(data)
        self.priorities.append(priority)
        
    def sample(self, batch_size):
        """
        基于优先级进行采样，返回 (batch, indices, weights)
        """
        if len(self.buffer) == 0:
            return [], [], []
            
        batch_size = min(batch_size, len(self.buffer))
        
        # 计算采样概率
        priorities = np.array(self.priorities)
        probabilities = priorities / priorities.sum()
        
        # 根据优先级采样
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities, replace=False)
        samples = [self.buffer[idx] for idx in indices]
        
        # 计算重要性采样权重
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()  # 归一化权重
        
        self.frame += 1
        
        return samples, indices, weights
    
    def update_priorities(self, indices, errors):
        """
        更新指定索引样本的优先级
        indices: 样本索引列表
        errors: 对应的 TD-error 列表
        """
        for idx, error in zip(indices, errors):
            priority = (abs(error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)

class HierarchicalDQN():
    def __init__(self, input_shape, rho=0.01, reward_multiplier=1.0, high_discount_factor=0.1, low_discount_factor=0.01, num_classes=9):
        # self.discount_factor = discount_factor
        self.high_discount_factor = high_discount_factor
        self.low_discount_factor = low_discount_factor
        self.mem_size = 50000
        self.rho = rho
        self.reward_multiplier = reward_multiplier
        self.num_classes = num_classes
        # 移除 OOD 相关设定
        self.total_low_actions = num_classes
        # self.ood_action = num_classes
        # self.ood_reward_scale = 0.1
        self.t_max = 120000
        # 分别定义高层和低层的软更新系数
        self.high_eta = 0.05  # 高层策略的软更新系数
        self.low_eta = 0.05   # 低层策略的软更新系数
        self.learning_rate = 0.00025
        self.batch_size = 64
        self.ratio = 1
        
        # 定义高层目标映射
        # 0类: 正常, K类(标签1): [2,3,5,6,8], M类(标签2): [1,4,7]
        self.high_level_mapping = {
            0: [0],           # 正常类
            1: [2, 3, 5, 6, 8],  # K类
            2: [1, 4, 7]      # M类
        }
        
        # 原始类别到高层目标的映射
        self.class_to_goal = {
            0: 0,  # 正常
            1: 2, 4: 2, 7: 2,  # M类
            2: 1, 3: 1, 5: 1, 6: 1, 8: 1  # K类
        }
        
        # 高层策略网络 (输出3个高层目标: 0/K/M)
        self.high_q_net = Transformer(input_shape, output_dim=3)
        self.high_target_net = Transformer(input_shape, output_dim=3)
        self.high_target_net.load_state_dict(self.high_q_net.state_dict())
        
        # 低层策略网络 (输入state+goal, 输出9个类别)
        self.low_q_net = HierarchicalTransformer(input_shape, goal_dim=3, output_dim=self.total_low_actions)
        self.low_target_net = HierarchicalTransformer(input_shape, goal_dim=3, output_dim=self.total_low_actions)
        self.low_target_net.load_state_dict(self.low_q_net.state_dict())
        
        # 优化器
        self.high_optimizer = optim.Adam(self.high_q_net.parameters(), lr=self.learning_rate)
        self.low_optimizer = optim.Adam(self.low_q_net.parameters(), lr=self.learning_rate)
        
        # [修改] 使用 PrioritizedReplayMemory 替代 ClassBalancedReplayMemory
        self.high_replay_memory = PrioritizedReplayMemory(capacity=self.mem_size, alpha=0.6)
        self.low_replay_memory = PrioritizedReplayMemory(capacity=self.mem_size, alpha=0.6)
        
        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.high_q_net.to(self.device)
        self.high_target_net.to(self.device)
        self.low_q_net.to(self.device)
        self.low_target_net.to(self.device)
        
        # 训练计数器
        self.step_count = 0
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / (self.t_max * self.ratio)
        
        # 损失历史
        self.high_loss_history = []
        self.low_loss_history = []
        
        # 奖励权重
        self.reward_weights = {}
        self.high_reward_weights = {}

    def set_reward_weights(self, reward_weights):
        """设置各类别的奖励权重"""
        self.reward_weights = reward_weights
        print(f"已设置奖励权重: {self.reward_weights}")

    def set_high_level_reward_weights(self, high_reward_weights):
        """设置高层奖励权重"""
        self.high_reward_weights = high_reward_weights or {}
        print(f"已设置高层奖励权重: {self.high_reward_weights}")

    def get_goal_from_label(self, label):
        """将原始标签转换为高层目标"""
        return self.class_to_goal[label]

    def get_action_mask(self, goal):
        """根据高层目标生成动作掩码"""
        mask = torch.zeros(self.total_low_actions, device=self.device)
        valid_actions = self.high_level_mapping[goal]
        mask[valid_actions] = 1.0
        return mask
    def select_high_level_action(self, state):
        """选择高层动作(目标)"""
        if random.random() < self.epsilon:
            return random.randint(0, 2)  # 随机选择0/1/2
        else:
            with torch.no_grad():
                q_values = self.high_q_net(state)
            return q_values.argmax().item()

    def select_low_level_action(self, state, goal):
        """选择低层动作(具体类别),受goal限制"""
        goal_onehot = F.one_hot(torch.tensor([goal]), num_classes=3).float().to(self.device)
        
        if random.random() < self.epsilon:
            valid_actions = list(self.high_level_mapping[goal])
            return random.choice(valid_actions)
        else:
            with torch.no_grad():
                q_values = self.low_q_net(state, goal_onehot)
                mask = self.get_action_mask(goal)
                masked_q = q_values.clone()
                masked_q[0, mask == 0] = -float('inf')
            return masked_q.argmax().item()

    def compute_reward(self, action, target, level="low"):
        """计算奖励, 不使用 OOD 机制"""
        weights = self.high_reward_weights if level == "high" else self.reward_weights
        weight = weights.get(target, 1.0)
        terminal = False
        update = True

        if action == target:
            reward = weight * self.reward_multiplier
        else:
            reward = -weight * self.reward_multiplier
            terminal = True
            update = False
        
        return reward, terminal, update

    def replay_high_level(self, update_target=True):
        """训练高层策略 - 使用优先级采样"""
        batch, indices, weights = self.high_replay_memory.sample(self.batch_size)
        if not batch:
            return
            
        states, actions, rewards, next_states, terminals = zip(*batch)
        
        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.stack(next_states).to(self.device)
        terminals = torch.tensor(terminals, dtype=torch.bool, device=self.device).unsqueeze(1)
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1)
        
        # 计算当前Q值
        current_q = self.high_q_net(states).gather(1, actions)
        
        # 计算目标Q值
        with torch.no_grad():
            next_q = self.high_target_net(next_states).max(1, keepdim=True)[0]
            target_q = rewards + self.high_discount_factor * next_q * (~terminals)
        
        # 计算 TD-error 并更新优先级
        td_errors = (current_q - target_q).detach().cpu().numpy().flatten()
        self.high_replay_memory.update_priorities(indices, td_errors)
        
        # 使用重要性采样权重计算损失
        loss = (weights * F.mse_loss(current_q, target_q, reduction='none')).mean()
        self.high_loss_history.append(loss.item())
        
        self.high_optimizer.zero_grad()
        loss.backward()
        self.high_optimizer.step()
        
        if update_target:
            for target_param, param in zip(self.high_target_net.parameters(), self.high_q_net.parameters()):
                target_param.data.copy_(self.high_eta * param.data + (1.0 - self.high_eta) * target_param.data)

    def replay_low_level(self, update_target=True):
        """训练低层策略 - 使用优先级采样"""
        batch, indices, weights = self.low_replay_memory.sample(self.batch_size)
        if not batch:
            return
            
        states, goals, actions, rewards, next_states, next_goals, terminals = zip(*batch)
        
        states = torch.stack(states).to(self.device)
        goals = torch.stack(goals).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.stack(next_states).to(self.device)
        next_goals = torch.stack(next_goals).to(self.device)
        terminals = torch.tensor(terminals, dtype=torch.bool, device=self.device).unsqueeze(1)
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1)
        
        # 计算当前Q值
        current_q = self.low_q_net(states, goals).gather(1, actions)
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.low_target_net(next_states, next_goals)
            batch_size = next_q_values.size(0)
            for i in range(batch_size):
                goal_idx = next_goals[i].argmax().item()
                mask = self.get_action_mask(goal_idx)
                next_q_values[i, mask == 0] = -float('inf')
            next_q = next_q_values.max(1, keepdim=True)[0]
            target_q = rewards + self.low_discount_factor * next_q * (~terminals)
        
        # 计算 TD-error 并更新优先级
        td_errors = (current_q - target_q).detach().cpu().numpy().flatten()
        self.low_replay_memory.update_priorities(indices, td_errors)
        
        # 使用重要性采样权重计算损失
        loss = (weights * F.mse_loss(current_q, target_q, reduction='none')).mean()
        self.low_loss_history.append(loss.item())
        
        self.low_optimizer.zero_grad()
        loss.backward()
        self.low_optimizer.step()
        
        if update_target:
            for target_param, param in zip(self.low_target_net.parameters(), self.low_q_net.parameters()):
                target_param.data.copy_(self.low_eta * param.data + (1.0 - self.low_eta) * target_param.data)

    def train(self, dataset):
        """训练分层DQN - 使用优先级经验回放"""
        dist_info = dataset.get_class_distribution()
        if dist_info["reward_weights"] is not None:
            self.set_reward_weights(dist_info["reward_weights"])
        if dist_info.get("high_reward_weights") is not None:
            self.set_high_level_reward_weights(dist_info["high_reward_weights"])
        
        self.step_count = 0
        episode = 0
        
        # [修改] 简化预热逻辑 - PER 不需要类别平衡预热
        min_samples_before_training = 1000  # 最少样本数后开始训练
        
        total_pbar = tqdm(total=self.t_max, desc="Training Progress", unit="step")
        
        while self.step_count < self.t_max:
            episode += 1
            train_loader, _ = dataset.get_dataloaders()
            
            for batch_idx, (states, labels) in enumerate(train_loader):
                if self.step_count >= self.t_max:
                    break
                
                states = states.float().to(self.device)
                labels = labels.to(self.device)
                
                batch_size = states.size(0)
                for i in range(batch_size - 1):
                    current_state = states[i:i+1]
                    current_label = labels[i].item()
                    next_state = states[i+1:i+2]
                    next_label = labels[i+1].item()
                    
                    true_goal = self.get_goal_from_label(current_label)
                    next_true_goal = self.get_goal_from_label(next_label)
                    
                    # 1. 高层策略选择目标
                    high_action = self.select_high_level_action(current_state)
                    goal_onehot = F.one_hot(torch.tensor([high_action]), num_classes=3).float()
                    
                    high_reward, high_terminal, high_update = self.compute_reward(
                        high_action, true_goal, level="high"
                    )
                    
                    # [修改] 使用 PER，不需要 label 参数
                    self.high_replay_memory.push((
                        current_state.squeeze(0).cpu().clone().detach(),
                        high_action,
                        high_reward,
                        next_state.squeeze(0).cpu().clone().detach(),
                        high_terminal
                    ))
                    
                    # 2. 低层策略选择动作
                    low_action = self.select_low_level_action(current_state, high_action)
                    
                    low_reward, low_terminal, low_update = self.compute_reward(
                        low_action, current_label, level="low"
                    )
                    
                    next_high_action = self.select_high_level_action(next_state)
                    next_goal_onehot = F.one_hot(torch.tensor([next_high_action]), num_classes=3).float()
                    
                    self.low_replay_memory.push((
                        current_state.squeeze(0).cpu().clone().detach(),
                        goal_onehot.squeeze(0).cpu().clone().detach(),
                        low_action,
                        low_reward,
                        next_state.squeeze(0).cpu().clone().detach(),
                        next_goal_onehot.squeeze(0).cpu().clone().detach(),
                        low_terminal
                    ))
                    
                    # 3. 训练网络（简化预热检查）
                    performed_update = False
                    
                    if len(self.high_replay_memory) >= min_samples_before_training:
                        self.replay_high_level(update_target=high_update)
                        performed_update = True
                    
                    if len(self.low_replay_memory) >= min_samples_before_training:
                        self.replay_low_level(update_target=low_update)
                        performed_update = True
                    
                    # 更新计数器
                    if performed_update:
                        self.step_count += 1
                        total_pbar.update(1)
                        total_pbar.set_postfix({
                            'Episode': episode,
                            'Epsilon': f'{self.epsilon:.4f}',
                            'High_Mem': len(self.high_replay_memory),
                            'Low_Mem': len(self.low_replay_memory)
                        })
                    
                    # 衰减探索率
                    if self.epsilon > self.epsilon_min:
                        self.epsilon -= self.epsilon_decay
        
        total_pbar.close()
        print("训练完成!")
        # print(f"最终高层经验池大小: {len(self.high_replay_memory)}")
        # print(f"最终低层经验池大小: {len(self.low_replay_memory)}")
        # if warmup_complete_high:
        #     print("高层策略完成预热训练")
        # else:
        #     print("警告: 高层策略未完成预热！")
        # if warmup_complete_low:
        #     print("低层策略完成预热训练")
        # else:
        #     print("警告: 低层策略未完成预热！")

    def plot_loss(self, save_path):
        """绘制损失曲线"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        ax1.plot(self.high_loss_history)
        ax1.set_xlabel('Training Steps', fontsize=18)
        ax1.set_ylabel('High-Level Loss', fontsize=18)
        ax1.tick_params(labelsize=18)
        ax1.grid(False)
        
        ax2.plot(self.low_loss_history)
        ax2.set_xlabel('Training Steps', fontsize=18)
        ax2.set_ylabel('Low-Level Loss', fontsize=18)
        ax2.tick_params(labelsize=18)
        ax2.grid(False)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"损失曲线已保存到 {save_path}")

def main():
    tbm_configs = [('TBM_0.01', 0.01)]
    reward_multipliers = [1]
    discount_factor_pairs = [(0.1, 0.1)]
    eta_pairs = [(0.05, 0.05)]
    model_variants = ['Hierarchical_Transformer']
    # 移除 OOD 奖励缩放遍历
    save_dir = '/workspace/RL/DRLimb-Multi/final_hierarchical_results/Prioritized_replay'
    os.makedirs(save_dir, exist_ok=True)
    
    for model_variant in model_variants:
        for dataset_name, rho in tbm_configs:
            for reward_multiplier in reward_multipliers:
                for high_discount_factor, low_discount_factor in discount_factor_pairs:
                    for high_eta, low_eta in eta_pairs:
                        print(f"\n{'='*70}")
                        print(f"开始处理数据集: {dataset_name}, 模型: {model_variant}")
                        print(f"奖励倍数: {reward_multiplier}, 高层折扣因子: {high_discount_factor}, 低层折扣因子: {low_discount_factor}")
                        print(f"高层eta: {high_eta}, 低层eta: {low_eta}")
                        print(f"{'='*70}")
                        
                        input_shape = (1024, 3)
                        
                        try:
                            dataset = ImbalancedDataset(dataset_name=dataset_name, rho=rho, batch_size=64)
                            _, test_loader = dataset.get_dataloaders()
                            num_classes = 9
                            num_runs = 5
                            training_ratio = 1
                            
                            if not TEST_ONLY:
                                for run in range(1, num_runs + 1):
                                    print(f"\n{'='*50}")
                                    print(f"开始第 {run} 次训练")
                                    print(f"{'='*50}")
                                    
                                    classifier = HierarchicalDQN(
                                        input_shape, rho=rho,
                                        reward_multiplier=reward_multiplier,
                                        high_discount_factor=high_discount_factor,
                                        low_discount_factor=low_discount_factor,
                                        num_classes=num_classes
                                    )
                                    classifier.high_eta = high_eta
                                    classifier.low_eta = low_eta
                                    
                                    classifier.train(dataset)
                                    
                                    model_filename = f'{dataset_name}_rho{rho}_{model_variant}_reward{reward_multiplier}_highGamma{high_discount_factor}_lowGamma{low_discount_factor}_highEta{high_eta}_lowEta{low_eta}_第{run}次.pth'
                                    model_path = os.path.join(save_dir, model_filename)
                                    
                                    torch.save({
                                        'high_q_net': classifier.high_q_net.state_dict(),
                                        'low_q_net': classifier.low_q_net.state_dict(),
                                        'high_eta': classifier.high_eta,
                                        'low_eta': classifier.low_eta
                                    }, model_path)
                                    print(f"模型已保存到 {model_path}")
                                    
                                    loss_filename = f'{dataset_name}_rho{rho}_{model_variant}_reward{reward_multiplier}_highGamma{high_discount_factor}_lowGamma{low_discount_factor}_highEta{high_eta}_lowEta{low_eta}_第{run}次_loss.png'
                                    loss_path = os.path.join(save_dir, loss_filename)
                                    classifier.plot_loss(loss_path)
                                    
                                    print(f"\n开始评估第 {run} 次训练的模型...")
                                    metrics, high_accuracy = evaluate_model_hierarchical(
                                        hierarchical_dqn=classifier,
                                        test_loader=test_loader,
                                        save_dir=save_dir,
                                        dataset_name=dataset_name,
                                        training_ratio=training_ratio,
                                        rho=rho,
                                        dataset_obj=dataset,
                                        run_number=run,
                                        model_type=model_variant,
                                        reward_multiplier=reward_multiplier,
                                        high_discount_factor=high_discount_factor,
                                        low_discount_factor=low_discount_factor,
                                        high_eta=high_eta,
                                        low_eta=low_eta
                                    )
                            else:
                                # TEST_ONLY模式: 只进行评估
                                print("\n进入评估模式...")
                                for run in range(1, num_runs + 1):
                                    print(f"\n{'='*50}")
                                    print(f"加载并评估第 {run} 次训练的模型")
                                    print(f"{'='*50}")
                                    
                                    # 加载模型
                                    model_filename = f'{dataset_name}_rho{rho}_{model_variant}_reward{reward_multiplier}_highGamma{high_discount_factor}_lowGamma{low_discount_factor}_highEta{high_eta}_lowEta{low_eta}_第{run}次.pth'
                                    model_path = os.path.join(save_dir, model_filename)
                                    
                                    if not os.path.exists(model_path):
                                        print(f"模型文件不存在: {model_path}")
                                        continue
                                    
                                    # 创建模型实例
                                    classifier = HierarchicalDQN(
                                        input_shape, rho=rho,
                                        reward_multiplier=reward_multiplier,
                                        high_discount_factor=high_discount_factor,
                                        low_discount_factor=low_discount_factor,
                                        num_classes=num_classes
                                    )
                                    
                                    # 加载权重
                                    checkpoint = torch.load(model_path)
                                    classifier.high_q_net.load_state_dict(checkpoint['high_q_net'])
                                    classifier.low_q_net.load_state_dict(checkpoint['low_q_net'])
                                    classifier.high_eta = checkpoint.get('high_eta', high_eta)
                                    classifier.low_eta = checkpoint.get('low_eta', low_eta)
                                    print(f"已加载模型: {model_path}")
                                    
                                    # 评估模型
                                    metrics, high_accuracy = evaluate_model_hierarchical(
                                        hierarchical_dqn=classifier,
                                        test_loader=test_loader,
                                        save_dir=save_dir,
                                        dataset_name=dataset_name,
                                        training_ratio=training_ratio,
                                        rho=rho,
                                        dataset_obj=dataset,
                                        run_number=run,
                                        model_type=model_variant,
                                        reward_multiplier=reward_multiplier,
                                        high_discount_factor=high_discount_factor,
                                        low_discount_factor=low_discount_factor,
                                        high_eta=high_eta,
                                        low_eta=low_eta
                                    )
                                    
                                    print(f"第 {run} 次模型评估完成")          
                        except Exception as e:
                            print(f"处理配置时出错: {e}")
                            import traceback
                            traceback.print_exc()
                            continue

if __name__ == "__main__":
    main()