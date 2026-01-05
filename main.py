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

class HierarchicalDQN():
    def __init__(self, input_shape, rho=0.01, reward_multiplier=1.0, high_discount_factor=0.1, low_discount_factor=0.01, num_classes=9):
        # self.discount_factor = discount_factor
        self.high_discount_factor = high_discount_factor
        self.low_discount_factor = low_discount_factor
        self.mem_size = 50000
        self.rho = rho
        self.reward_multiplier = reward_multiplier
        self.num_classes = num_classes
        self.ood_action = num_classes  # OOD 类别索引
        self.total_low_actions = num_classes + 1
        self.ood_reward_scale = 0.1
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
        
        # [修改] 高层和低层都使用类别平衡经验回放池
        self.high_replay_memory = ClassBalancedReplayMemory(capacity=self.mem_size, num_classes=3)  # 高层3个目标
        self.low_replay_memory = ClassBalancedReplayMemory(capacity=self.mem_size, num_classes=num_classes)
        
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

    def get_action_mask(self, goal, allow_ood=True):
        """根据高层目标生成动作掩码
        当 goal=0(正常类) 时，屏蔽 OOD 动作
        其他目标时，根据 allow_ood 参数决定是否允许 OOD
        """
        mask = torch.zeros(self.total_low_actions, device=self.device)
        valid_actions = self.high_level_mapping[goal]
        mask[valid_actions] = 1.0
        
        # 目标为0时始终屏蔽OOD，其他目标根据allow_ood参数决定
        if goal != 0 and allow_ood:
            mask[self.ood_action] = 1.0
        
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
        """选择低层动作(具体类别),受goal限制
        当 goal=0 时，不允许选择 OOD 动作
        """
        goal_onehot = F.one_hot(torch.tensor([goal]), num_classes=3).float().to(self.device)
        
        if random.random() < self.epsilon:
            valid_actions = list(self.high_level_mapping[goal])
            # 目标为0时不添加OOD动作，其他目标添加
            if goal != 0:
                valid_actions.append(self.ood_action)
            return random.choice(valid_actions)
        else:
            with torch.no_grad():
                q_values = self.low_q_net(state, goal_onehot)
                # 应用掩码时，目标为0会自动屏蔽OOD
                mask = self.get_action_mask(goal)
                masked_q = q_values.clone()
                masked_q[0, mask == 0] = -float('inf')
            return masked_q.argmax().item()

    def compute_reward(self, action, target, level="low", high_action=None, true_goal=None):
        """计算奖励, 根据层级选择权重"""
        weights = self.high_reward_weights if level == "high" else self.reward_weights
        weight = weights.get(target, 1.0)
        terminal = False
        update = True

        # 低层 OOD 特殊情况：仅处理高层错误时的纠错奖励
        if level == "low" and high_action is not None and true_goal is not None:
            # 如果高层选择错误
            if high_action != true_goal:
                if action == self.ood_action:
                    # 纠错成功：给予带权重的正奖励（较小）
                    reward = self.reward_multiplier * self.ood_reward_scale * weight
                    update = True
                else:
                    # 错上加错：给予标准惩罚
                    reward = -weight * self.reward_multiplier
                    update = False
                terminal = True  # 高层错误时必须终止
                return reward, terminal, update
                
        # 正常奖励计算
        # 涵盖了：
        # 1. 高层策略的奖励计算
        # 2. 高层正确时，低层策略的奖励计算（包括选对、选错类、选OOD）
        if action == target:
            reward = weight * self.reward_multiplier
        else:
            # 选错（包括在高层正确时选了OOD）
            reward = -weight * self.reward_multiplier
            terminal = True
            update = False
        
        return reward, terminal, update

    def replay_high_level(self, update_target=True):
        """训练高层策略"""
        # [修改] 使用类别平衡采样
        batch = self.high_replay_memory.sample(self.batch_size)
        if not batch:
            return
            
        states, actions, rewards, next_states, terminals = zip(*batch)
        
        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.stack(next_states).to(self.device)
        terminals = torch.tensor(terminals, dtype=torch.bool, device=self.device).unsqueeze(1)
        
        # 计算当前Q值
        current_q = self.high_q_net(states).gather(1, actions)
        
        # 计算目标Q值 - 使用高层折扣因子
        with torch.no_grad():
            next_q = self.high_target_net(next_states).max(1, keepdim=True)[0]
            target_q = rewards + self.high_discount_factor * next_q * (~terminals)
        
        # 计算损失并更新
        loss = F.mse_loss(current_q, target_q)
        self.high_loss_history.append(loss.item())
        
        self.high_optimizer.zero_grad()
        loss.backward()
        self.high_optimizer.step()
        
        # 软更新目标网络 - 使用高层eta
        if update_target:
            for target_param, param in zip(self.high_target_net.parameters(), self.high_q_net.parameters()):
                target_param.data.copy_(self.high_eta * param.data + (1.0 - self.high_eta) * target_param.data)

    def replay_low_level(self, update_target=True):
        """训练低层策略"""
        # if len(self.low_replay_memory) < self.batch_size:
        #     return
            
        # [修改] 使用自定义的 sample 方法
        batch = self.low_replay_memory.sample(self.batch_size)
        states, goals, actions, rewards, next_states, next_goals, terminals = zip(*batch)
        
        states = torch.stack(states).to(self.device)
        goals = torch.stack(goals).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.stack(next_states).to(self.device)
        next_goals = torch.stack(next_goals).to(self.device)
        terminals = torch.tensor(terminals, dtype=torch.bool, device=self.device).unsqueeze(1)
        
        # 计算当前Q值
        current_q = self.low_q_net(states, goals).gather(1, actions)
        
        # 计算目标Q值 - 使用低层折扣因子
        with torch.no_grad():
            next_q_values = self.low_target_net(next_states, next_goals)
            # 应用掩码到next_q
            batch_size = next_q_values.size(0)
            for i in range(batch_size):
                goal_idx = next_goals[i].argmax().item()
                mask = self.get_action_mask(goal_idx)
                next_q_values[i, mask == 0] = -float('inf')
            next_q = next_q_values.max(1, keepdim=True)[0]
            target_q = rewards + self.low_discount_factor * next_q * (~terminals)
        
        # 计算损失并更新
        loss = F.mse_loss(current_q, target_q)
        self.low_loss_history.append(loss.item())
        
        self.low_optimizer.zero_grad()
        loss.backward()
        self.low_optimizer.step()
        
        # 软更新目标网络 - 使用低层eta
        if update_target:
            for target_param, param in zip(self.low_target_net.parameters(), self.low_q_net.parameters()):
                target_param.data.copy_(self.low_eta * param.data + (1.0 - self.low_eta) * target_param.data)

    def train(self, dataset):
        """训练分层DQN - 高层和低层各做一次决策"""
        dist_info = dataset.get_class_distribution()
        if dist_info["reward_weights"] is not None:
            self.set_reward_weights(dist_info["reward_weights"])
        if dist_info.get("high_reward_weights") is not None:
            self.set_high_level_reward_weights(dist_info["high_reward_weights"])
        
        self.step_count = 0
        episode = 0
        
        # [修改] 高层和低层各自的预热参数
        min_samples_per_class_high = 25  # 高层每个类别需要25个样本
        min_samples_per_class_low = 10   # 低层每个类别需要10个样本
        warmup_complete_high = False
        warmup_complete_low = False
        
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
                    
                    # [修改] 高层使用类别平衡池，添加label参数
                    self.high_replay_memory.push((
                        current_state.squeeze(0).cpu().clone().detach(),
                        high_action,
                        high_reward,
                        next_state.squeeze(0).cpu().clone().detach(),
                        high_terminal
                    ), label=true_goal)  # 使用true_goal作为标签
                    
                    # 2. 低层策略选择动作
                    low_action = self.select_low_level_action(current_state, high_action)
                    
                    low_reward, low_terminal, low_update = self.compute_reward(
                        low_action, current_label, level="low",
                        high_action=high_action, true_goal=true_goal
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
                    ), label=current_label)
                    
                    # 3. 检查预热期状态
                    # [修改] 检查高层预热状态
                    if not warmup_complete_high:
                        if self.high_replay_memory.is_ready(min_samples_per_class_high):
                            warmup_complete_high = True
                            print(f"\n高层经验池预热完成！所有目标类别都有至少 {min_samples_per_class_high} 个样本")
                            print(f"当前高层经验池大小: {len(self.high_replay_memory)}")
                        else:
                            if len(self.high_replay_memory) % 100 == 0 and len(self.high_replay_memory) > 0:
                                missing = self.high_replay_memory.get_missing_classes(min_samples_per_class_high)
                                print(f"\n高层预热中... 样本不足的目标类别: {missing}")
                                print(f"当前高层经验池: {len(self.high_replay_memory)}")
                    
                    # 检查低层预热状态
                    if not warmup_complete_low:
                        if self.low_replay_memory.is_ready(min_samples_per_class_low):
                            warmup_complete_low = True
                            print(f"\n低层经验池预热完成！所有类别都有至少 {min_samples_per_class_low} 个样本")
                            print(f"当前低层经验池大小: {len(self.low_replay_memory)}")
                        else:
                            if len(self.low_replay_memory) % 500 == 0 and len(self.low_replay_memory) > 0:
                                missing = self.low_replay_memory.get_missing_classes(min_samples_per_class_low)
                                print(f"\n低层预热中... 样本不足的类别: {missing}")
                                print(f"当前低层经验池: {len(self.low_replay_memory)}")
                    
                    # 4. 训练网络
                    performed_update = False
                    
                    # [修改] 高层策略：需要预热完成后才开始训练
                    if warmup_complete_high and len(self.high_replay_memory) >= self.batch_size:
                        self.replay_high_level(update_target=high_update)
                        performed_update = True
                    
                    # 低层策略：需要预热完成后才开始训练
                    if warmup_complete_low and len(self.low_replay_memory) >= self.batch_size:
                        self.replay_low_level(update_target=low_update)
                        performed_update = True
                    
                    # 更新计数器（仅在至少一个网络更新时）
                    if performed_update:
                        self.step_count += 1
                        total_pbar.update(1)
                        total_pbar.set_postfix({
                            'Episode': episode,
                            'Epsilon': f'{self.epsilon:.4f}',
                            'High_Mem': len(self.high_replay_memory),
                            'Low_Mem': len(self.low_replay_memory),
                            'High_Ready': '✓' if warmup_complete_high else '✗',
                            'Low_Ready': '✓' if warmup_complete_low else '✗'
                        })
                    
                    # 衰减探索率
                    if self.epsilon > self.epsilon_min:
                        self.epsilon -= self.epsilon_decay
        
        total_pbar.close()
        print("训练完成!")
        print(f"最终高层经验池大小: {len(self.high_replay_memory)}")
        print(f"最终低层经验池大小: {len(self.low_replay_memory)}")
        if warmup_complete_high:
            print("高层策略完成预热训练")
        else:
            print("警告: 高层策略未完成预热！")
        if warmup_complete_low:
            print("低层策略完成预热训练")
        else:
            print("警告: 低层策略未完成预热！")

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

class ClassBalancedReplayMemory:
    """
    基于类别的经验回放池
    确保采样时包含所有类别的样本，防止尾部样本被遗忘
    """
    def __init__(self, capacity, num_classes):
        self.capacity = capacity
        self.num_classes = num_classes
        self.per_class_capacity = capacity // num_classes
        self.buffers = defaultdict(lambda: deque(maxlen=self.per_class_capacity))
        self.count = 0
        # 新增：记录每个类别的样本数
        self.class_counts = defaultdict(int)

    def push(self, data, label):
        """
        data: (state, action, reward, next_state, terminal) 等元组
        label: 用于分类存储的标签 (int)
        """
        self.buffers[label].append(data)
        self.class_counts[label] = len(self.buffers[label])
        self.count += 1

    def is_ready(self, min_samples_per_class=10):
        """
        检查是否所有类别都有足够的样本
        min_samples_per_class: 每个类别最少需要的样本数
        """
        for cls in range(self.num_classes):
            if self.class_counts[cls] < min_samples_per_class:
                return False
        return True
    
    def get_missing_classes(self, min_samples_per_class=10):
        """返回样本不足的类别列表"""
        missing = []
        for cls in range(self.num_classes):
            if self.class_counts[cls] < min_samples_per_class:
                missing.append(cls)
        return missing

    def sample(self, batch_size):
        """
        分层采样：尝试从每个类别中均匀采样
        """
        samples = []
        active_classes = [k for k, v in self.buffers.items() if len(v) > 0]
        if not active_classes:
            return []
            
        samples_per_class = batch_size // len(active_classes)
        remainder = batch_size % len(active_classes)

        for cls in active_classes:
            n = samples_per_class + (1 if remainder > 0 else 0)
            remainder -= 1
            
            buffer = self.buffers[cls]
            if len(buffer) >= n:
                samples.extend(random.sample(buffer, n))
            else:
                samples.extend(random.choices(buffer, k=n))
        
        random.shuffle(samples)
        return samples

    def __len__(self):
        return sum(len(b) for b in self.buffers.values())

def main():
    tbm_configs = [('TBM_0.01', 0.01)]
    reward_multipliers = [1]
    # 将高层和低层折扣因子成对定义
    discount_factor_pairs = [(0.1, 0.1)]  # (高层折扣因子, 低层折扣因子)
    # 新增：高层和低层eta的成对定义，用于参数敏感性分析
    eta_pairs = [(0.05, 0.05)]  # (high_eta, low_eta)
    model_variants = ['Hierarchical_Transformer']
    # 新增OOD奖励缩放系数的遍历
    # ood_reward_scales = list(np.linspace(0.1, 1, 5))
    ood_reward_scales = [0.5]
    
    save_dir = '/workspace/RL/DRLimb-Multi/final_hierarchical_results/new_replay'
    os.makedirs(save_dir, exist_ok=True)
    
    for model_variant in model_variants:
        for dataset_name, rho in tbm_configs:
            for reward_multiplier in reward_multipliers:
                for high_discount_factor, low_discount_factor in discount_factor_pairs:
                    for high_eta, low_eta in eta_pairs:  # 新增eta遍历
                        for ood_reward_scale in ood_reward_scales:
                            print(f"\n{'='*70}")
                            print(f"开始处理数据集: {dataset_name}, 模型: {model_variant}")
                            print(f"奖励倍数: {reward_multiplier}, 高层折扣因子: {high_discount_factor}, 低层折扣因子: {low_discount_factor}")
                            print(f"高层eta: {high_eta}, 低层eta: {low_eta}")  # 新增打印
                            print(f"OOD奖励缩放: {ood_reward_scale}")
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
                                        # 设置OOD奖励缩放系数和eta参数
                                        classifier.ood_reward_scale = ood_reward_scale
                                        classifier.high_eta = high_eta
                                        classifier.low_eta = low_eta
                                        
                                        classifier.train(dataset)
                                        
                                        # 保存模型（文件名包含eta参数）
                                        model_filename = f'{dataset_name}_rho{rho}_{model_variant}_reward{reward_multiplier}_highGamma{high_discount_factor}_lowGamma{low_discount_factor}_highEta{high_eta}_lowEta{low_eta}_oodScale{ood_reward_scale}_第{run}次.pth'
                                        model_path = os.path.join(save_dir, model_filename)
                                        
                                        torch.save({
                                            'high_q_net': classifier.high_q_net.state_dict(),
                                            'low_q_net': classifier.low_q_net.state_dict(),
                                            'ood_reward_scale': classifier.ood_reward_scale,
                                            'high_eta': classifier.high_eta,  # 新增保存
                                            'low_eta': classifier.low_eta      # 新增保存
                                        }, model_path)
                                        print(f"模型已保存到 {model_path}")
                                        
                                        # 绘制损失曲线
                                        loss_filename = f'{dataset_name}_rho{rho}_{model_variant}_reward{reward_multiplier}_highGamma{high_discount_factor}_lowGamma{low_discount_factor}_highEta{high_eta}_lowEta{low_eta}_oodScale{ood_reward_scale}_第{run}次_loss.png'
                                        loss_path = os.path.join(save_dir, loss_filename)
                                        classifier.plot_loss(loss_path)
                                        
                                        # 评估模型
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
                                            high_eta=high_eta,  # 新增参数
                                            low_eta=low_eta,    # 新增参数
                                            ood_reward_scale=ood_reward_scale
                                        )
                                
                                else:
                                    # TEST_ONLY模式: 只进行评估
                                    print("\n进入评估模式...")
                                    for run in range(1, num_runs + 1):
                                        print(f"\n{'='*50}")
                                        print(f"加载并评估第 {run} 次训练的模型")
                                        print(f"{'='*50}")
                                        
                                        # 加载模型
                                        model_filename = f'{dataset_name}_rho{rho}_{model_variant}_reward{reward_multiplier}_highGamma{high_discount_factor}_lowGamma{low_discount_factor}_highEta{high_eta}_lowEta{low_eta}_oodScale{ood_reward_scale}_第{run}次.pth'
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
                                        # 加载参数
                                        if 'ood_reward_scale' in checkpoint:
                                            classifier.ood_reward_scale = checkpoint['ood_reward_scale']
                                        else:
                                            classifier.ood_reward_scale = ood_reward_scale
                                        if 'high_eta' in checkpoint:
                                            classifier.high_eta = checkpoint['high_eta']
                                        else:
                                            classifier.high_eta = high_eta
                                        if 'low_eta' in checkpoint:
                                            classifier.low_eta = checkpoint['low_eta']
                                        else:
                                            classifier.low_eta = low_eta
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
                                            high_eta=high_eta,  # 新增参数
                                            low_eta=low_eta,    # 新增参数
                                            ood_reward_scale=ood_reward_scale
                                        )
                                        
                                        print(f"第 {run} 次模型评估完成")          
                            except Exception as e:
                                print(f"处理配置时出错: {e}")
                                import traceback
                                traceback.print_exc()
                                continue

if __name__ == "__main__":
    main()