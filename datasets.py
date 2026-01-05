import torch
import torchvision
import numpy as np
import h5py
import os
from torch.utils.data import DataLoader, Subset, TensorDataset
# from sklearn.model_selection import train_test_split

class ImbalancedDataset:
    def __init__(self, dataset_name="mnist", rho=0.01, batch_size=64, seed=42):
        """
        初始化数据集处理类
        :param dataset_name: 数据集名称 (e.g., "mnist", "cifar10", "TBM_K_M_Noise")
        :param rho: 不平衡因子 (用于某些数据集)
        :param batch_size: DataLoader 批次大小
        :param seed: 随机种子（确保可复现）
        """
        self.dataset_name = dataset_name
        self.rho = rho
        self.batch_size = batch_size
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # 加载并预处理数据
        self.train_data, self.test_data = self.load_raw_data()
        
        # 添加存储类别样本数的字典
        self.class_counts = {}
        
        # 预处理数据（针对多分类）
        self._preprocess_data()

    def load_raw_data(self):
        """加载原始数据集（需扩展时在此添加新数据集）"""
        if self.dataset_name == "mnist":
            # MNIST数据集处理（略）
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)) #对每个像素进行归一化，0.1307和0.3081分别是MNIST训练集的均值和标准差。这样可以让模型训练更稳定、收敛更快。
            ])
            
            print("正在下载MNIST训练集...")
            train_set = torchvision.datasets.MNIST(
                root='./data', train=True, download=True, transform=transform
            )
            
            print("正在下载MNIST测试集...")
            test_set = torchvision.datasets.MNIST(
                root='./data', train=False, download=True, transform=transform
            )
            return train_set, test_set
        elif self.dataset_name == "TBM_0.01":
            # 所有TBM数据集使用统一的文件路径
            print(f"正在加载TBM训练集...")
            train_data, train_labels = self._load_h5_file('/datasets/TBM/train_data/data/train_dataset0.3_1024_512_standard_snr5_prob0.3_amp0.05_ratio0.01_head10000.h5')
            
            print(f"正在加载TBM测试集...")
            test_data, test_labels = self._load_h5_file('/datasets/TBM/train_data/data/test_dataset0.3_1024_512_standard_snr5_prob0.3_amp0.05.h5')
            
            # 创建训练集和测试集
            train_set = self._create_dataset_from_arrays(train_data, train_labels)
            test_set = self._create_dataset_from_arrays(test_data, test_labels)
            
            return train_set, test_set
        elif self.dataset_name == "TBM_fault_0.01":
            # 加载TBM数据集并过滤掉label=0的样本
            print(f"正在加载TBM故障训练集（过滤正常样本）...")
            train_data, train_labels = self._load_h5_file('/datasets/TBM/train_data/data/train_dataset0.3_1024_512_standard_snr5_prob0.3_amp0.05_ratio0.01_head10000.h5')
            
            # 过滤掉标签为0的样本
            fault_indices = train_labels != 0
            train_data = train_data[fault_indices]
            train_labels = train_labels[fault_indices]
            
            print(f"正在加载TBM故障测试集（过滤正常样本）...")
            test_data, test_labels = self._load_h5_file('/datasets/TBM/train_data/data/test_dataset0.3_1024_512_standard_snr5_prob0.3_amp0.05.h5')
            
            # 过滤掉标签为0的样本
            fault_indices = test_labels != 0
            test_data = test_data[fault_indices]
            test_labels = test_labels[fault_indices]
            
            # 创建训练集和测试集
            train_set = self._create_dataset_from_arrays(train_data, train_labels)
            test_set = self._create_dataset_from_arrays(test_data, test_labels)
            
            return train_set, test_set
        elif self.dataset_name== "TBM_0.001":
            # 所有TBM数据集使用统一的文件路径
            print(f"正在加载TBM训练集...")
            train_data, train_labels = self._load_h5_file('/datasets/TBM/train_data/data/train_dataset0.3_1024_512_standard_snr5_prob0.3_amp0.05_ratio0.001_head10000.h5')
            
            print(f"正在加载TBM测试集...")
            test_data, test_labels = self._load_h5_file('/datasets/TBM/train_data/data/test_dataset0.3_1024_512_standard_snr5_prob0.3_amp0.05.h5')
            
            # 创建训练集和测试集
            train_set = self._create_dataset_from_arrays(train_data, train_labels)
            test_set = self._create_dataset_from_arrays(test_data, test_labels)
            
            return train_set, test_set
        elif self.dataset_name == "TBM_fault_0.001":
            # 加载TBM数据集并过滤掉label=0的样本
            print(f"正在加载TBM故障训练集（过滤正常样本）...")
            train_data, train_labels = self._load_h5_file('/datasets/TBM/train_data/data/train_dataset0.3_1024_512_standard_snr5_prob0.3_amp0.05_ratio0.001_head10000.h5')
            
            # 过滤掉标签为0的样本
            fault_indices = train_labels != 0
            train_data = train_data[fault_indices]
            train_labels = train_labels[fault_indices]
            
            print(f"正在加载TBM故障测试集（过滤正常样本）...")
            test_data, test_labels = self._load_h5_file('/datasets/TBM/train_data/data/test_dataset0.3_1024_512_standard_snr5_prob0.3_amp0.05.h5')
            
            # 过滤掉标签为0的样本
            fault_indices = test_labels != 0
            test_data = test_data[fault_indices]
            test_labels = test_labels[fault_indices]
            
            # 创建训练集和测试集
            train_set = self._create_dataset_from_arrays(train_data, train_labels)
            test_set = self._create_dataset_from_arrays(test_data, test_labels)
            
            return train_set, test_set
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
            
    def _load_h5_file(self, file_path):
        """从h5文件中加载数据和标签"""
        with h5py.File(file_path, 'r') as h5f:
            data = h5f['data'][:]
            labels = h5f['labels'][:]
        return data, labels
        
    def _create_dataset_from_arrays(self, data, labels):
        """从NumPy数组创建一个类似torchvision数据集的对象"""
        # 创建一个具有类似torchvision数据集接口的对象
        dataset = type('', (), {})()
        dataset.data = data
        dataset.targets = labels
        return dataset

    def _preprocess_data(self):
        """
        核心预处理：处理多分类数据
        - 保留原始标签（0-8）
        - 计算每个类别的样本数
        """
        # 获取标签数据 - 处理不同数据集的标签格式
        if isinstance(self.train_data.targets, list):
            train_labels = np.array(self.train_data.targets)
        elif isinstance(self.train_data.targets, np.ndarray):
            train_labels = self.train_data.targets
        else:
            train_labels = self.train_data.targets.numpy()
            
        if isinstance(self.test_data.targets, list):
            test_labels = np.array(self.test_data.targets)
        elif isinstance(self.test_data.targets, np.ndarray):
            test_labels = self.test_data.targets
        else:
            test_labels = self.test_data.targets.numpy()
        
        # 计算训练集中每个类别的样本数
        unique_classes, class_counts = np.unique(train_labels, return_counts=True)
        self.class_counts = {int(cls): int(count) for cls, count in zip(unique_classes, class_counts)}
        self.min_class = min(self.class_counts.items(), key=lambda x: x[1])[0]
        self.min_count = self.class_counts[self.min_class]
        self.reward_weights = {
            cls: (self.min_count / count) if count > 0 else 0.0
            for cls, count in self.class_counts.items()
        }
        self.high_level_mapping = {
            0: [0],
            1: [2, 3, 5, 6, 8],
            2: [1, 4, 7]
        }
        self.high_class_counts = {
            goal: int(sum(self.class_counts.get(cls, 0) for cls in cls_list))
            for goal, cls_list in self.high_level_mapping.items()
        }
        valid_high_counts = [cnt for cnt in self.high_class_counts.values() if cnt > 0]
        self.high_min_count = min(valid_high_counts)
        self.high_reward_weights = {
            goal: (self.high_min_count / count) if count > 0 else 0.0
            for goal, count in self.high_class_counts.items()
        }

        if not isinstance(self.train_data.data, torch.Tensor):
            train_data = torch.tensor(self.train_data.data)
        else:
            train_data = self.train_data.data 
        self.train_data = TensorDataset(train_data, torch.tensor(train_labels))
        
        # 处理测试集
        if not isinstance(self.test_data.data, torch.Tensor):
            test_data = torch.tensor(self.test_data.data)
        else:
            test_data = self.test_data.data
            
        self.test_data = TensorDataset(test_data, torch.tensor(test_labels))

    def get_dataloaders(self):
        """
        生成训练和测试 DataLoader
        :return: (train_loader, test_loader)
        """
        
        train_loader = DataLoader(
            self.train_data, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        test_loader = DataLoader(
            self.test_data, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        return train_loader, test_loader
        
    def get_full_dataset(self):
        """
        直接返回完整的训练和测试数据集
        :return: (train_data, train_labels, test_data, test_labels)
        """
        train_data = self.train_data.tensors[0]
        train_labels = self.train_data.tensors[1]
        test_data = self.test_data.tensors[0]
        test_labels = self.test_data.tensors[1]
        return train_data, train_labels, test_data, test_labels

    def get_class_distribution(self):
        """返回处理后的类别分布（用于验证）"""
        train_labels = self.train_data.tensors[1].numpy()
        test_labels = self.test_data.tensors[1].numpy()
        
        # 获取所有唯一的类别标签
        all_classes = sorted(np.unique(np.concatenate([train_labels, test_labels])))
        
        # 计算训练集中每个类别的数量
        train_counts = np.bincount(train_labels, minlength=max(all_classes)+1)
        test_counts = np.bincount(test_labels, minlength=max(all_classes)+1)
        
        # 返回按标签分组的计数
        return {
            "train": train_counts,
            "test": test_counts,
            "classes": all_classes,
            "reward_weights": self.reward_weights if hasattr(self, 'reward_weights') else None,
            "min_class": self.min_class if hasattr(self, 'min_class') else None,
            "min_count": self.min_count if hasattr(self, 'min_count') else None,
            "high_reward_weights": self.high_reward_weights if hasattr(self, 'high_reward_weights') else None
        }