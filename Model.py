import torch
import torch.nn as nn
import math

class BiLSTM(nn.Module):
    def __init__(self, input_shape, output_dim=2):
        super(BiLSTM, self).__init__()
        len_window, feature_dim = input_shape
        input_size = feature_dim
        hidden_size = 64
        num_layers = 2
        num_classes = output_dim
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # BiLSTM
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, 
            batch_first=True, bidirectional=True
        )
        
        # Dueling架构 - 分别设置状态值和优势函数分支
        feature_size = hidden_size * 2  # *2 因为双向
        self.feature_layer = nn.Linear(feature_size, 128)
        self.relu = nn.ReLU()
        
        # 状态值分支 - 输出单个值
        self.value_stream = nn.Linear(128, 1)
        
        # 优势分支 - 输出每个动作的优势
        self.advantage_stream = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # 确保输入在正确的设备上
        device = next(self.parameters()).device
        x = x.to(device)
        
        # 处理输入维度: [batch, channels, length] -> [batch, length, channels]
        # if len(x.shape) == 4:
        #     batch_size, channels, height, width = x.shape
        #     x = x.squeeze(2).permute(0, 2, 1)  # [batch, channels, length] -> [batch, length, channels]
        # elif len(x.shape) == 3:
        #     # 假设输入是 [batch, channels, length]，需要转为 [batch, length, channels]
        # x = x.permute(0, 2, 1)

        
        # LSTM 期望输入形状为 [batch, seq_len, input_size]
        # 初始化隐藏状态
        x = x.float()
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        
        # 前向传播BiLSTM
        out, _ = self.lstm(x, (h0, c0)) #out的维度: [batch, seq_len, hidden_size*2]
        
        # 取最后一个时间步的输出作为特征
        features = self.relu(self.feature_layer(out[:, -1, :]))
        
        # 计算状态值
        values = self.value_stream(features)
        
        # 计算优势函数
        advantages = self.advantage_stream(features)
        
        # 组合状态值和优势函数: Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
        q_values = values + advantages - advantages.mean(dim=1, keepdim=True)
        
        return q_values

class Transformer(nn.Module):
    def __init__(self, input_shape, output_dim=2, d_model=128, n_heads=2, num_layers=2, dropout=0.1, extra_dim=0):
        super().__init__()
        len_window, feature_dim = input_shape
        # 直接在输入特征维拼接额外特征
        self.input_proj = nn.Linear(feature_dim + extra_dim, d_model)
        self.register_buffer("pos_encoding", self._build_positional_encoding(len_window, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.feature_layer = nn.Linear(d_model, 128)
        self.relu = nn.ReLU()
        self.value_stream = nn.Linear(128, 1)
        self.advantage_stream = nn.Linear(128, output_dim)

    def _build_positional_encoding(self, length, d_model):
        position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(length, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x, extra_features=None):
        device = next(self.parameters()).device
        x = x.to(device).float()
        # 将额外特征在时间维复制后与输入的特征维拼接
        if extra_features is not None:
            ef = extra_features.to(device).float()                     # [B, F_extra]
            ef_expanded = ef.unsqueeze(1).expand(-1, x.size(1), -1)   # [B, T, F_extra]
            x = torch.cat([x, ef_expanded], dim=-1)                    # [B, T, F+F_extra]
        x = self.input_proj(x)
        x = x + self.pos_encoding[:, : x.size(1)].to(x.dtype)
        out = self.encoder(x)
        features = self.relu(self.feature_layer(out[:, -1, :]))
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_values = values + advantages - advantages.mean(dim=1, keepdim=True)
        return q_values