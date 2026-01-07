import torch
import torch.nn as nn
import math
import torch.nn.functional as F

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
        device = next(self.parameters()).device
        x = x.to(device).float()
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0)) # [batch, seq_len, hidden_size*2]
        features = self.relu(self.feature_layer(out[:, -1, :]))
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_values = values + advantages - advantages.mean(dim=1, keepdim=True)
        return q_values

class Transformer(nn.Module):
    def __init__(self, input_shape, output_dim=2, d_model=128, n_heads=2, num_layers=2, dropout=0.1, extra_dim=0):
        super().__init__()
        len_window, feature_dim = input_shape
        self.d_model = d_model
        # 状态投影
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.register_buffer("pos_encoding", self._build_positional_encoding(len_window, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 引入 cross attention 以融合 Goal（extra_dim>0 时生效）
        self.extra_dim = extra_dim
        if extra_dim > 0:
            # 修改：实现基于类别的语义Attention
            # Key: 将振动数据投影到类别空间 (0/K/M)，维度为 extra_dim
            self.goal_embedding = nn.Linear(extra_dim, d_model)  # 学习goal表示
            self.key_proj = nn.Linear(d_model, d_model)  # 保持特征维度
            self.value_proj = nn.Linear(d_model, d_model)
            
            self.dropout = nn.Dropout(dropout)
            self.norm_cross = nn.LayerNorm(d_model)
            # 移除原有的 MultiheadAttention，因为我们要手动控制 Q/K 的维度匹配

        # Dueling 头
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
        x = self.input_proj(x)
        x = x + self.pos_encoding[:, : x.size(1)].to(x.dtype)

        # 状态编码
        enc_output = self.encoder(x)  # [B, T, d_model]

        # 使用 Goal 作为 Query 进行 cross attention
        if self.extra_dim > 0 and extra_features is not None:
            # extra_features (Goal): [B, extra_dim] (One-hot vector)
            
            # 1. Query: 直接使用 Goal (0/K/M)
            # [B, 1, extra_dim]
            query = self.goal_embedding(extra_features).unsqueeze(1)  # [B, 1, d_model]
            
            # 2. Key: 将 State 投影到类别空间
            # 物理含义：预测每个时间步属于哪个类别
            # [B, T, extra_dim]
            keys = self.key_proj(enc_output)  # [B, T, d_model]
            
            # 3. Value: State 的特征表示
            # [B, T, d_model]
            values = self.value_proj(enc_output)
            
            # 4. 计算 Attention Scores: Query * Key^T
            # [B, 1, extra_dim] @ [B, extra_dim, T] -> [B, 1, T]
            # 含义：Goal类别 与 每个时间步的类别预测值 的匹配程度
            scores = torch.bmm(query, keys.transpose(1, 2)) / (self.d_model ** 0.5)
            
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # 5. 加权求和得到 Context
            # [B, 1, T] @ [B, T, d_model] -> [B, 1, d_model]
            context = torch.bmm(attn_weights, values)
            
            # 融合: 将 Context 与最后一个时间步的特征相加 (残差连接) 并归一化
            # 这样既保留了全局信息(last step)，又融入了Goal关注的特定片段信息
            features_input = self.norm_cross(context.squeeze(1) + enc_output[:, -1, :])
        else:
            features_input = enc_output[:, -1, :]                 # [B, d_model]

        features = self.relu(self.feature_layer(features_input))
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_values = values + advantages - advantages.mean(dim=1, keepdim=True)
        return q_values

# class ImprovedTransformer(nn.Module):
#     def __init__(self, input_shape, output_dim=2, d_model=128, n_heads=2, num_layers=2, dropout=0.1, extra_dim=0):
#         super().__init__()
#         len_window, feature_dim = input_shape
        
#         # 状态投影
#         self.input_proj = nn.Linear(feature_dim, d_model)
#         self.register_buffer("pos_encoding", self._build_positional_encoding(len_window, d_model))
        
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model,
#             nhead=n_heads,
#             dim_feedforward=d_model * 4,
#             dropout=dropout,
#             batch_first=True,
#         )
#         self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

#         # 引入 cross attention 以融合 Goal（extra_dim>0 时生效）
#         self.extra_dim = extra_dim
#         if extra_dim > 0:
#             # 修改：实现基于类别的语义Attention
#             # Key: 将振动数据投影到类别空间 (0/K/M)，维度为 extra_dim
#             self.goal_embedding = nn.Linear(extra_dim, d_model)  # 学习goal表示
#             self.key_proj = nn.Linear(d_model, d_model)  # 保持特征维度
#             self.value_proj = nn.Linear(d_model, d_model)
            
#             self.dropout = nn.Dropout(dropout)
#             self.norm_cross = nn.LayerNorm(d_model)
#             # 移除原有的 MultiheadAttention，因为我们要手动控制 Q/K 的维度匹配

#         # Dueling 头
#         self.feature_layer = nn.Linear(d_model, 128)
#         self.relu = nn.ReLU()
#         self.value_stream = nn.Linear(128, 1)
#         self.advantage_stream = nn.Linear(128, output_dim)

#         # 新增：Goal门控机制
#         self.goal_gate_proj = nn.Linear(d_model, d_model)

#     def _build_positional_encoding(self, length, d_model):
#         position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
#         pe = torch.zeros(length, d_model, dtype=torch.float32)
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         return pe.unsqueeze(0)

#     def forward(self, x, extra_features=None):
#         device = next(self.parameters()).device
#         x = x.to(device).float()
#         x = self.input_proj(x)
#         x = x + self.pos_encoding[:, : x.size(1)].to(x.dtype)

#         # 状态编码
#         enc_output = self.encoder(x)  # [B, T, d_model]

#         # 使用 Goal 作为 Query 进行 cross attention
#         if self.extra_dim > 0 and extra_features is not None:
#             # extra_features (Goal): [B, extra_dim] (One-hot vector)
            
#             # 1. Query: 直接使用 Goal (0/K/M)
#             # [B, 1, extra_dim]
#             query = self.goal_embedding(extra_features).unsqueeze(1)  # [B, 1, d_model]
            
#             # 2. Key: 将 State 投影到类别空间
#             # 物理含义：预测每个时间步属于哪个类别
#             # [B, T, extra_dim]
#             keys = self.key_proj(enc_output)  # [B, T, d_model]
            
#             # 3. Value: State 的特征表示
#             # [B, T, d_model]
#             values = self.value_proj(enc_output)
            
#             # 4. 计算 Attention Scores: Query * Key^T
#             # [B, 1, extra_dim] @ [B, extra_dim, T] -> [B, 1, T]
#             # 含义：Goal类别 与 每个时间步的类别预测值 的匹配程度
#             scores = torch.bmm(query, keys.transpose(1, 2)) / (d_model ** 0.5)
            
#             attn_weights = F.softmax(scores, dim=-1)
#             attn_weights = self.dropout(attn_weights)
            
#             # 5. 加权求和得到 Context
#             # [B, 1, T] @ [B, T, d_model] -> [B, 1, d_model]
#             context = torch.bmm(attn_weights, values)
            
#             # 融合: 将 Context 与最后一个时间步的特征相加 (残差连接) 并归一化
#             # 这样既保留了全局信息(last step)，又融入了Goal关注的特定片段信息
#             features_input = self.norm_cross(context.squeeze(1) + enc_output[:, -1, :])
#         else:
#             features_input = enc_output[:, -1, :]                 # [B, d_model]

#         # 1. 早期融合：Goal直接约束输入
#         if extra_features is not None:
#             goal_embed = self.goal_embedding(extra_features)  # [B, d_model]
#             # 方式1：Goal作为初始偏置
#             x = x + goal_embed.unsqueeze(1) * 0.3  # 早期约束
        
#         # 2. 中期融合：Goal门控机制
#         enc_output = self.encoder(x)
#         if extra_features is not None:
#             goal_gate = torch.sigmoid(
#                 self.goal_gate_proj(goal_embed)
#             )  # [B, d_model]
#             enc_output = enc_output * goal_gate.unsqueeze(1)
        
#         # 3. 晚期融合：原有的attention
#         if extra_features is not None:
#             context = self._cross_attention(enc_output, goal_embed)
#             features_input = self.norm_cross(context + enc_output[:, -1, :])
#         else:
#             features_input = enc_output[:, -1, :]
        
#         return self.dueling_head(features_input)