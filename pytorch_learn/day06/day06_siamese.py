import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# --- 1. 定义匹配数据集 ---
class MapMatchingDataset(Dataset):
    def __init__(self, num_samples=2000, seq_len=20):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.t = np.linspace(0, 10, seq_len)
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 策略：50% 匹配(1)，50% 不匹配(0)
        is_match = (np.random.rand() > 0.5)
        
        # --- 1. 动态生成“底图道路” ---
        # 我们不再预先存死 roads，而是现场生成，这样可以随机覆盖左转和右转
        
        # 随机决定是“直行”还是“弯道”
        if np.random.rand() > 0.5:
            # 直行: y = 0
            map_road = np.stack([self.t, np.zeros_like(self.t)], axis=1)
            road_type = 'straight'
        else:
            # 弯道: y = a * x^2
            # 【关键修改】：这里的 curvature (曲率) 随机正负！
            # 随机范围在 -0.1 到 0.1 之间，不再只是 0.1
            curvature = np.random.uniform(-0.15, 0.15) 
            # 避免产生太平直的弯道（如果不幸随机到0）
            if abs(curvature) < 0.05: curvature = 0.1
            
            map_road = np.stack([self.t, curvature * self.t**2], axis=1)
            road_type = 'curve'
            
        # --- 2. 生成 GPS ---
        if is_match:
            # 匹配：GPS 基于同一条路
            base_road = map_road
            label = 1.0
        else:
            # 不匹配：GPS 基于“相反”类型的路
            if road_type == 'straight':
                # 如果底图是直的，GPS 就生成一个弯的（随机左或右）
                c = np.random.uniform(-0.15, 0.15)
                if abs(c) < 0.05: c = 0.1
                base_road = np.stack([self.t, c * self.t**2], axis=1)
            else:
                # 如果底图是弯的，GPS 就生成一个直的
                base_road = np.stack([self.t, np.zeros_like(self.t)], axis=1)
            label = 0.0
            
        # 给 GPS 加噪声
        gps_noise = np.random.normal(0, 0.3, size=base_road.shape)
        gps_traj = base_road + gps_noise
        
        # 转 Tensor
        seq1 = torch.tensor(gps_traj, dtype=torch.float32)
        seq2 = torch.tensor(map_road, dtype=torch.float32)
        target = torch.tensor([label], dtype=torch.float32)
        
        return seq1, seq2, target

# 测试一下
ds = MapMatchingDataset()
s1, s2, lab = ds[0]
print(f"GPS Shape: {s1.shape}, Map Shape: {s2.shape}, Label: {lab}")

class SiameseLSTMMatcher(nn.Module):
    def __init__(self):
        super().__init__()
        
        # --- 子网络 (Feature Extractor) ---
        # 用于提取轨迹特征，就像“视网膜”
        # 我们用 LSTM 把 [20, 2] 的序列变成 [64] 的特征向量
        self.lstm = nn.LSTM(input_size=2, hidden_size=64, batch_first=True)
        
        # --- 比较网络 (Comparator) ---
        # 输入：两个特征向量的差异
        # 输出：0~1 的相似度概率
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid() # 关键！把输出压缩到 0-1 之间作为概率
        )
        
    def forward_one(self, x):
        # 辅助函数：只跑一次 LSTM
        out, _ = self.lstm(x)
        return out[:, -1, :] # 取最后一个时间步的特征 [Batch, 64]

    def forward(self, x1, x2):
        # x1: GPS 轨迹
        # x2: Map 道路形状
        
        # 1. 孪生结构核心：同一个 LSTM，调用两次！
        # 这意味着它们共享同一套权重，用同样的“标准”去审视两个输入
        feat1 = self.forward_one(x1)
        feat2 = self.forward_one(x2)
        
        # 2. 计算差异
        # 我们可以计算两个向量的绝对差值 |v1 - v2|
        # 如果两个形状很像，这个差值应该接近 0
        diff = torch.abs(feat1 - feat2)
        
        # 3. 这里的 diff 是 [Batch, 64]
        # 把它喂给分类器，让它判断“这个差异算不算大？”
        score = self.classifier(diff)
        
        return score

# 实例化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseLSTMMatcher().to(device)
print(model)

# 数据加载器
loader = DataLoader(MapMatchingDataset(num_samples=2000), batch_size=32, shuffle=True)

# 优化器和Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss() # 二分类交叉熵

print(">>> 开始训练地图匹配模型...")

for epoch in range(10):
    total_loss = 0
    for gps_batch, map_batch, label_batch in loader:
        # 搬运到 GPU
        gps_batch = gps_batch.to(device)
        map_batch = map_batch.to(device)
        label_batch = label_batch.to(device)
        
        optimizer.zero_grad()
        
        # 前向传播：同时喂入两个序列
        probs = model(gps_batch, map_batch)
        
        # 计算 Loss
        loss = criterion(probs, label_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    print(f"Epoch {epoch+1}: Avg Loss = {total_loss / len(loader):.4f}")

print(">>> 训练完成！")

# --- 2. 保存模型 (Saving) ---
# 我们不仅要保存模型权重，还要保存 mean 和 std！
# 所以我们创建一个字典来打包所有东西
checkpoint = {
    'model_state_dict': model.state_dict(), # 模型的权重 (w, b)
}

save_path = 'day06_model_v1.pth'
torch.save(checkpoint, save_path)
print(f">>> 模型全套参数已保存至: {save_path}")