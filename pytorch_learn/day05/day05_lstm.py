import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 定义时序数据集
class TrajectoryDataset(Dataset):
    def __init__(self, num_samples=1000, seq_len=20):
        self.num_samples = num_samples
        self.seq_len = seq_len # 序列长度，比如观测过去 20 个点
        
        # --- 模拟数据生成 ---
        # 我们生成三种轨迹，每种轨迹包含 (x, y) 坐标
        
        # 1. 直行 (Straight): x 随时间增加, y 只有小波动
        # Shape: [样本数/3, 序列长, 2]
        x_straight = np.linspace(0, 10, seq_len) # 0 到 10 均匀分布
        # 广播机制：让每个样本的基础 x 都一样
        X_0 = np.tile(x_straight, (num_samples//3, 1)) 
        Y_0 = np.random.normal(0, 0.2, size=X_0.shape) # y 在 0 附近波动
        
        # 2. 左转 (Left): x 增加, y 向上弯曲 (y = 0.1 * x^2)
        X_1 = np.tile(x_straight, (num_samples//3, 1))
        Y_1 = 0.1 * (X_1 ** 2) + np.random.normal(0, 0.2, size=X_1.shape)
        
        # 3. 右转 (Right): x 增加, y 向下弯曲 (y = -0.1 * x^2)
        X_2 = np.tile(x_straight, (num_samples//3, 1))
        Y_2 = -0.1 * (X_2 ** 2) + np.random.normal(0, 0.2, size=X_2.shape)
        
        # --- 合并数据 ---
        # 现在的 Shape 需要合并成: [1000, 20, 2]
        # 这里的 stack 稍微有点绕，只要记得我们需要 (Batch, Seq, Feat)
        
        # 先合并 X 和 Y
        # data_0 shape: [333, 20, 2]
        data_0 = np.stack([X_0, Y_0], axis=2)
        data_1 = np.stack([X_1, Y_1], axis=2)
        data_2 = np.stack([X_2, Y_2], axis=2)
        
        # 合并所有样本
        self.inputs = np.concatenate([data_0, data_1, data_2], axis=0).astype(np.float32)
        
        # 生成标签 (0, 1, 2)
        self.labels = np.array([0]*(num_samples//3) + [1]*(num_samples//3) + [2]*(num_samples//3), dtype=np.int64)
        
        # --- 同样要记得归一化！(针对所有点的 x 和 y) ---
        # 为了简单，这里只打印 shape 确认一下
        print(f"数据集构建完成。Shape: {self.inputs.shape}") # 应该是 [1000, 20, 2]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 拿出来直接就是 [20, 2] 的 Tensor
        return torch.from_numpy(self.inputs[idx]), torch.tensor(self.labels[idx])

# 实例化
dataset = TrajectoryDataset(num_samples=1500, seq_len=20)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

class TrajectoryClassifier(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_classes=3):
        super().__init__()
        
        # --- 定义 LSTM 层 ---
        # input_size=2 : 输入特征是 x,y
        # hidden_size=64 : LSTM 内部的记忆容量 (可以理解为提取了 64 个特征)
        # batch_first=True : 告诉 PyTorch 输入数据的第一个维度是 Batch (最常用)
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            batch_first=True)
        
        # --- 定义全连接层 (分类头) ---
        # 把 LSTM 提取的 64 个特征，映射到 3 个分类上
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x 的形状: [Batch, Seq_Len, Feature] -> [32, 20, 2]
        
        # 1. 过 LSTM
        # LSTM 返回两个值: 
        #   out: 所有时间步的输出 [32, 20, 64]
        #   (h_n, c_n): 最后一个时间步的记忆状态
        out, (h_n, c_n) = self.lstm(x)
        
        # 2. 取“最后一个时间点”的输出
        # 因为我们看完了整段轨迹，要在最后时刻做决定
        # out[:, -1, :] 意思是：所有样本(:)，最后一个时间步(-1)，所有特征(:)
        # 形状变成: [32, 64]
        last_time_step_feature = out[:, -1, :]
        
        # 3. 过分类层
        x = self.fc(last_time_step_feature)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TrajectoryClassifier().to(device)

# 使用 Adam (你已经知道它比 SGD 快了)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

print(">>> 开始训练轨迹分类模型...")

for epoch in range(10): # 10轮就够了，LSTM 学这种简单规律很快
    total_loss = 0
    correct = 0
    total = 0
    
    for i, (inputs, labels) in enumerate(loader):
        inputs = inputs.to(device) # Shape: [32, 20, 2]
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs) # Forward
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # 计算精度
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    print(f"Epoch {epoch+1}: Loss={total_loss/len(loader):.4f}, Acc={100*correct/total:.2f}%")

print(">>> 训练完成！")

# --- 2. 保存模型 (Saving) ---
# 我们不仅要保存模型权重，还要保存 mean 和 std！
# 所以我们创建一个字典来打包所有东西
checkpoint = {
    'model_state_dict': model.state_dict(), # 模型的权重 (w, b)
}

save_path = 'turn_model_v1.pth'
torch.save(checkpoint, save_path)
print(f">>> 模型全套参数已保存至: {save_path}")
