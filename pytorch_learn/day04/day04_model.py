import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

# --- 1. 快速复现昨天的训练过程 ---
# 为了演示，我们直接生成归一化好的数据，简化流程
num_samples = 1500
# 生成三条车道数据 (Lane 0: 0, Lane 1: 10, Lane 2: 20)
X = np.vstack((
    np.random.normal(0, 1, (500, 1)),
    np.random.normal(10, 1, (500, 1)),
    np.random.normal(20, 1, (500, 1))
))
Y = np.random.uniform(0, 100, (1500, 1)) # 噪声维度
inputs_raw = np.hstack((X, Y)).astype(np.float32)
labels = torch.tensor([0]*500 + [1]*500 + [2]*500, dtype=torch.long)

# *** 关键：计算并记录归一化参数 ***
mean = inputs_raw.mean(axis=0)
std = inputs_raw.std(axis=0)
print(f"[训练阶段] 计算出的 Mean: {mean}, Std: {std}")

# 归一化
inputs_norm = (inputs_raw - mean) / (std + 1e-6)
inputs_tensor = torch.from_numpy(inputs_norm)

# 定义模型
model = nn.Linear(2, 3) # 输入2，输出3分类
optimizer = optim.Adam(model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()

print(">>> 开始快速训练...")

for i in range(100): # 训练100次
    # --- 标准训练步骤 (不变) ---
    optimizer.zero_grad()
    outputs = model(inputs_tensor)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # --- 新增：每 20 轮 (Epoch) 打印一次调试信息 ---
    # (i + 1) % 20 == 0 表示：当 i+1 能被 20 整除时 (即第20, 40, 60...次)
    if (i + 1) % 20 == 0:
        
        # 1. 计算准确率 (Accuracy)
        # torch.max(outputs, 1) 返回两个值：(最大数值, 最大值的索引)
        # 我们只需要索引 (predicted)，因为它代表类别 (0, 1, 2)
        _, predicted = torch.max(outputs, 1)
        
        # 计算猜对的数量
        correct = (predicted == labels).sum().item()
        # 计算总数量
        total = labels.size(0)
        # 算出百分比
        accuracy = 100 * correct / total
        
        # 2. 打印 Loss 和 Accuracy
        print(f"Epoch [{i+1}/100]: Loss = {loss.item():.4f}, Accuracy = {accuracy:.2f}%")

print(">>> 训练完成。")

# --- 2. 保存模型 (Saving) ---
# 我们不仅要保存模型权重，还要保存 mean 和 std！
# 所以我们创建一个字典来打包所有东西
checkpoint = {
    'model_state_dict': model.state_dict(), # 模型的权重 (w, b)
    'input_mean': mean,                     # 数据的均值
    'input_std': std                        # 数据的标准差
}

save_path = 'lane_model_v1.pth'
torch.save(checkpoint, save_path)
print(f">>> 模型全套参数已保存至: {save_path}")