import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import torch.optim as optim

# 1. 定义我们自己的数据集类
class LaneDataset(Dataset):
    def __init__(self, num_samples=1500):
        # --- 在这里模拟读取硬盘上的数据 ---
        # 实际项目中，这里通常是: self.data = pandas.read_csv('gps_logs.csv')
        
        # 模拟生成 1000 个样本
        # 假设左车道(Lane 0)的 x 在 0 附近，右车道(Lane 1)的 x 在 10 附近
        self.num_samples = num_samples
        
        # 生成 GPS 坐标 (Inputs)
        # 500辆车在左边，500辆车在右边
        left_lane_x = np.random.normal(0, 1, size=(num_samples//3, 1))
        mid_lane_x = np.random.normal(10, 1, size=(num_samples//3, 1))
        right_lane_x = np.random.normal(20, 1, size=(num_samples//3, 1))
        
        # y 坐标随机分布 (路上前后都有车)
        y_coords = np.random.uniform(0, 100, size=(num_samples, 1))
        
        # 合并 x 坐标
        x_coords = np.vstack((left_lane_x, mid_lane_x, right_lane_x))
        
        # 最终输入数据: Shape [1000, 2] -> (x, y)
        self.inputs = np.hstack((x_coords, y_coords)).astype(np.float32)

        # ============ 新增：归一化 (Normalization) ============
        # 这一步是提升准确率的关键！
        # 计算均值和标准差
        mean = self.inputs.mean(axis=0)
        std = self.inputs.std(axis=0)
        
        # 执行 Z-Score 归一化： (X - Mean) / Std
        # 加上 1e-6 是为了防止除以 0
        self.inputs = (self.inputs - mean) / (std + 1e-6)
        
        print(f"数据已归一化。Mean: {mean}, Std: {std}")
        # ====================================================
        
        # 生成标签 (Targets)
        # 前500个是 0 (左车道)，后500个是 1 (右车道)
        # 注意：分类任务的标签通常用 long (整数) 类型
        self.labels = np.array([0] * (num_samples//3) + [1] * (num_samples//3) + [2] * (num_samples//3), dtype=np.int64)

    def __len__(self):
        # 告诉 PyTorch 数据集有多长
        return self.num_samples

    def __getitem__(self, idx):
        # --- 核心：拿取第 idx 条数据 ---
        
        # 1. 拿到原始数据
        x_data = self.inputs[idx]
        y_label = self.labels[idx]
        
        # 2. 这里的 x_data 是 numpy，为了给模型训练，必须转为 Tensor
        # 注意：在这里转 Tensor 比在 __init__ 里转更省内存
        sample = torch.from_numpy(x_data)
        label = torch.tensor(y_label) # 标量转 Tensor
        
        return sample, label

# 实例化数据集
my_dataset = LaneDataset(num_samples=1500)

# 测试一下能不能拿到数据
print(f"数据总量: {len(my_dataset)}")
first_data, first_label = my_dataset[0]
print(f"第0条数据: 输入={first_data}, 标签={first_label}")

# 2. 定义 DataLoader
# batch_size=32: 每次训练喂给模型 32 辆车的数据
# shuffle=True: 每次 Epoch 开始时，把数据顺序打乱 (洗牌)
train_loader = DataLoader(dataset=my_dataset, batch_size=32, shuffle=True)

# 测试 DataLoader
# iter() 和 next() 是 Python 获取迭代器第一个内容的写法
first_batch_inputs, first_batch_labels = next(iter(train_loader))

print(f"\n--- DataLoader 测试 ---")
print(f"一个 Batch 的输入形状: {first_batch_inputs.shape}") # 应该是 [32, 2]
print(f"一个 Batch 的标签形状: {first_batch_labels.shape}") # 应该是 [32]



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 定义分类模型
class LaneClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # 输入: (x, y) 2个特征
        # 输出: 2个类别 (Lane 0 的得分, Lane 1 的得分)
        self.layer = nn.Linear(2, 3) 
        
    def forward(self, x):
        return self.layer(x)

model = LaneClassifier().to(device)

# 2. 损失函数与优化器
# CrossEntropyLoss 内部会自动计算 Softmax，所以模型输出不需要加 Softmax
criterion = nn.CrossEntropyLoss() 
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("\n--- 开始训练 (Batch Training) ---")

# 3. 训练循环
for epoch in range(5): 
    total_loss = 0
    correct_count = 0  # 记录猜对的数量
    total_count = 0    # 记录总数量

    for i, (batch_inputs, batch_labels) in enumerate(train_loader):
        batch_inputs = batch_inputs.to(device)
        batch_labels = batch_labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

        # --- 新增：计算准确率 ---
        # 1. 获取得分最高的那个类别的索引 (0, 1, 或 2)
        # argmax(dim=1) 表示在第1维度(列)上找最大值的索引
        _, predicted = torch.max(outputs.data, 1)
        
        # 2. 统计猜对的个数
        total_count += batch_labels.size(0)
        correct_count += (predicted == batch_labels).sum().item()
        
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct_count / total_count
    
    print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%")

print("训练完成！")