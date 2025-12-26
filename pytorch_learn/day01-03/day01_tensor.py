import torch

# --- 场景模拟 ---
# 假设我们有 一个 Batch (批次) 的数据，包含 4 辆车。
# 每辆车有 3 个特征：[经度 x, 纬度 y, 车速 v]
# 数据来自于 CPU (比如刚才用 Pandas 读取的 csv)
data_cpu = [
    [10.0, 20.0, 30.0], # Car 1
    [11.0, 21.0, 35.0], # Car 2
    [10.5, 20.5, 0.0],  # Car 3 (静止)
    [12.0, 22.0, 60.0]  # Car 4
]

# 1. 将列表转为 Tensor
x = torch.tensor(data_cpu)

print(f"数据类型: {x.dtype}") # 默认通常是 float32
print(f"数据形状: {x.shape}") # 输出应该是 torch.Size([4, 3])

# 2. 数据搬运
# 检查是否有 GPU，没有就用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 将 Tensor 移动到显存中
x_gpu = x.to(device)

print(f"当前设备: {x_gpu.device}")

# 注意：存储在 GPU 上的 Tensor 不能直接转 numpy，必须先转回 CPU
# print(x_gpu.numpy()) # 这行会报错
print(x_gpu.cpu().numpy()) # 正确做法

# 取出所有车辆的位置 (所有行，前两列)
# ":" 表示取所有，"0:2" 表示取索引0到1
car_positions = x_gpu[:, 0:2]  # Shape: [4, 2]

# 定义路口坐标
target_intersection = torch.tensor([11.0, 21.0]).to(device) # Shape: [2]

# --- 计算欧式距离 ---
# 1. 计算差值 (这里发生了广播：[2] 自动扩展成了 [4, 2] 去进行减法)
diff = car_positions - target_intersection 

# 2. 平方
sq_diff = diff ** 2

# 3. 求和 (注意：要在 dim=1 上求和，即把 x^2 和 y^2 加起来，保持 batch 维度不变)
dist_sq = torch.sum(sq_diff, dim=1) 

# 4. 开根号
distances = torch.sqrt(dist_sq)

print("--- 距离计算结果 ---")
print(f"位置差值:\n{diff}")
print(f"每辆车到路口的距离: {distances}")

# 创建一个需要求导的参数 (假设这是我们模型预测的初始位置)
# requires_grad=True 告诉 PyTorch：请盯着这个变量的一举一动
loc_param = torch.tensor([10.0, 20.0], device=device, requires_grad=True)
target = torch.tensor([11.0, 21.0], device=device)

# 模拟 Loss 计算 (均方误差)
loss = torch.sum((loc_param - target) ** 2)

print(f"当前 Loss: {loss.item()}")

# 反向传播 (魔法时刻)
loss.backward()

# 查看梯度
# 梯度告诉我们：loc_param 里的数据应该怎么变，才能让 Loss 变小
print(f"位置参数的梯度 (Gradient): {loc_param.grad}")

# 也就是：2 * (10 - 11) = -2, 2 * (20 - 21) = -2