import torch
import torch.nn as nn
import torch.optim as optim
import geopandas as gpd
from shapely.geometry import LineString
import numpy as np

# 1. 定义一个简单的全连接网络
class TrajectoryPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义层：线性层 (Linear Layer)
        # 输入特征数 2 (x, y)，输出特征数 2 (next_x, next_y)
        # 这相当于公式: output = input * Weight + Bias
        self.layer = nn.Linear(in_features=3, out_features=3)
        
    def forward(self, x):
        # 前向传播：数据怎么流过网络
        return self.layer(x)

# 2. 实例化模型并移至 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TrajectoryPredictor().to(device)

print("--- 模型结构 ---")
print(model)
# 查看初始化的随机参数 (Weight 和 Bias)
print(f"初始权重: \n{model.layer.weight.data}")

# --- 模拟数据 ---
# 随机生成 100 个点作为输入 (Batch Size = 100)
inputs = torch.randn(100, 3).to(device)
# 生成真实的标签 (Ground Truth)
# 假设真实规律是：下个点 = 当前点 * 2 + 1 (我们希望模型学会这个规律)
targets = inputs * 2 + 1 
# --- 训练组件 ---
# 1. 损失函数: 使用 MSE (均方误差)
criterion = nn.MSELoss()

# 2. 优化器: 使用 SGD (随机梯度下降)，学习率 lr=0.1
# 这一行代替了昨天的 "w = w - lr * grad"
optimizer = optim.SGD(model.parameters(), lr=0.1)

print("\n--- 开始训练 ---")
# 训练 1000 轮 (Epochs)
for epoch in range(1001):
    # A. 梯度清零 (非常重要！否则梯度会累加)
    optimizer.zero_grad()
    
    # B. 前向传播 (Forward)
    outputs = model(inputs)
    
    # C. 计算损失 (Loss)
    loss = criterion(outputs, targets)
    
    # D. 反向传播 (Backward)
    loss.backward()
    
    # E. 更新参数 (Step)
    optimizer.step()
    
    # 每 200 轮打印一次进度
    if epoch % 200 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

print("\n--- 训练后验证 ---")
# 测试一个新数据 [1.0, 1.0]
# 真实答案应该是 [3.0, 3.0] (因为 1*2+1=3)
test_input = torch.tensor([[1.0, 1.0, 1.0]]).to(device)
prediction = model(test_input)
print(f"输入: [1.0, 1.0, 1.0]")
print(f"模型预测: {prediction.cpu().detach().numpy()}")
print(f"真实目标: [3.0, 3.0, 3.0]")

print("\n--- 地图数据处理实战 ---")

# 1. [模拟] 创建一个假的 SD 地图数据
# 在实际项目中，你会用 gpd.read_file('map.shp')
# 这里我们模拟 3 条道路，坐标是大的墨卡托坐标 (单位: 米)
road_data = {
    'road_id': [101, 102, 103],
    'geometry': [
        LineString([(400000, 3000000), (400100, 3000100)]), # 路段 A
        LineString([(400100, 3000100), (400200, 3000200)]), # 路段 B
        LineString([(400200, 3000200), (400300, 3000000)])  # 路段 C
    ]
}
gdf_map = gpd.GeoDataFrame(road_data)
print("原始地图数据:\n", gdf_map)

# 2. 提取坐标点并转为 Tensor
# 深度学习模型不喜欢巨大的数字 (如 400000)，会导致梯度爆炸。
# 必须进行 [归一化 Normalization]。

def map_to_tensor(gdf):
    all_points = []
    
    for geom in gdf.geometry:
        # shapely 的 .coords 属性可以拿到 [(x,y), (x,y)...]
        points = list(geom.coords)
        all_points.extend(points)
        
    # 转为 Numpy 数组
    np_points = np.array(all_points, dtype=np.float32)
    
    # --- 关键步骤: 归一化 ---
    # 找到地图的中心点，把所有坐标减去中心点，使其分布在 (0,0) 附近
    mean_xy = np.mean(np_points, axis=0)
    std_xy = np.std(np_points, axis=0)
    
    normalized_points = (np_points - mean_xy) / (std_xy + 1e-6) # 加 1e-6 防止除以0
    
    # 转为 PyTorch Tensor 并放入 GPU
    tensor_map = torch.from_numpy(normalized_points).to(device)
    
    return tensor_map, mean_xy, std_xy

# 执行转换
map_tensor, map_mean, map_std = map_to_tensor(gdf_map)

print(f"\n转换后的 Tensor (前5个点):\n{map_tensor[:5]}")
print(f"Shape: {map_tensor.shape}")
print(f"归一化参数 (Mean): {map_mean}")

# 3. 模拟应用：反归一化 (模型预测出来后，还原回真实地图坐标)
# 假设模型输出说车辆在 [0.1, -0.1] (归一化后的坐标)
model_output = torch.tensor([[0.1, -0.1]]).to(device)

# 还原
restored_pos = model_output.cpu().numpy() * map_std + map_mean
print(f"\n模型输出 (归一化): {model_output.cpu().numpy()}")
print(f"还原回地图真实坐标: {restored_pos}")