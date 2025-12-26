import torch
import geopandas as gpd
from shapely.geometry import LineString, Point
import numpy as np

print(">>> 开始环境联动测试...")

# 1. [CPU任务] 模拟构建一个简单的 SD 地图 (两条车道线)
# 假设是墨卡托投影坐标 (米)
lane_1 = LineString([(0, 0), (100, 0)])   # 直行车道
lane_2 = LineString([(0, 4), (100, 4)])   # 左侧车道 (宽4米)
gdf_map = gpd.GeoDataFrame({'lane_id': [1, 2], 'geometry': [lane_1, lane_2]})
print(f"[GeoPandas] 地图构建成功，包含 {len(gdf_map)} 条车道。")

# 2. [CPU任务] 模拟车辆 GPS 位置 (带有一些噪声)
car_x, car_y = 50.0, 1.5  # 车辆在 x=50, y=1.5 的位置
car_point = Point(car_x, car_y)

# 3. [CPU任务] 利用几何库计算车辆到每条车道的物理距离
# (这是未来数据预处理的关键步骤)
distances = gdf_map.distance(car_point)
print(f"[Shapely] 车辆距离车道1: {distances[0]:.2f}米, 距离车道2: {distances[1]:.2f}米")

# 4. [GPU任务] 将数据转入 PyTorch 进行计算
# 假设我们要训练一个模型来判断车辆在哪条车道，这里模拟一次 Tensor 运算
try:
    # 将距离数据转为 Tensor 并送入 GPU
    dist_tensor = torch.tensor(distances.values, dtype=torch.float32).cuda()

    # 模拟一个简单的 Softmax 概率计算 (距离越近，概率越大)
    # 取负号是因为距离越小越好
    probs = torch.nn.functional.softmax(-dist_tensor, dim=0)

    print(f"[PyTorch] GPU 运算成功! (Device: {dist_tensor.device})")
    print(f"[模型输出] 车辆在车道1的概率: {probs[0]:.4f}, 在车道2的概率: {probs[1]:.4f}")

    if probs[0] > probs[1]:
        print(">>> 结论: 车辆当前位于 [车道 1]")
    else:
        print(">>> 结论: 车辆当前位于 [车道 2]")

except Exception as e:
    print(f"!!! GPU 运算失败: {e}")

print(">>> 测试结束。")
