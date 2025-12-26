import torch
import torch.nn as nn
import numpy as np
import time
import random
import matplotlib.pyplot as plt

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
    
def generate_data():
    """生成模拟数据 (修正版: X=前向, Y=横向)"""
    
    # 1. 模拟实时 GPS 轨迹 (直行，向东/向 X 轴正方向)
    # 生成 20 个点，X 从 0 到 10
    t = np.linspace(0, 10, 20) 
    
    # 【修正点 1】注意这里多加了一层括号 () 组成元组
    # 第一列是 t (X轴), 第二列是 0 (Y轴)
    current_gps = np.column_stack((t, np.zeros_like(t))) 
    
    # 添加噪声 (针对 X 和 Y 都加一点)
    current_gps += np.random.normal(0, 0.3, current_gps.shape) 
    
    # 2. 候选道路
    
    # 候选 A: 直行道路 (y = 0)
    # 【修正点 2】同样加括号
    road_straight = np.column_stack((t, np.zeros_like(t)))
    
    # 候选 B: 左转道路 
    # (在 X 轴前进时，通常定义 Y 正方向为左，或者根据你的习惯定义)
    # 这里沿用你的逻辑：用曲线改变 Y 值
    y_left = 0.1 * t**2  # 假设左转是 Y 变大 (你可以改成负数，取决于坐标系定义)
    road_left = np.column_stack((t, y_left))
    
    # 候选 C: 右转道路
    y_right = -0.1 * t**2 # 假设右转是 Y 变小
    road_right = np.column_stack((t, y_right))
    
    candidate_roads = {
        "Candidate_Straight": road_straight,
        "Candidate_Left": road_left,
        "Candidate_Right": road_right
    }
    
    return current_gps, candidate_roads

# 2. 可视化模块 (新增)
# ==========================================
def plot_scenario(current_gps, candidate_roads):
    """
    绘制 GPS 轨迹和候选道路
    """
    plt.figure(figsize=(10, 6), dpi=100) # 设置画布大小和清晰度
    
    # --- 绘制候选道路 ---
    # 定义一些颜色，方便区分
    colors = ['green', 'blue', 'orange']
    line_styles = ['-', '-', '-'] # 都是实线

    # 遍历字典绘制每一条道路
    for i, (name, road_data) in enumerate(candidate_roads.items()):
        # road_data[:, 0] 是所有点的 X 坐标
        # road_data[:, 1] 是所有点的 Y 坐标
        plt.plot(road_data[:, 0], road_data[:, 1], 
                 label=name,          # 图例名称
                 color=colors[i % len(colors)], # 循环使用颜色
                 linestyle=line_styles[i % len(line_styles)],
                 linewidth=2.5,       # 线宽一点，表示是地图基准
                 alpha=0.7)           # 稍微透明一点

    # --- 绘制实时 GPS 轨迹 ---
    # 使用醒目的红色，带圆点标记(o)和虚线(--)连接，表示它是离散的观测值
    plt.plot(current_gps[:, 0], current_gps[:, 1], 
             'r--o',                  # 红色虚线带圆点
             label='Real-time GPS (Noisy)', 
             linewidth=1.5, 
             markersize=6,            # 标记点的大小
             zorder=10)               # zorder=10 保证它画在最上层

    # --- 图表设置 ---
    plt.title("Map Matching Scenario: Which road is the car on?", fontsize=14)
    plt.xlabel("X Position (Forward Direction)", fontsize=12)
    plt.ylabel("Y Position (Lateral Offset)", fontsize=12)
    
    plt.legend(fontsize=10) # 显示图例
    plt.grid(True, linestyle='--', alpha=0.6) # 显示网格
    
    # 【重要】设置坐标轴比例相等
    # 这样 1米在横向和纵向上看起来长度是一样的，真实反映弯道形状
    plt.axis('equal') 
    
    plt.tight_layout() # 自动调整布局防止重叠
    print("Plot generated. Check the popup window.")
    plt.show() # 显示图像




if __name__ == "__main__":

    # 1.模型加载
    model = SiameseLSTMMatcher()
    try:
        checkpoint = torch.load('day06_model_v1.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(">>> 模型权重加载成功！")
    except:
        print(">>> [警告] 未找到模型文件，使用随机初始化模型演示流程")

    model.eval()

    # 2.数据生成
    gps_data, roads_map = generate_data()
    print("GPS Shape:", gps_data.shape)
    print("First 3 points of GPS:\n", gps_data[:3])
    print(">>> 数据生成完成")

    # 3.模型预测
    results = {} # 用来存分数
    
    # 【修正1】不需要 enumerate，直接遍历 items() 获取 名字 和 数据
    for name, road_data in roads_map.items():
        
        # 【修正2】GPS 数据应该来自外部生成的 gps_data，而不是循环变量 name
        # 注意：gps_data 是我们之前 generate_data 返回的那个变量
        gps_tensor = torch.tensor(gps_data, dtype=torch.float32)
        
        # 【修正3】road_data 现在纯粹是 numpy 数组了，可以转换
        road_tensor = torch.tensor(road_data, dtype=torch.float32)

        # 2. 增加 Batch 维度 (Batch Size, Sequence Length, Features)
        # 变成 (1, 20, 2)
        gps_tensor = gps_tensor.unsqueeze(0)  
        road_tensor = road_tensor.unsqueeze(0)

        with torch.no_grad():
            score = model(gps_tensor, road_tensor)

        final_score = score.item()
        results[name] = final_score
        print(f"候选道路: {name:20} | 模型打分: {final_score:.4f}")
    
    # --- 找出分数最高的 ---
    # 使用 Python 的 max 函数找出 value 最大的 key
    best_road = max(results, key=results.get)
    print("-" * 30)
    print(f" 最终匹配结果: 车辆在 [{best_road}] 上！")

    # 画图
    plot_scenario(gps_data, roads_map)