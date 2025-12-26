import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# 1. 模型定义 (保持与 Day 6 一致)
# ===============================
class SiameseLSTMMatcher(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=64, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward_one(self, x):
        out, _ = self.lstm(x)
        return out[:, -1, :] 

    def forward(self, x1, x2):
        feat1 = self.forward_one(x1)
        feat2 = self.forward_one(x2)
        diff = torch.abs(feat1 - feat2)
        score = self.classifier(diff)
        return score

# ===============================
# 2. 长距离数据生成 (Long Trajectory)
# ===============================
def generate_long_scenario():
    """
    生成一个长达 100 个点的行程
    场景：0-40直行，40-60向左变道/弯道，60-100直行
    """
    total_steps = 100
    t = np.linspace(0, 50, total_steps) # 时间/纵向位移
    
    # --- 构造真实的 GPS 轨迹 (车走了个 S 型) ---
    true_y = np.zeros_like(t)
    # 前 40 个点是直的，y=0
    # 中间 20 个点 (idx 40-60) 发生偏移
    # 使用 sigmoid 函数模拟平滑换道/弯道
    sigmoid_part = 1 / (1 + np.exp(-(t - 25))) # 简单的 S 型过渡
    true_y = sigmoid_part * 5 # 最终偏移 5 米
    
    gps_long = np.column_stack((t, true_y))
    # 加噪声
    gps_long += np.random.normal(0, 0.15, gps_long.shape)
    
    # --- 构造候选道路 ---
    
    # 候选路 A (Correct): 完美匹配这一条 S 型路径
    road_correct = np.column_stack((t, true_y)) # 理想状态无噪声
    
    # 候选路 B (Wrong): 一直是直的 (即在中间段会偏离)
    # 它只在头尾看起来像，中间完全对不上
    road_wrong = np.column_stack((t, np.zeros_like(t))) 
    
    return gps_long, road_correct, road_wrong

# ===============================
# 3. 滑动窗口推理引擎
# ===============================
def run_sliding_window_inference(model, gps_long, road_correct, road_wrong, window_size=20):
    
    scores_correct = []
    scores_wrong = []
    time_steps = []
    
    total_len = len(gps_long)
    
    print(f">>> 开始滑动窗口推理 (总长度 {total_len}, 窗口 {window_size})...")
    
    # 从第 20 个点开始，每次往后滑 1 格
    for i in range(window_size, total_len):
        # 1. 切片 (Slicing) - 取过去 20 个点
        # slice_gps: [20, 2]
        slice_gps = gps_long[i-window_size : i]
        slice_road_correct = road_correct[i-window_size : i]
        slice_road_wrong = road_wrong[i-window_size : i]
        
        # 2. 转 Tensor 并增加 Batch 维度 -> [1, 20, 2]
        t_gps = torch.tensor(slice_gps, dtype=torch.float32).unsqueeze(0)
        t_road_c = torch.tensor(slice_road_correct, dtype=torch.float32).unsqueeze(0)
        t_road_w = torch.tensor(slice_road_wrong, dtype=torch.float32).unsqueeze(0)
        
        # 3. 推理
        with torch.no_grad():
            s_c = model(t_gps, t_road_c).item()
            s_w = model(t_gps, t_road_w).item()
            
        scores_correct.append(s_c)
        scores_wrong.append(s_w)
        time_steps.append(i)
        
    return time_steps, scores_correct, scores_wrong

# ===============================
# 4. 动态分析可视化
# ===============================
def plot_results(gps, road_c, road_w, time_steps, score_c, score_w):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # --- 子图 1: 地图视角 ---
    ax1.set_title("Map View: Trajectory vs Candidates")
    ax1.plot(road_c[:, 0], road_c[:, 1], 'g-', label="Road A (Correct - S shape)", linewidth=3, alpha=0.5)
    ax1.plot(road_w[:, 0], road_w[:, 1], 'orange', label="Road B (Wrong - Straight)", linewidth=3, alpha=0.5)
    ax1.plot(gps[:, 0], gps[:, 1], 'r.', label="GPS Trace", markersize=2)
    ax1.legend()
    ax1.grid(True)
    ax1.set_xlabel("X (Forward)")
    ax1.set_ylabel("Y (Lateral)")
    
    # --- 子图 2: 随着时间变化的置信度 ---
    ax2.set_title("Real-time Confidence Score (Sliding Window)")
    ax2.plot(time_steps, score_c, 'g-', label="Score for Road A", linewidth=2)
    ax2.plot(time_steps, score_w, 'orange', label="Score for Road B", linewidth=2)
    
    # 画一条 0.5 的阈值线
    ax2.axhline(y=0.5, color='gray', linestyle='--')
    
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Matching Probability")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# ===============================
# 主程序
# ===============================
if __name__ == "__main__":
    # 1. 加载模型
    model = SiameseLSTMMatcher()
    try:
        # 请确保这个路径是你昨天重新训练过的那个模型
        model.load_state_dict(torch.load('day06_model_v1.pth')['model_state_dict'])
        print(">>> 模型加载成功！")
    except:
        print(">>> [警告] 未找到模型，使用随机参数演示（结果会是乱的）")
    
    model.eval()
    
    # 2. 生成数据
    gps, road_correct, road_wrong = generate_long_scenario()
    
    # 3. 运行滑动窗口
    steps, sc, sw = run_sliding_window_inference(model, gps, road_correct, road_wrong)
    
    # 4. 画图分析
    plot_results(gps, road_correct, road_wrong, steps, sc, sw)