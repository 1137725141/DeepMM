import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ==========================================
# 1. 场景生成器 (Y-Split Scenario)
# ==========================================
def generate_y_split_scenario():
    """
    生成一个 Y 字形路口场景
    0-40米：重合直行
    40-100米：分岔
    """
    # 生成 100 个时间步
    t = np.linspace(0, 100, 100)
    
    # --- 1. 构造道路几何 ---
    # Road A (Left): 在 x>40 后向左弯 (y 增加)
    y_left = np.zeros_like(t)
    mask = t > 40
    y_left[mask] = 0.05 * (t[mask] - 40)**1.2 # 指数增长，模拟弯道
    road_left = np.column_stack((t, y_left))
    
    # Road B (Right): 在 x>40 后向右弯 (y 减少)
    y_right = np.zeros_like(t)
    y_right[mask] = -0.05 * (t[mask] - 40)**1.2
    road_right = np.column_stack((t, y_right))
    
    # --- 2. 构造 GPS 轨迹 ---
    # 假设车主要走的是 Left Road，但在分岔口有些犹豫和抖动
    gps_trace = road_left.copy()
    
    # 添加较大的随机噪声 (模拟定位飘移)
    np.random.seed(42) # 固定随机种子，保证每次运行效果一致
    noise_level = 1.5 
    gps_trace[:, 1] += np.random.normal(0, noise_level, t.shape)
    
    candidates = {
        "Road A (Left)": road_left,
        "Road B (Right)": road_right
    }
    
    return gps_trace, candidates

# ==========================================
# 2. 模拟模型与平滑引擎
# ==========================================
class SmoothMatchingEngine:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.score_memory = {} # 记忆库
        
    def mock_model_predict(self, gps_point, road_point):
        """
        这里模拟一个 '神经质' 的 AI 模型。
        它基于距离打分，但是加了很多随机抖动，模拟模型的不稳定性。
        """
        # 计算欧氏距离
        dist = np.linalg.norm(gps_point - road_point)
        
        # 基础分：距离越近分越高 (Exp decay)
        base_score = np.exp(-0.2 * dist)
        
        # 添加 '模型不确定性' 噪声 (Model Uncertainty)
        # 这模拟了 LSTM 在模糊地带的跳变
        model_jitter = np.random.normal(0, 0.1)
        final_score = base_score + model_jitter
        
        # 限制在 0~1 之间
        return np.clip(final_score, 0.0, 1.0)

    def step(self, current_gps, current_roads):
        """
        处理一帧数据：先推理，再平滑
        """
        raw_scores = {}
        smoothed_scores = {}
        
        for name, road_pt in current_roads.items():
            # 1. 模拟模型推理 (Raw)
            raw_s = self.mock_model_predict(current_gps, road_pt)
            raw_scores[name] = raw_s
            
            # 2. 获取上一时刻的记忆 (Memory)
            last_s = self.score_memory.get(name, 0.5) # 默认从 0.5 开始
            
            # 3. 惯性平滑公式 (The Momentum)
            # alpha 越小，惯性越大，越平滑
            smooth_s = self.alpha * raw_s + (1 - self.alpha) * last_s
            
            # 4. 更新记忆
            self.score_memory[name] = smooth_s
            smoothed_scores[name] = smooth_s
            
        return raw_scores, smoothed_scores

# ==========================================
# 3. 主程序与可视化 (核心优化部分)
# ==========================================
def main():
    # 1. 准备数据
    gps_full, roads = generate_y_split_scenario()
    engine = SmoothMatchingEngine(alpha=0.15) # 设置较强的平滑系数
    
    history_raw = {name: [] for name in roads}
    history_smooth = {name: [] for name in roads}
    time_steps = range(len(gps_full))
    
    # 2. 模拟实时运行
    print(">>> 引擎启动，处理中...")
    for t in time_steps:
        # 取当前时刻的点
        curr_gps = gps_full[t]
        # 取当前时刻道路上的对应点 (简化处理，假设已匹配到最近点)
        curr_roads = {name: rd[t] for name, rd in roads.items()}
        
        raw, smooth = engine.step(curr_gps, curr_roads)
        
        for name in roads:
            history_raw[name].append(raw[name])
            history_smooth[name].append(smooth[name])

    # 3. 高级可视化布局 (3个子图)
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1]) 
    
    # --- 图 1: 空间地图 (上部分，占据整行) ---
    ax_map = fig.add_subplot(gs[0, :])
    ax_map.set_title("1. Spatial View: GPS Trace vs. Roads (The Reality)", fontsize=14, fontweight='bold')
    
    # 画路
    colors = {'Road A (Left)': 'green', 'Road B (Right)': 'orange'}
    for name, rd in roads.items():
        ax_map.plot(rd[:, 0], rd[:, 1], color=colors[name], linewidth=3, alpha=0.6, label=name)
        
    # 画 GPS
    ax_map.plot(gps_full[:, 0], gps_full[:, 1], 'r.-', linewidth=0.5, markersize=4, alpha=0.8, label="Noisy GPS Trace")
    
    # 标注分岔区域
    ax_map.axvline(x=40, color='gray', linestyle='--', alpha=0.5)
    ax_map.text(42, 5, "Split Point", color='gray')
    
    ax_map.legend()
    ax_map.grid(True, linestyle=':')
    ax_map.set_xlabel("X (Forward Distance)")
    ax_map.set_ylabel("Y (Lateral Offset)")
    ax_map.axis('equal') # 关键！保持比例

    # --- 图 2: 原始分数 (左下) ---
    ax_raw = fig.add_subplot(gs[1, 0])
    ax_raw.set_title("2. Raw Model Output (The 'Flickering')", fontsize=12)
    for name in roads:
        ax_raw.plot(time_steps, history_raw[name], color=colors[name], alpha=0.4, linewidth=1.5, label=f"{name} Raw")
    
    ax_raw.set_ylim(-0.1, 1.1)
    ax_raw.set_ylabel("Similarity Score")
    ax_raw.set_xlabel("Time Step")
    ax_raw.grid(True)
    ax_raw.legend(fontsize=8)

    # --- 图 3: 平滑分数 (右下) ---
    ax_smooth = fig.add_subplot(gs[1, 1])
    ax_smooth.set_title("3. Smoothed Output (The 'Decision')", fontsize=12, fontweight='bold')
    for name in roads:
        # 画粗线，表示这是最终决策
        ax_smooth.plot(time_steps, history_smooth[name], color=colors[name], linewidth=3, label=f"{name} Smoothed")
        
    # 画 0.5 决策线
    ax_smooth.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    ax_smooth.set_ylim(-0.1, 1.1)
    ax_smooth.set_xlabel("Time Step")
    ax_smooth.grid(True)
    ax_smooth.legend(fontsize=8)

    plt.tight_layout()
    print(">>> 可视化生成完毕！")
    plt.show()

if __name__ == "__main__":
    main()