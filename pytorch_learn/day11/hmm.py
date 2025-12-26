import numpy as np
import matplotlib.pyplot as plt
import math

# ==========================================
# 1. 场景构建：平行世界 (高架 vs 辅路)
# ==========================================
def generate_parallel_roads():
    # 长度 20 个点
    t = np.linspace(0, 100, 20)
    
    # Road A (高架): y = 10
    road_high = np.column_stack((t, np.full_like(t, 10)))
    
    # Road B (辅路): y = 0
    road_low = np.column_stack((t, np.zeros_like(t)))
    
    # GPS: 真实在 High 上，但慢慢飘到了 Low 上
    # 前 5 个点准，中间飘，后 5 个点完全在 Low 上
    gps_trace = road_high.copy()
    drift = np.linspace(0, -10, 20) # 逐渐向下偏离 10米
    gps_trace[:, 1] += drift
    
    # 候选路字典
    candidates = {"High": road_high, "Low": road_low}
    
    return gps_trace, candidates

# ==========================================
# 2. 概率计算引擎
# ==========================================
def get_emission_prob(gps_pt, road_pt):
    """
    观测概率：GPS 离路越近，概率越大
    """
    dist = np.linalg.norm(gps_pt - road_pt)
    # 标准差 sigma=5米
    prob = (1 / (np.sqrt(2 * np.pi) * 5)) * np.exp(-0.5 * (dist / 5)**2)
    return prob

def get_transition_prob(prev_road_name, curr_road_name):
    """
    转移概率：核心中的核心！
    定义地图的连通性。
    """
    # 情况 1: 同一条路 (High -> High) 或者 (Low -> Low)
    # 这是最正常的，概率很高
    if prev_road_name == curr_road_name:
        return 0.9 
    
    # 情况 2: 换路 (High -> Low)
    # 在这个简易地图里，高架和辅路是不通的！
    # 实际上可能通过出口连通，这里为了演示，设为极小概率
    else:
        return 1e-10 # 几乎不可能发生瞬移！

# ==========================================
# 3. 两种算法对比
# ==========================================

# --- A. 贪心算法 (Day 6-10 的逻辑) ---
def match_greedy(gps_trace, candidates):
    results = []
    for i, gps_pt in enumerate(gps_trace):
        best_road = None
        max_prob = -1
        
        # 每一帧只看当下谁分高，不管上一帧
        for name, road_data in candidates.items():
            road_pt = road_data[i] # 简化：假设索引对齐
            prob = get_emission_prob(gps_pt, road_pt)
            if prob > max_prob:
                max_prob = prob
                best_road = name
        results.append(best_road)
    return results

# --- B. 维特比算法 (HMM - Day 11 的逻辑) ---
def match_viterbi(gps_trace, candidates):
    road_names = list(candidates.keys())
    n_obs = len(gps_trace)
    n_states = len(road_names)
    
    # dp[t][state_idx] 存储 t 时刻到达 state 的最大累积概率
    # 为了防止下溢出 (probability 乘积越来越小)，通常取 log，变成加法
    # 这里为了直观演示，直接用乘法 (数据短没关系)
    V = np.zeros((n_obs, n_states))
    path = np.zeros((n_obs, n_states), dtype=int) # 记录路径指针
    
    # 1. 初始化 (t=0)
    for s_idx, name in enumerate(road_names):
        road_pt = candidates[name][0]
        # 初始只看观测概率
        V[0, s_idx] = get_emission_prob(gps_trace[0], road_pt)
        
    # 2. 递归 (Recursion)
    for t in range(1, n_obs):
        for curr_s, curr_name in enumerate(road_names):
            # 这里的 road_pt 依然简化假设索引对齐
            emit_p = get_emission_prob(gps_trace[t], candidates[curr_name][t])
            
            # 寻找从上一时刻的哪个状态跳过来概率最大
            max_trans_prob = -1
            best_prev_s = -1
            
            for prev_s, prev_name in enumerate(road_names):
                trans_p = get_transition_prob(prev_name, curr_name)
                # 核心公式：前一刻累积概率 * 转移概率 * 当前观测概率
                prob = V[t-1, prev_s] * trans_p * emit_p
                
                if prob > max_trans_prob:
                    max_trans_prob = prob
                    best_prev_s = prev_s
            
            V[t, curr_s] = max_trans_prob
            path[t, curr_s] = best_prev_s
            
    # 3. 回溯 (Backtracking) - 找出全局最优路径
    best_path_indices = []
    # 找到最后时刻概率最大的状态
    last_state = np.argmax(V[-1, :])
    best_path_indices.append(last_state)
    
    # 倒着往回找
    for t in range(n_obs-1, 0, -1):
        prev_state = path[t, best_path_indices[-1]]
        best_path_indices.append(prev_state)
        
    best_path_indices.reverse()
    return [road_names[i] for i in best_path_indices]

# ==========================================
# 4. 可视化
# ==========================================
def main():
    gps, candidates = generate_parallel_roads()
    
    # 运行两种匹配
    res_greedy = match_greedy(gps, candidates)
    res_viterbi = match_viterbi(gps, candidates)
    
    # 绘图
    plt.figure(figsize=(10, 6))
    
    # 画路
    plt.plot(candidates['High'][:,0], candidates['High'][:,1], 'g-', lw=3, label='High Road (y=10)')
    plt.plot(candidates['Low'][:,0], candidates['Low'][:,1], 'orange', lw=3, label='Low Road (y=0)')
    
    # 画 GPS
    plt.plot(gps[:,0], gps[:,1], 'r--o', alpha=0.5, label='Drifting GPS')
    
    # 画匹配结果 (用散点叠加在路上)
    # 贪心结果 (蓝色 x)
    for i, name in enumerate(res_greedy):
        y = 10 if name == 'High' else 0
        plt.scatter(gps[i,0], y+0.5, c='blue', marker='x', s=100, label='Greedy Match' if i==0 else "")
        
    # HMM 结果 (黑色圆圈)
    for i, name in enumerate(res_viterbi):
        y = 10 if name == 'High' else 0
        plt.scatter(gps[i,0], y-0.5, c='black', marker='o', s=80, facecolors='none', lw=2, label='HMM Viterbi' if i==0 else "")

    plt.title("HMM vs Greedy: Handling Impossible Jumps (The Overpass Problem)")
    plt.legend()
    plt.grid(True)
    plt.ylim(-5, 15)
    plt.show()

if __name__ == "__main__":
    main()