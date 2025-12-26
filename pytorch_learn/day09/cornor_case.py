import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_aabb_false_positive():
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # --- 1. 构造一条斜着的“坏”路 (Road) ---
    # 从 (100, 100) 到 (900, 900) 的对角线
    road_x = np.linspace(100, 900, 10)
    road_y = np.linspace(100, 900, 10)
    
    # 它的 AABB 盒子 (覆盖了大片区域)
    road_box = (100, 900, 100, 900) # min_x, max_x, min_y, max_y
    
    # --- 2. 构造一个刁钻的 GPS 点 ---
    # 它在 (200, 800)，也就是左上角
    # 在几何上，它离对角线非常远！(离路很远)
    # 但是，它确实在 Road 的 AABB 盒子范围内 (100~900 之间)
    gps_center = np.array([200, 800])
    buffer = 50
    search_box = (gps_center[0]-buffer, gps_center[0]+buffer, 
                  gps_center[1]-buffer, gps_center[1]+buffer)
    
    # --- 3. 绘图 ---
    
    # A. 画路 (实线)
    ax.plot(road_x, road_y, 'b-', linewidth=3, label='The Road (Geometry)')
    
    # B. 画路的 AABB 盒子 (蓝色虚线框)
    # 这个框很大，包含了很多空白
    rect_road = patches.Rectangle((100, 100), 800, 800, 
                                  linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.1, linestyle='--',
                                  label="Road's AABB (The Trap)")
    ax.add_patch(rect_road)
    
    # C. 画 GPS 点 (红点)
    ax.plot(gps_center[0], gps_center[1], 'ro', markersize=10, label='GPS Point')
    
    # D. 画 GPS 的搜索框 (红色虚线框)
    rect_search = patches.Rectangle((search_box[0], search_box[2]), 100, 100, 
                                    linewidth=2, edgecolor='red', facecolor='red', alpha=0.3,
                                    label="Search Buffer")
    ax.add_patch(rect_search)
    
    # E. 画出“虚报”的距离
    # 视觉辅助线，展示实际距离有多远
    ax.plot([200, 500], [800, 500], 'k:', label="Actual Distance (Far!)")

    ax.set_title("AABB False Positive: Boxes Overlap, but Geometry Doesn't", fontsize=14)
    ax.legend(loc='lower right')
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    plot_aabb_false_positive()