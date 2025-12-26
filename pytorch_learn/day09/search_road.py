import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ==========================================
# 1. 模拟地图数据库 (Map Database)
# ==========================================
class RoadSegment:
    def __init__(self, id, points):
        self.id = id
        self.points = points # Shape: [N, 2]
        
        # --- 核心：预计算包围盒 (Min/Max X, Min/Max Y) ---
        # 这一步通常在地图加载时做一次，之后查询就很快
        self.min_x = np.min(points[:, 0])
        self.max_x = np.max(points[:, 0])
        self.min_y = np.min(points[:, 1])
        self.max_y = np.max(points[:, 1])

def generate_city_map(num_roads=50):
    roads = []
    for i in range(num_roads):
        # 随机生成一些散落在 (0,0) 到 (1000,1000) 区域的路
        start_x = np.random.uniform(0, 1000)
        start_y = np.random.uniform(0, 1000)
        
        # 让路稍微长一点，随机延伸
        length = np.random.uniform(50, 200)
        angle = np.random.uniform(0, 2 * np.pi)
        
        end_x = start_x + length * np.cos(angle)
        end_y = start_y + length * np.sin(angle)
        
        # 插值生成几个点，模拟一条路
        t = np.linspace(0, 1, 10)
        x = start_x + (end_x - start_x) * t
        y = start_y + (end_y - start_y) * t
        
        road_points = np.column_stack((x, y))
        roads.append(RoadSegment(f"Road_{i}", road_points))
    return roads

# ==========================================
# 2. 空间检索引擎 (Spatial Indexer)
# ==========================================
def find_candidates(gps_trace, all_roads, buffer_radius=50):
    """
    输入: 
      gps_trace: 当前的 20个 GPS 点
      all_roads: 整个城市的道路列表
      buffer_radius: 搜索半径 (比如只看周围 50米)
    输出:
      candidates: 落入搜索范围的道路列表
      search_box: 搜索框坐标 (用于画图)
    """
    candidates = []
    
    # 1. 计算 GPS 轨迹的包围盒
    g_min_x = np.min(gps_trace[:, 0])
    g_max_x = np.max(gps_trace[:, 0])
    g_min_y = np.min(gps_trace[:, 1])
    g_max_y = np.max(gps_trace[:, 1])
    
    # 2. 向外扩充半径 (Buffer) -> 形成搜索框 (Search Box)
    search_min_x = g_min_x - buffer_radius
    search_max_x = g_max_x + buffer_radius
    search_min_y = g_min_y - buffer_radius
    search_max_y = g_max_y + buffer_radius
    
    search_box = (search_min_x, search_max_x, search_min_y, search_max_y)
    
    # 3. 快速筛选 (Broad Phase)
    # 遍历所有路，看它们的盒子是否和搜索框相交
    for road in all_roads:
        # AABB 重叠测试算法:
        # 如果 A的左边 > B的右边，或者 A的右边 < B的左边... 也就是完全没挨着
        if (road.min_x > search_max_x or road.max_x < search_min_x or
            road.min_y > search_max_y or road.max_y < search_min_y):
            continue # 绝对不相交，跳过
        
        # 如果盒子相交，暂时认为是候选 (虽然可能只是擦肩而过，但足以进入下一轮 LSTM)
        candidates.append(road)
        
    return candidates, search_box

# ==========================================
# 3. 主程序与可视化
# ==========================================
if __name__ == "__main__":
    # 1. 造城
    city_roads = generate_city_map(num_roads=50)
    
    # 2. 造一个 GPS 轨迹 (假装我们在地图中间)
    gps_center = np.array([[500, 500]])
    gps_trace = gps_center + np.random.normal(0, 10, (20, 2)) # 20个点的一团
    
    # 3. 执行检索 (这是今天的重点!)
    found_roads, s_box = find_candidates(gps_trace, city_roads, buffer_radius=60)
    
    print(f"地图总道路数: {len(city_roads)}")
    print(f"筛选出的候选道路数: {len(found_roads)}")
    print("候选ID:", [r.id for r in found_roads])
    
    # 4. 可视化
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 画所有路 (灰色，表示未选中)
    for road in city_roads:
        ax.plot(road.points[:, 0], road.points[:, 1], c='lightgray', lw=1, zorder=1)
        
    # 画候选路 (蓝色，加粗)
    for road in found_roads:
        ax.plot(road.points[:, 0], road.points[:, 1], c='blue', lw=2, zorder=2, label='Candidates')
        
    # 画 GPS (红色)
    ax.plot(gps_trace[:, 0], gps_trace[:, 1], 'r.', zorder=3, label='GPS')
    
    # 画搜索框 (红色虚线矩形)
    # s_box = (min_x, max_x, min_y, max_y)
    rect_width = s_box[1] - s_box[0]
    rect_height = s_box[3] - s_box[2]
    rect = patches.Rectangle((s_box[0], s_box[2]), rect_width, rect_height, 
                             linewidth=2, edgecolor='red', facecolor='none', linestyle='--', label='Search Buffer')
    ax.add_patch(rect)
    
    # 避免图例重复
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    
    ax.set_title("Spatial Indexing: Finding Candidates Efficiently")
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    plt.show()