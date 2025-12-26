import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import math

# ==========================================
# 1. 模拟真实的 GPX 文件内容 (XML格式)
# ==========================================
gpx_data_content = """
<gpx version="1.1">
  <trk>
    <name>Real World Test Drive</name>
    <trkseg>
      <trkpt lat="39.98470" lon="116.31841"></trkpt>
      <trkpt lat="39.98482" lon="116.31838"></trkpt>
      <trkpt lat="39.98501" lon="116.31835"></trkpt>
      <trkpt lat="39.98555" lon="116.31810"></trkpt>
      <trkpt lat="39.98610" lon="116.31790"></trkpt>
      <trkpt lat="39.98688" lon="116.31755"></trkpt>
      <trkpt lat="39.98750" lon="116.31720"></trkpt>
      <trkpt lat="39.98820" lon="116.31690"></trkpt>
      <trkpt lat="39.98850" lon="116.31750"></trkpt> 
      <trkpt lat="39.98860" lon="116.31880"></trkpt>
      <trkpt lat="39.98865" lon="116.31990"></trkpt>
      <trkpt lat="39.98870" lon="116.32100"></trkpt>
      <trkpt lat="39.98872" lon="116.32250"></trkpt>
      <trkpt lat="39.98875" lon="116.32350"></trkpt>
    </trkseg>
  </trk>
</gpx>
"""

# ==========================================
# 2. 解析器 (Parsing)
# ==========================================
def parse_gpx(xml_string):
    root = ET.fromstring(xml_string)
    points = []
    
    # 查找所有的 trkpt (Track Point) 标签
    # namespace 处理有时候很麻烦，这里假设是最简结构
    for trkpt in root.findall('.//trkpt'):
        lat = float(trkpt.get('lat'))
        lon = float(trkpt.get('lon'))
        points.append((lat, lon))
        
    return np.array(points)

# ==========================================
# 3. 坐标转换器 (Lat/Lon -> Meters)
# ==========================================
def latlon_to_meters(gps_points):
    """
    将经纬度列表转换为以第一个点为原点的局部平面坐标 (ENU)
    """
    origin_lat = gps_points[0][0]
    origin_lon = gps_points[0][1]
    
    # 将原点纬度转换为弧度，用于计算 X 轴的压缩比
    lat_rad = math.radians(origin_lat)
    
    # 地球半径常量 (简化版)
    DEG_TO_M_LAT = 111000 # 纬度 1度 ≈ 111km
    DEG_TO_M_LON = 111000 * math.cos(lat_rad) # 经度 1度 ≈ 111km * cos(lat)
    
    xy_points = []
    
    for lat, lon in gps_points:
        # 1. 减去原点 (Delta)
        d_lat = lat - origin_lat
        d_lon = lon - origin_lon
        
        # 2. 转换为米
        pos_y = d_lat * DEG_TO_M_LAT
        pos_x = d_lon * DEG_TO_M_LON
        
        xy_points.append([pos_x, pos_y])
        
    return np.array(xy_points)

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    # 1. 解析
    print(">>> 解析 GPX 数据...")
    raw_gps = parse_gpx(gpx_data_content)
    print(f"原始数据 (Lat/Lon) 前3个点:\n{raw_gps[:3]}")
    
    # 2. 投影转换
    print("\n>>> 执行坐标投影 (Lat/Lon -> Meters)...")
    xy_gps = latlon_to_meters(raw_gps)
    print(f"投影数据 (Meters) 前3个点:\n{xy_gps[:3]}")
    
    # 3. 可视化
    plt.figure(figsize=(10, 5))
    
    # 左图：原始经纬度 (看起来虽然也像，但在数学上是变形的)
    plt.subplot(1, 2, 1)
    plt.plot(raw_gps[:, 1], raw_gps[:, 0], 'r-o')
    plt.title("Raw Lat/Lon (Degrees)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    # 强制不自动调整比例，看看会发生什么（通常看起来会很扁或很长）
    
    # 右图：米制坐标 (这是真实的物理形状)
    plt.subplot(1, 2, 2)
    plt.plot(xy_gps[:, 0], xy_gps[:, 1], 'b-o')
    plt.title("Projected Local Plane (Meters)")
    plt.xlabel("X (East-West) meters")
    plt.ylabel("Y (North-South) meters")
    plt.grid(True)
    plt.axis('equal') # 关键！保持 1米:1米 的比例
    
    plt.tight_layout()
    plt.show()