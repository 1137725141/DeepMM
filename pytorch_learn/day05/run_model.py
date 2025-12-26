import torch
import torch.nn as nn
import numpy as np
from collections import deque  # <--- 神器在这里
import time
import random

# ==========================================
# 1. 模型定义 (必须和训练时一致)
# ==========================================
class TrajectoryClassifier(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        last_time_step = out[:, -1, :]
        x = self.fc(last_time_step)
        return x

# ==========================================
# 2. 核心类：滑动窗口推断器
# ==========================================
class RealTimePredictor:
    def __init__(self, model_path, mean, std):
        # A. 加载模型
        self.model = TrajectoryClassifier()
        # 这里为了演示，如果找不到文件就不加载，实际使用请确保文件存在
        try:
            checkpoint = torch.load(model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(">>> 模型权重加载成功！")
        except:
            print(">>> [警告] 未找到模型文件，使用随机初始化模型演示流程")
            
        self.model.eval() # 开启推理模式
        
        # B. 记录归一化参数 (必须用训练集的!)
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        
        # C. 初始化滑动窗口 (关键点!)
        # maxlen=20: 保证列表永远最多只有 20 个元素
        # 一旦第 21 个进来，第 1 个自动被踢走
        self.buffer = deque(maxlen=20)
        
    def process_new_point(self, x, y):
        # 1. 将新点加入窗口
        self.buffer.append([x, y])
        
        # 2. 检查数据够不够 20 个
        if len(self.buffer) < 20:
            print(f"\r[积攒数据] 当前缓冲池大小: {len(self.buffer)}/20", end="", flush=True)
            return None # 数据不够，不预测
            
        # 3. 数据够了，开始推理！
        # 3.1 转为 Numpy [20, 2]
        seq_data = np.array(self.buffer, dtype=np.float32)
        
        # 3.2 归一化 (关键!)
        seq_norm = (seq_data - self.mean) / (self.std + 1e-6)
        
        # 3.3 转 Tensor 并增加 Batch 维度
        # [20, 2] -> [1, 20, 2]
        input_tensor = torch.from_numpy(seq_norm).unsqueeze(0)
        
        # 3.4 模型预测
        with torch.no_grad():
            scores = self.model(input_tensor)
            probs = torch.nn.functional.softmax(scores, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            
        return pred_idx, probs.numpy()[0]

# ==========================================
# 3. 主程序：模拟传感器数据流
# ==========================================
if __name__ == "__main__":
    # 假设的训练集统计数据 (你需要填入你训练时的真实值)
    # 这里随便写几个防报错
    saved_mean = [5.0, 0.0]  
    saved_std = [2.0, 1.0]
    
    # 实例化推断器
    predictor = RealTimePredictor('turn_model_v1.pth', saved_mean, saved_std)
    
    print("\n>>> 开始接收传感器数据流...")
    
    # 模拟一个左转弯的过程 (X增加, Y平方增加)
    # 生成 100 个连续的时间点
    for i in range(100):
        # --- 模拟传感器产生数据 ---
        t = i * 0.5
        sensor_x = t
        sensor_y = 0.05 * (t ** 2) + random.uniform(-0.1, 0.1) # 加点噪声
        
        # --- 喂给推断器 ---
        result = predictor.process_new_point(sensor_x, sensor_y)
        
        # --- 如果有结果就打印 ---
        if result is not None:
            pred_class, conf = result
            labels = ["直行", "左转", "右转"]
            
            # 打印结果 (用 \r 覆盖上一行，做出动态效果)
            print(f"\r>>> 时刻 {i}: 坐标({sensor_x:.1f}, {sensor_y:.1f}) | "
                  f"预测: [{labels[pred_class]}] | "
                  f"置信度: 左{conf[1]:.2f} / 直{conf[0]:.2f} / 右{conf[2]:.2f}   ", end="")
        
        # 模拟 0.1 秒的传感器间隔
        time.sleep(0.1)

    print("\n>>> 模拟结束")