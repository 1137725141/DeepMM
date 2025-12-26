import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

print("\n" + "="*30)
print("   模拟车辆上路 (推理阶段)   ")
print("="*30)

# --- 1. 初始化一个空模型 ---
# 必须和训练时的结构完全一致 (Linear 2->3)
loaded_model = nn.Linear(2, 3) 

# --- 2. 加载文件 ---
if os.path.exists('lane_model_v1.pth'):
    # 加载字典
    checkpoint = torch.load('lane_model_v1.pth')
    
    # A. 恢复模型权重
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    
    # B. 恢复预处理参数
    saved_mean = checkpoint['input_mean']
    saved_std = checkpoint['input_std']
    
    print(">>> 模型加载成功！预处理参数也已就位。")
else:
    print("!!! 找不到模型文件")
    exit()

# --- 3. 切换到推理模式 (非常重要) ---
# 这会通知 PyTorch：“我不学习了，我现在只工作”。
# 对于 Linear 层没区别，但对于 Dropout/BatchNorm 层至关重要。
loaded_model.eval() 

# --- 4. 模拟新数据到来 ---
# 假设现在来了一辆车，GPS 显示 (x=10.5, y=66.6)
# 我们知道这应该是 Lane 1 (中间车道)
new_car_raw = np.array([[0.5, 66.6]], dtype=np.float32)

print(f"\n[新车信号] 原始坐标: {new_car_raw}")

# --- 5. 预处理 (使用保存的参数) ---
# 千万不能用新数据的 mean/std，必须用训练时的！
new_car_norm = (new_car_raw - saved_mean) / (saved_std + 1e-6)
new_car_tensor = torch.from_numpy(new_car_norm)

print(f"[预处理] 归一化后坐标: {new_car_tensor.numpy()}")

# --- 6. 执行预测 ---
# torch.no_grad(): 告诉 PyTorch 别计算梯度了，省内存，提速
with torch.no_grad():
    # 前向传播
    scores = loaded_model(new_car_tensor)
    
    # 计算概率 (Softmax)
    probabilities = torch.nn.functional.softmax(scores, dim=1)
    
    # 获取最大概率的类别
    predicted_lane = torch.argmax(probabilities, dim=1).item()

# --- 7. 输出结果 ---
lanes = ["左车道 (0)", "中车道 (1)", "右车道 (2)"]
print(f"\n>>> 预测结果: 车辆位于 [{lanes[predicted_lane]}]")
print(f">>> 置信度: {probabilities.numpy()}")