import numpy as np
import matplotlib.pyplot as plt

def plot_distortion_test():
    # 模拟数据：在直路上变道
    # Y 轴：走了 1000 米
    y = np.linspace(0, 1000, 100)
    # X 轴：慢慢变道，横向只移动了 5 米
    x = 5 * (1 / (1 + np.exp(-(y - 500)/50))) 
    
    plt.figure(figsize=(10, 4))
    
    # 1. 错误的画法 (Auto Scaling - 默认)
    plt.subplot(1, 2, 1)
    plt.plot(x, y, 'r-', linewidth=3)
    plt.title("No 'equal' (Distorted)")
    plt.grid(True)
    # Matplotlib 会把 X轴的 0-5米 拉伸得和 Y轴的 0-1000米 一样长！
    # 看起来像是在过一个 45度 的超级大急弯！
    
    # 2. 正确的画法 (Equal Aspect Ratio)
    plt.subplot(1, 2, 2)
    plt.plot(x, y, 'g-', linewidth=3)
    plt.title("With 'equal' (True Shape)")
    plt.grid(True)
    plt.axis('equal') 
    # 强制 1米 X轴长度 = 1米 Y轴长度
    # 看起来就是一条几乎直的线（这是真相）

    plt.tight_layout()
    plt.show()

plot_distortion_test()