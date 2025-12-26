import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==========================================
# 1. 模拟测试数据集 (Test Set Generation)
# ==========================================
def generate_evaluation_data():
    """
    生成一段测试数据。
    场景：车子从 Road_A -> Road_B -> Road_C
    """
    # 真实标签序列 (Ground Truth)
    # 0-30s 在 A，30-60s 在 B，60-100s 在 C
    true_labels = (['Road_A'] * 30) + (['Road_B'] * 30) + (['Road_C'] * 40)
    
    # --- 模拟模型的预测结果 (Predicted Labels) ---
    # 假设我们的模型准确率不错，但在切换路段时有延迟 (Lag)，偶尔还有抖动
    pred_labels = true_labels.copy()
    
    # 1. 模拟 "滞后" (Lag) - HMM 常见的副作用
    # 当真值从 A 变 B 时 (index 30)，模型还在犹豫，多报了 3 秒的 A
    pred_labels[30:33] = ['Road_A'] * 3
    
    # 2. 模拟 "误切" (Flicker) - 噪声导致的跳变
    # 在 Road_B 中间，突然跳到了 Road_A 两次
    pred_labels[45] = 'Road_A'
    pred_labels[46] = 'Road_A'
    
    # 3. 模拟 "混淆" (Confusion)
    # Road_C 和 Road_B 可能靠得很近，最后阶段模型经常搞混
    # 把一部分 C 判成了 B
    for i in range(80, 90):
        pred_labels[i] = 'Road_B'
        
    return true_labels, pred_labels

# ==========================================
# 2. 评测可视化引擎
# ==========================================
def evaluate_performance(y_true, y_pred):
    # 1. 计算总准确率
    acc = accuracy_score(y_true, y_pred)
    print(f">>> 总体准确率 (Accuracy): {acc:.2%}")
    print("-" * 30)
    
    # 2. 打印详细分类报告 (Precision/Recall/F1)
    print(">>> 详细报告:")
    print(classification_report(y_true, y_pred))
    
    # 3. 绘制混淆矩阵 (The Heatmap)
    # 获取唯一的标签列表，保证顺序一致
    labels = sorted(list(set(y_true)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(8, 6))
    
    # 使用 Seaborn 画热力图
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    
    plt.title(f"Confusion Matrix (Acc: {acc:.2%})")
    plt.xlabel("Predicted Road (Model Thought)")
    plt.ylabel("True Road (Actually Is)")
    plt.show()

# ==========================================
# 3. 错误案例分析可视化 (Error Analysis)
# ==========================================
def plot_timeline_comparison(y_true, y_pred):
    """
    画出时间轴对比图，直观看到哪里错了
    """
    # 把字符串标签转为数字以便画图
    label_map = {label: i for i, label in enumerate(sorted(list(set(y_true))))}
    num_true = [label_map[l] for l in y_true]
    num_pred = [label_map[l] for l in y_pred]
    
    plt.figure(figsize=(12, 4))
    
    # 画真值 (绿色实线)
    plt.plot(num_true, 'g-', linewidth=4, alpha=0.5, label='Ground Truth')
    
    # 画预测 (红色虚线)
    plt.plot(num_pred, 'r--', linewidth=2, label='Prediction')
    
    # 标注 Y 轴
    plt.yticks(list(label_map.values()), list(label_map.keys()))
    plt.xlabel("Time Step")
    plt.title("Timeline Analysis: Where did the model fail?")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # 1. 获取数据
    y_true, y_pred = generate_evaluation_data()
    
    # 2. 执行评测
    evaluate_performance(y_true, y_pred)
    
    # 3. 时间轴分析
    plot_timeline_comparison(y_true, y_pred)