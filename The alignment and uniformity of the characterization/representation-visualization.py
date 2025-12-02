# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 21:25:51 2025

@author: y
"""

# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.family"] = ["Arial", "sans-serif"]
# 读取 CSV 文件
file_path = 'F:\\PFAS-KANO\\骨架可视化\\DNN\\DNN_TOP5_scaffolds.csv'
try:
    data = pd.read_csv(file_path)
    print(f"数据加载成功，形状: {data.shape}")
    
    # 提取特征和标签
    X = data.iloc[:, 1:33].values
    y = data.iloc[:, 0].values
    
    # 标准化特征
    scaler = StandardScaler()
    X1 = scaler.fit_transform(X)
    
    # t-SNE 降维
    print("\n执行 t-SNE 降维...")
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X1)
    
    # 计算 DB 指数
    unique_labels = np.unique(y)
    db_index_tsne = None
    if len(unique_labels) >= 2:
        db_index_tsne = davies_bouldin_score(X_tsne, y)
        print(f"DB 指数计算完成: {db_index_tsne:.4f}")
    
    # 可视化
    plt.figure(figsize=(8, 6))
    plt.gca().set_facecolor('#EAEAF3')
    
    # 设置颜色映射
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        indices = y == label
        plt.scatter(X_tsne[indices, 0], X_tsne[indices, 1], 
                    label=label, color=colors[i], alpha=0.7, s=40, zorder=3)
    
    plt.title('DNN emb')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    
    # 设置网格线
    plt.grid(True, linestyle='-', alpha=0.6, color='white', zorder=1)
    
    # 显示 DB 指数
    if db_index_tsne is not None:
        plt.text(0.02, 0.97, f'DB index: {db_index_tsne:.2f}', 
                 transform=plt.gca().transAxes, verticalalignment='top',
                 fontsize=11,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.4),
                 zorder=5)
    
    # 隐藏四周边框线
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    
    # 隐藏坐标轴刻度线
    plt.tick_params(axis='both', which='both', length=0)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
               borderaxespad=0., title='标签', frameon=True)
    
    plt.tight_layout()
    
    # 保存图片
    save_path = 'F:\\PFAS-KANO\\骨架可视化\\DNN\\DNN-TSNE.png'
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    print(f"\n图片已保存至: {save_path}")
    
    plt.show()
    
except FileNotFoundError:
    print(f"错误：文件未找到 - {file_path}")
except Exception as e:
    print(f"发生未知错误: {e}")