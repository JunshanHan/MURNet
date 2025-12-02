# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 20:16:39 2025

@author: y
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 22:57:32 2025

@author: y
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import StandardScaler

# 读取 CSV 文件
file_path = 'F:\\PFAS-KANO\\骨架可视化\\pubchem\\pubchem_TOP5.csv'  # 替换为你的 CSV 文件路径
data = pd.read_csv(file_path)

# 假设最后一列为标签列，其余为特征列
X = data.iloc[:, 1:].values
scaler = StandardScaler()
X1 = scaler.fit_transform(X)

y = data.iloc[:, 0].values

# 使用 PCA 进行降维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X1)
# 计算 PCA 降维后的 DB 指数
db_index_pca = davies_bouldin_score(X_pca, y)

# 使用 t-SNE 进行降维
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X1)
# 计算 t-SNE 降维后的 DB 指数
db_index_tsne = davies_bouldin_score(X_tsne, y)

# 可视化 PCA 降维结果
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
unique_labels = np.unique(y)
for label in unique_labels:
    indices = y == label
    plt.scatter(X_pca[indices, 0], X_pca[indices, 1], label=label)
plt.title('PCA Dimensionality Reduction')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
# 在左上角显示 PCA 的 DB 指数
plt.text(0.02, 0.95, f'DB Index: {db_index_pca:.2f}', transform=plt.gca().transAxes,
         verticalalignment='top')
plt.legend()
plt.savefig('F:\\PFAS-KANO\\骨架可视化\\pubchem\\pubchem-PCA.png',dpi=600,bbox_inches='tight')


# 可视化 t-SNE 降维结果
plt.figure(figsize=(8, 6))
#plt.subplot(1, 2, 2)
for label in unique_labels:
    indices = y == label
    plt.scatter(X_tsne[indices, 0], X_tsne[indices, 1], label=label)
plt.title('t-SNE Dimensionality Reduction')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
# 在左上角显示 t-SNE 的 DB 指数
plt.text(0.02, 0.97, f'DB Index: {db_index_tsne:.2f}', transform=plt.gca().transAxes,
         verticalalignment='top')
plt.legend(bbox_to_anchor=(1.05, 1),  # 图例位置相对于坐标轴的坐标(1,1)为右上角
    loc='upper left',          # 图例自身的锚点位置
    borderaxespad=0.,          # 边界填充
    title='Labels',            # 添加图例标题（可选）
    frameon=True )

plt.tight_layout()

plt.savefig('F:\\PFAS-KANO\\骨架可视化\\pubchem\\pubchem-TSNE.png',dpi=600,bbox_inches='tight')
plt.show()

