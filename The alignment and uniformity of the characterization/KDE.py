# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 20:26:55 2025

@author: y
"""
import argparse
from copy import deepcopy
from matplotlib.colors import Normalize, ListedColormap
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.manifold import TSNE
from matplotlib import cm

import seaborn as sns

import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, kde, gaussian_kde, binned_statistic_2d

smiles=open("F:\\PFAS-KANO\\骨架可视化\\ECFP6\\ECFP6.csv")
df=pd.read_csv(smiles)


# 假设最后一列为标签列，其余为特征列
X = df.iloc[:, :].values
scaler = StandardScaler()
X1 = scaler.fit_transform(X)

# 使用 t-SNE 进行降维
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X1)


# 计算密度
x = X_tsne[:, 0]
y = X_tsne[:, 1]

# 绘制角度密度估计图
angles = np.arctan2(y, x)
kde_angles = gaussian_kde(angles)
angle_range = np.linspace(-np.pi, np.pi, 100)
density_angles = kde_angles(angle_range)

plt.figure(figsize=(20, 12))
plt.xlabel('Angles', fontsize=40)
#plt.ylabel('Density', fontsize=40)
plt.yticks(fontsize=40)
plt.ylim([0, 0.30])
plt.xticks(fontsize=40)
plt.plot(angle_range, density_angles, color='lightblue')
plt.fill_between(angle_range, density_angles, color='lightblue', alpha=0.1)

plt.savefig('F:\\PFAS-KANO\\骨架可视化\\ECFP6\\ECFP6-angle1.png',dpi=600,bbox_inches='tight')
plt.show()


# 创建直角坐标系下的图
fig, ax = plt.subplots(figsize=(20, 20))

# 生成圆环的点并绘制
for i in range(len(angle_range) - 1):
    theta_start = angle_range[i]
    theta_end = angle_range[i + 1]
    radii = np.linspace(0.9, 1, 100)  # 进一步调整起始和终止半径，以减少圆环的宽度

    for radius in radii:
        x_start = radius * np.cos(theta_start)
        y_start = radius * np.sin(theta_start)
        x_end = radius * np.cos(theta_end)
        y_end = radius * np.sin(theta_end)

        ax.plot([x_start, x_end], [y_start, y_end], color=plt.cm.pink(1 - density_angles[i] / max(density_angles)), lw=2)

# 设置颜色条
# norm = plt.Normalize(vmin=min(density_angles), vmax=max(density_angles))
# sm = plt.cm.ScalarMappable(cmap='pink_r', norm=norm)
# sm.set_array([])
# fig.colorbar(sm, ax=ax, label='Density')

# 设置标题和标签
ax.set_xlabel('Features', fontsize=40)
#ax.set_ylabel('Features', fontsize=40)
ax.set_aspect('equal')

# 设置x轴和y轴的刻度
ax.set_xticks([-1, 0, 1])
ax.set_yticks([-1.0, -0.5, 0, 0.5, 1])
ax.tick_params(axis='both', which='major', labelsize=40)

# 隐藏网格
ax.grid(False)

# 显示四个框
ax.spines['top'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['right'].set_visible(True)

plt.savefig('F:\\PFAS-KANO\\骨架可视化\\ECFP6\\ECFP6-yuanhuan1.png',dpi=600,bbox_inches='tight')
plt.show()