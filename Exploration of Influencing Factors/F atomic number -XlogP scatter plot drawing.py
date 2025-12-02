# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 17:05:21 2025

@author: y
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取CSV文件（假设文件名为data.csv）
df = pd.read_csv('F:\\PFAS-KANO\\影响因素\\OECD-predict_xlogp_F_count-high_class增加.csv')  # 替换为你的文件路径

# 设置颜色映射
palette = {
    'Low Class': '#F7941D',     # 橙色
    'High Class': '#00CFFF'     # 蓝色
}

# 开始绘图
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x='F_count',
    y='XLogP',
    hue='class',
    palette=palette,
    alpha=0.8,
    edgecolor='none',
    s=40,
    style='class',  # 可根据 class 设置不同形状
    markers={'Low Class': 'o', 'High Class': '^'}  # 与图中形状一致
)

# 设置坐标轴标签与图例
plt.xlabel('Number of fluorine atom', fontsize=20)
plt.ylabel('XLogP', fontsize=20)
plt.legend(title='', loc='best', fontsize=20,markerscale=2)
# 调整坐标轴刻度字体大小
plt.xticks(fontsize=17)  # x轴刻度字体大小
plt.yticks(fontsize=17)  # y轴刻度字体大小
plt.xlim(0, 60)
plt.ylim(-5, 35)
plt.tight_layout()
plt.savefig("F:\\PFAS-KANO\\影响因素\\氟原子数-LogP", dpi=600, bbox_inches='tight')
plt.show()
