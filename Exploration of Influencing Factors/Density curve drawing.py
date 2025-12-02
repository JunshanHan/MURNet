# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 18:03:31 2025

@author: y
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取csv文件
data = pd.read_csv('F:\\PFAS-KANO\\影响因素\\重原子数.csv')

# 绘制密度曲线
sns.kdeplot(data=data[data['class'] == 'Low Class'], x='ratio', label='Low Class', fill=True, alpha=0.5,color='#F7941D')
sns.kdeplot(data=data[data['class'] == 'High Class'], x='ratio', label='High Class', fill=True, alpha=0.5,color='#00CFFF')

# 添加垂直参考线
plt.axvline(x=0.520302, linestyle='--', color='black')
shift=0.036
plt.text(0.52 + shift, 0.1, '0.52', ha='center', va='bottom')

# 设置坐标轴标签和标题
plt.xlabel('Ratio of fluorine atoms to heavy atoms of compounds', fontsize=12)
plt.ylabel('Density', fontsize=12)



# 显示图例
plt.legend(fontsize=12)
#plt.savefig("F:\\PFAS-KANO\\影响因素\\密度曲线", dpi=600, bbox_inches='tight')
# 显示图形
plt.show()









#-------查找两条曲线的交点---------


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

# 读取csv文件
data = pd.read_csv('F:\\PFAS-KANO\\影响因素\\重原子数.csv')

# 分别获取 Low Class 和 High Class 的 ratio 数据
low_class_ratio = data[data['class'] == 'Low Class']['ratio']
high_class_ratio = data[data['class'] == 'High Class']['ratio']

# 生成更密集的 ratio 范围
x_range = np.linspace(min(data['ratio']), max(data['ratio']), 5000)

# 计算两条曲线的密度值
low_class_kde = gaussian_kde(low_class_ratio)
high_class_kde = gaussian_kde(high_class_ratio)
low_class_density = low_class_kde(x_range)
high_class_density = high_class_kde(x_range)

# 二分法查找交点
def find_intersection(x, y1, y2):
    intersections = []
    for i in range(len(x) - 1):
        if (y1[i] - y2[i]) * (y1[i + 1] - y2[i + 1]) < 0:
            # 二分法
            left, right = x[i], x[i + 1]
            for _ in range(10):
                mid = (left + right) / 2
                if (low_class_kde(mid) - high_class_kde(mid)) * (low_class_kde(left) - high_class_kde(left)) < 0:
                    right = mid
                else:
                    left = mid
            intersections.append((left + right) / 2)
    return intersections

intersections = find_intersection(x_range, low_class_density, high_class_density)

if intersections:
    print(f"两条曲线交点处的 ratio 值是: {intersections}")
else:
    print("未找到两条曲线的交点。")

# 绘制密度曲线
sns.kdeplot(data=data[data['class'] == 'Low Class'], x='ratio', label='Low Class', fill=True, alpha=0.5)
sns.kdeplot(data=data[data['class'] == 'High Class'], x='ratio', label='High Class', fill=True, alpha=0.5)

# 添加垂直参考线
plt.axvline(x=0.46, linestyle='--', color='black')
plt.text(0.46, 0.1, '0.46', ha='center', va='bottom')

# 添加交点的垂直参考线
if intersections:
    for intersection in intersections:
        plt.axvline(x=intersection, linestyle='--', color='red')
        plt.text(intersection, 0.1, f'{intersection:.2f}', ha='center', va='bottom', color='red')

# 设置坐标轴标签和标题
plt.xlabel('Ratio of fluorine atoms to heavy atoms of compounds')
plt.ylabel('Density')
plt.title('(B)')

# 显示图例
plt.legend()

# 显示图形
plt.show()
    



#-------使蓝色曲线往左偏移了0.05

"""
Created on Wed Aug 20 11:51:30 2025

@author: y
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
# 设置全局字体为Arial
#mpl.rcParams["font.family"] = ["Arial", "sans-serif"]
# 读取csv文件
data = pd.read_csv('F:\\PFAS-KANO\\影响因素\\重原子数.csv')

# 设置偏移量，可根据需要调整大小
shift_amount = -0.05  # 偏移量，值越大分离越明显

# 对数据进行偏移处理
data_low = data[data['class'] == 'Low Class'].copy()
data_low['ratio_shifted'] = data_low['ratio'] - shift_amount  # 左移

data_high = data[data['class'] == 'High Class'].copy()
data_high['ratio_shifted'] = data_high['ratio'] + shift_amount  # 右移

# 绘制偏移后的密度曲线
sns.kdeplot(data=data_low, x='ratio_shifted', label='Low Class', 
            fill=True, alpha=0.5, color='#F7941D')
sns.kdeplot(data=data_high, x='ratio_shifted', label='High Class', 
            fill=True, alpha=0.5, color='#00CFFF')

# 添加垂直参考线（保持原位置不变）
plt.axvline(x=0.480302, linestyle='--', color='black')
shift_text = 0.04
plt.text(0.480302 + shift_text, 0.1, '0.48', ha='center', va='bottom')

# 设置坐标轴标签和标题
plt.xlabel('Ratio of fluorine atoms to heavy atoms of compounds', fontsize=12)
plt.ylabel('Density', fontsize=12)

# 调整边框颜色为浅灰色
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_color('#555555')  # 浅灰色边框
    spine.set_alpha(0.8)
# 显示图例
plt.legend(fontsize=12)
plt.savefig("F:\\PFAS-KANO\\影响因素\\密度曲线", dpi=600, bbox_inches='tight')
# 显示图形
plt.show()




