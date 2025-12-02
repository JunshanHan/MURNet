# -*- coding: utf-8 -*-
"""
Created on Fri May 16 09:49:19 2025

@author: y
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import rdmolops
from scipy import stats
import itertools

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def get_longest_carbon_chain(smiles):
    """计算SMILES中最长连续碳链的长度"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        
        # 创建只包含碳的子图
        carbon_mol = rdmolops.DeleteSubstructs(mol, Chem.MolFromSmarts('[!#6]'))
        
        # 计算最长碳链
        max_path = 0
        for n in range(1, 50):
            paths = Chem.FindAllPathsOfLengthN(carbon_mol, n, useBonds=False, useHs=False)
            if not paths:
                break
            max_path = n
        
        return max_path
    except Exception as e:
        print(f"处理SMILES '{smiles}' 时出错: {e}")
        return 0

def perform_significance_test(data, group_col, value_col, alpha=0.05):
    """
    对分组数据进行显著性差异检验
    
    参数:
    data: DataFrame，包含分组列和数值列
    group_col: 分组列名
    value_col: 数值列名
    alpha: 显著性水平，默认0.05
    """
    # 按标签分组
    groups = data.groupby(group_col)[value_col].apply(list)
    group_names = list(groups.index)
    
    # 存储所有比较结果
    results = []
    
    # 对所有组对进行比较
    for group1, group2 in itertools.combinations(group_names, 2):
        data1 = groups[group1]
        data2 = groups[group2]
        
        # 1. 正态性检验 (Shapiro-Wilk)
        _, p_norm1 = stats.shapiro(data1)
        _, p_norm2 = stats.shapiro(data2)
        both_normal = (p_norm1 > alpha) and (p_norm2 > alpha)
        
        # 2. 方差齐性检验 (Levene)
        _, p_levene = stats.levene(data1, data2)
        equal_var = p_levene > alpha
        
        # 3. 根据数据分布选择检验方法
        if both_normal and equal_var:
            # 参数检验: 独立样本t检验
            test_name = "独立样本t检验"
            stat, p_value = stats.ttest_ind(data1, data2)
        elif both_normal and not equal_var:
            # 参数检验: Welch's t检验 (不等方差)
            test_name = "Welch's t检验"
            stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)
        else:
            # 非参数检验: Mann-Whitney U检验
            test_name = "Mann-Whitney U检验"
            stat, p_value = stats.mannwhitneyu(data1, data2)
        
        # 4. 确定显著性
        significant = p_value < alpha
        significance_level = ""
        if p_value < 0.001:
            significance_level = "*** (p < 0.001)"
        elif p_value < 0.01:
            significance_level = "** (p < 0.01)"
        elif p_value < 0.05:
            significance_level = "* (p < 0.05)"
        else:
            significance_level = "不显著"
        
        # 5. 记录结果
        results.append({
            "组别1": group1,
            "组别2": group2,
            "检验方法": test_name,
            "统计量": stat,
            "p值": p_value,
            "是否显著": significant,
            "显著性水平": significance_level
        })
    
    # 转换为DataFrame并返回
    return pd.DataFrame(results)

def plot_chain_length_vs_label(csv_path, output_image_path=None, perform_tests=True, x_labels=None):
    """
    读取CSV文件，计算分子链长并绘制与label的关系图，可选择执行显著性检验
    
    参数:
    csv_path: CSV文件路径，包含'smiles'和'label'列
    output_image_path: 可选，图像保存路径
    perform_tests: 是否执行显著性检验，默认True
    """
    # 读取数据
    df = pd.read_csv(csv_path)
    
    # 确保列名正确
    if 'smiles' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV文件必须包含'smiles'和'label'列")
    
    # 计算链长
    print("正在计算分子链长...")
    df['chain_length'] = df['smiles'].apply(get_longest_carbon_chain)
    
    # 数据摘要
    print(f"数据摘要:")
    print(df[['smiles', 'label', 'chain_length']].head())
    
    # 创建图形
    plt.figure(figsize=(4.5, 9))
    
    # 检查label类型（分类或连续）
    unique_labels = df['label'].nunique()
    
    if unique_labels <= 10:  # 假设是分类变量
        # 箱线图
        ax1 = plt.subplot(2, 1, 1)
        sns.boxplot(x='label', y='chain_length', data=df, width=0.3, palette=['#F7941D', '#00CFFF'])
        #plt.title('分子链长与Label的关系（箱线图）')
        plt.xlabel('', fontsize=20)
        plt.ylabel('Carbon Chain Length', fontsize=20)
        
        # 调整坐标轴刻度字体大小
        plt.xticks(fontsize=18)  # x轴刻度字体大小
        plt.yticks(fontsize=15)  # y轴刻度字体大小
        plt.xlim(-0.5, unique_labels-0.5)  # 紧贴左右边缘
        
        # 自定义X轴标签
        if x_labels is not None:
            if len(x_labels) == unique_labels:
                ax1.set_xticklabels(x_labels)
            else:
                print(f"警告：提供的X轴标签数量({len(x_labels)})与实际类别数量({unique_labels})不匹配，将使用原始标签")
        
        
# =============================================================================
#         # 小提琴图
#         plt.subplot(2, 1, 2)
#         sns.violinplot(x='label', y='chain_length', data=df)
#         plt.title('分子链长与Label的关系（小提琴图）')
#         plt.xlabel('Label')
#         plt.ylabel('碳链长度')
# =============================================================================
        
        # 如果需要，执行显著性检验
        if perform_tests:
            print("\n执行显著性差异检验...")
            test_results = perform_significance_test(df, 'label', 'chain_length')
            
            # 打印结果
            print("\n显著性检验结果:")
            print(test_results[['组别1', '组别2', '检验方法', 'p值', '显著性水平']])
            
            # 在箱线图上添加显著性标记
            ax = plt.subplot(2, 1, 1)
            height_factor = 1.05
            y_max = df['chain_length'].max()
            
            for i, row in test_results.iterrows():
                if row['是否显著']:
                    # 获取组别的索引
                    group1_idx = df['label'].unique().tolist().index(row['组别1'])
                    group2_idx = df['label'].unique().tolist().index(row['组别2'])
                    
                    # 计算连接线位置
                    x1, x2 = group1_idx, group2_idx
                    y, h, col = y_max * height_factor, y_max * 0.05, 'k'
                    
                    # 绘制横线
                    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
                    
                    # 添加显著性标记
                    sig_text = row['显著性水平'].split()[0]
                    ax.text((x1+x2)*0.5, y+h, sig_text, ha='center', va='bottom', c=col)
                    
                    # 调整下一个比较的高度
                    height_factor += 0.15
            
            plt.tight_layout()
    
    else:  # 假设是连续变量
        # 散点图
        plt.subplot(2, 1, 1)
        sns.scatterplot(x='label', y='chain_length', data=df)
        plt.title('分子链长与Label的关系（散点图）')
        #plt.xlabel('Label', fontsize=12)
        plt.ylabel('碳链长度', fontsize=12)
        
        # 回归图
        plt.subplot(2, 1, 2)
        sns.regplot(x='label', y='chain_length', data=df)
        plt.title('分子链长与Label的关系（回归图）')
        #plt.xlabel('Label', fontsize=12)
        plt.ylabel('碳链长度', fontsize=12)
    
    plt.tight_layout()
    
    # 保存图像（如果指定了路径）
    if output_image_path:
        plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存至: {output_image_path}")
    
    plt.show()
    
    # 返回处理后的数据框
    return df

if __name__ == "__main__":
    # 使用示例
    csv_path = "F:\\PFAS-KANO\\KANO-5.8\\predict_OECD_HSA.csv"  # 替换为你的CSV文件路径
    output_image = "F:\\PFAS-KANO\\影响因素\\chain_length_vs_label1.png" 
    
    try:
        processed_data = plot_chain_length_vs_label(csv_path, output_image,x_labels=['Low Class', 'High Class'])
        print("\n分析完成！")
    except Exception as e:
        print(f"程序运行出错: {e}")