# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 16:26:20 2025

@author: y
"""

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen

# 读取CSV文件，假设有一列叫 "SMILES"
df = pd.read_csv('F:\\PFAS-KANO\\结果\\OECD-predict.csv')  # 替换为你的路径

# 定义函数：根据 SMILES 计算 XLogP
def calculate_xlogp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Crippen.MolLogP(mol)

# 应用函数，创建新列 "XLogP"
df['XLogP'] = df['SMILES'].apply(calculate_xlogp)

# 输出前几行看看结果
print(df[['SMILES', 'XLogP']].head())

# 可选：保存结果
df.to_csv('F:\\PFAS-KANO\\结果\\OECD-predict_xlogp.csv', index=False)
