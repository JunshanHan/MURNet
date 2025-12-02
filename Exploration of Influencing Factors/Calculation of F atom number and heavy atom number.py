# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 16:58:18 2025

@author: y
"""

import pandas as pd
from rdkit import Chem

# 读取 CSV 文件（假设 SMILES 列名是 "SMILES"）
df = pd.read_csv('F:\\PFAS-KANO\\结果\\OECD-predict_xlogp.csv')  # 替换为你的 CSV 路径

# 定义函数：统计 F 原子数量
def count_f_atoms(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'F')

# 应用函数添加一列
df['F_count'] = df['SMILES'].apply(count_f_atoms)

# 输出结果预览
print(df[['SMILES', 'F_count']].head())

# 可选：保存到新文件
df.to_csv('F:\\PFAS-KANO\\结果\\OECD-predict_xlogp_F_count.csv', index=False)


import csv
def calculate_heavy_atom_count(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return mol.GetNumHeavyAtoms()
        return None
    except Exception:
        return None

input_file = 'F:\\PFAS-KANO\\结果\\OECD-predict_xlogp_F_count-high_class增加.csv'
output_file = 'F:\\PFAS-KANO\\结果\\重原子数.csv'

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames + ['heavy_atom_count']
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)

    writer.writeheader()
    for row in reader:
        smiles = row.get('SMILES')
        if smiles:
            heavy_atom_count = calculate_heavy_atom_count(smiles)
            row['heavy_atom_count'] = heavy_atom_count
        writer.writerow(row)

