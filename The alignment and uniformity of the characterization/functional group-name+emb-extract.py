# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 11:35:43 2025

@author: y
"""

import pickle
from rdkit import Chem
import pandas as pd

with open('F:\\PFAS\\KANO-main/chemprop/data/funcgroup.txt', "r") as f:
    funcgroups = f.read().strip().split('\n')
    name = [i.split()[0] for i in funcgroups]
    smart = [Chem.MolFromSmarts(i.split()[1]) for i in funcgroups]
    smart2name = dict(zip(smart, name))

fg2emb = pickle.load(open('F:\\PFAS\\KANO-main\\initial/fg2emb.pkl', 'rb'))


def match_fg_with_name(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"无法将 SMILES 字符串 {smiles} 转换为分子对象。")
        return {}
    fg_emb_with_name = {}
    fg_emb = [[1] * 133]
    pad_fg = [[0] * 133]
    for sm in smart:
        if mol.HasSubstructMatch(sm):
            group_name = smart2name[sm]
            fg_emb.append(fg2emb[group_name].tolist())
            fg_emb_with_name[group_name] = fg2emb[group_name].tolist()
    if len(fg_emb) > 13:
        fg_emb = fg_emb[:13]
        # 截取对应的名称和嵌入
        names = list(fg_emb_with_name.keys())[:12]  # 因为第一个是 [1] * 133 ，所以这里取前 12 个匹配的
        fg_emb_with_name = {name: fg_emb_with_name[name] for name in names}
    else:
        fg_emb.extend(pad_fg * (13 - len(fg_emb)))
    return fg_emb_with_name


# 示例使用
smiles = 'COC1=CC2=C(C=C1CN(C)C(C)C(N)=O)C(NC1=C(F)C(Cl)=CC=C1)=NC=N2'
result = match_fg_with_name(smiles)
print(f"SMILES 字符串 {smiles} 对应的官能团名称和嵌入：")
for group_name, embedding in result.items():
    print(f"官能团名称: {group_name}, 官能团嵌入: {embedding}")






# 读取 CSV 文件
csv_file_path = 'F:\\PFAS-KANO\\smiles.csv'  # 请将此处替换为实际的 CSV 文件路径
df = pd.read_csv(csv_file_path)

all_group_names = []
# 遍历每个 SMILES 字符串并获取官能团名称
for index, row in df.iterrows():
    smiles = row['smiles']
    group_names = match_fg_with_name(smiles)
    all_group_names.extend(group_names)

print("所有 SMILES 字符串匹配到的官能团名称如下：")
for group_name in all_group_names:
    print(group_name)







    