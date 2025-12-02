# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 21:10:08 2025

@author: y
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from rdkit import Chem
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# ---------- Step 1: SMILES to Graph ----------
def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    atom_features = [[atom.GetAtomicNum()] for atom in mol.GetAtoms()]
    edge_index = []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.append([start, end])
        edge_index.append([end, start])
    
    x = torch.tensor(atom_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index)

# ---------- Step 2: GCN 模型 ----------
class GCNClassifier(nn.Module):
    def __init__(self, input_dim=1):
        super(GCNClassifier, self).__init__()
        self.conv1 = GCNConv(input_dim, 64)
        self.conv2 = GCNConv(64, 32)
        self.conv3 = GCNConv(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.relu(self.conv1(x, edge_index))
        x = self.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        x = torch.sum(x, dim=0)
        return self.sigmoid(x)

# ---------- Step 3: 读取数据 ----------
# CSV 中需要至少两列：SMILES 和 label
df = pd.read_csv('F:\\PFAS-KANO\\PFAS与血浆蛋白结合情况预测.csv')  # 替换为你的数据路径
smiles_list = df['SMILES'].tolist()
labels = df['active_label'].tolist()  # label 需要是 0 / 1

# 转换成图
graph_data = [smiles_to_graph(s) for s in smiles_list]
labels = torch.tensor(labels, dtype=torch.float).view(-1, 1)

# ---------- Step 4: 交叉验证 ----------
kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = {'acc': [], 'f1': [], 'precision': [], 'recall': [], 'roc_auc': []}

for train_idx, test_idx in kf.split(graph_data):
    model = GCNClassifier()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.BCELoss()

    # 训练
    for epoch in range(20):
        model.train()
        total_loss = 0
        for i in train_idx:
            data = graph_data[i]
            label = labels[i]
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{20}, Loss: {total_loss / len(train_idx)}')

    # 测试
    model.eval()
    y_true, y_pred, y_proba = [], [], []
    with torch.no_grad():
        for i in test_idx:
            data = graph_data[i]
            label = labels[i].item()
            output = model(data).item()
            y_true.append(label)
            y_proba.append(output)
            y_pred.append(1 if output >= 0.5 else 0)

    results['acc'].append(accuracy_score(y_true, y_pred))
    results['f1'].append(f1_score(y_true, y_pred))
    results['precision'].append(precision_score(y_true, y_pred))
    results['recall'].append(recall_score(y_true, y_pred))
    results['roc_auc'].append(roc_auc_score(y_true, y_proba))

# ---------- Step 5: 结果输出 ----------
for metric, scores in results.items():
    print(f"{metric}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
