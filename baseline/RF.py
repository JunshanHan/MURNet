# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 14:55:58 2025

@author: y
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    recall_score, accuracy_score, precision_score
)
import warnings
warnings.filterwarnings("ignore")

# 导入数据
filename = 'F:\\PFAS-KANO\\PFAS与血浆蛋白结合情况预测-ECFP4.csv'
data = pd.read_csv(filename)

# 模型参数
n_estimators_best = 100  # 决策树的数量
max_depth_best = None  # 树的最大深度
seed = 42

# 只取一列作为二分类标签，例如第一列
y = data.iloc[:, 0]
X = data.iloc[:, 1:2049]

# 初始化交叉验证器
kf = KFold(n_splits=10, shuffle=True, random_state=seed)

# 存储每折的结果
all_f1, all_acc, all_pre, all_rec, all_roc_auc, all_pr_auc = [], [], [], [], [], []

for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # 训练模型
    model = RandomForestClassifier(n_estimators=n_estimators_best, max_depth=max_depth_best, random_state=seed)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # 评估指标
    all_f1.append(f1_score(y_test, y_pred))
    all_acc.append(accuracy_score(y_test, y_pred))
    all_pre.append(precision_score(y_test, y_pred))
    all_rec.append(recall_score(y_test, y_pred))
    try:
        all_roc_auc.append(roc_auc_score(y_test, y_prob))
    except:
        all_roc_auc.append(np.nan)
    try:
        all_pr_auc.append(average_precision_score(y_test, y_prob))
    except:
        all_pr_auc.append(np.nan)

# 输出平均结果与标准差
print("acc:{:.4f}±{:.4f}\nf1:{:.4f}±{:.4f}\nrec:{:.4f}±{:.4f}\nroc_auc:{:.4f}±{:.4f}\npr_auc:{:.4f}±{:.4f}\npre:{:.4f}±{:.4f}".format(
    np.nanmean(all_acc), np.nanstd(all_acc),
    np.nanmean(all_f1), np.nanstd(all_f1),
    np.nanmean(all_rec), np.nanstd(all_rec),
    np.nanmean(all_roc_auc), np.nanstd(all_roc_auc),
    np.nanmean(all_pr_auc), np.nanstd(all_pr_auc),
    np.nanmean(all_pre), np.nanstd(all_pre)
))
    