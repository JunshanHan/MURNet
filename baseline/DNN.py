import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    recall_score, accuracy_score, precision_score
)
import torch
import torch.nn as nn
import torch.optim as optim

# 导入数据
filename = 'F:\\PFAS-KANO\\PFAS与血浆蛋白结合情况预测-ECFP4.csv'
data = pd.read_csv(filename)

# 模型参数
seed = 42
epochs = 50
batch_size = 32
learning_rate = 0.001

# 只取一列作为二分类标签，例如第一列
y = data.iloc[:, 0].values
X = data.iloc[:, 1:2049].values

# 初始化交叉验证器
kf = KFold(n_splits=10, shuffle=True, random_state=seed)

# 存储每折的结果
all_f1, all_acc, all_pre, all_rec, all_roc_auc, all_pr_auc = [], [], [], [], [], []


# 定义 DNN 模型
class DNN(nn.Module):
    def __init__(self, input_size):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out


for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # 转换为 PyTorch 张量
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train).view(-1, 1)
    X_test = torch.FloatTensor(X_test)

    # 初始化模型、损失函数和优化器
    model = DNN(X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    # 预测
    with torch.no_grad():
        y_pred_proba = model(X_test).numpy().flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)

    # 评估指标
    all_f1.append(f1_score(y_test, y_pred))
    all_acc.append(accuracy_score(y_test, y_pred))
    all_pre.append(precision_score(y_test, y_pred))
    all_rec.append(recall_score(y_test, y_pred))
    try:
        all_roc_auc.append(roc_auc_score(y_test, y_pred_proba))
    except:
        all_roc_auc.append(np.nan)
    try:
        all_pr_auc.append(average_precision_score(y_test, y_pred_proba))
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
    