

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, precision_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os

# 设置随机种子确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 定义数据集类
class CSVDataset(Dataset):
    def __init__(self, csv_path):
        # 读取CSV数据
        data = pd.read_csv(csv_path)
        
        # 假设最后一列是标签，其余列是特征
        self.features = data.iloc[:, :-1].values.astype(np.float32)
        self.labels = data.iloc[:, -1].values.astype(np.float32).reshape(-1, 1)
        
        # 数据标准化
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'features': torch.tensor(self.features[idx], dtype=torch.float32),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float32)
        }

# 创建两层前馈神经网络
def create_ffn(input_size, hidden_size=128, dropout=0.2):
    """创建一个两层的前馈神经网络"""
    return nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_size, 1)  # 二分类输出
    )

# 计算所有评估指标
def calculate_metrics(labels, probs):
    preds = (probs >= 0.5).astype(int)
    return {
        'auc': roc_auc_score(labels, probs),
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds),
        'recall': recall_score(labels, preds),
        'precision': precision_score(labels, preds)
    }

# 训练模型 - 保存所有epoch的验证结果和训练结果
def train_model(model, train_loader, val_loader, epochs=200, lr=0.0001, save_dir=None, fold_idx=None):
    criterion = nn.BCEWithLogitsLoss()  # 包含Sigmoid的二分类损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    all_train_metrics = []
    all_val_metrics = []
    best_val_auc = 0.0
    best_epoch = 0
    best_model = None
    
    # 创建保存模型的目录
    if save_dir and fold_idx is not None:
        os.makedirs(f"{save_dir}/fold_{fold_idx}", exist_ok=True)
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_labels_list = []
        train_preds_list = []
        
        for batch in train_loader:
            features = batch['features']
            labels = batch['labels']
            
            optimizer.zero_grad()
            outputs = model(features).squeeze()
            loss = criterion(outputs, labels.squeeze())
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * features.size(0)
            
            # 收集训练预测结果（添加detach()）
            train_labels_list.extend(labels.cpu().numpy())
            train_preds_list.extend(torch.sigmoid(outputs).detach().cpu().numpy())
        
        train_loss /= len(train_loader.dataset)
        
        # 计算训练集指标
        train_labels_np = np.array(train_labels_list).flatten()
        train_preds_np = np.array(train_preds_list).flatten()
        train_metrics = calculate_metrics(train_labels_np, train_preds_np)
        train_metrics['loss'] = train_loss
        all_train_metrics.append(train_metrics)
        
        # 验证阶段
        model.eval()
        val_labels_list = []
        val_preds_list = []
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features']
                labels = batch['labels']
                
                outputs = model(features).squeeze()
                preds = torch.sigmoid(outputs)  # 转换为概率
                
                # 收集验证预测结果（添加detach()，虽然在no_grad()中，但为保险起见）
                val_labels_list.extend(labels.cpu().numpy())
                val_preds_list.extend(preds.detach().cpu().numpy())
        
        # 计算验证集指标
        val_labels_np = np.array(val_labels_list).flatten()
        val_preds_np = np.array(val_preds_list).flatten()
        val_metrics = calculate_metrics(val_labels_np, val_preds_np)
        all_val_metrics.append(val_metrics)
        
        # 保存最佳模型
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_epoch = epoch
            best_model = model.state_dict().copy()
        
        if save_dir and fold_idx is not None:
            # 保存当前epoch的模型
            torch.save(model.state_dict(), f"{save_dir}/fold_{fold_idx}/model_epoch_{epoch}.pth")
            # 保存当前epoch的指标
            metrics_df = pd.DataFrame({
                'train_loss': [train_metrics['loss']],
                'train_auc': [train_metrics['auc']],
                'train_accuracy': [train_metrics['accuracy']],
                'train_f1': [train_metrics['f1']],
                'train_recall': [train_metrics['recall']],
                'train_precision': [train_metrics['precision']],
                'val_auc': [val_metrics['auc']],
                'val_accuracy': [val_metrics['accuracy']],
                'val_f1': [val_metrics['f1']],
                'val_recall': [val_metrics['recall']],
                'val_precision': [val_metrics['precision']]
            })
            metrics_df.to_csv(f"{save_dir}/fold_{fold_idx}/metrics_epoch_{epoch}.csv", index=False)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val AUC: {val_metrics["auc"]:.4f}')
    
    print(f"Fold最佳Epoch: {best_epoch+1}, Best Val AUC: {best_val_auc:.4f}")
    
    return best_model, best_val_auc, best_epoch, all_train_metrics, all_val_metrics

# 评估模型
def evaluate_model(model, test_loader):
    model.eval()
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features']
            labels = batch['labels']
            
            outputs = model(features).squeeze()
            probs = torch.sigmoid(outputs)  # 转换为概率
            
            # 添加detach()
            all_labels.extend(labels.detach().cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())
    
    all_labels = np.array(all_labels).flatten()
    all_probs = np.array(all_probs).flatten()
    
    # 计算各项评估指标
    return calculate_metrics(all_labels, all_probs)

# 十折交叉验证
def kfold_cross_validation(dataset, k=10, epochs=200, lr=0.0001, hidden_size=128, dropout=0.2, save_dir=None):
    input_size = dataset[0]['features'].shape[0]
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_metrics = []
    best_epochs = []
    all_fold_train_metrics = []
    all_fold_val_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(dataset)))):
        print(f"\n===== 开始第 {fold+1}/{k} 折训练 =====")
        
        # 创建训练集和验证集
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=50, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=50)
        
        # 创建模型
        model = create_ffn(input_size, hidden_size=hidden_size, dropout=dropout)
        
        # 训练模型，保存所有epoch的训练和验证指标
        best_state, best_val_auc, best_epoch, train_metrics, val_metrics = train_model(
            model, train_loader, val_loader, epochs=epochs, lr=lr, 
            save_dir=save_dir, fold_idx=fold
        )
        model.load_state_dict(best_state)
        
        # 评估模型
        metrics = evaluate_model(model, val_loader)
        fold_metrics.append(metrics)
        best_epochs.append(best_epoch + 1)  # 转换为1-based索引
        all_fold_train_metrics.append(train_metrics)
        all_fold_val_metrics.append(val_metrics)
        
        # 打印当前折的评估结果
        print(f"第 {fold+1} 折评估结果 (Epoch {best_epoch+1}):")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
    
    # 计算平均指标和标准差
    avg_metrics = {key: np.mean([fold[key] for fold in fold_metrics]) for key in fold_metrics[0].keys()}
    std_metrics = {key: np.std([fold[key] for fold in fold_metrics]) for key in fold_metrics[0].keys()}
    
    print("\n===== 十折交叉验证平均结果 =====")
    for key in avg_metrics:
        print(f"{key}: {avg_metrics[key]:.4f} ± {std_metrics[key]:.4f}")
    
    print("\n===== 各折最佳Epoch =====")
    for i, epoch in enumerate(best_epochs):
        print(f"Fold {i+1}: Epoch {epoch}")
    
    # 保存各折的指标曲线
    if save_dir:
        # 绘制AUC曲线
        plt.figure(figsize=(12, 8))
        for i, fold_aucs in enumerate([[epoch['auc'] for epoch in fold] for fold in all_fold_val_metrics]):
            plt.plot(range(1, len(fold_aucs)+1), fold_aucs, label=f'Fold {i+1}')
        plt.xlabel('Epoch')
        plt.ylabel('Validation AUC')
        plt.title('Validation AUC per Epoch for Each Fold')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{save_dir}/auc_curves.png")
        
        # 绘制准确率曲线
        plt.figure(figsize=(12, 8))
        for i, fold_acc in enumerate([[epoch['accuracy'] for epoch in fold] for fold in all_fold_val_metrics]):
            plt.plot(range(1, len(fold_acc)+1), fold_acc, label=f'Fold {i+1}')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Accuracy')
        plt.title('Validation Accuracy per Epoch for Each Fold')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{save_dir}/accuracy_curves.png")
        
        # 绘制F1分数曲线
        plt.figure(figsize=(12, 8))
        for i, fold_f1 in enumerate([[epoch['f1'] for epoch in fold] for fold in all_fold_val_metrics]):
            plt.plot(range(1, len(fold_f1)+1), fold_f1, label=f'Fold {i+1}')
        plt.xlabel('Epoch')
        plt.ylabel('Validation F1 Score')
        plt.title('Validation F1 Score per Epoch for Each Fold')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{save_dir}/f1_curves.png")
        
        # 绘制召回率曲线
        plt.figure(figsize=(12, 8))
        for i, fold_recall in enumerate([[epoch['recall'] for epoch in fold] for fold in all_fold_val_metrics]):
            plt.plot(range(1, len(fold_recall)+1), fold_recall, label=f'Fold {i+1}')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Recall')
        plt.title('Validation Recall per Epoch for Each Fold')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{save_dir}/recall_curves.png")
        
        # 绘制精确率曲线
        plt.figure(figsize=(12, 8))
        for i, fold_precision in enumerate([[epoch['precision'] for epoch in fold] for fold in all_fold_val_metrics]):
            plt.plot(range(1, len(fold_precision)+1), fold_precision, label=f'Fold {i+1}')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Precision')
        plt.title('Validation Precision per Epoch for Each Fold')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{save_dir}/precision_curves.png")
        
        print(f"指标曲线已保存至 {save_dir}")
    
    # 保存所有指标数据到CSV
    if save_dir:
        for fold in range(k):
            # 创建训练指标的DataFrame
            train_df = pd.DataFrame(all_fold_train_metrics[fold])
            train_df.to_csv(f"{save_dir}/fold_{fold}/all_train_metrics.csv", index_label='epoch')
            
            # 创建验证指标的DataFrame
            val_df = pd.DataFrame(all_fold_val_metrics[fold])
            val_df.to_csv(f"{save_dir}/fold_{fold}/all_val_metrics.csv", index_label='epoch')
            
            # 创建合并的指标DataFrame
            combined_df = pd.concat([
                train_df.add_prefix('train_'),
                val_df.add_prefix('val_')
            ], axis=1)
            combined_df.to_csv(f"{save_dir}/fold_{fold}/all_metrics.csv", index_label='epoch')
        
        print(f"所有指标数据已保存至 {save_dir}/fold_*/all_metrics.csv")
    
    return avg_metrics, fold_metrics, best_epochs, all_fold_train_metrics, all_fold_val_metrics

# 主函数
def main():
    # 数据路径和保存路径
    data_path = 'F:\\PFAS-KANO\\骨架可视化\\KANO+pubchem+描述符.csv'
    save_dir = 'F:\\PFAS-KANO\\结果\\cross_validation_results-3种结合'
    
    # 加载数据集
    dataset = CSVDataset(data_path)
    
    # 创建保存结果的目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 执行十折交叉验证
    print("开始十折交叉验证...")
    avg_metrics, fold_metrics, best_epochs, train_metrics, val_metrics = kfold_cross_validation(
        dataset,
        k=10,
        epochs=100,
        hidden_size=128,
        dropout=0.2,
        save_dir=save_dir
    )
    
    # 保存交叉验证结果
    results_df = pd.DataFrame(fold_metrics)
    results_df['best_epoch'] = best_epochs
    results_df.loc['mean'] = results_df.mean()
    results_df.loc['std'] = results_df.std()
    results_df.to_csv(f"{save_dir}/cross_validation_results.csv", index_label='fold')
    
    print(f"\n交叉验证结果已保存至 {save_dir}/cross_validation_results.csv")
    print(f"各折详细指标已保存至 {save_dir}/fold_*/all_metrics.csv")

if __name__ == "__main__":
    main()