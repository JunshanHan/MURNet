
import pandas as pd

def find_smiles_index(csv_path, target_smiles):
    """在CSV文件中找到特定SMILES的索引"""
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    # 检查是否包含 'smiles' 列
    if "Scaffold_SMILES" not in df.columns:
        raise ValueError("CSV 文件缺少 'smiles' 列")
    
    # 查找目标 SMILES 的索引
    index_list = df[df["Scaffold_SMILES"] == target_smiles].index.tolist()
    
    # 如果找到了索引，则返回索引列表，否则返回空列表
    if index_list:
        return index_list
    else:
        print(f"未在CSV中找到 SMILES: {target_smiles}")
        return []

def get_rows_from_indices(csv_path, indices):
    """根据索引列表从CSV文件中获取对应的行"""
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    # 获取指定索引的行
    rows = df.iloc[indices]
    return rows

def save_rows_to_csv(rows, output_csv_path):
    """将提取的行保存到CSV文件"""
    rows.to_csv(output_csv_path, index=False)
    print(f"行已保存至: {output_csv_path}")

# 示例：查找特定SMILES的索引并在另一个CSV文件中获取对应的行
csv_file_1 = "F:\\PFAS-KANO\\骨架可视化\\scaffold_counts.csv"  # 替换为你的第一个CSV文件路径
csv_file_2 = "F:\\PFAS-KANO\\骨架可视化\\KANO+pubchem.csv"  # 替换为你的第二个CSV文件路径
target_smiles = "c1ccc(Nc2ccnc3ncnn23)cc1"  # 替换为你要查找的SMILES
#c1ccc(Nc2ccnc3ncnn23)cc1、c1ccccc1、c1ccc(-c2ccccc2)cc1、c1ccc(Nc2ccnc3ccc(N4CCNCC4)cc23)cc1、O=c1ccn(-c2ccccc2)c2cc(N3CC[NH2+]CC3)ccc12

# 获取第一个CSV文件中SMILES的索引
indices = find_smiles_index(csv_file_1, target_smiles)

if indices:
    # 使用这些索引从第二个CSV文件中获取对应的行
    corresponding_rows = get_rows_from_indices(csv_file_2, indices)
    
    # 保存提取的行到新的CSV文件
    output_csv_file = "F:\\PFAS-KANO\\骨架可视化\\pubchem\\pubchem_21.csv"  # 输出文件路径
    save_rows_to_csv(corresponding_rows, output_csv_file)
else:
    print("没有找到目标SMILES的索引。")
