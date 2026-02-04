import pandas as pd
from sklearn.model_selection import train_test_split
import os

# 获取项目根目录
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 读取数据
def split_data(input_file, train_output, test_output, test_size=0.3):
    print(f"读取数据文件: {input_file}")
    df = pd.read_csv(input_file)
    
    print(f"原始数据形状: {df.shape}")
    
    # 自动检测目标列
    target_columns = ['is_risky', 'label', 'target', 'TARGET', 'risk_flag', 'is_default', 'default']
    target_col = None
    
    for col in target_columns:
        if col in df.columns:
            target_col = col
            break
    
    if not target_col:
        print("错误: 未找到目标列")
        return False
    
    print(f"使用目标列: {target_col}")
    print(f"原始标签分布:")
    print(df[target_col].value_counts())
    
    # 分层抽样，保持原有正负比例
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        stratify=df[target_col], 
        random_state=42
    )
    
    print(f"\n训练集形状: {train_df.shape}")
    print(f"训练集标签分布:")
    print(train_df[target_col].value_counts())
    
    print(f"\n测试集形状: {test_df.shape}")
    print(f"测试集标签分布:")
    print(test_df[target_col].value_counts())
    
    # 保存文件
    train_df.to_csv(train_output, index=False, encoding='utf-8')
    test_df.to_csv(test_output, index=False, encoding='utf-8')
    
    print(f"\n数据分割完成！")
    print(f"训练集已保存到: {train_output}")
    print(f"测试集已保存到: {test_output}")
    return True

if __name__ == '__main__':
    # 使用相对路径
    input_file = os.path.join(base_dir, 'data', 'application_data.csv')
    train_output = os.path.join(base_dir, 'data', 'train.csv')
    test_output = os.path.join(base_dir, 'data', 'test.csv')
    
    print(f"项目根目录: {base_dir}")
    print(f"输入文件路径: {input_file}")
    print(f"训练输出路径: {train_output}")
    print(f"测试输出路径: {test_output}")
    
    # 验证文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 输入文件不存在: {input_file}")
    else:
        split_data(
            input_file=input_file,
            train_output=train_output,
            test_output=test_output,
            test_size=0.3
        )
