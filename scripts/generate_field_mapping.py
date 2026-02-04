#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成字段映射文件，从Excel文档中提取字段的中文名称
"""

import pandas as pd
import json
import os

# 确保数据目录存在
os.makedirs('data', exist_ok=True)

# Excel文件路径
excel_file = 'data/贷款数据集字段翻译文档.xlsx'

# 读取Excel文件
try:
    print(f"正在读取Excel文件: {excel_file}")
    df = pd.read_excel(excel_file)
    print(f"Excel文件读取成功，行数: {len(df)}")
except Exception as e:
    print(f"Excel文件读取失败: {e}")
    exit(1)

# 生成字段映射
field_mapping = {}
for index, row in df.iterrows():
    # 假设第一列是英文名称，第二列是中文名称
    if len(row) >= 2:
        english_name = row.iloc[0]
        chinese_name = row.iloc[1]
        if pd.isna(chinese_name):
            chinese_name = '--'
        field_mapping[english_name] = chinese_name

# 保存字段映射到JSON文件
output_file = 'data/field_mapping.json'
try:
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(field_mapping, f, ensure_ascii=False, indent=2)
    print(f"字段映射已保存到: {output_file}")
    print(f"映射数量: {len(field_mapping)}")
except Exception as e:
    print(f"保存字段映射失败: {e}")
    exit(1)

print("字段映射生成完成！")
