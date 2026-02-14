import pandas as pd
import json
import os

# 路径
data_dir = r"d:\code\dryRun\0214\dryRun\data"
analysis_results_path = os.path.join(data_dir, "analysis_results.json")
time_series_path = os.path.join(data_dir, "experience_cashflow.xlsx")
unstructured_path = os.path.join(data_dir, "非结构化数据.xlsx")

# 读取 JSON
try:
    with open(analysis_results_path, "r", encoding="utf-8") as f:
        results = json.load(f)
except FileNotFoundError:
    print(f"File not found: {analysis_results_path}")
    exit(1)

# 读取时序数据
try:
    df_ts = pd.read_excel(time_series_path)
    # 只保留特定列
    cols = ['企业名称', 'Month_1', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6']
    df_ts = df_ts[[c for c in cols if c in df_ts.columns]]
    # 取前5行
    ts_sample = df_ts.head(5).where(pd.notnull(df_ts), None).to_dict(orient="records")
    results["time_series_sample"] = ts_sample
    print("Added time_series_sample")
except Exception as e:
    print(f"Error reading time series: {e}")

# 读取非结构化数据
try:
    df_un = pd.read_excel(unstructured_path)
    # 取前5行
    un_sample = df_un.head(5).where(pd.notnull(df_un), None).to_dict(orient="records")
    results["unstructured_sample"] = un_sample
    print("Added unstructured_sample")
except Exception as e:
    print(f"Error reading unstructured data: {e}")

# 保存 JSON
with open(analysis_results_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("Done")