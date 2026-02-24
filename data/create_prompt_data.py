import csv
import json
import pandas as pd
import os

def _load_translation_map(excel_path: str) -> dict:
    """
    从Excel文件加载中英文特征对照表，使用更灵活的方式。
    """
    try:
        df = pd.read_excel(excel_path, header=0) # 确保第一行被当作表头
        if df.shape[1] < 2:
            print("警告: Excel文件少于两列，无法建立映射。")
            return {}
        
        # 直接将第一列作为key，第二列作为value
        english_col = df.columns[0]
        chinese_col = df.columns[1]
        
        # 移除key或value为空的行
        df.dropna(subset=[english_col, chinese_col], inplace=True)
        
        return pd.Series(df[chinese_col].values, index=df[english_col]).to_dict()

    except FileNotFoundError:
        print(f"警告: 找不到中英文对照文件 {excel_path}。将使用原始英文特征名。")
        return {}
    except Exception as e:
        print(f"读取Excel文件时出错: {e}")
        return {}

def create_prompt_json(prompt_file, csv_file, excel_file, output_file, num_records):
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            base_prompt = f.read()
    except FileNotFoundError:
        print(f"错误: 找不到 prompt 文件 at {prompt_file}")
        return

    translation_map = _load_translation_map(excel_file)
    if not translation_map:
        print("未能成功加载翻译映射，将使用英文原名。")

    prompts_list = []
    try:
        with open(csv_file, 'r', encoding='utf-8-sig') as f_csv:
            reader = csv.reader(f_csv)
            headers = next(reader)
            for i, row in enumerate(reader):
                if i >= num_records:
                    break
                
                record_parts = []
                for header, value in zip(headers, row):
                    if header not in ['Selected', 'Default', 'xx']:
                        display_header = translation_map.get(header, header)
                        record_parts.append(f"{display_header}:{value}")
                
                record_str = ", ".join(record_parts)
                final_prompt = f"{base_prompt}\\n\\n【企业贷款申请数据报告】\\n{record_str}"
                prompts_list.append({"prompt": final_prompt})

    except FileNotFoundError:
        print(f"错误: 找不到 CSV 文件 at {csv_file}")
        return

    try:
        with open(output_file, 'w', encoding='utf-8') as f_out:
            json.dump(prompts_list, f_out, ensure_ascii=False, indent=4)
    except IOError as e:
        print(f"写入文件时出错: {e}")
        return
    
    print(f"成功创建 {output_file} 文件，其中包含 {len(prompts_list)} 条记录。")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    prompt_file_path = os.path.join(BASE_DIR, 'optimized_prompt.txt')
    csv_file_path = os.path.join(BASE_DIR, 'SBAcase.11.13.17.csv')
    excel_translation_file = os.path.join(BASE_DIR, '中英文对照.xlsx')
    output_file_path = os.path.join(BASE_DIR, 'prompt_data.json')
    records_to_process = 5
    
    create_prompt_json(prompt_file_path, csv_file_path, excel_translation_file, output_file_path, records_to_process)
