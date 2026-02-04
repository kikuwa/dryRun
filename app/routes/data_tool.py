from flask import Blueprint, request, jsonify, current_app, send_from_directory, render_template
import pandas as pd
import numpy as np
import subprocess
import os
import sys

data_tool_bp = Blueprint('data_tool', __name__)

ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Pages
@data_tool_bp.route('/')
def index():
    return render_template('data_tool/dataset.html', active_page='dataset')

@data_tool_bp.route('/feature_selection')
def feature_selection():
    return render_template('data_tool/feature_selection.html', active_page='feature_selection')



# APIs
@data_tool_bp.route('/upload_file', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': '没有文件上传'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    if file and allowed_file(file.filename):
        # Script expects data/train.csv
        file_path = os.path.join(current_app.config['PROJECT_ROOT'], 'data', 'uploads', 'train.csv')
        # Also copy to data/train.csv as scripts use it
        file.save(file_path)
        import shutil
        shutil.copy(file_path, os.path.join(current_app.config['PROJECT_ROOT'], 'data', 'train.csv'))
        return jsonify({'success': '文件上传成功'})
    
    return jsonify({'error': '只允许上传CSV文件'}), 400

@data_tool_bp.route('/run_feature_selection', methods=['POST'])
def run_feature_selection():
    try:
        print("Starting run_feature_selection...")
        
        import time
        start_time = time.time()
        print(f"Start time: {start_time}")
        
        # Import and call run_feature_selection from fast_model.py
        print("Importing fast_model...")
        # 使用相对导入
        import importlib.util
        spec = importlib.util.spec_from_file_location("fast_model", os.path.join(current_app.config['PROJECT_ROOT'], 'app', 'services', 'data_core', 'fast_model.py'))
        fast_model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fast_model)
        
        print("Running fast_model...")
        feature_importance_df = fast_model.run_feature_selection()
        
        end_time = time.time()
        training_time = round(end_time - start_time, 2)
        print(f"End time: {end_time}")
        print(f"Training time: {training_time} seconds")
        
        if feature_importance_df is not None:
            # Process feature importance data
            feature_importance = []
            fi_df = feature_importance_df
                
            # Load Chinese feature names from Excel file
            excel_path = os.path.join(current_app.config['PROJECT_ROOT'], 'data', '贷款数据集字段翻译文档.xlsx')
            chinese_names = {}
            if os.path.exists(excel_path):
                try:
                    df_excel = pd.read_excel(excel_path)
                    # Assuming Excel has columns 'English' and 'Chinese' or similar
                    # Adjust column names based on actual Excel structure
                    if 'English' in df_excel.columns and 'Chinese' in df_excel.columns:
                        for _, row in df_excel.iterrows():
                            chinese_names[row['English']] = row['Chinese']
                    elif '字段名' in df_excel.columns and '中文名称' in df_excel.columns:
                        for _, row in df_excel.iterrows():
                            chinese_names[row['字段名']] = row['中文名称']
                    elif 'feature' in df_excel.columns and '中文' in df_excel.columns:
                        for _, row in df_excel.iterrows():
                            chinese_names[row['feature']] = row['中文']
                    elif '英文字段名' in df_excel.columns and '中文字段名' in df_excel.columns:
                        for _, row in df_excel.iterrows():
                            chinese_names[row['英文字段名']] = row['中文字段名']
                except Exception as e:
                    print(f"Error reading Excel file: {e}")
                
            # Add Chinese names to features
            feature_importance = []
            for _, row in fi_df.iterrows():
                feature_dict = row.to_dict()
                feature_dict['chinese_name'] = chinese_names.get(row['feature'], '')
                feature_importance.append(feature_dict)
            
            # Load original dataset to get feature count
            original_feature_count = 0
            sample_count = 0
            positive_ratio = 0
            train_path = os.path.join(current_app.config['PROJECT_ROOT'], 'data', 'train.csv')
            if os.path.exists(train_path):
                df_train = pd.read_csv(train_path, nrows=10000)
                sample_count = len(df_train)
                # Get original feature count (excluding target)
                if 'TARGET' in df_train.columns:
                    original_feature_count = len(df_train.columns) - 1
                    positive_count = df_train['TARGET'].sum()
                    if sample_count > 0:
                        positive_ratio = round((positive_count / sample_count) * 100, 2)
                elif 'target' in df_train.columns:
                    original_feature_count = len(df_train.columns) - 1
                    positive_count = df_train['target'].sum()
                    if sample_count > 0:
                        positive_ratio = round((positive_count / sample_count) * 100, 2)
                else:
                    original_feature_count = len(df_train.columns)
            
            # Calculate feature compression rate
            feature_compression_rate = 0
            if original_feature_count > 0:
                feature_compression_rate = round((50 / original_feature_count) * 100, 2)
            
            return jsonify({
                'success': True,
                'log': '',
                'stderr': '',
                'original_feature_count': original_feature_count,
                'sample_count': sample_count,
                'positive_ratio': positive_ratio,
                'feature_compression_rate': feature_compression_rate,
                'training_time': training_time,
                'top_features': feature_importance[:50], # Return top 50 features
            })
        else:
            return jsonify({
                'success': False,
                'error': '模型运行失败',
                'log': '',
                'stderr': '模型运行失败'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@data_tool_bp.route('/download_feature_importance')
def download_feature_importance():
    results_path = os.path.join(current_app.config['PROJECT_ROOT'], 'app', 'services', 'data_core', 'data', 'results', 'pu_learning')
    return send_from_directory(results_path, "feature_importance.csv", as_attachment=True)


