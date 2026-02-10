from flask import Blueprint, request, jsonify, current_app, send_from_directory, render_template
import pandas as pd
import numpy as np
import subprocess
import os
import sys
import importlib.util

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

        # Dynamically import and run the feature selection script
        spec = importlib.util.spec_from_file_location(
            "fast_model",
            os.path.join(current_app.config['PROJECT_ROOT'], 'app', 'services', 'data_core', 'fast_model.py')
        )
        fast_model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fast_model)
        
        results = fast_model.run_feature_selection()
        
        end_time = time.time()
        training_time = round(end_time - start_time, 2)

        if results and "feature_importance_df" in results:
            fi_df = results["feature_importance_df"]
            
            # Convert DataFrame to list of dicts for JSON response
            top_features = fi_df.head(30).to_dict(orient='records')

            # Calculate feature compression rate
            original_feature_count = results.get("original_feature_count", 0)
            feature_compression_rate = 0
            if original_feature_count > 0:
                feature_compression_rate = round((len(top_features) / original_feature_count) * 100, 2)

            return jsonify({
                'success': True,
                'log': '', 
                'stderr': '',
                'original_feature_count': original_feature_count,
                'sample_count': results.get("sample_count", 0),
                'positive_ratio': results.get("positive_ratio", 0),
                'feature_compression_rate': feature_compression_rate,
                'training_time': training_time,
                'top_features': top_features,
            })
        else:
            return jsonify({
                'success': False,
                'error': '模型运行失败或未返回预期的结果',
                'log': '',
                'stderr': '模型运行失败'
            })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        })

@data_tool_bp.route('/download_feature_importance')
def download_feature_importance():
    results_path = os.path.join(current_app.config['PROJECT_ROOT'], 'app', 'services', 'data_core', 'data', 'results', 'pu_learning')
    return send_from_directory(results_path, "feature_importance.csv", as_attachment=True)


