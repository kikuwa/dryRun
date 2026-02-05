from flask import Blueprint, render_template, redirect, url_for, jsonify
import json
import os

views_bp = Blueprint('risk_cot_views', __name__)

@views_bp.route('/')
def index():
    return redirect(url_for('risk_cot_views.inference'))

@views_bp.route('/prompt_design')
def prompt_design():
    return render_template('risk_cot/prompt_design.html', active_page='prompt_design')

@views_bp.route('/model_diff')
def model_diff():
    return render_template('risk_cot/modelDiff.html')

@views_bp.route('/inference')
def inference():
    return render_template('risk_cot/inference.html', active_page='cot_synthesis')

@views_bp.route('/distillation')
def distillation():
    return render_template('risk_cot/distillation.html', active_page='model_distillation')

@views_bp.route('/multi_source_data')
def multi_source_data():
    return render_template('risk_cot/multi_source_data.html', active_page='multi_source_data')

@views_bp.route('/get_analysis_results')
def get_analysis_results():
    """
    获取数据分析结果
    """
    # 使用相对路径
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    analysis_file = os.path.join(base_dir, 'data', 'analysis_results.json')
    
    if not os.path.exists(analysis_file):
        # 如果分析结果文件不存在，运行分析脚本
        from app.services.data_core.data_analyzer import DataAnalyzer
        data_file = os.path.join(base_dir, 'data', 'application_data.csv')
        analyzer = DataAnalyzer(data_file)
        analyzer.run_analysis()
    
    # 读取分析结果
    try:
        with open(analysis_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 添加数据预览
        import pandas as pd
        data_file = os.path.join(base_dir, 'data', 'application_data.csv')
        df = pd.read_csv(data_file)
        preview = df.head(5).to_dict('records')
        
        # 处理NaN值，替换为null
        def replace_nan(obj):
            if isinstance(obj, float) and pd.isna(obj):
                return None
            elif isinstance(obj, dict):
                return {k: replace_nan(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_nan(item) for item in obj]
            else:
                return obj
        
        # 处理所有数据中的NaN值
        data = replace_nan(data)
        preview = replace_nan(preview)
        data['preview'] = preview
        
        return jsonify(data)
    except Exception as e:
        print(f"Error reading analysis results: {e}")
        return jsonify({}), 500
