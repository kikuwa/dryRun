from flask import Blueprint, render_template, redirect, url_for, jsonify, request
from app.services.risk_cot.cot_synthesis_service import CotSynthesisService
import json
import os

views_bp = Blueprint('risk_cot_views', __name__)
cot_service = CotSynthesisService()

@views_bp.route('/')
def index():
    return redirect(url_for('risk_cot_views.inference'))

@views_bp.route('/prompt_design')
def prompt_design():
    return render_template('risk_cot/prompt_design.html', active_page='prompt_design')

@views_bp.route('/model_diff')
def model_diff():
    return render_template('risk_cot/modelDiff.html', active_page='model_diff')

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

@views_bp.route('/cot_synthesis')
def cot_synthesis():
    return render_template('risk_cot/cot_synthesis.html', active_page='cot_data_synthesis')

@views_bp.route('/api/cot/get_sample', methods=['GET'])
def get_cot_sample():
    index = request.args.get('index', type=int)
    try:
        result = cot_service.get_data_sample(index)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@views_bp.route('/api/cot/generate', methods=['POST'])
def generate_cot():
    data = request.json
    index = data.get('index')
    cot_type = data.get('cot_type') # 'original', 'optimized', 'expert'
    custom_prompt = data.get('custom_prompt') # For expert mode
    
    if index is None or not cot_type:
        return jsonify({'error': 'Missing parameters'}), 400
        
    try:
        result = cot_service.generate_cot(index, cot_type, custom_prompt)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@views_bp.route('/api/cot/optimize_prompt', methods=['POST'])
def optimize_prompt():
    data = request.json
    expert_advice = data.get('expert_advice')
    
    if not expert_advice:
        return jsonify({'error': 'Missing expert advice'}), 400
        
    try:
        result = cot_service.optimize_prompt_with_expert(expert_advice)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@views_bp.route('/api/cot/clear_cache', methods=['POST'])
def clear_cot_cache():
    try:
        cot_service.clear_cache()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
