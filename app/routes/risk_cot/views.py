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
        data_file = os.path.join(base_dir, 'data', 'SBAcase.11.13.17.csv')
        analyzer = DataAnalyzer(data_file)
        analyzer.run_analysis()
    
    # 读取分析结果
    try:
        with open(analysis_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
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
    expert_advice = data.get('expert_advice') # For expert mode from our UI
    
    if index is None or not cot_type:
        return jsonify({'error': 'Missing parameters'}), 400
        
    try:
        result = cot_service.generate_cot(index, cot_type, custom_prompt, expert_advice)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@views_bp.route('/api/cot/generate_expert_from_results', methods=['POST'])
def generate_expert_from_results():
    data = request.json
    loan_id = data.get('loan_id')
    expert_advice = data.get('expert_advice')
    
    if not loan_id or not expert_advice:
        return jsonify({'error': 'Missing loan_id or expert_advice'}), 400
        
    try:
        result = cot_service.generate_expert_cot_from_results(loan_id, expert_advice)
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

@views_bp.route('/api/cot/fetch_expert_response', methods=['POST'])
def fetch_expert_response():
    import time
    data = request.json
    loan_id = data.get('loan_id')
    
    if not loan_id:
        return jsonify({'error': 'Missing loan_id'}), 400
        
    # 使用相对路径读取 results.json
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    results_file = os.path.join(base_dir, 'data', 'results.json')
    
    if not os.path.exists(results_file):
        return jsonify({'error': 'Results file not found'}), 404
        
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
            
        # 查找匹配的 id
        target_response = None
        for item in results:
            if str(item.get('id')) == str(loan_id):
                target_response = item.get('response')
                break
        
        if target_response:
            # 延迟 30 秒
            time.sleep(30)
            return jsonify({
                'content': target_response,
                'timestamp': time.time(),
                'cached': False
            })
        else:
            return jsonify({'error': 'Loan ID not found'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@views_bp.route('/api/cot/fetch_original_response', methods=['POST'])
def fetch_original_response():
    import time
    data = request.json
    loan_id = data.get('loan_id')
    
    if not loan_id:
        return jsonify({'error': 'Missing loan_id'}), 400
        
    # 使用相对路径读取 results_ori.json
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    results_file = os.path.join(base_dir, 'data', 'results_ori.json')
    
    if not os.path.exists(results_file):
        return jsonify({'error': 'Results file not found'}), 404
        
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
            
        # 查找匹配的 id
        target_response = None
        for item in results:
            if str(item.get('id')) == str(loan_id):
                target_response = item.get('response')
                break
        
        if target_response:
            # 延迟 15 秒
            time.sleep(15)
            return jsonify({
                'content': target_response,
                'timestamp': time.time(),
                'cached': False
            })
        else:
            return jsonify({'error': 'Loan ID not found'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@views_bp.route('/api/cot/clear_cache', methods=['POST'])
def clear_cot_cache():
    try:
        cot_service.clear_cache()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
