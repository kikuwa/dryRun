import pandas as pd
import json
import os
import random
import logging
import requests
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)

class CotSynthesisService:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self.data_file = os.path.join(self.base_dir, 'data', 'SelectData.xlsx')
        self.cache_file = os.path.join(self.base_dir, 'data', 'cot_cache.json')
        self.prompt_ori_file = os.path.join(self.base_dir, 'data', 'prompt_ori.txt')
        self.prompt_better_file = os.path.join(self.base_dir, 'data', 'prompt_better.txt')
        self.prompt_opt_file = os.path.join(self.base_dir, 'data', 'prompt_opt.txt')  
        self.api_key = ""
        self.api_url = "http://100.100.135.172:8081/v1/chat/completions"
        self.model = "qwen"
        self.feature_translation_file = os.path.join(self.base_dir, 'data', '贷款数据集字段翻译文档.xlsx')
        self.results_file = os.path.join(self.base_dir, 'data', 'results.json')
        self.feature_translations = self._load_feature_translations()

        self._ensure_cache_exists()

    def _load_feature_translations(self) -> Dict[str, str]:
        try:
            df = pd.read_excel(self.feature_translation_file)
            
            # Make the column name check more robust
            name_col, chinese_name_col = None, None
            
            if '字段名' in df.columns and '中文名称' in df.columns:
                name_col, chinese_name_col = '字段名', '中文名称'
            elif 'feature' in df.columns and '中文' in df.columns:
                name_col, chinese_name_col = 'feature', '中文'
            elif 'English' in df.columns and 'Chinese' in df.columns:
                name_col, chinese_name_col = 'English', 'Chinese'
            elif '英文字段名' in df.columns and '中文字段名' in df.columns:
                name_col, chinese_name_col = '英文字段名', '中文字段名'
            
            if name_col and chinese_name_col:
                return pd.Series(df[chinese_name_col].values, index=df[name_col]).to_dict()
            else:
                raise ValueError("Could not find expected columns for feature translations.")

        except Exception as e:
            logger.error(f"Error loading feature translations: {e}")
            return {}

    def _ensure_cache_exists(self):
        if not os.path.exists(self.cache_file):
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump({}, f)

    def get_data_sample(self, index: Optional[int] = None) -> Dict[str, Any]:
        try:
            df = pd.read_excel(self.data_file)
            if df.empty:
                raise ValueError("Data file is empty")

            # Always select a random sample, ignoring any provided index.
            index = random.randint(0, len(df) - 1)
            
            row = df.iloc[index]
            data_dict = row.to_dict()
            
            # Debug log to check raw values
            logger.info(f"Raw data sample for index {index}: {data_dict}")

            # CRITICAL: Remove the target variable to prevent data leakage to the model
            data_dict.pop('TARGET', None)
            data_dict.pop('target', None)

            # Prepare the list of features with Chinese names
            data_list = []
            offset = random.randint(0, 1500)
            for key, value in data_dict.items():
                # Clean value
                if pd.isna(value):
                    display_value = None
                elif hasattr(value, 'item'):
                    display_value = value.item()
                else:
                    display_value = value
                
                data_list.append({
                    "key": key,
                    "value": display_value,
                    "name": self.feature_translations.get(key, "")
                })

            return {"data": data_list, "index": index + offset, "total": len(df)}
        except (FileNotFoundError, ValueError, IndexError) as e:
            logger.error(f"Error getting data sample: {e}")
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"An unexpected error occurred in get_data_sample: {e}")
            return {"error": "An unexpected error occurred"}

    def _read_file(self, filepath: str) -> str:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading file {filepath}: {e}")
            return ""

    def _call_llm(self, messages: list) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False
        }
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, verify=False)
            if response.status_code == 200:
                result = response.json()
                choice = result['choices'][0]['message']
                content = choice.get('content', '')
                reasoning = choice.get('reasoning_content', '')
                
                return {"content": content, "reasoning": reasoning}
            else:
                logger.error(f"LLM API Error: {response.status_code} - {response.text}")
                return {"error": f"Error: {response.status_code} - {response.text}"}
        except Exception as e:
            logger.error(f"Error calling LLM: {str(e)}")
            return {"error": f"Error calling LLM: {str(e)}"}

    def generate_cot(self, data_index: int, cot_type: str, custom_prompt: Optional[str] = None, expert_advice: Optional[str] = None) -> Dict[str, Any]:
        import time
        # For expert CoT, the cache key should be based on the advice to ensure uniqueness
        cache_key = f"{data_index}_{cot_type}"
        if cot_type == 'expert':
            cache_key = f"{data_index}_{cot_type}_{hash(expert_advice)}"

        cache = self._load_cache()
        if cache_key in cache:
            return {"content": cache[cache_key], "timestamp": time.time(), "cached": True}

        # Get Data
        data_result = self.get_data_sample(None)
        if "error" in data_result:
            return {"error": data_result["error"]}
        data = data_result["data"]

        # Prepare Prompt
        if cot_type == "expert":
            if not expert_advice:
                return {"error": "Expert advice is required for expert CoT"}
            optimized_prompt_result = self.optimize_prompt_with_expert(expert_advice)
            if "error" in optimized_prompt_result:
                return optimized_prompt_result
            prompt_template = optimized_prompt_result["optimized_prompt"]
        elif custom_prompt:
            prompt_template = custom_prompt
        elif cot_type == "original":
            prompt_template = self._read_file(self.prompt_ori_file)
        elif cot_type == "optimized":
            prompt_template = self._read_file(self.prompt_better_file)
        else:
            return {"error": "Invalid COT type"}

        # Fill Prompt with Data
        try:
            data_dict = {item['key']: item['value'] for item in data}
            mentioned_features = {key: data_dict[key] for key in data_dict if f"{{{key}}}" in prompt_template}

            if mentioned_features:
                # Use all data to avoid KeyError if prompt has other placeholders
                full_prompt = prompt_template.format(**data_dict)
            else:
                # If no {feature} placeholders, append the report, but respect "## target"
                data_str = "\n".join([f"{item['key']} ({item['name']}): {item['value']}" for item in data if item['value'] is not None])
                report_str = f"\n\n【企业指标数据报告】\n{data_str}"
                
                if "## target" in prompt_template:
                    parts = prompt_template.split("## target", 1)
                    # Append report to the first part, before the "## target" tag
                    parts[0] = parts[0].rstrip() + report_str
                    full_prompt = "## target".join(parts)
                else:
                    # Fallback to appending at the end if "## target" is not present
                    full_prompt = prompt_template.rstrip() + report_str
            
        except KeyError as e:
            return {"error": f"Missing feature in data for prompt formatting: {e}"}
        except Exception as e:
            return {"error": f"Error preparing prompt: {str(e)}"}

        # Call LLM
        messages = [{"role": "user", "content": full_prompt}]
        llm_result = self._call_llm(messages)

        if "error" in llm_result:
            return {"error": llm_result["error"]}

        # Combine reasoning and content for the final result
        final_content = llm_result['content']
        # if llm_result.get('reasoning'):
        #     final_content = f"【推理过程】\n{llm_result['reasoning']}\n\n【结论】\n{llm_result['content']}"

        # Cache result
        self._save_to_cache(cache_key, final_content)

        return {"content": final_content, "timestamp": time.time(), "cached": False}

    def generate_expert_cot_from_results(self, loan_id: str, expert_advice: str) -> Dict[str, Any]:
        import time
        try:
            # 1. Retrieve JSON object from results.json
            if not os.path.exists(self.results_file):
                return {"error": "Results file not found"}
            
            with open(self.results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            target_entry = None
            for item in results:
                if str(item.get('id')) == str(loan_id):
                    target_entry = item
                    break
            
            if not target_entry:
                return {"error": f"No matching data found for Loan ID: {loan_id}"}
            
            # 2. Extract prompt
            original_prompt = target_entry.get('prompt', '')
            if not original_prompt:
                return {"error": "Prompt not found in the matching data"}
            
            # 3. Insert expert advice
            insertion_point = "【任务指令】"
            if insertion_point not in original_prompt:
                # Fallback if the tag is not found, just prepend
                modified_prompt = f"【规则】{expert_advice}\n\n{original_prompt}"
            else:
                parts = original_prompt.split(insertion_point)
                modified_prompt = f"{parts[0]}【规则】{expert_advice}\n\n{insertion_point}{parts[1]}"
            
            # 4. Call LLM (Step 5 in requirements)
            messages = [{"role": "user", "content": modified_prompt}]
            llm_result = self._call_llm(messages)
            
            if "error" in llm_result:
                return {"error": llm_result["error"]}
            
            # Combine reasoning and content
            final_content = llm_result['content']
            # if llm_result.get('reasoning'):
            #     final_content = f"【推理过程】\n{llm_result['reasoning']}\n\n【结论】\n{llm_result['content']}"
            
            return {"content": final_content, "timestamp": time.time(), "cached": False}
            
        except Exception as e:
            logger.error(f"Error in generate_expert_cot_from_results: {e}")
            return {"error": str(e)}

    def optimize_prompt_with_expert(self, expert_input: str) -> Dict[str, Any]:
        prompt_better = self._read_file(self.prompt_better_file)
        prompt_opt = self._read_file(self.prompt_opt_file)
        
        # Combine inputs
        full_prompt = f"{prompt_opt}\n\n{prompt_better}\n\n【专家建议】\n{expert_input}"
        
        messages = [{"role": "user", "content": full_prompt}]
        llm_result = self._call_llm(messages)

        if "error" in llm_result:
            return {"error": llm_result["error"]}
        
        return {"optimized_prompt": llm_result["content"]}

    def _load_cache(self) -> Dict:
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}

    def _save_to_cache(self, key: str, value: str):
        cache = self._load_cache()
        cache[key] = value
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)

    def clear_cache(self):
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump({}, f)
