#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据分析师脚本
用于分析application_data.csv文件的全面数据特征

功能包括：
1. 数据类型统计及占比分析
2. 数据质量评估
3. 缺失值分布分析
4. IQR(四分位距)算法分析数据分布特征
5. 异常值检测
6. 基本统计描述
"""

import pandas as pd
import numpy as np
import os
import json
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class DataAnalyzer:
    """数据分析师类"""
    
    def __init__(self, file_path: str):
        """
        初始化数据分析师
        
        Args:
            file_path: CSV文件路径
        """
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.file_path = file_path
        self.data = None
        self.analysis_results = {}
        self.field_mapping = self.load_field_mapping()
    
    def load_field_mapping(self) -> Dict[str, str]:
        """
        加载字段映射，从Excel文档中提取字段的中文名称
        
        Returns:
            Dict: 字段映射，键为英文名称，值为中文名称
        """
        field_mapping = {}
        excel_file = os.path.join(self.project_root, "data", "中英文对照.xlsx")
        
        if os.path.exists(excel_file):
            try:
                logger.info(f"正在加载字段映射文件: {excel_file}")
                df = pd.read_excel(excel_file)
                logger.info(f"字段映射文件加载成功，行数: {len(df)}")
                logger.info(f"Excel文件列数: {len(df.columns)}")
                logger.info(f"Excel文件列名: {list(df.columns)}")
                
                # 打印前5行数据，了解Excel文件的格式
                logger.info("Excel文件前5行数据:")
                logger.info(df.head())
                
                # 尝试使用列名来识别正确的列
                # 假设英文字段名在"英文字段名"列，中文字段名在"中文字段名"列
                if ('英文字段名' in df.columns and '中文字段名' in df.columns) or \
                   ('英文' in df.columns and '中文' in df.columns) or \
                   ('英文列名' in df.columns and '中文列名' in df.columns):
                    
                    if '英文字段名' in df.columns:
                        english_col = '英文字段名'
                        chinese_col = '中文字段名'
                    elif '英文' in df.columns:
                        english_col = '英文'
                        chinese_col = '中文'
                    else:
                        english_col = '英文列名'
                        chinese_col = '中文列名'

                    logger.info(f"使用列名'{english_col}'和'{chinese_col}'来提取字段映射")
                    for index, row in df.iterrows():
                        english_name = row[english_col]
                        chinese_name = row[chinese_col]
                        if isinstance(english_name, str) and len(english_name) > 0:
                            if pd.isna(chinese_name):
                                chinese_name = '--'
                            elif not isinstance(chinese_name, str):
                                chinese_name = str(chinese_name)
                            field_mapping[english_name] = chinese_name
                # 尝试使用列索引来识别正确的列
                elif len(df.columns) >= 4:
                    logger.info("使用列索引来提取字段映射，假设第3列是英文字段名，第4列是中文字段名")
                    for index, row in df.iterrows():
                        english_name = row.iloc[2]  # 第3列（索引2）是英文字段名
                        chinese_name = row.iloc[3]  # 第4列（索引3）是中文字段名
                        if isinstance(english_name, str) and len(english_name) > 0:
                            if pd.isna(chinese_name):
                                chinese_name = '--'
                            elif not isinstance(chinese_name, str):
                                chinese_name = str(chinese_name)
                            field_mapping[english_name] = chinese_name
                else:
                    logger.warning("无法识别Excel文件的格式，无法提取字段映射")
                
                logger.info(f"字段映射加载完成，映射数量: {len(field_mapping)}")
                # 打印前5个字段映射，了解映射的质量
                logger.info("前5个字段映射:")
                for i, (k, v) in enumerate(list(field_mapping.items())[:5]):
                    logger.info(f"{k}: {v}")
            except Exception as e:
                logger.error(f"字段映射加载失败: {e}")
        else:
            logger.warning(f"字段映射文件不存在: {excel_file}")
        
        return field_mapping
    
    def load_data(self) -> bool:
        """
        加载数据
        
        Returns:
            bool: 加载是否成功
        """
        try:
            logger.info(f"正在加载数据文件: {self.file_path}")
            self.data = pd.read_csv(self.file_path)
            logger.info(f"数据加载成功，形状: {self.data.shape}")
            return True
        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            return False
    
    def analyze_data_types(self) -> Dict[str, any]:
        """
        分析数据类型统计及占比
        
        Returns:
            Dict: 数据类型统计结果
        """
        logger.info("=== 数据类型统计分析 ===")
        
        # 获取数据类型统计
        dtype_counts = self.data.dtypes.value_counts()
        dtype_percentages = (dtype_counts / len(self.data.columns)) * 100
        
        # 获取每个数据类型对应的特征列表
        feature_by_dtype = self.data.columns.to_series().groupby(self.data.dtypes).apply(list)
        
        # 转换为字典格式
        dtype_stats = {}
        for dtype, count in dtype_counts.items():
            dtype_str = str(dtype)
            dtype_stats[dtype_str] = {
                "count": int(count),
                "percentage": f"{dtype_percentages[dtype]:.2f}%",
                "features": feature_by_dtype.get(dtype, [])
            }
        
        # 统计总数据类型数
        total_types = len(dtype_counts)
        
        result = {
            "total_types": total_types,
            "type_stats": dtype_stats
        }
        
        logger.info(f"总数据类型数: {total_types}")
        for dtype, stats in dtype_stats.items():
            logger.info(f"{dtype}: {stats['count']}个特征, 占比{stats['percentage']}")
        
        self.analysis_results["data_types"] = result
        return result
    
    def analyze_data_quality(self) -> Dict[str, any]:
        """
        分析数据质量和缺失值分布
        
        Returns:
            Dict: 数据质量分析结果
        """
        logger.info("=== 数据质量评估 ===")
        
        # 计算缺失值统计
        missing_values = self.data.isnull().sum()
        missing_percentages = (missing_values / len(self.data)) * 100
        
        # 按缺失率从高到低排序
        missing_df = pd.DataFrame({
            "count": missing_values,
            "percentage": missing_percentages
        })
        missing_df = missing_df.sort_values("percentage", ascending=False)
        
        # 计算数据完整性
        total_cells = self.data.size
        total_missing = missing_values.sum()
        completeness = ((total_cells - total_missing) / total_cells) * 100
        
        # 转换为字典格式
        missing_stats = []
        for feature, row in missing_df.iterrows():
                # 如果缺失率大于0但小于1%，则将其设置为一个较小的值（例如1%），以确保在UI上可见
                percentage_val = row["percentage"]
                if 0 < percentage_val < 1:
                    percentage_val = 1
                else:
                    percentage_val = round(percentage_val, 2)

                missing_stats.append({
                    "feature": feature,
                    "count": int(row["count"]),
                    "percentage": f"{row['percentage']:.2f}%",
                    "percentageValue": percentage_val,  # 使用调整后的百分比值
                    "total": len(self.data)
                })
        
        result = {
            "completeness": f"{completeness:.2f}%",
            "total_missing": int(total_missing),
            "missing_stats": missing_stats
        }
        
        logger.info(f"数据完整性: {result['completeness']}")
        logger.info(f"总缺失值数量: {result['total_missing']}")
        logger.info(f"有缺失值的特征数: {len(missing_stats)}")
        
        if len(missing_stats) > 0:
            logger.info("缺失率最高的10个特征:")
            for i, item in enumerate(missing_stats[:10]):
                logger.info(f"{i+1}. {item['feature']}: {item['count']}个缺失值, 占比{item['percentage']}")
        
        self.analysis_results["data_quality"] = result
        return result
    
    def analyze_iqr_and_outliers(self) -> Dict[str, any]:
        """
        使用IQR算法分析数据分布特征和异常值检测
        
        Returns:
            Dict: IQR分析和异常值检测结果
        """
        logger.info("=== IQR数据分布分析和异常值检测 ===")
        
        # 分析所有特征
        all_features = self.data.columns
        iqr_results = []
        outlier_results = []
        
        for feature in all_features:
            # 尝试分析特征，如果是数值型则计算统计量，否则跳过
            try:
                # 检查特征是否为数值型
                if pd.api.types.is_numeric_dtype(self.data[feature]):
                    # 计算基本统计量
                    Q1 = self.data[feature].quantile(0.25)
                    Q3 = self.data[feature].quantile(0.75)
                    IQR = Q3 - Q1
                    min_val = self.data[feature].min()
                    max_val = self.data[feature].max()
                    median = self.data[feature].median()
                    mean = self.data[feature].mean()
                    
                    # 计算异常值边界
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    # 检测异常值
                    outliers = self.data[(self.data[feature] < lower_bound) | (self.data[feature] > upper_bound)]
                    outlier_count = len(outliers)
                    outlier_percentage = (outlier_count / len(self.data)) * 100
                    
                    # 计算范围百分比（用于可视化）
                    value_range = max_val - min_val
                    if value_range > 0:
                        range_percentage = ((Q3 - Q1) / value_range) * 100
                    else:
                        range_percentage = 100
                    
                    # 添加IQR分析结果
                    iqr_results.append({
                        "feature": feature,
                        "min": float(min_val),
                        "q1": float(Q1),
                        "median": float(median),
                        "mean": float(mean),
                        "q3": float(Q3),
                        "max": float(max_val),
                        "iqr": float(IQR),
                        "range_percentage": round(range_percentage, 2)
                    })
                    
                    # 添加异常值检测结果
                    if outlier_count > 0:
                        outlier_results.append({
                            "feature": feature,
                            "outlier_count": outlier_count,
                            "outlier_percentage": f"{outlier_percentage:.2f}%"
                        })
                else:
                    # 对于非数值型特征，添加基本信息
                    iqr_results.append({
                        "feature": feature,
                        "min": None,
                        "q1": None,
                        "median": None,
                        "mean": None,
                        "q3": None,
                        "max": None,
                        "iqr": None,
                        "range_percentage": 0
                    })
            except Exception as e:
                # 如果分析失败，添加基本信息
                logger.error(f"分析特征 {feature} 时出错: {e}")
                iqr_results.append({
                    "feature": feature,
                    "min": None,
                    "q1": None,
                    "median": None,
                    "mean": None,
                    "q3": None,
                    "max": None,
                    "iqr": None,
                    "range_percentage": 0
                })
        
        # 按异常值比例排序
        outlier_results.sort(key=lambda x: float(x["outlier_percentage"].replace("%", "")) if "outlier_percentage" in x else 0, reverse=True)
        
        result = {
            "iqr_analysis": iqr_results,
            "outlier_detection": outlier_results
        }
        
        logger.info(f"分析的特征数: {len(all_features)}")
        logger.info(f"检测到异常值的特征数: {len(outlier_results)}")
        
        if len(outlier_results) > 0:
            logger.info("异常值比例最高的5个特征:")
            for i, item in enumerate(outlier_results[:5]):
                logger.info(f"{i+1}. {item['feature']}: {item['outlier_count']}个异常值, 占比{item['outlier_percentage']}")
        
        self.analysis_results["iqr_and_outliers"] = result
        return result
    
    def analyze_basic_stats(self) -> Dict[str, any]:
        """
        分析基本统计描述
        
        Returns:
            Dict: 基本统计描述结果
        """
        logger.info("=== 基本统计描述 ===")
        
        # 只分析数值型特征
        numeric_features = self.data.select_dtypes(include=[np.number]).columns
        
        # 计算基本统计量
        basic_stats = self.data[numeric_features].describe().round(2)
        
        # 转换为字典格式
        stats_dict = {}
        for feature in numeric_features:
            feature_stats = {}
            for stat in basic_stats.index:
                if pd.notna(basic_stats.loc[stat, feature]):
                    feature_stats[stat] = float(basic_stats.loc[stat, feature])
            stats_dict[feature] = feature_stats
        
        result = {
            "numeric_features_count": len(numeric_features),
            "stats": stats_dict
        }
        
        logger.info(f"数值型特征数: {len(numeric_features)}")
        logger.info("基本统计描述已计算完成")
        
        self.analysis_results["basic_stats"] = result
        return result
    
    def generate_overview(self) -> Dict[str, any]:
        """
        生成数据概览
        
        Returns:
            Dict: 数据概览结果
        """
        logger.info("=== 数据概览 ===")
        
        overview = {
            "sample_count": len(self.data),
            "feature_count": len(self.data.columns),
            "completeness": self.analysis_results.get("data_quality", {}).get("completeness", "0%"),
            "data_type_count": self.analysis_results.get("data_types", {}).get("total_types", 0)
        }
        
        logger.info(f"样本量: {overview['sample_count']}")
        logger.info(f"特征数: {overview['feature_count']}")
        logger.info(f"数据完整性: {overview['completeness']}")
        logger.info(f"数据类型数: {overview['data_type_count']}")
        
        self.analysis_results["overview"] = overview
        return overview
    
    def generate_preview(self) -> List[Dict[str, any]]:
        """
        生成数据预览
        
        Returns:
            List[Dict]: 数据预览列表
        """
        logger.info("=== 生成数据预览 ===")
        
        # 获取全部数据
        preview_df = self.data
        
        # 处理NaN值，替换为None
        preview = preview_df.where(pd.notnull(preview_df), None).to_dict('records')
        
        self.analysis_results["preview"] = preview
        logger.info(f"成功生成数据预览，共 {len(preview)} 行")
        return preview
    
    def _replace_nan_recursive(self, obj):
        """
        递归替换对象中的NaN值为None
        """
        if isinstance(obj, float) and (np.isnan(obj) or pd.isna(obj)):
            return None
        elif isinstance(obj, dict):
            return {k: self._replace_nan_recursive(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._replace_nan_recursive(item) for item in obj]
        return obj

    def load_time_series_sample(self) -> List[Dict[str, any]]:
        """
        加载时序数据样例
        
        Returns:
            List[Dict]: 时序数据样例列表
        """
        logger.info("=== 加载时序数据样例 ===")
        excel_file = os.path.join(self.project_root, "data", "experience_cashflow.xlsx")
        
        if os.path.exists(excel_file):
            try:
                # 指定要读取的列
                target_columns = ['企业名称', 'Month_1', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6']
                df = pd.read_excel(excel_file)
                
                # 检查列是否存在，只保留存在的列
                existing_columns = [col for col in target_columns if col in df.columns]
                if len(existing_columns) < len(target_columns):
                    logger.warning(f"警告: 部分目标列未在Excel文件中找到。缺失列: {set(target_columns) - set(existing_columns)}")
                
                # 获取全部数据，仅包含指定列
                sample_df = df[existing_columns]
                
                # 处理NaN值
                sample = sample_df.where(pd.notnull(sample_df), None).to_dict('records')
                
                self.analysis_results["time_series_sample"] = sample
                logger.info(f"成功加载时序数据样例，行数: {len(sample)}")
                return sample
            except Exception as e:
                logger.error(f"加载时序数据样例失败: {e}", exc_info=True)
                return []
        else:
            logger.warning(f"时序数据文件不存在: {excel_file}")
            return []

    def load_unstructured_sample(self) -> List[Dict[str, any]]:
        """
        加载非结构化数据样例
        
        Returns:
            List[Dict]: 非结构化数据样例列表
        """
        logger.info("=== 加载非结构化数据样例 ===")
        excel_file = os.path.join(self.project_root, "data", "非结构化数据.xlsx")
        
        if os.path.exists(excel_file):
            try:
                df = pd.read_excel(excel_file)
                # 获取全部数据
                sample_df = df
                # 处理NaN值
                sample = sample_df.where(pd.notnull(sample_df), None).to_dict('records')
                
                self.analysis_results["unstructured_sample"] = sample
                logger.info(f"成功加载非结构化数据样例，行数: {len(sample)}")
                return sample
            except Exception as e:
                logger.error(f"加载非结构化数据样例失败: {e}")
                return []
        else:
            logger.warning(f"非结构化数据文件不存在: {excel_file}")
            return []

    def save_results(self, output_path: str = None):
        """
        保存分析结果
        
        Args:
            output_path: 输出文件路径
        """
        if output_path is None:
            output_path = os.path.join(os.path.dirname(self.file_path), "analysis_results.json")
        
        # 处理所有数据中的NaN值
        clean_results = self._replace_nan_recursive(self.analysis_results)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(clean_results, f, ensure_ascii=False, indent=2)
            logger.info(f"分析结果已保存至: {output_path}")
        except Exception as e:
            logger.error(f"保存结果失败: {e}", exc_info=True)
    
    def run_analysis(self):
        """
        运行完整的分析流程
        """
        if not self.load_data():
            return False
        
        # 执行各项分析
        self.analyze_data_types()
        self.analyze_data_quality()
        self.analyze_iqr_and_outliers()
        self.analyze_basic_stats()
        self.generate_overview()
        self.generate_preview()
        self.load_time_series_sample()
        self.load_unstructured_sample()
        
        # 添加字段映射到分析结果
        self.analysis_results["field_mapping"] = self.field_mapping
        
        # 保存结果
        self.save_results()
        
        logger.info("=== 分析完成 ===")
        return True

if __name__ == "__main__":
    # 使用相对路径
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    data_file = os.path.join(base_dir, 'data', 'SBAcase.11.13.17.csv')
    
    # 创建并运行分析器
    analyzer = DataAnalyzer(data_file)
    analyzer.run_analysis()
