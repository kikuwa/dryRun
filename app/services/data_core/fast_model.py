import pandas as pd
import numpy as np
import lightgbm as lgb
import os
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

# Get project root directory
base_dir = os.path.dirname(os.path.abspath(__file__))

class MISelector:
    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        self.feature_importance_df = None

    def fit(self, X, y):
        print("正在计算 MI 特征重要性...")
        mi_scores = mutual_info_classif(X, y, random_state=self.random_seed)
        self.feature_importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': mi_scores
        }).sort_values(by='importance', ascending=False)
        print("MI 特征重要性计算完成")

    def get_feature_importance(self):
        if self.feature_importance_df is None:
            raise ValueError("模型未训练，请先调用fit()方法")
        return self.feature_importance_df

class RandomForestSelector:
    def __init__(self, random_seed=42, n_jobs=-1):
        self.random_seed = random_seed
        self.n_jobs = n_jobs
        self.feature_importance_df = None

    def fit(self, X, y):
        print("正在训练随机森林模型以计算特征重要性...")
        rf = RandomForestClassifier(random_state=self.random_seed, n_jobs=self.n_jobs)
        rf.fit(X, y)
        rf_scores = rf.feature_importances_
        self.feature_importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_scores
        }).sort_values(by='importance', ascending=False)
        print("随机森林特征重要性计算完成")

    def get_feature_importance(self):
        if self.feature_importance_df is None:
            raise ValueError("模型未训练，请先调用fit()方法")
        return self.feature_importance_df

class RandomForestSelector:
    def __init__(self, random_seed=42, n_jobs=-1):
        self.random_seed = random_seed
        self.n_jobs = n_jobs
        self.feature_importance_df = None

    def fit(self, X, y):
        print("正在训练随机森林模型以计算特征重要性...")
        rf = RandomForestClassifier(random_state=self.random_seed, n_jobs=self.n_jobs)
        rf.fit(X, y)
        rf_scores = rf.feature_importances_
        self.feature_importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_scores
        }).sort_values(by='importance', ascending=False)
        print("随机森林特征重要性计算完成")

    def get_feature_importance(self):
        if self.feature_importance_df is None:
            raise ValueError("模型未训练，请先调用fit()方法")
        return self.feature_importance_df

class LGBMBaggingPUSelector:
    def __init__(self, n_estimators=10, random_seed=42):
        self.n_estimators = n_estimators
        self.random_seed = random_seed
        self.models = []
        self.feature_importance_df = None

    def fit(self, X, y):
        print(f"开始训练 LightGBM Bagging PU 模型（共{self.n_estimators}个子模型）")
        
        importances = np.zeros(len(X.columns))

        for i in range(self.n_estimators):
            # 快速参数
            params = {
                'objective': 'binary',
                'metric': 'average_precision',
                'verbosity': -1,
                'learning_rate': 0.1,
                'num_leaves': 10,
                'n_jobs': -1,
                'scale_pos_weight': 2,
                'max_depth': 3,
                'min_child_samples': 30,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'boosting_type': 'gbdt',
                'seed': self.random_seed + i
            }
            
            # 训练LightGBM
            dtrain = lgb.Dataset(X, label=y)
            model = lgb.train(params, dtrain, num_boost_round=100)
            self.models.append(model)
            
            # 累加特征重要性
            importances += model.feature_importance(importance_type='gain')
            
            print(f"已完成 {i + 1}/{self.n_estimators} 个模型")
        
        # 计算平均特征重要性
        importances /= self.n_estimators
        self.feature_importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': importances
        }).sort_values(by='importance', ascending=False)

    def get_feature_importance(self):
        """获取特征重要性"""
        if self.feature_importance_df is None:
            raise ValueError("模型未训练，请先调用fit()方法")
        return self.feature_importance_df

class LGBMBaggingPUSelector:
    def __init__(self, n_estimators=10, random_seed=42):
        self.n_estimators = n_estimators
        self.random_seed = random_seed
        self.models = []
        self.feature_importance_df = None

    def fit(self, X, y):
        print(f"开始训练 LightGBM Bagging PU 模型（共{self.n_estimators}个子模型）")
        
        importances = np.zeros(len(X.columns))

        for i in range(self.n_estimators):
            # 快速参数
            params = {
                'objective': 'binary',
                'metric': 'average_precision',
                'verbosity': -1,
                'learning_rate': 0.1,
                'num_leaves': 10,
                'n_jobs': -1,
                'scale_pos_weight': 2,
                'max_depth': 3,
                'min_child_samples': 30,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'boosting_type': 'gbdt',
                'seed': self.random_seed + i
            }
            
            # 训练LightGBM
            dtrain = lgb.Dataset(X, label=y)
            model = lgb.train(params, dtrain, num_boost_round=100)
            self.models.append(model)
            
            # 累加特征重要性
            importances += model.feature_importance(importance_type='gain')
            
            print(f"已完成 {i + 1}/{self.n_estimators} 个模型")
        
        # 计算平均特征重要性
        importances /= self.n_estimators
        self.feature_importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': importances
        }).sort_values(by='importance', ascending=False)

    def get_feature_importance(self):
        """获取特征重要性"""
        if self.feature_importance_df is None:
            raise ValueError("模型未训练，请先调用fit()方法")
        return self.feature_importance_df

class EnsembleFeatureSelector:
    def __init__(self, selectors, weights):
        if len(selectors) != len(weights):
            raise ValueError("选择器和权重的数量必须相同")
        self.selectors = selectors
        self.weights = weights
        self.feature_importance_df = None

    def fit(self, X, y):
        print("开始集成特征选择...")
        ensemble_scores = {}
        feature_names = X.columns.tolist()

        for selector, weight in zip(self.selectors, self.weights):
            selector.fit(X, y)
            importance_df = selector.get_feature_importance()
            
            # 将重要性分数转换为排名
            importance_df['rank'] = importance_df['importance'].rank(ascending=False)
            rank_dict = pd.Series(importance_df['rank'].values, index=importance_df['feature']).to_dict()

            for feature in feature_names:
                if feature not in ensemble_scores:
                    ensemble_scores[feature] = 0
                # 基于排名的加权分数
                ensemble_scores[feature] += rank_dict.get(feature, 0) * weight
        
        # 创建 DataFrame 并排序
        self.feature_importance_df = pd.DataFrame(list(ensemble_scores.items()), columns=['feature', 'importance'])
        self.feature_importance_df = self.feature_importance_df.sort_values(by='importance', ascending=True) # 排名越小越好
        print("集成特征选择完成")

    def get_feature_importance(self):
        if self.feature_importance_df is None:
            raise ValueError("模型未训练，请先调用fit()方法")
        return self.feature_importance_df

def preprocess_dataframe(df):
    """简化的数据预处理函数"""
    df_processed = df.copy()
    
    # 处理分类变量
    for col in df_processed.columns:
        if df_processed[col].dtype == 'object':
            # 因子化编码
            codes, uniques = pd.factorize(df_processed[col])
            df_processed[f"{col}_encoded"] = codes
            df_processed = df_processed.drop(columns=[col])
    
    # 填充缺失值
    numeric_cols = df_processed.select_dtypes(include=['number']).columns
    df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].median())
    
    return df_processed



def load_feature_mapping(file_path):
    """从Excel文件中加载英文特征名到中文特征名的映射"""
    try:
        df = pd.read_excel(file_path)
        # 确定英文和中文列名
        english_col, chinese_col = '', ''
        if '英文列名' in df.columns and '中文列名' in df.columns:
            english_col, chinese_col = '英文列名', '中文列名'
        elif '英文' in df.columns and '中文' in df.columns:
            english_col, chinese_col = '英文', '中文'
        elif '英文字段名' in df.columns and '中文字段名' in df.columns:
            english_col, chinese_col = '英文字段名', '中文字段名'
        
        if english_col and chinese_col:
            mapping = pd.Series(df[chinese_col].values, index=df[english_col]).to_dict()
            print(f"成功加载 {len(mapping)} 个特征映射")
            return mapping
        else:
            print("错误: 无法在Excel文件中找到匹配的列名")
            return {}
            
    except Exception as e:
        print(f"加载特征映射失败: {e}")
        return {}

def run_feature_selection():
    """运行特征筛选"""
    try:
        # 1. 加载数据
        print("加载数据...")
        # 使用相对路径，确保在任何环境中都能正确找到文件
        train_path = os.path.join(base_dir, '..', '..', '..', 'data', 'SBAcase.11.13.17.csv')
        
        # 加载完整数据集
        df_train = pd.read_csv(train_path)
        print(f"Train.csv 加载成功，形状: {df_train.shape}")
        
        # 基于 Default 列创建 TARGET
        df_train['TARGET'] = df_train['Default']
        df_train = df_train.drop(columns=['Default'])
        print("已根据 Default 创建 TARGET 列")
        
        # 2. 数据预处理
        print("\n数据预处理...")
        processed_train = preprocess_dataframe(df_train)
        print(f"处理后数据形状: {processed_train.shape}")
        
        # 3. 构建PU场景
        print("\n构建PU场景...")
        if 'TARGET' in processed_train.columns:
            y = processed_train['TARGET'].astype(int).copy()
            X = processed_train.drop(columns=['TARGET'])
        else:
            print("错误: 未找到 'TARGET' 列")
            return None
        
        # 确保y是数值型
        y = pd.to_numeric(y, errors='coerce').fillna(0)
        
        # 构建PU场景
        positive_indices = y[y == 1].index
        if len(positive_indices) == 0:
            print("错误: 未找到正样本，无法构建PU场景")
            return None
        
        # 随机选择80%作为已知正样本，20%作为隐藏正样本
        np.random.seed(42)
        hidden_ratio = 0.2
        hidden_size = int(len(positive_indices) * hidden_ratio)
        hidden_positive_indices = np.random.choice(positive_indices, hidden_size, replace=False)
        known_positive_indices = list(set(positive_indices) - set(hidden_positive_indices))
        
        X_p = X.loc[known_positive_indices]
        y_p = pd.Series(1, index=X_p.index)
        
        u_indices = list(set(X.index) - set(known_positive_indices))
        X_u = X.loc[u_indices]
        y_u = pd.Series(0, index=X_u.index)
        
        print(f"PU场景构建:")
        print(f"已知风险客户(P): {len(X_p)} 个")
        print(f"未标记数据(U): {len(X_u)} 个 (包含 {len(hidden_positive_indices)} 个隐藏风险客户)")
        
        # 4. 训练MI模型
        print("\n训练MI模型...")
        mi_selector = MISelector(random_seed=42)
        mi_selector.fit(X, y)
        feature_importance = mi_selector.get_feature_importance()
        
        # 打印MI评估指标
        print("\n--- MI独立评估指标 ---")
        print(feature_importance.head(5))
        
        # 6. 输出经过验证的重要特征排序结果
        print("\n=== 重要特征排序结果 ===")
        print("排序依据: 特征重要性得分（基于LightGBM的split gain）")
        print("说明: 特征重要性得分越高，表示该特征对模型预测性能的贡献越大")
        print("\n特征排序:")
        
        # 只输出前30个重要特征，确保结果简洁
        top_features = feature_importance.head(30)
        for idx, row in top_features.iterrows():
            print(f"{idx + 1}. {row['feature']}: {row['importance']:.4f}")
        
        # 加载特征映射
        mapping_path = os.path.join(base_dir, '..', '..', '..', 'data', '中英文对照.xlsx')
        feature_map = load_feature_mapping(mapping_path)
        
        # 将中文名称合并到结果中
        # 对于后缀"_encoded"的特征名，匹配中文名的时候把"_encoded"删掉
        feature_importance['chinese_name'] = feature_importance['feature'].apply(
            lambda x: feature_map.get(x.replace('_encoded', ''), '无')
        )
        
        # 7. 保存并返回结果
        output_dir = os.path.join(base_dir, 'data', 'results', 'pu_learning')
        os.makedirs(output_dir, exist_ok=True)
        
        importance_file = os.path.join(output_dir, 'feature_importance.csv')
        feature_importance.to_csv(importance_file, index=False)
        print(f"\n特征重要性已保存到: {importance_file}")

        # 计算统计数据
        original_feature_count = len(X.columns)
        sample_count = len(df_train)
        positive_count = df_train['TARGET'].sum()
        positive_ratio = round((positive_count / sample_count) * 100, 2) if sample_count > 0 else 0
        
        return {
            "feature_importance_df": feature_importance,
            "original_feature_count": original_feature_count,
            "sample_count": sample_count,
            "positive_ratio": positive_ratio
        }
        
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """主函数"""
    run_feature_selection()

if __name__ == "__main__":
    main()