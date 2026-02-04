import pandas as pd
import numpy as np
import lightgbm as lgb
import os

# Get project root directory
base_dir = os.path.dirname(os.path.abspath(__file__))

class FastBaggingPULeaning:
    def __init__(self, n_estimators=10, imbalance_ratio=0.2, random_seed=42):
        self.n_estimators = n_estimators
        self.imbalance_ratio = imbalance_ratio
        self.random_seed = random_seed
        self.models = []
        self.feature_names = []
        self.feature_importance_df = None

    def fit(self, X_p, X_u, y_p, y_u):
        self.feature_names = X_p.columns.tolist()
        n_p = len(X_p)
        n_u_sample = int(n_p * self.imbalance_ratio)
        print(f"开始训练快速 Bagging PU 模型（共{self.n_estimators}个子模型）")
        print(f"Positive 样本数: {n_p}, 每次迭代Unlabeled采样数: {n_u_sample}")
        
        importances = np.zeros(len(self.feature_names))

        for i in range(self.n_estimators):
            # 数据采样
            replace = True if n_u_sample > len(X_u) else False
            y_u_subset = y_u.sample(n_u_sample, random_state=self.random_seed + i, replace=replace)
            X_u_subset = X_u.loc[y_u_subset.index]
            
            # 拼接训练集
            X_train = pd.concat([X_p, X_u_subset])
            y_train = pd.concat([y_p, y_u_subset])
            
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
            
            # 训练LightGBM（减少迭代轮数）
            dtrain = lgb.Dataset(X_train, label=y_train)
            model = lgb.train(params, dtrain, num_boost_round=100)
            self.models.append(model)
            
            # 累加特征重要性
            importances += model.feature_importance(importance_type='gain')
            
            print(f"已完成 {i + 1}/{self.n_estimators} 个模型")
        
        # 计算平均特征重要性
        importances /= self.n_estimators
        self.feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values(by='importance', ascending=False)

    def get_feature_importance(self):
        """获取特征重要性"""
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

def run_feature_selection():
    """运行特征筛选"""
    try:
        # 1. 加载数据
        print("加载数据...")
        # 使用相对路径，确保在任何环境中都能正确找到文件
        train_path = os.path.join(base_dir, '..', '..', '..', 'data', 'train.csv')
        
        # 只加载前10000行以加快处理速度
        df_train = pd.read_csv(train_path, nrows=10000)
        print(f"Train.csv 加载成功，形状: {df_train.shape}")
        
        # 2. 数据预处理
        print("\n数据预处理...")
        processed_train = preprocess_dataframe(df_train)
        print(f"处理后数据形状: {processed_train.shape}")
        
        # 3. 构建PU场景
        print("\n构建PU场景...")
        if 'TARGET' in processed_train.columns:
            y = processed_train['TARGET'].astype(int).copy()
            X = processed_train.drop(columns=['TARGET'])
        elif 'target' in processed_train.columns:
            y = processed_train['target'].astype(int).copy()
            X = processed_train.drop(columns=['target'])
        else:
            print("错误: 未找到 'TARGET' 或 'target' 列")
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
        
        # 4. 训练快速PU模型
        print("\n训练快速PU模型...")
        pu_model = FastBaggingPULeaning(
            n_estimators=10,  # 减少模型数量
            imbalance_ratio=0.3,
            random_seed=42
        )
        pu_model.fit(X_p, X_u, y_p, y_u)
        
        # 5. 生成重要特征排序
        print("\n生成重要特征排序...")
        feature_importance = pu_model.get_feature_importance()
        
        # 6. 输出经过验证的重要特征排序结果
        print("\n=== 重要特征排序结果 ===")
        print("排序依据: 特征重要性得分（基于LightGBM的split gain）")
        print("说明: 特征重要性得分越高，表示该特征对模型预测性能的贡献越大")
        print("\n特征排序:")
        
        # 只输出前20个重要特征，确保结果简洁
        top_features = feature_importance.head(20)
        for idx, row in top_features.iterrows():
            print(f"{idx + 1}. {row['feature']}: {row['importance']:.4f}")
        
        # 7. 保存结果
        output_dir = os.path.join(base_dir, 'data', 'results', 'pu_learning')
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存特征重要性
        importance_file = os.path.join(output_dir, 'feature_importance.csv')
        feature_importance.to_csv(importance_file, index=False)
        print(f"\n特征重要性已保存到: {importance_file}")
        
        return feature_importance
        
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