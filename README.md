# 金融风控AI系统

本项目是一个基于Flask和现代前端技术的全栈Web应用，旨在为金融风控领域提供一套完整的AI驱动解决方案。系统深度集成了大语言模型（LLM），实现了从数据工程到模型训练、再到智能推理的端到端工作流。

## 核心功能模块

系统主要围绕两大核心流水线进行构建：**数据工程流水线** 和 **模型训练流水线**。

### 1. 数据工程流水线

此流水线负责处理和准备用于模型训练和分析的数据。

-   **多源数据 (`multi_source_data`)**:
    -   集成和管理来自不同来源的数据。
    -   提供数据统计、分析和可视化功能。
-   **特征工程 (`feature_selection`)**:
    -   提供自动化的特征筛选和处理能力。
    -   利用 `fast_model.py` 等服务进行快速特征重要性评估。
-   **COT数据合成 (`cot_synthesis`)**:
    -   **核心功能**：利用大语言模型（如 `deepseek-reasoner`）自动生成高质量的思维链（Chain of Thought）推理样本。
    -   支持基于原始Prompt、优化Prompt和专家经验三种模式生成思维链。
    -   实现了服务端和客户端双重缓存机制，提升性能和数据一致性。

### 2. 模型训练流水线

此流水线专注于大模型的训练、优化和评估。

-   **模型后训练流水线 (`model_diff`)**:
    -   提供模型效果评估和多模型差异对比分析功能。
    -   支持对不同版本的模型进行性能和行为的可视化比较。

## 技术栈

-   **后端**:
    -   **框架**: Flask
    -   **核心服务**:
        -   `cot_synthesis_service.py`: 负责思维链数据的生成、缓存和与大模型的交互。
        -   `data_analyzer.py`: 提供数据分析功能。
        -   `fast_model.py`: 用于快速特征筛选和模型训练。
-   **前端**:
    -   **框架**: Alpine.js
    -   **样式**: Tailwind CSS
    -   **模板引擎**: Jinja2
-   **数据处理**:
    -   Pandas
    -   Numpy

## 项目结构概览

```
app/
├── routes/                  # Flask蓝图和路由定义
│   ├── main.py              # 核心路由（如首页）
│   ├── data_tool.py         # 数据工具相关路由
│   └── risk_cot/            # 大模型风控模块路由
│       ├── views.py         # 主要页面和API路由
│       └── ...
├── services/                # 后端核心服务逻辑
│   ├── data_core/           # 数据处理和分析服务
│   │   ├── data_analyzer.py
│   │   └── fast_model.py
│   └── risk_cot/            # 大模型风控相关服务
│       └── cot_synthesis_service.py
├── templates/               # Jinja2前端模板
│   ├── base.html            # 基础布局模板
│   ├── home.html            # 首页
│   ├── data_tool/           # 数据工具页面
│   └── risk_cot/            # 大模型风控页面
│       └── cot_synthesis.html
└── __init__.py              # Flask应用工厂
```

## 快速开始

1.  **安装依赖**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **配置API密钥**:
    在启动应用之前，您需要配置大语言模型的API密钥。请打开文件 `app/services/risk_cot/cot_synthesis_service.py` 并填入您的密钥：

    ```python
    # app/services/risk_cot/cot_synthesis_service.py:L20-22
    self.api_key = "YOUR_DEEPSEEK_API_KEY"
    self.api_url = "https://api.deepseek.com/chat/completions"
    self.model = "deepseek-reasoner"
    ```

3.  **启动应用**:
    ```bash
    flask run
    ```

4.  **访问系统**:
    在浏览器中打开 `http://127.0.0.1:5000` 即可访问系统首页。

---
*Version 2.4.0-LTS*
*© 2024 金融风控AI系统 | 核心架构模组*
