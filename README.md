# Asset Allocation Model

一个用于资产配置分析和优化的Python项目。

## 项目结构

```
asset-allocation-model/
├── src/                    # 源代码目录
│   ├── __init__.py
│   └── portfolio.py       # 资产组合核心代码
├── examples/              # 示例代码
│   └── basic_example.py   # 基础使用示例
├── requirements.txt       # Python依赖包
├── .gitignore            # Git忽略文件配置
└── README.md             # 项目说明文档
```

## 快速开始

### 1. 创建虚拟环境

```bash
# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
# macOS/Linux:
source venv/bin/activate
# Windows:
# venv\Scripts\activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 运行示例

```bash
python examples/basic_example.py
```

## 功能说明

- **数据获取**: 使用 yfinance 获取股票历史数据
- **收益计算**: 计算资产的历史收益率
- **风险分析**: 计算波动率、协方差矩阵
- **资产配置**: 实现简单的等权重配置策略

## 依赖说明

- `numpy`: 数值计算
- `pandas`: 数据处理和分析
- `matplotlib`: 数据可视化
- `yfinance`: 获取金融市场数据
- `scipy`: 科学计算和优化算法

## 下一步计划

- [ ] 实现均值-方差优化（Markowitz模型）
- [ ] 添加风险平价策略
- [ ] 实现回测框架
- [ ] 添加更多可视化功能
