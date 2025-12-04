好的，遵照您的要求，本次将**仅针对“企业主财富传承 (平衡型长期投资者)”这一个客户**进行服务。

---

### **1. 标准化输入：结构化JSON**

```json
{
  "client_id": "Client_002",
  "profile_name": "企业主财富传承 (平衡型长期投资者)",
  "demographics": {
    "age": 45,
    "status": "企业主"
  },
  "investment_objectives": [
    "长期资本增值",
    "跑赢通胀",
    "为子女未来教育/婚嫁提供资金"
  ],
  "risk_tolerance": {
    "risk_profile": "平衡型",
    "max_acceptable_drawdown": "15%"
  },
  "investment_horizon": "长期 (>10年)",
  "liquidity_needs": "低",
  "capital_size": {
    "initial_investment": "5000万元人民币",
    "future_contributions": "有"
  },
  "existing_portfolio": {
    "asset_allocation": [
      { "asset_class": "Private Business Equity", "percentage": 60 },
      { "asset_class": "A-Shares", "percentage": 20 },
      { "asset_class": "Residential Real Estate", "percentage": 15 },
      { "asset_class": "Cash", "percentage": 5 }
    ]
  }
}
```

---

### **2. JSON字段详解说明**

下表详细解释了上述JSON中每个字段的含义、作用以及为什么它对量化宏观资产配置至关重要。

| 字段路径 | 字段名称 (示例) | 含义与作用 |
| :--- | :--- | :--- |
| `client_id` | `"Client_002"` | **客户唯一标识符**。用于在系统中区分和追踪不同客户，是数据管理的基础。 |
| `profile_name` | `"企业主财富传承 (平衡型长期投资者)"` | **客户画像的简明概括**。便于快速识别客户类型，常用于报告和沟通。 |
| `demographics.age` | `45` | **客户的年龄**。是评估生命周期、风险承受能力和投资期限的关键人口统计学变量。45岁通常意味着仍有较长的投资期和一定的风险承担能力。 |
| `demographics.status` | `"企业主"` | **客户的职业或身份**。揭示其收入来源、财富构成和潜在风险（如企业经营风险）。企业主可能已有大量非流动性资产（如企业股权），影响整体风险敞口。 |
| `investment_objectives[0]`<br>`...` | `"长期资本增值"`<br>`"跑赢通胀"`<br>`"为子女未来教育/婚嫁提供资金"` | **客户的核心财务目标列表**。这是所有投资决策的最终导向。决定了收益来源的侧重（增长 vs 收入）和投资策略的激进程度。多个目标需要权衡。 |
| `risk_tolerance.risk_profile` | `"平衡型"` | **客户的风险偏好等级（定性）**。一个初步的分类锚点，帮助快速理解客户的风险态度，常作为问卷结果。 |
| `risk_tolerance.max_acceptable_drawdown` | `"15%"` | **客户能承受的最大亏损幅度（定量）**。这是**最关键的风险约束**，是量化模型中的硬性边界。直接决定了组合的整体风险预算和仓位上限。 |
| `investment_horizon` | `"长期 (>10年)"` | **资金的投资期限**。决定了能否跨越完整的经济周期。长期限允许采用更积极的战略轮动，并能平滑短期市场波动带来的影响，是运用美林时钟的前提。 |
| `liquidity_needs` | `"低"` | **客户对资金流动性的需求程度**。高流动性需求会迫使配置更多现金类资产，牺牲潜在回报。低需求则允许投资于流动性稍差但收益更高的资产（如长久期债券、私募基金）。 |
| `capital_size.initial_investment` | `"5000万元人民币"` | **初始投资金额的规模**。影响策略的选择（如是否能有效分散化）、交易成本（冲击成本）以及可投资产的范围（如某些私募产品有最低门槛）。 |
| `capital_size.future_contributions` | `"有"` | **未来是否有追加投资**。影响现金流管理和投资策略（如有追加计划，可采用更灵活的建仓或定投策略）。 |
| `existing_portfolio.asset_allocation[0].asset_class`<br>`...` | `"Private Business Equity"`<br>`"A-Shares"`<br>`"Residential Real Estate"`<br>`"Cash"` | **现有投资组合中各主要大类资产的类别**。识别客户当前的风险敞口来源。例如，已持有60%的企业股权，意味着其总财富对企业经营风险高度暴露。 |
| `existing_portfolio.asset_allocation[0].percentage`<br>`...` | `60`<br>`20`<br>`15`<br>`5` | **现有各资产类别的市值占比**。这是进行**增量配置**的基础。新配置建议必须考虑现有持仓，以计算**整体组合**的风险和收益，避免重复或过度集中。 |

此JSON结构确保了所有必要信息都以机器可读、无歧义的方式提供，为后续的量化分析和自动化决策流程奠定了坚实基础。