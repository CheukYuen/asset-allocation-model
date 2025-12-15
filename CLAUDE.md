# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Asset Allocation Model** is a Python project for tactical and strategic asset allocation analysis. It provides:
- Strategic Asset Allocation (SAA) with 420 portfolio configurations across risk levels
- Tactical Asset Allocation (TAA) signal engine for dynamic weight adjustments
- Backtesting framework for portfolio performance evaluation
- Monte Carlo simulation for long-term scenario analysis

The project uses financial data (historical returns, macro indicators, CME expectations) to optimize asset allocations across 16 strategies organized into 4 asset classes (Cash, Bonds, Equities, Alternatives).

## Critical Compatibility Requirements

**IMPORTANT**: This project has strict Python version requirements for production:

### Local Development
- Python **3.11**

### Production Environment (Online)
- Python **3.9.x**
- **SciPy is NOT available** in production — must use numpy-only implementations
- **Fixed library versions** (numpy 1.26.4, pandas 2.2.3)

### When Writing Code
1. **Forbidden in Python 3.10+**:
   - ❌ `match/case` pattern matching
   - ❌ Type union syntax: `int | float` (use `Union[int, float]`)
   - ❌ Simplified typing: `list[str]`, `dict[str, int]` (use `List[str]`, `Dict[str, int]`)

2. **No SciPy** — implement alternatives:
   - Use numpy for linear algebra (matrix operations, eigenvalues)
   - Use numpy for statistical distributions (normal PDF/CDF)
   - Use numpy for optimization (gradient descent if needed)

3. **Use Python 3.9 compatible syntax**:
   - `from typing import Union, List, Dict, Tuple, Optional`
   - Standard if/elif conditionals
   - numpy operations for numerical tasks

See `AGENTS.md` for the complete compatibility specification.

## Project Structure

```
asset-allocation-model/
├── src/                          # Core portfolio analysis module
│   ├── portfolio.py              # Basic Portfolio class (historical analysis)
│
├── taa_learning_project/         # TAA (Tactical Asset Allocation) system
│   ├── core/
│   │   ├── taa_signal_engine.py  # Weight adjustment engine (16 strategies → 4 macro quadrants)
│   │   ├── backtest_engine.py    # Portfolio performance metrics & backtesting
│   │   ├── mc_simulation.py      # Monte Carlo simulation for scenarios
│   │   └── mock_data.py          # Generate synthetic market data
│   └── scripts/
│       ├── run_backtest.py       # Run backtests on multiple SAA configs
│       └── run_mock_data.py      # Generate mock data for testing
│
├── mock/                         # Data generation utilities
│   └── mu_annual.py              # Annual return expectations
│
├── agent/                        # Research & specification documents
│   ├── merrill_lynch_clock/      # Macro quadrant definitions & data requirements
│   ├── risk_parity/              # Risk parity strategy calculations
│   ├── brenchmark/               # Benchmark metrics & examples
│   ├── 产品数据/                  # Product data & standardization specs
│   └── prompt/                   # AI prompt templates for product selection
│
├── PRD/                          # Product Requirements Documents
│   ├── 四大类说明.md              # 4 asset classes definition
│   ├── asset_classes_and_strategies_4x16.md  # 16 strategies mapping
│   └── 工程拆分任务.md            # Engineering task breakdown
│
├── examples/                     # Example usage scripts
│   └── basic_example.py          # Portfolio class usage demo
│
├── requirements.txt              # Python dependencies
├── AGENTS.md                     # Development environment & compatibility rules
└── README.md                     # Project introduction (Chinese)
```

## Key Concepts

### 4 Asset Classes + 16 Strategies
- **Cash** (1 strategy): Cash deposits
- **Bonds** (5 strategies): Fixed income products at different risk levels
- **Equities** (5 strategies): Domestic and overseas equity allocations
- **Alternatives** (5 strategies): Commodities, real estate, hedge funds

### 4 Macro Quadrants
The system detects macro state and applies weight tilts:
1. **Recovery** — growth accelerating
2. **Overheat** — growth too fast, risk of correction
3. **Stagflation** — growth slowing, inflation high
4. **Recession** — growth negative or weak

### SAA vs TAA
- **SAA (Strategic)**: Long-term baseline weights for each portfolio profile (420 configurations)
- **TAA (Tactical)**: Monthly weight adjustments based on macro quadrant detection

### Core TAA Flow
```
Input: SAA weights (baseline) + Current macro quadrant
  ↓
Apply quadrant-specific tilt rules to 4 asset classes
  ↓
Distribute class tilts to 16 strategies (proportionally by SAA weight)
  ↓
Apply sensitivity adjustments (β multipliers per strategy)
  ↓
Normalize weights (non-negative, sum to 1)
  ↓
Output: TAA weights (monthly rebalanced)
```

## Common Development Tasks

### Setup
```bash
# Create virtual environment
python3 -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run Examples
```bash
# Basic Portfolio analysis
python examples/basic_example.py

# Generate mock market data
python taa_learning_project/scripts/run_mock_data.py

# Run backtests on all SAA configurations
python taa_learning_project/scripts/run_backtest.py
```

### TAA Engine Usage
```python
from taa_learning_project.core.taa_signal_engine import compute_taa_weights

# Input: SAA baseline weights, current macro quadrant
saa_weights = {...}  # dict of 16 strategies
quadrant = "Recovery"  # or "Overheat", "Stagflation", "Recession"

taa_weights = compute_taa_weights(saa_weights, quadrant)
# Output: TAA weights (normalized, ready to use)
```

### Backtesting
```python
from taa_learning_project.core.backtest_engine import BacktestEngine

engine = BacktestEngine()
results = engine.backtest(
    returns_df=monthly_returns,      # 16 strategy returns over time
    saa_weights=saa_config,           # SAA baseline
    quadrants_timeseries=quadrants,   # Monthly macro state
    strategy_names=STRATEGY_NAMES
)
# Returns: Sharpe ratio, max drawdown, cumulative return, etc.
```

## Data Flow & Key Files

### Mock Data Generation (`taa_learning_project/core/mock_data.py`)
Generates synthetic monthly returns for the 16 strategies. Used for testing and simulation.

### TAA Signal Engine (`taa_learning_project/core/taa_signal_engine.py`)
- Maps 4 macro quadrants → 4 asset class tilts
- Distributes tilts to 16 strategies by SAA weight proportion
- Applies sensitivity multipliers (β) per strategy
- Enforces constraints (non-negative, normalized)
- **Standalone module** — no dependencies on other project code

### Backtest Engine (`taa_learning_project/core/backtest_engine.py`)
- Computes portfolio returns from strategy weights & returns
- Calculates performance metrics (annualized return/volatility, Sharpe, max drawdown, Calmar ratio)
- Supports batch backtesting across multiple SAA configurations

### Monte Carlo Simulation (`taa_learning_project/core/mc_simulation.py`)
- Generates long-term market scenarios using CME expectations (mean/covariance)
- Computes SAA & TAA performance paths
- Outputs confidence intervals, downside risk, probability of success

## Testing & Validation

When modifying TAA engine logic, verify:
- **β = 0** → weights should equal SAA exactly (no tilt)
- **Δw = 0** (zero delta in quadrant rules) → no change from SAA
- **Weight sum ≈ 1.0** after normalization
- **No negative weights** after constraint enforcement
- **Consistency across 420 SAA configurations**

## Important Notes

- **Use pandas/numpy only** for all numerical operations (no SciPy in production)
- **Monthly data frequency** — all returns, quadrants, and metrics are monthly
- **16 strategy order is fixed** — must use consistent ordering in all DataFrames
- **Avoid external API calls** during backtest batch runs (mock data for testing)
- **Strategy configuration** — each of 16 strategies has specific risk characteristics documented in `agent/` subdirectories
