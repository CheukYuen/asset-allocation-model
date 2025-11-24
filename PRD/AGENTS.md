# 🟧 【Cursor 项目规则｜Python 本地 3.11，本番环境 3.9】

在本项目中，开发环境与线上环境不同。

**请你（AI）在生成全部 Python 代码时，严格遵守以下兼容性规则：**

---

## 一、语言版本要求

### 本地开发环境
* Python **3.11**

### 线上运行环境
* Python **3.9.x**

### 因此：禁止使用任何 3.10+ 才支持的语法特性：

1. ❌ `match / case` 模式匹配
2. ❌ 类型联合写法（PEP 604）：`int | float`
3. ❌ typing 简化写法：`list[str]`, `dict[str, int]`
4. ❌ 新增字符串方法，如部分 `removeprefix`, `removesuffix`
5. ❌ 依赖 Python 3.10+ 行为改变的函数/模块
6. ❌ numpy 2.x / pandas 对 Python 3.10+ 才兼容的 API

### 必须使用 Python 3.9 可兼容语法：

* `typing.Union`, `typing.List`, `typing.Dict`
* 传统 if/elif，不使用 match-case
* 不调用 Python 3.10+ 的特性或函数

---

## 二、线上可用依赖库版本（必须严格遵守）

线上环境已固定以下库并不可升级，生成代码时必须兼容：

### ✔ numpy
```
numpy == 1.26.4
```

### ✔ pandas
```
pandas == 2.2.3
```

### ✘ scipy（线上没有）
* **禁止依赖 SciPy**
* 所有需要用到 `scipy.stats`、`scipy.optimize`、`scipy.linalg` 的功能，必须用：
  1. numpy 替代实现
  2. 自己写纯 Python/numpy 版本
  3. 避免使用只在 scipy 提供的功能

（例如：优化问题可以使用自己实现的简单梯度下降或闭式公式；统计分布可以手动写正态分布 PDF/CI 等。）

---

## 三、库兼容性硬规则

### 必须保证所有代码在 Python 3.9 + numpy 1.26.4 + pandas 2.2.3 下可运行

* 不允许使用 numpy 2.x 的 API
* 不允许使用 pandas 2.3+ 功能
* 不允许引入不支持 3.9 的依赖项

代码生成后，请你（AI）自动检查这些兼容性。

---

## 四、项目开发流程要求

### 生成的代码必须满足：

1. 在 Python 3.11 本地可运行
2. 但**不能使用 Python 3.11 独有特性**（保证升级不会破坏 3.9）
3. 在 Python 3.9 本番环境推理时也能运行
4. 所有公式计算必须基于 numpy/pandas 实现

---

## 五、Cursor 内部行为要求（非常重要）

### 你必须：

* 自动检查你生成的代码是否依赖 Python 3.10+ 特性
* 自动检查是否用到了 scipy
* 自动检查是否使用了 numpy/pandas 的 3.9 不兼容写法
* 自动给出必要的替代实现（如 scipy 替代公式）
* 避免使用未来行为（例如 pandas 3.x 设计）

---

## 六、如果用户要求不兼容特性，你必须回应：

> "本项目线上环境是 Python 3.9，并不支持当前请求的语法/库，我会给出兼容方案。"

并自动生成兼容 Python 3.9 的替代实现。

---

