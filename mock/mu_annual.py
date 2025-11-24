import numpy as np

def generate_series(mu_annual, sigma_annual, months=120):
    mu_m = mu_annual / 12
    sigma_m = sigma_annual / np.sqrt(12)
    return np.random.normal(mu_m, sigma_m, months)

print("=" * 60)
print("测试 1：基础调用")
print("=" * 60)
result = generate_series(0.08, 0.15)  # 年化收益率8%，年化波动率15%
print(f"生成的数据长度: {len(result)}")
print(f"前 10 个数据点: {result[:10]}")
print(f"平均值: {result.mean():.6f}")
print(f"标准差: {result.std():.6f}")