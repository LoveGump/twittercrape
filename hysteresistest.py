import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.regression.quantile_regression import QuantReg
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from pyinform.transferentropy import transfer_entropy
from scipy.stats import permutation_test

file_path = r'C:\Users\zhang\Desktop\data.csv'
data = pd.read_csv(file_path)
print(data.head())

# 标准化
data = data.dropna()
scaler = StandardScaler()
data['sentiment'] = scaler.fit_transform(data[['sentiment']])
data['price'] = scaler.fit_transform(data[['price']])
# 创建滞后变量
data['sentiment_lag1'] = data['sentiment'].shift(1)
data['price_lag1'] = data['price'].shift(1)
data = data.dropna()  # 移除滞后后的NaN
quantiles=np.linspace(0.05, 0.95, 10)
poly = PolynomialFeatures(degree=3)  # 更高阶多项式
X_poly = poly.fit_transform(data[['sentiment', 'sentiment_lag1']])
X=data['sentiment']
Y = data['price']
results = {}
bootstrap_params_list = []

for q in quantiles:
    model = QuantReg(Y, X_poly).fit(q=q)
    results[f'Quantile_{q}'] = model.params
    print(f"Quantile {q} coefficients: {model.params}")
# Bootstrap计算置信区间
    n_bootstrap = 1000
    bootstrap_params = []
    for _ in range(n_bootstrap):
        sample = data.sample(n=len(data), replace=True)
        X_sample = poly.fit_transform(sample['sentiment'].values.reshape(-1, 1))
        model_boot = QuantReg(sample['price'], X_sample).fit(q=q)
        bootstrap_params.append(model_boot.params)
    bootstrap_params_list.append(np.array(bootstrap_params))
    lower_ci = np.percentile(bootstrap_params, 2.5, axis=0)
    upper_ci = np.percentile(bootstrap_params, 97.5, axis=0)
    print(f"95% CI for Quantile {q}: [{lower_ci[1]}, {upper_ci[1]}]")

# 转移熵分析
for k in range(1, 4):
    te_xy = transfer_entropy(X, Y, k=k)
    print(f"Transfer Entropy from Sentiment to Price (k={k}): {te_xy}")
    result = permutation_test((X, Y), lambda x, y: transfer_entropy(x, y, k), n_resamples=100, alternative='greater')
    print(f"P-value for Transfer Entropy (k={k}): {result.pvalue}")

plt.figure(figsize=(10, 6))
plt.scatter(X, Y, alpha=0.5, label='Data')
for q in quantiles:
    model = QuantReg(Y, X_poly).fit(q=q)
    plt.plot(X, model.predict(X_poly), label=f'Quantile {q}')
plt.legend()
plt.xlabel('Sentiment')
plt.ylabel('Price')
plt.title('Quantile Regression for Sentiment and Price')
plt.show()

# 因果效应可视化
df_ci = pd.DataFrame({
    'Quantile': quantiles,
    'Causal_Effect': [results[f'Quantile_{q}'][1] for q in quantiles], # 因果效应（多项式第二项）
    'Lower_CI': [bootstrap_params_list[i][:, 1].min() for i in range(len(quantiles))],  # 简化CI计算
    'Upper_CI': [bootstrap_params_list[i][:, 1].max() for i in range(len(quantiles))]
})
plt.figure(figsize=(10, 6))
sns.lineplot(x='Quantile', y='Causal_Effect', data=df_ci, marker='o', label='Causal Effect')
plt.fill_between(df_ci['Quantile'], df_ci['Lower_CI'], df_ci['Upper_CI'], alpha=0.2, label='95% CI')
plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
plt.xlabel('Quantile (τ)')
plt.ylabel('Causal Effect (β(τ))')
plt.title('Quantile Causal Effect of Sentiment on Cryptocurrency Price')
plt.grid(True)
plt.legend()
plt.show()

