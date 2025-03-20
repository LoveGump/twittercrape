import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller  # 用于检验平稳性
from sklearn.neural_network import MLPRegressor  # 非线性模型
from sklearn.preprocessing import StandardScaler  # 标准化数据
from sklearn.metrics import mean_squared_error  # 计算均方误差
from scipy.stats import permutation_test  # 置换检验
import matplotlib.pyplot as plt

# 读取数据
file_path = r'C:\Users\zhang\Desktop\data.csv'
data = pd.read_csv(file_path)
print(data.head())

# 检查数据平稳性并进行差分处理
def check_stationarity(series, series_name):
    result = adfuller(series)
    print(f'{series_name} ADF Statistic: {result[0]}, p-value: {result[1]}')
    p_value = result[1]  # 明确提取 p 值
    return p_value > 0.05

# 检查 sentiment 和 price 的平稳性
if check_stationarity(data['sentiment'], 'Sentiment') or check_stationarity(data['price'], 'Price'):
    data['sentiment_diff'] = data['sentiment'].diff().dropna()
    data['price_diff'] = data['price'].diff().dropna()
    data = data.dropna()  # 移除 NaN
else:
    data['sentiment_diff'] = data['sentiment']
    data['price_diff'] = data['price']

# 标准化数据
data = data.dropna()
scaler = StandardScaler()
data['sentiment_diff'] = scaler.fit_transform(data[['sentiment_diff']])
data['price_diff'] = scaler.fit_transform(data[['price_diff']])

# 创建滞后变量
max_lag = 3  # 假设滞后长度为 3，可以根据数据调整
for lag in range(1, max_lag + 1):
    data[f'sentiment_diff_lag{lag}'] = data['sentiment_diff'].shift(lag)
    data[f'price_diff_lag{lag}'] = data['price_diff'].shift(lag)
data = data.dropna()  # 移除滞后后的 NaN

# 提取变量（使用差分后的数据）
x = data['sentiment_diff']
y = data['price_diff']
x_sentiment_lags = data[[f'sentiment_diff_lag{lag}' for lag in range(1, max_lag + 1)]].values
x_price_lags = data[[f'price_diff_lag{lag}' for lag in range(1, max_lag + 1)]].values

# 非线性 Granger 因果检验
def fit_nonlinear_model(x_data, y_data):
    model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    model.fit(x_data, y_data)
    return model

# 步骤 1：仅使用价格的滞后值预测价格（无情感信息）
model_price_only = fit_nonlinear_model(x_price_lags, y)
y_prediction_price_only = model_price_only.predict(x_price_lags)
mse_price_only = mean_squared_error(y, y_prediction_price_only)

# 步骤 2：使用价格和情感的滞后值预测价格（有情感信息）
x_combined = np.hstack((x_price_lags, x_sentiment_lags))  # 组合价格和情感的滞后值
model_combined = fit_nonlinear_model(x_combined, y)
y_prediction_combined = model_combined.predict(x_combined)
mse_combined = mean_squared_error(y, y_prediction_combined)

# 计算非线性 Granger 因果统计量（MSE 差异）
granger_stat = mse_price_only - mse_combined
print(f"Nonlinear Granger Causality Statistic (MSE difference): {granger_stat}")

# 置换检验验证显著性
def statistic(x_price_perm, x_sentiment, y_data):
    x_combined_perm = np.hstack((x_price_perm, x_sentiment))
    model_perm = fit_nonlinear_model(x_combined_perm, y_data)
    y_prediction_perm = model_perm.predict(x_combined_perm)
    mse_perm = mean_squared_error(y_data, y_prediction_perm)
    return mse_price_only - mse_perm  # 零假设：无因果关系

nonlinear_granger_result = permutation_test((x_price_lags, x_sentiment_lags, y),
                                            statistic,
                                            n_resamples=100,  # 减少计算量
                                            alternative='greater')
print(f"P-value for Nonlinear Granger Causality: {nonlinear_granger_result.pvalue}")

# 可视化：比较预测误差（非线性 Granger 因果）
plt.figure(figsize=(10, 6))
plt.plot(y, label='Actual Price Changes', alpha=0.5)
plt.plot(y_prediction_price_only, label='Predicted (Price Only)', alpha=0.5)
plt.plot(y_prediction_combined, label='Predicted (Price + Sentiment)', alpha=0.5)
plt.legend()
plt.xlabel('Time')
plt.ylabel('Scaled Price Changes')
plt.title('Nonlinear Granger Causality: Prediction Comparison')
plt.show()

# 输出最终结论
if nonlinear_granger_result.pvalue < 0.05:
    print("结论：情感指数（sentiment）非线性 Granger 导致价格（price），因果关系显著（p < 0.05）。")
else:
    print("结论：情感指数（sentiment）不显著非线性 Granger 导致价格（price），因果关系不显著（p >= 0.05）。")