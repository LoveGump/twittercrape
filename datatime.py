import pandas as pd
sentiment_path = r'C:\Users\zhang\Desktop\sentiment.csv'
price_path = r'C:\Users\zhang\Desktop\price.csv'

# 读取情感指标数据
sentiment_data = pd.read_csv(sentiment_path)
price_data = pd.read_csv(price_path)

# 2. 对齐数据
# 假设两个数据集都有一个 'Date' 列
merged_data = pd.merge(sentiment_data, price_data, on='Date', how='inner')  # 按日期对齐
merged_data['Date'] = pd.to_datetime(merged_data['Date'])
merged_data = merged_data.sort_values(by='Date')
merged_data = merged_data.dropna()
print("删除缺失值后的数据：")
print(merged_data.head())

output_path = r'C:\Users\zhang\Desktop\merged_data.csv'
merged_data.to_csv(output_path, index=False)
print(f"对齐后的数据已保存到：{output_path}")