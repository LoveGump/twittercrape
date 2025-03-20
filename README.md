# TwitterCrape - 加密货币情感分析与价格预测工具

## 项目概述
本项目通过爬取Twitter上关于加密货币的讨论数据，分析社交媒体情感与加密货币价格之间的因果关系。利用多种统计和机器学习方法，探索情感指标是否能够预测或影响加密货币价格波动。

## 主要功能

### 数据收集
- **justasking.py**: 使用Twitter API爬取加密货币相关推文和媒体，包括文本、图片和视频内容

### 数据处理
- **datatime.py**: 合并和对齐情感数据与价格数据，确保两个时间序列可以进行后续分析

### 数据分析
- **crawlusingsnscrape.py**: 使用量化回归和转移熵分析情感与价格关系
  - 多项式量化回归分析不同分位数下的因果效应
  - Bootstrap方法计算置信区间
  - 转移熵分析信息流动方向

- **crawlusingsnscrape (1).py**: 非线性Granger因果检验
  - 检查数据平稳性并进行差分处理
  - 使用神经网络模型(MLPRegressor)进行非线性建模
  - 通过置换检验验证结果显著性

## 技术栈
- Python数据分析：Pandas, NumPy, Matplotlib, Seaborn
- 统计分析：statsmodels (量化回归)
- 机器学习：scikit-learn (神经网络模型)
- 信息理论：pyinform (转移熵)
- 推特API：Tweepy

## 使用方法
1. 配置Twitter API凭证（在justasking.py中）
2. 运行数据收集脚本获取推文数据
3. 处理并对齐情感数据和价格数据
4. 运行因果分析脚本分析情感与价格的关系

## 注意事项
- 使用前需要安装所有必要的Python依赖
- Twitter API的使用需要有效的开发者账号和认证
