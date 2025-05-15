## 社交媒体情绪与股价分析研究
本项目分析社交媒体情绪与股票价格之间的关系，通过使用自然语言处理和统计分析技术，探索情绪指标对股价预测的潜在价值。

### 环境要求
#### Python 版本
Python 3.8+
#### 核心依赖包

numpy==2.2.5

pandas==2.2.3

pandas_datareader==0.10.0

seaborn==0.13.2

statsmodels==0.14.4

tqdm==4.67.1

transformers==4.51.3

您可以使用以下命令安装所有依赖：

pip install -r requirements.txt

### 文件运行

先运行sentiment_factor.py，可以自行选择使用GPU进行处理，得到tweets_with_sentiment.csv。
运行daily_sentiment.py，得到daily_sentiment.csv。
运行rolling.py，得到daily_sentiment.csv
运行merge_prices.py，得到merged.csv
最后运行analysis.py，可以看到输出的结果和图表
运行sentiment_reacts_to_returns.py，会得到股票收益对情绪影响的相关结果。

示例输出已经放在sample文件夹中