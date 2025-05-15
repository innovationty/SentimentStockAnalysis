## Social Media Sentiment and Stock Price Analysis Research
This project analyzes the relationship between social media sentiment and stock prices, exploring the potential value of sentiment indicators for stock price prediction through natural language processing and statistical analysis techniques.

### Environment Requirements
#### Python Version
Python 3.8+
#### Core Dependencies

numpy==2.2.5

pandas==2.2.3

pandas_datareader==0.10.0

seaborn==0.13.2
w
statsmodels==0.14.4

tqdm==4.67.1

transformers==4.51.3

You can install all dependencies using the following command:

pip install -r requirements.txt

### Running the Files

First run sentiment_factor.py, you can choose to use GPU for processing, to get tweets_with_sentiment.csv.

Run daily_sentiment.py to get daily_sentiment.csv.
Run rolling.py to get daily_sentiment.csv
Run merge_prices.py to get merged.csv
Finally, run analysis.py to see the output results and charts
Run sentiment_reacts_to_returns.py to get results on how stock returns affect sentiment.

Sample outputs are available in the sample folder