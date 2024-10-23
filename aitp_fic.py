# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from datetime import date
import yfinance as yf  # Importing yfinance for American stock data
plt.style.use('fivethirtyeight')  # Setting matplotlib style

# Defining Parameters
stocksymbols = ['AAPL', 'NVDA'] # Example symbols 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'FB', 'NFLX', 'NVDA'
startdate = date(2018, 10, 14)
end_date = date.today()
print(end_date)
print(f"You have {len(stocksymbols)} assets in your portfolio")

# Fetching Data
df = pd.DataFrame()
for symbol in stocksymbols:
    data = yf.download(symbol, start=startdate, end=end_date)[['Close']]
    
    if data.empty:  # Check if data is empty
        print(f"No data found for {symbol}. Skipping.")
        continue  # Skip to the next symbol if no data is found
    
    data.rename(columns={'Close': symbol}, inplace=True)  # Rename the 'Close' column to the symbol
    if df.empty:  # If df is empty, initialize it
        df = data
    else:
        df = df.join(data)
        
print(df)

# # Analysis
# fig, ax = plt.subplots(figsize=(15, 8))

# for i in df.columns.values:
#     ax.plot(df[i], label=i)

# ax.set_title("Portfolio Close Price History")
# ax.set_xlabel('Date', fontsize=18)
# ax.set_ylabel('Close Price)', fontsize=18)
# ax.legend(df.columns.values, loc='upper left')

# # Correlation Matrix

# correlation_matrix = df.corr(method='pearson')
# print(correlation_matrix)

# fig1 = plt.figure()
# sb.heatmap(correlation_matrix, 
#            xticklabels=correlation_matrix.columns, 
#            yticklabels=correlation_matrix.columns,
#            cmap='YlGnBu', ###this is just the color of the heatmap 
#            annot=True, 
#            linewidth=0.5)

# print('Correlation between Stocks in your portfolio')

# # Risk & Return
# #Calculate daily simple returns
# daily_simple_return = df.pct_change(1)
# daily_simple_return.dropna(inplace=True)

# # Print daily simple returns
# print(daily_simple_return)
# print('Daily simple returns')

# # Plot daily simple returns
# fig, ax = plt.subplots(figsize=(15, 8))

# for i in daily_simple_return.columns.values:
#     ax.plot(daily_simple_return[i], lw=2, label=i)

# ax.legend(loc='upper right', fontsize=10)
# ax.set_title('Volatility in Daily Simple Returns')
# ax.set_xlabel('Date')
# ax.set_ylabel('Daily Simple Returns')

# # Average Daily returns
# print('Average Daily returns(%) of stocks in your portfolio')
# Avg_daily = daily_simple_return.mean()
# print(Avg_daily*100)

# # Risk Box-Plot
# # Box plot for daily simple returns
# daily_simple_return.plot(kind="box", figsize=(20, 10), title="Risk Box Plot")


# # Print annualized standard deviation
# print('Annualized Standard Deviation (Volatility(%), 252 trading days) of individual stocks in your portfolio based on daily simple returns:')
# print(daily_simple_return.std() * np.sqrt(252) * 100)

# # Return Per Unit Of Risk
# print(Avg_daily / (daily_simple_return.std() * np.sqrt(252)) * 100)

# # Cumulative Returns
# daily_cumulative_simple_return = (daily_simple_return + 1).cumprod()
# print(daily_cumulative_simple_return)

# # Visualize the daily cumulative simple return
# fig, ax = plt.subplots(figsize=(18, 8))

# for column in daily_cumulative_simple_return.columns:
#     ax.plot(daily_cumulative_simple_return[column], label=str(column))

# ax.legend(loc='upper left', fontsize=12)
# ax.set_title('Daily Cumulative Simple Returns')
# ax.set_xlabel('Date')
# ax.set_ylabel('Cumulative Returns')

# # Visualize the daily cumulative simple return
# fig, ax = plt.subplots(figsize=(18, 8))

# for i in daily_cumulative_simple_return.columns.values:
#     ax.plot(daily_cumulative_simple_return[i], lw=2, label=i)

# ax.legend(loc='upper left', fontsize=10)
# ax.set_title('Daily Cumulative Simple Returns/Growth of Investment')
# ax.set_xlabel('Date')
# ax.set_ylabel('Growth of ₨ 1 Investment')

# plt.show()
