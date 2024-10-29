import pandas as pd
import yfinance as yf  # Importing yfinance for American stock data
from datetime import date

stocksymbols = ['AAPL', 'NVDA','MSFT'] # Example symbols 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'FB', 'NFLX', 'NVDA'
startdate = date(2018, 10, 14)
end_date = date.today()
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
        
df.to_csv("ticker_data.csv", index=False)