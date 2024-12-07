import pandas as pd
import matplotlib.pyplot as plt
from pycoingecko import CoinGeckoAPI
from datetime import datetime, timedelta

# Initialize CoinGecko API client
cg = CoinGeckoAPI()

# Define parameters
crypto_id = 'bitcoin'
vs_currency = 'usd'
days = 'max'  # 'max' retrieves the full available history

# Fetch historical market data
data = cg.get_coin_market_chart_by_id(id=crypto_id, vs_currency=vs_currency, days=days)

# Convert data to DataFrame
prices = data['prices']
df = pd.DataFrame(prices, columns=['timestamp', 'price'])

# Convert timestamp to datetime
df['date'] = pd.to_datetime(df['timestamp'], unit='ms')

# Filter data for the desired date range
start_date = '2020-01-01'
end_date = '2024-12-06'
mask = (df['date'] >= start_date) & (df['date'] <= end_date)
df = df.loc[mask]

# Save to CSV with meaningful name
csv_filename = f'{crypto_id}_price_data_{start_date}_to_{end_date}.csv'
df.to_csv(csv_filename, index=False)

# Plot closing prices
plt.figure(figsize=(10, 5))
plt.plot(df['date'], df['price'], label='Closing Price')
plt.xlabel('Date')
plt.ylabel(f'Price ({vs_currency.upper()})')
plt.title(f'{crypto_id.capitalize()} Price from {start_date} to {end_date}')
plt.legend()
plt.grid(True)
plt.show()
