import polars as pl
import matplotlib.pyplot as plt    
import pathlib
import mplfinance as mpf

def plot_close_price(df: pl.DataFrame, symbol: str):
    """
    Plots the closing price over time for a given symbol.

    Args:
        df (pl.DataFrame): The input DataFrame containing the data.
        symbol (str): The symbol to filter the data by.
    """
    # Filter the DataFrame for the specified symbol
    symbol_df = df.filter(pl.col("symbol") == symbol)

    # Convert Polars DataFrame to Pandas for plotting
    pandas_df = symbol_df.to_pandas()

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(pandas_df['time'], pandas_df['close'], label='Close Price')
    plt.title(f'Closing Price Over Time for {symbol}')
    plt.xlabel('Time')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid()
    plt.savefig("close_price.png")
    
def plot_volume_histogram(df: pl.DataFrame, symbol: str):
    """
    Plots a histogram of trading volumes for a given symbol.

    Args:
        df (pl.DataFrame): The input DataFrame containing the data.
        symbol (str): The symbol to filter the data by.
    """
    # Filter the DataFrame for the specified symbol
    symbol_df = df.filter(pl.col("symbol") == symbol)

    # Convert Polars DataFrame to Pandas for plotting
    pandas_df = symbol_df.to_pandas()

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.hist(pandas_df['volume'], bins=50, alpha=0.7, color='blue')
    plt.title(f'Trading Volume Distribution for {symbol}')
    plt.xlabel('Volume')
    plt.ylabel('Frequency')
    plt.grid()
    plt.savefig("volume_histogram.png")
    
def plot_candlestick_chart(df: pl.DataFrame, symbol: str):
    """
    Plots a candlestick chart for a given symbol.

    Args:
        df (pl.DataFrame): The input DataFrame containing the data.
        symbol (str): The symbol to filter the data by.
    """
    import mplfinance as mpf

    # Filter the DataFrame for the specified symbol
    symbol_df = df.filter(pl.col("symbol") == symbol)

    # Convert Polars DataFrame to Pandas for plotting
    pandas_df = symbol_df.to_pandas()
    pandas_df.set_index('time', inplace=True)

    # Prepare data for mplfinance
    ohlc_data = pandas_df[['open', 'high', 'low', 'close']]

    # Plotting
    mpf.plot(ohlc_data, type='candle', style='charles',
             title=f'Candlestick Chart for {symbol}',
             ylabel='Price',
             volume=True)
# Define the schema for the DataFrame
col_schema = {
    "time": pl.Datetime("ms"),
    "open": pl.Float32,
    "high": pl.Float32,
    "low": pl.Float32,
    "close": pl.Float32,
    "volume": pl.Float32,
    "quote_volume": pl.Float32,
    "count": pl.Float32,
    "taker_buy_volume": pl.Float32,
    "taker_buy_quote_volume": pl.Float32,
    "symbol": pl.Utf8
}

file_path = pathlib.Path("data/futures/dataset/train/")


data = pl.scan_parquet(str(file_path/ "*.parquet"), glob=True)

df = data.filter(pl.col("symbol") == "BTCUSDT").collect()

plot_close_price(df, "BTCUSDT")
plot_volume_histogram(df, "BTCUSDT")
# plot_candlestick_chart(df, "BTCUSDT")

