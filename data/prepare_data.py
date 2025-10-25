import os
import pandas as pd
import pathlib
import argparse
import logging
import polars as pl
from typing import Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# init_schema = pl.Schema({
#     "open_time": pl.Int64,
#     "open": pl.Float32,
#     "high": pl.Float32,
#     "low": pl.Float32,
#     "close": pl.Float32,
#     "volume": pl.Float32,
#     "close_time": pl.Int64,
#     "quote_volume": pl.Float32,
#     "count": pl.Int32,
#     "taker_buy_volume": pl.Float32,
#     "taker_buy_quote_volume": pl.Float32,
#     "ignore": pl.Float32
# })

schema = pl.Schema({
    "open_time": pl.Datetime("ms"),
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
})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for analysis")
    parser.add_argument("--input", type=str, default="data/futures/raw", help="Path to the input data file")
    parser.add_argument("--output", type=str, default="data/futures/dataset/", help="Path to the output data file")
    parser.add_argument("--max_scaling", type=bool, default=True, help="Whether to apply max scaling")
    parser.add_argument("--log_scaling", type=bool, default=True, help="Whether to apply log scaling")
    parser.add_argument("--bias_removal", type=bool, default=True, help="Whether to apply bias removal")
    parser.add_argument("--eps", type=float, default=1e-8, help="Small value to avoid log(0)")
    args = parser.parse_args()
    
    input_path = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)
    
    if not output_path.exists() or not any(output_path.iterdir()):
        logger.info(f"Processing data from {input_path} to {output_path}")
        
        if not input_path.exists():
            logger.error(f"Input file {input_path} does not exist.")
            exit(1)
    
        for dir in input_path.iterdir():
            if not (dir.is_dir() and any(dir.iterdir())):
                continue
                
            logger.info(f"Processing {dir.name}")
            
            col = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 
                'quote_volume', 'count', 'taker_buy_volume', 'taker_buy_quote_volume', 'ignore']
            
            # Read all CSVs in directory
            df = pl.read_csv(str(dir / "*.csv"),  has_header=False, new_columns=col).filter(pl.col("open_time").ne("open_time"))
            
            

            df = df.filter(pl.col("open_time").ne("open_time"))
            
            df = df.select(
                pl.col("open_time").cast(pl.Int64).alias("time"),
                pl.col("open").cast(pl.Float32),
                pl.col("high").cast(pl.Float32),
                pl.col("low").cast(pl.Float32),
                pl.col("close").cast(pl.Float32),
                pl.col("volume").cast(pl.Float32),
                pl.col("quote_volume").cast(pl.Float32),
                pl.col("count").cast(pl.Float32),
                pl.col("taker_buy_volume").cast(pl.Float32),
                pl.col("taker_buy_quote_volume").cast(pl.Float32),
            )
            
            df = df.with_columns(
                pl.from_epoch(pl.col("time"), "ms").alias("time"),
                pl.lit(dir.name).alias("symbol")
            )

            df = df.unique().drop_nulls().drop_nans()
            
            
            df = df.sort("time")

            df = df.with_columns(
                pl.col("time").diff().shift(-1).alias("time_diff")
            )




            if df["time_diff"][1:].min() < pd.Timedelta("1min"):
                logger.warning(f"Duplicate timestamps found in {dir.name}")

            if df["time_diff"].max() > pd.Timedelta("1min"):
                logger.warning(f"Missing timestamps found in {dir.name}")
                logger.warning(f"Time differences: {df["time_diff"].unique().sort()}")
                logger.warning(f"Missing timestamps at: {df.filter(pl.col('time').diff().shift(-1).gt(pd.Timedelta('1min')))}")

            
            # df = df.join(pl.DataFrame({"time": pd.date_range(start=df["time"].min(), end=df["time"].max(), freq="1min", unit="ms")}), on="time", how="right").sort("time")
            
            
            df = df.drop("time_diff")
            
            test_period = pd.DateOffset(months=1)
            val_period = pd.DateOffset(months=1)

            max_date = df["time"].max()
            test_start = max_date - test_period
            val_start = test_start - val_period
            
            df = df.with_row_index("idx")
            
            # Pass to log scale
            for col_name in ["open", "high", "low", "close", "volume", "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume"]:
                df = df.with_columns(
                        pl.when(pl.col(col_name) > 0)
                        .then(pl.col(col_name).log())
                        .otherwise((pl.col(col_name) + args.eps).log())
                        .alias(col_name)
                    )

            if args.max_scaling:
                max_vals = df.filter(pl.col("time") < val_start).select([
                    pl.col("open").max().alias("open_max"),
                    pl.col("high").max().alias("high_max"),
                    pl.col("low").max().alias("low_max"),
                    pl.col("close").max().alias("close_max"),
                    pl.col("volume").max().alias("volume_max"),
                    pl.col("quote_volume").max().alias("quote_volume_max"),
                    pl.col("count").max().alias("count_max"),
                    pl.col("taker_buy_volume").max().alias("taker_buy_volume_max"),
                    pl.col("taker_buy_quote_volume").max().alias("taker_buy_quote_volume_max"),
                ]).to_dicts()[0]
                
                for key, value in max_vals.items():
                    col_name = key.replace("_max", "")
                    df = df.with_columns(
                        (pl.col(col_name) - value).alias(col_name) # log scal: - = /
                    )
                    
            if args.bias_removal:
                mean_vals = df.filter(pl.col("time") < val_start).select([
                    pl.col("open").diff().mean().alias("open_mean"),
                    pl.col("high").diff().mean().alias("high_mean"),
                    pl.col("low").diff().mean().alias("low_mean"),
                    pl.col("close").diff().mean().alias("close_mean"),
                    pl.col("volume").diff().mean().alias("volume_mean"),
                    pl.col("quote_volume").diff().mean().alias("quote_volume_mean"),
                    pl.col("count").diff().mean().alias("count_mean"),
                    pl.col("taker_buy_volume").diff().mean().alias("taker_buy_volume_mean"),
                    pl.col("taker_buy_quote_volume").diff().mean().alias("taker_buy_quote_volume_mean"),
                ]).to_dicts()[0]
                
                max_train_idx = df.filter(pl.col("time") < val_start)["idx"].max()
                
                for key, value in mean_vals.items():
                    col_name = key.replace("_mean", "")
                    
                    df = df.with_columns(
                        pl.when(pl.col("time") < val_start)
                        .then(pl.col(col_name) - pl.col("idx") * value)
                        .otherwise(pl.col(col_name) - (max_train_idx * value))
                        .alias(col_name)
                    )
                    
            if not args.log_scaling:
                for col_name in ["open", "high", "low", "close", "volume", "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume"]:
                    df = df.with_columns(
                        pl.col(col_name).exp().alias(col_name)
                    )


            
            # Split data correctly
            df.filter(pl.col("time") >= test_start).write_parquet(output_path / "test" / f"{dir.name}.parquet", mkdir=True)
            df.filter((pl.col("time") >= val_start) & (pl.col("time") < test_start)).write_parquet(output_path / "val" / f"{dir.name}.parquet", mkdir=True)
            df.filter(pl.col("time") < val_start).write_parquet(output_path / "train" / f"{dir.name}.parquet", mkdir=True)
            
            logger.info(f"Saved {dir.name}")
    else:
        logger.info(f"Output directory {output_path} already exists and is not empty. Skipping processing.")
        
