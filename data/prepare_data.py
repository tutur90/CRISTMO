import os
import pandas as pd
import pathlib
import argparse
import logging
import polars as pl
from typing import Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    parser.add_argument("--input", type=str, default="data/futures/raw", help="Path to the input data directory")
    parser.add_argument("--output", type=str, default="data/futures/dataset/", help="Path to the output data directory")
    parser.add_argument("--no-overwrite", action="store_true", help="Skip processing if output directory exists and is not empty")
    parser.add_argument("--no-max-scaling", action="store_true", help="Skip max scaling")
    parser.add_argument("--no-log-scaling", action="store_true", help="Skip log scaling")
    parser.add_argument("--no-bias-removal", action="store_true", help="Skip bias removal")
    parser.add_argument("--eps", type=float, default=1e-20, help="Small value to avoid log(0)")
    args = parser.parse_args()
    
    input_path = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)
    
    # Check if we should process data
    if not output_path.exists() or not any(output_path.iterdir()) or not args.no_overwrite:
        logger.info(f"Processing data from {input_path} to {output_path}")
        
        if not input_path.exists():
            logger.error(f"Input directory {input_path} does not exist.")
            exit(1)
    
        for dir in input_path.iterdir():
            if not (dir.is_dir() and any(dir.iterdir())):
                continue
                
            logger.info(f"Processing {dir.name}")
            
            col = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 
                'quote_volume', 'count', 'taker_buy_volume', 'taker_buy_quote_volume', 'ignore']
            
            # Read all CSVs in directory
            df = pl.read_csv(
                str(dir / "*.csv"), 
                has_header=False, 
                new_columns=col
            ).filter(pl.col("open_time").ne("open_time"))
            
            # Cast columns to appropriate types
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
            
            # Convert timestamp and add symbol
            df = df.with_columns(
                pl.from_epoch(pl.col("time"), "ms").alias("time"),
                pl.lit(dir.name).alias("symbol")
            )

            # Clean data
            df = df.unique().drop_nulls().drop_nans()
            df = df.sort("time")

            # Check for data quality issues
            df = df.with_columns(
                pl.col("time").diff().shift(-1).alias("time_diff")
            )

            if df["time_diff"][1:].min() < pd.Timedelta("1min"):
                logger.warning(f"Duplicate timestamps found in {dir.name}")

            if df["time_diff"].max() > pd.Timedelta("1min"):
                logger.warning(f"Missing timestamps found in {dir.name}")
                logger.warning(f"Time differences: {df['time_diff'].unique().sort()}")
                missing = df.filter(pl.col('time').diff().shift(-1).gt(pd.Timedelta('1min')))
                logger.warning(f"Missing timestamps at: {missing}")

            df = df.drop("time_diff")
            
            # Define train/val/test splits
            test_period = pd.DateOffset(months=1)
            val_period = pd.DateOffset(months=1)

            max_date = df["time"].max()
            test_start = max_date - test_period
            val_start = test_start - val_period
            
            df = df.with_row_index("idx")
            
            # Apply log scaling
            if not args.no_log_scaling:
                logger.info(f"Applying log scaling to {dir.name}")
                for col_name in ["open", "high", "low", "close"]:
                    df = df.with_columns(
                        pl.col(col_name).log().alias(col_name)
                    )
                    
                for col_name in ["volume", "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume"]:
                    df = df.with_columns(
                        (pl.col(col_name) + args.eps).log().alias(col_name)
                    )

            # # Apply max scaling
            # if not args.no_max_scaling:
            #     logger.info(f"Applying max scaling to {dir.name}")
            #     max_vals = df.filter(pl.col("time") < val_start).select([
            #         pl.col("open").max().alias("open_max"),
            #         pl.col("high").max().alias("high_max"),
            #         pl.col("low").max().alias("low_max"),
            #         pl.col("close").max().alias("close_max"),
            #         pl.col("volume").max().alias("volume_max"),
            #         pl.col("quote_volume").max().alias("quote_volume_max"),
            #         pl.col("count").max().alias("count_max"),
            #         pl.col("taker_buy_volume").max().alias("taker_buy_volume_max"),
            #         pl.col("taker_buy_quote_volume").max().alias("taker_buy_quote_volume_max"),
            #     ]).to_dicts()[0]
                
            #     for key, value in max_vals.items():
            #         col_name = key.replace("_max", "")
            #         if args.no_log_scaling:
            #             df = df.with_columns(
            #                 (pl.col(col_name) / value).alias(col_name)
            #             )
            #         else:
            #             df = df.with_columns(
            #             (pl.col(col_name) - value).alias(col_name)  # log scale: - = /
            #         )

            print(df.drop_nans())

            # Apply bias removal
            if not args.no_bias_removal:
                logger.info(f"Applying bias removal to {dir.name}")
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
                
                print("mean_vals", mean_vals)
                
                max_train_idx = df.filter(pl.col("time") < val_start)["idx"].max()
                
                
                
                for key, value in mean_vals.items():
                    col_name = key.replace("_mean", "")
                    
                    df = df.with_columns(
                        pl.when(pl.col("time") < val_start)
                        .then(pl.col(col_name) - pl.col("idx") * value)
                        .otherwise(pl.col(col_name) - (max_train_idx * value))
                        .alias(col_name)
                    )
                    
            df = df.drop("idx")
            
            # Save split datasets
            df.filter(pl.col("time") >= test_start).write_parquet(
                output_path / "test" / f"{dir.name}.parquet", 
                mkdir=True
            )
            df.filter((pl.col("time") >= val_start) & (pl.col("time") < test_start)).write_parquet(
                output_path / "val" / f"{dir.name}.parquet", 
                mkdir=True
            )
            df.filter(pl.col("time") < val_start).write_parquet(
                output_path / "train" / f"{dir.name}.parquet", 
                mkdir=True
            )
            
            logger.info(f"Saved {dir.name} to train/val/test splits")
    else:
        logger.info(f"Output directory {output_path} already exists and is not empty. Skipping processing.")
        logger.info(f"Use --no-overwrite=False or delete the output directory to reprocess.")