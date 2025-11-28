#!/bin/bash

# Parallel Binance klines downloader with improved error handling

set -euo pipefail

# Configuration
symbols=("BNBUSDT" "BTCUSDT" "ETHUSDT" "XRPUSDT" "SOLUSDT" "DOGEUSDT" "TRXUSDT" "ADAUSDT" "LINKUSDT" "SUIUSDT" "XLMUSDT" "AVAXUSDT" "BCHUSDT" "HYPEUSDT" "LTCUSDT" "HBARUSDT" "SHIBUSDT" "TONUSDT" "XMRUSDT" "UNIUSDT" "AAVEUSDT" "ENAUSDT" "NEARUSDT" "TAOUSDT" "OKBUSDT" "ZECUSDT" "ETCUSDT" "POLUSDT")
intervals=("aggTrades")
years=("2017" "2018" "2019" "2020" "2021" "2022" "2023" "2024" "2025")
months=("01" "02" "03" "04" "05" "06" "07" "08" "09" "10" "11" "12")
type="futures" # "spot" or "futures"
max_parallel=64  # Adjust based on your system and network

# Set base URL
if [[ "${type}" == "spot" ]]; then
  baseurl="https://data.binance.vision/data/spot/monthly/aggTrades"
elif [[ "${type}" == "futures" ]]; then
  baseurl="https://data.binance.vision/data/futures/um/monthly/aggTrades"
else
  echo "Error: Invalid type '${type}'. Must be 'spot' or 'futures'."
  exit 1
fi

# Download function for parallel execution
download_file() {
  local symbol=$1
  local interval=$2
  local year=$3
  local month=$4
  local baseurl=$5
  local type=$6
  local filename="${symbol}-${interval}-${year}-${month}.zip"
  local url="${baseurl}/${symbol}/${filename}"
  local dir="../data/aggTrades/${type}/raw/${symbol}"

  
  mkdir -p "${dir}"

  # echo url="${url}"
  
  rm -f "${dir}/${filename}"

  # Download with wget
  if wget -q -P "${dir}" "${url}" 2>/dev/null; then
    echo "[SUCCESS] ${filename}"
    # Unzip and remove archive
    if unzip -oq "${dir}/${filename}" -d "${dir}" 2>/dev/null; then
      rm "${dir}/${filename}"
    else
      echo "[ERROR] Failed to unzip: ${filename}"
    fi
  else
    echo "[SKIP] Not available: ${filename}"
  fi
}

export -f download_file

# Generate all combinations and process in parallel
echo "Starting parallel download (max ${max_parallel} concurrent jobs)..."
echo "Type: ${type}"
echo "Symbols: ${#symbols[@]}, Intervals: ${#intervals[@]}, Years: ${#years[@]}, Months: ${#months[@]}"
echo ""

for symbol in "${symbols[@]}"; do
  for interval in "${intervals[@]}"; do
    for year in "${years[@]}"; do
      for month in "${months[@]}"; do
        echo "${symbol} ${interval} ${year} ${month}"
      done
    done
  done
done | xargs -P "${max_parallel}" -n 4 bash -c 'download_file "$@" "'"${baseurl}"'" "'"${type}"'"' _

echo ""
echo "Download complete!"
