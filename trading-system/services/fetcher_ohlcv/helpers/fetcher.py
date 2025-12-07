import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Optional

BYBIT_KLINE_URL = "https://api.bybit.com/v5/market/kline"
USER_AGENT = "ohlcv-collector/1.0"

def fetch_ohlcv_bybit(symbol: str = "BTCUSDT", interval_min: int = 60, limit: int = 60) -> pd.DataFrame:
    """
    Fetch OHLCV from Bybit REST and return a DataFrame with columns:
      ['ticker', 'candle_time'(UTC datetimetz), 'open','high','low','close','volume']
    candle_time will be timezone-aware UTC datetime (no tz info stripped).
    """
    params = {"category": "linear", "symbol": symbol, "interval": str(interval_min), "limit": limit}
    headers = {"User-Agent": USER_AGENT}
    
    resp = requests.get(BYBIT_KLINE_URL, params=params, headers=headers, timeout=15)
    resp.raise_for_status()
    
    j = resp.json()
    # V5 K-Line data is always in the 'list' key under 'result'
    data = j.get("result", {}).get("list")
    
    # Check if data is a non-empty list
    if not isinstance(data, list) or not data:
        print(f"DEBUG: API returned no K-Line data for {symbol}.")
        return pd.DataFrame(columns=["ticker","candle_time","open","high","low","close","volume"])

    rows = []
    
    # Bybit V5 K-Line list format: [timestamp (ms), open, high, low, close, volume, turnover]
    for item in data:
        # We only expect the list/tuple format from V5
        if isinstance(item, (list, tuple)) and len(item) >= 6:
            ts, o, h, l, c, v = item[0], item[1], item[2], item[3], item[4], item[5]
        else:
            # Skip unexpected formats
            continue

        if isinstance(ts, (str, int, float)):
            try:
                ts_ms = int(ts)
                ts_seconds = ts_ms / 1000.0
                dt = datetime.fromtimestamp(ts_seconds, tz=timezone.utc)
            except (ValueError, TypeError, OverflowError):
                print(f"DEBUG: Failed to convert timestamp {ts}")
                continue
        else:
            continue

        try:
            open_val = float(o)
            high_val = float(h)
            low_val = float(l)
            close_val = float(c)
            volume_val = float(v)
        except (ValueError, TypeError):
            # Skip this row if prices/volume are invalid
            continue
            
        rows.append({
            "ticker": symbol,
            "candle_time": dt,
            "open": open_val,
            "high": high_val,
            "low": low_val,
            "close": close_val,
            "volume": volume_val,
        })

    if not rows:
        print(f"DEBUG: All rows were filtered out for {symbol}.")
        return pd.DataFrame(columns=["ticker","candle_time","open","high","low","close","volume"])

    df = pd.DataFrame(rows)
    # sort, set index by time (UTC)
    df = df.sort_values("candle_time").reset_index(drop=True)
    # ensure candle_time dtype is pandas datetime UTC
    df["candle_time"] = pd.to_datetime(df["candle_time"], utc=True)
    return df