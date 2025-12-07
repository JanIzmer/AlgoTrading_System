import pandas as pd
import numpy as np

def detect_cross(series_a: pd.Series, series_b: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Returns two boolean series: (cross_up, cross_down) for detecting crosses.
    (cross_up: series_a crosses above series_b)
    (cross_down: series_a crosses below series_b)
    """
    prev_a = series_a.shift(1)
    prev_b = series_b.shift(1)

    # Cross up: Current A > Current B AND Previous A <= Previous B
    cross_up = (series_a > series_b) & (prev_a <= prev_b)
    # Cross down: Current A < Current B AND Previous A >= Previous B
    cross_down = (series_a < series_b) & (prev_a >= prev_b)
    return cross_up.fillna(False), cross_down.fillna(False)


def calculate_indicators_and_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates technical indicators and trading flags based on OHLCV data.
    Expects df to be sorted by candle_time in ascending order for a single ticker.
    
    Args:
        df (pd.DataFrame): DataFrame containing "open", "high", "low", "close", "volume" columns.
        
    Returns:
        pd.DataFrame: DataFrame containing the calculated indicators and boolean flags (as integers).
    """
    # Create a copy to avoid SettingWithCopyWarning and work with a clean DF
    df_result = df.copy()

    # --- 1. Indicator Calculation (Logic from compute_indicators) ---
    
    # 1.1. Ensure all necessary columns are numeric
    for c in ("open", "high", "low", "close", "volume"):
        df_result[c] = pd.to_numeric(df_result[c], errors="coerce")

    # 1.2. EMA12, EMA26, MACD
    df_result["ema12"] = df_result["close"].ewm(span=12, adjust=False).mean()
    df_result["ema26"] = df_result["close"].ewm(span=26, adjust=False).mean()
    df_result["macd_line"] = df_result["ema12"] - df_result["ema26"]
    df_result["macd_signal"] = df_result["macd_line"].ewm(span=9, adjust=False).mean()
    df_result["macd_hist"] = df_result["macd_line"] - df_result["macd_signal"]

    # 1.3. SMA50, SMA20, STD20, Bollinger Bands
    df_result["sma50"] = df_result["close"].rolling(window=50, min_periods=1).mean()
    df_result["sma20"] = df_result["close"].rolling(window=20, min_periods=1).mean()
    df_result["std20"] = df_result["close"].rolling(window=20, min_periods=1).std(ddof=0)
    df_result["bb_upper"] = df_result["sma20"] + 2 * df_result["std20"]
    df_result["bb_lower"] = df_result["sma20"] - 2 * df_result["std20"]

    # 1.4. RSI14 (Wilder smoothing)
    delta = df_result["close"].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    # Wilder's smoothing uses alpha = 1/14
    roll_up = up.ewm(alpha=1/14, adjust=False).mean()
    roll_down = down.ewm(alpha=1/14, adjust=False).mean()
    rs = roll_up / roll_down
    df_result["rsi14"] = 100 - (100 / (1 + rs))

    # 1.5. ATR14 (Wilder's smoothing)
    prev_close = df_result["close"].shift(1)
    # True Range (TR)
    df_result["tr"] = pd.concat([
        (df_result["high"] - df_result["low"]).abs(),
        (df_result["high"] - prev_close).abs(),
        (df_result["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    # ATR14
    df_result["atr14"] = df_result["tr"].ewm(alpha=1/14, adjust=False).mean()

    # 1.6. ROC9 (rate of change over 9 periods)
    df_result["roc9"] = df_result["close"].pct_change(periods=9)

    # 1.7. Volume pct change 1
    df_result["volume_pct_change_1"] = df_result["volume"].pct_change(periods=1)

    # 1.8. VWAP per candle (typical price weighted by volume)
    typical_price = (df_result["high"] + df_result["low"] + df_result["close"]) / 3.0
    # Note: VWAP here is for the single candle's typical price weighted by volume
    df_result["vwap"] = (typical_price * df_result["volume"]) / df_result["volume"].replace({0: np.nan})

    # --- 2. Flag Generation (Logic from build_flags) ---
    
    # 2.1. EMA crosses
    ema_up, ema_down = detect_cross(df_result["ema12"], df_result["ema26"])
    df_result["ema12_cross_ema26_up"] = ema_up.astype(int)
    df_result["ema12_cross_ema26_down"] = ema_down.astype(int)

    # 2.2. SMA50 crosses on close
    sma50_up, sma50_down = detect_cross(df_result["close"], df_result["sma50"])
    df_result["close_cross_sma50_up"] = sma50_up.astype(int)
    df_result["close_cross_sma50_down"] = sma50_down.astype(int)

    # 2.3. MACD cross with signal
    macd_up, macd_down = detect_cross(df_result["macd_line"], df_result["macd_signal"])
    df_result["macd_cross_signal_up"] = macd_up.astype(int)
    df_result["macd_cross_signal_down"] = macd_down.astype(int)

    # 2.4. Bollinger band crosses on close
    # Cross above upper band
    bb_upper_up, _ = detect_cross(df_result["close"], df_result["bb_upper"])
    df_result["close_cross_upper_bb"] = bb_upper_up.astype(int)
    # Cross below lower band
    _, bb_lower_down = detect_cross(df_result["close"], df_result["bb_lower"])
    df_result["close_cross_lower_bb"] = bb_lower_down.astype(int)

    # 2.5. RSI thresholds
    df_result["rsi_overbought"] = (df_result["rsi14"] > 70).astype(int)
    df_result["rsi_oversold"] = (df_result["rsi14"] < 30).astype(int)

    # 2.6. Strong trend heuristic
    df_result["strong_trend"] = (
        ((df_result["macd_hist"] > 0) & (df_result["rsi14"] > 50) & (df_result["close"] > df_result["sma50"])) | # Bullish trend
        ((df_result["macd_hist"] < 0) & (df_result["rsi14"] < 50) & (df_result["close"] < df_result["sma50"]))  # Bearish trend
    ).astype(int)

    # --- 3. Final Column Selection ---

    # Define all relevant columns for the final output
    pk_cols = [c for c in ["ticker", "candle_time"] if c in df_result.columns]

    # All major indicators
    output_indicator_cols = [
        "ema12", "ema26", "macd_line", "macd_signal", "macd_hist", "sma50", "rsi14",
        "atr14", "roc9", "volume_pct_change_1", "sma20", "std20",
        "bb_upper", "bb_lower", "vwap"
    ]
    # All binary flags
    output_flag_cols = [
        "ema12_cross_ema26_up", "ema12_cross_ema26_down",
        "close_cross_sma50_up", "close_cross_sma50_down",
        "macd_cross_signal_up", "macd_cross_signal_down",
        "close_cross_upper_bb", "close_cross_lower_bb",
        "rsi_overbought", "rsi_oversold", "strong_trend",
    ]
    # OLHCV columns
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    # Combine all final columns

    final_cols = pk_cols + output_indicator_cols + output_flag_cols + ohlcv_cols

    # Return only the columns that exist in the resulting DataFrame
    return df_result[[c for c in final_cols if c in df_result.columns]].copy()