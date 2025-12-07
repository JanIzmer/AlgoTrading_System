from sqlalchemy.engine import Engine
import pandas as pd
from sqlalchemy.sql import text 
import uuid
from sqlalchemy.types import String, DateTime, Float, Boolean, Integer 

# --- CONSTANTS ---
KLINE_TABLE_COLS = [
    "ticker", "candle_time", "open", "high", "low", "close", "volume"]

TECHNICAL_INDICATOR_COLS = [
    "ticker", "candle_time", "atr14", 
    "ema12_cross_ema26_up" ,
    "ema12_cross_ema26_down" , 
    "close_cross_sma50_up" ,
    "close_cross_sma50_down" ,
    "macd_cross_signal_up" ,
    "macd_cross_signal_down" ,
    "rsi_overbought",
    "rsi_oversold",
    "close_cross_upper_bb" ,
    "close_cross_lower_bb" , 
    "strong_trend"
]

# --- HELPER FUNCTION FOR ROBUST SQL COLUMN LISTING ---
def format_cols(cols):
    """Formats column names, quoting reserved keywords with backticks (` `) for MySQL."""
    return ', '.join(f'`{c}`' if c in ['open', 'high', 'low', 'close', 'volume', 'atr14'] else c for c in cols)

def load_data_to_db(df: pd.DataFrame, engine: Engine):
    """
    Splits the DataFrame and loads it into two separate database tables using 
    UPSERT (INSERT IGNORE) suitable for MySQL, within a single atomic transaction.
    """
    
    # 1. Prepare DataFrames, Constants, and Dtype Mapping
    ohlcv_df = df[KLINE_TABLE_COLS]
    indicator_df = df[TECHNICAL_INDICATOR_COLS]
    unique_suffix = str(uuid.uuid4()).replace('-', '')[:8]
    
    TEMP_KLINE_TABLE = f"temp_kline_upload_{unique_suffix}"
    TARGET_KLINE_TABLE = "kline_data"
    TARGET_INDICATOR_TABLE = "technical_indicators"
    TEMP_INDICATORS_TABLE = f"temp_indicator_upload_{unique_suffix}"
    
    ohlcv_dtype = {
        'ticker': String(10),
        'candle_time': DateTime,
        'open': Float,
        'high': Float,
        'low': Float,
        'close': Float,
        'volume': Float,
    }

    indicator_dtype = {
        'ticker': String(10),
        'candle_time': DateTime,
        'atr14': Float,
        **{col: Integer for col in TECHNICAL_INDICATOR_COLS if col not in ['ticker', 'candle_time', 'atr14']}
    }
    
    # 2. Generate Robust SQL Strings
    kline_cols_formatted = format_cols(KLINE_TABLE_COLS)
    indicator_cols_formatted = format_cols(TECHNICAL_INDICATOR_COLS)

    # OHLCV UPSERT SQL (ИСПОЛЬЗУЕМ INSERT IGNORE INTO)
    upsert_kline_sql = text(f"""
        INSERT IGNORE INTO {TARGET_KLINE_TABLE} ({kline_cols_formatted})
        SELECT {kline_cols_formatted}
        FROM {TEMP_KLINE_TABLE};
    """)

    # INDICATORS UPSERT SQL (ИСПОЛЬЗУЕМ INSERT IGNORE INTO)
    upsert_indicator_sql = text(f"""
        INSERT IGNORE INTO {TARGET_INDICATOR_TABLE} ({indicator_cols_formatted})
        SELECT {indicator_cols_formatted}
        FROM {TEMP_INDICATORS_TABLE};
    """)
    
    # 3. Main Transaction (Atomic Load)
    try:
        with engine.begin() as connection:
            
            # --- A. OHLCV Data Processing (UPSERT) ---
            
            try:
                # Load into temporary table (1)
                ohlcv_df.to_sql(
                    TEMP_KLINE_TABLE,
                    con=connection,
                    if_exists="replace", 
                    index=False,
                    chunksize=1000,
                    dtype=ohlcv_dtype 
                )
            except Exception as e:
                print(f"Error uploading OHLCV data to temporary table: {e}")
                raise
                
            # Execute UPSERT (2)
            result_kline = connection.execute(upsert_kline_sql)
            print(f"Inserted/Ignored {result_kline.rowcount} OHLCV rows into '{TARGET_KLINE_TABLE}'.")
            
            # Clean up temporary table (3)
            connection.execute(text(f"DROP TABLE IF EXISTS {TEMP_KLINE_TABLE};"))

            # --- B. Indicators Data Processing (UPSERT) ---
            
            # Load into temporary table (1)
            indicator_df.to_sql(
                TEMP_INDICATORS_TABLE,
                con=connection,
                if_exists="replace",
                index=False,
                chunksize=1000,
                dtype=indicator_dtype 
            )
            
            # Execute UPSERT (2)
            result_indicator = connection.execute(upsert_indicator_sql)
            print(f"Inserted/Ignored {result_indicator.rowcount} indicator rows into '{TARGET_INDICATOR_TABLE}'.")
            
            # Clean up temporary indicators table (3)
            connection.execute(text(f"DROP TABLE IF EXISTS {TEMP_INDICATORS_TABLE};"))
            
    except Exception as e:
        print(f"Database load failed. Transaction rolled back: {e}")
        raise 
    
    print("Both tables successfully updated.")