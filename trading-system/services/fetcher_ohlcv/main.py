from sql.config.db_config import get_db_engine
from services.fetcher_ohlcv.helpers.fetcher import fetch_ohlcv_bybit
from services.fetcher_ohlcv.helpers.calculate_indicators import calculate_indicators_and_flags
from services.fetcher_ohlcv.helpers.data_loader import load_data_to_db
import asyncio

KLINE_FETCH_INTERVAL_MIN = 30  # minutes

async def main_loop():
    while True:
        # Connect to database 
        engine = get_db_engine()

        # Fetch new OHLCV data from exchange (Extract)
        new_data = fetch_ohlcv_bybit(symbol="BTCUSDT", interval_min=60, limit=100)
        print(new_data.head)

        # Calculate indicators (Transform)
        if not new_data.empty:
            transformed_data = calculate_indicators_and_flags(new_data)
            print(transformed_data.head)
            print(transformed_data.columns)
            # Store transformed data back to database (Load)
            try:
                load_data_to_db(transformed_data, engine)
            except Exception as e:
                print(f"Error loading data to DB: {e}")
        else:
            print("No new data fetched.")

        await asyncio.sleep(KLINE_FETCH_INTERVAL_MIN)

if __name__ == "__main__":
    asyncio.run(main_loop())