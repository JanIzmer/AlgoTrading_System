from sqlalchemy.engine import Engine
import pandas as pd
from sqlalchemy.sql import text 
import uuid
from sqlalchemy.types import String, DateTime, Float, Integer, JSON 
from typing import Dict
from datetime import datetime
from sqlalchemy.engine import Engine

# --- CONSTANTS ---

# Expected columns in the incoming DataFrame (after sentiment analysis)
SENTIMENT_DF_COLS = [
    'ticker', 'publication_time', 'candle_time', 'title', 'description', 
    'source', 'vader_score', 'finbert_label', 'finbert_score', 'finbert_probs'
]

# Mapping between sentiment DataFrame columns and the target database table columns
SENTIMENT_MAPPING = {
    'title': 'headline',
    'description': 'content',
    'vader_score': 'sentiment_score',
}

# --- HELPER FUNCTIONS ---

def get_source_data(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts unique source names and URL bases for the news_source table."""
    # Assuming 'source' in the DF contains the name, and we don't have url_base yet, 
    # we use the source name as a placeholder for url_base (optional)
    source_df = df[['source']].drop_duplicates().copy()
    source_df.rename(columns={'source': 'name'}, inplace=True)
    # Adding a placeholder for url_base if not present
    source_df['url_base'] = "Cryptopanic"  # Placeholder, can be modified if actual URL bases are available
    return source_df

def upsert_source(source_df: pd.DataFrame, connection) -> None:
    """Inserts new news sources using INSERT...ON DUPLICATE KEY UPDATE."""
    
    # SQL UPSERT: Tries to insert the source name. If the name is duplicated, it updates url_base.
    # We use a temporary table to avoid executing thousands of small SQL statements.
    
    TEMP_SOURCE_TABLE = f"temp_source_upload_{str(uuid.uuid4()).replace('-', '')[:8]}"
    TARGET_SOURCE_TABLE = "news_source"
    
    source_dtype = {
        'name': String(100),
        'url_base': String(255)
    }
    
    # 1. Load data into temporary table
    source_df.to_sql(
        TEMP_SOURCE_TABLE,
        con=connection,
        if_exists="replace",
        index=False,
        chunksize=1000,
        dtype=source_dtype
    )

    # 2. Execute UPSERT from temporary table to target table
    upsert_source_sql = text(f"""
        INSERT INTO {TARGET_SOURCE_TABLE} (`name`, `url_base`)
        SELECT `name`, `url_base` 
        FROM {TEMP_SOURCE_TABLE}
        ON DUPLICATE KEY UPDATE
            `url_base` = VALUES(`url_base`); 
    """)
    
    result = connection.execute(upsert_source_sql)
    print(f"Upserted {result.rowcount} unique news sources into '{TARGET_SOURCE_TABLE}'.")
    
    # 3. Clean up temporary table
    connection.execute(text(f"DROP TABLE IF EXISTS {TEMP_SOURCE_TABLE};"))


def load_data_to_db(df: pd.DataFrame, engine: Engine):
    """
    Loads news sources (UPSERT) and then market sentiment data (INSERT) 
    within a single atomic transaction.
    """
    
    # --- Prepare Sentiment Data ---
    # Select columns and rename for DB schema match
    sentiment_df = df[SENTIMENT_DF_COLS].copy() 
    sentiment_df.rename(columns=SENTIMENT_MAPPING, inplace=True)
    
    if pd.api.types.is_datetime64_any_dtype(sentiment_df['publication_time']) and sentiment_df['publication_time'].dt.tz is not None:
        sentiment_df['publication_time'] = sentiment_df['publication_time'].dt.tz_localize(None)
    if pd.api.types.is_datetime64_any_dtype(sentiment_df['candle_time']) and sentiment_df['candle_time'].dt.tz is not None:
        sentiment_df['candle_time'] = sentiment_df['candle_time'].dt.tz_localize(None)

    # --- Database Transaction ---
    try:
        with engine.begin() as connection:
            
            # 1. Process News Sources (UPSERT)
            source_df = get_source_data(sentiment_df)
            upsert_source(source_df, connection)
            
            # 2. Retrieve Source IDs and Merge with Sentiment Data
            
            # Fetch all sources to get the IDs
            source_query = text("SELECT source_id, name FROM news_source;")
            sources_in_db = pd.read_sql(source_query, connection, index_col='name')
            
            # Map source name to source_id
            sentiment_df['source_id'] = sentiment_df['source'].map(sources_in_db['source_id'])
            
            # Filter out posts where source_id couldn't be mapped (shouldn't happen if upsert_source ran correctly)
            sentiment_df.dropna(subset=['source_id'], inplace=True)
            sentiment_df['source_id'] = sentiment_df['source_id'].astype(int)
            
            # 3. Prepare Final Sentiment DataFrame for Insertion
            FINAL_SENTIMENT_COLS = [
                'publication_time', 'source_id', 'headline', 'content', 'sentiment_score', 
                'finbert_label', 'finbert_score', 'finbert_probs', 'ticker', 'candle_time'
            ]
            final_insert_df = sentiment_df[FINAL_SENTIMENT_COLS]
            
            # Define SQL Alchemy DTypes
            sentiment_dtype: Dict[str, type] = {
                'publication_time': DateTime,
                'source_id': Integer,
                'headline': String(500),
                'content': String, 
                'sentiment_score': Float, 
                'finbert_label': String(20),
                'finbert_score': Float, 
                'finbert_probs': JSON,
                'ticker': String(10),
                'candle_time': DateTime
            }
            
            # 4. Load Sentiment Data directly (using append)
            TARGET_SENTIMENT_TABLE = "market_sentiment"
            
            # (though sentiment data is usually unique per post).
            rows_before = len(final_insert_df)
            final_insert_df.to_sql(
                TARGET_SENTIMENT_TABLE,
                con=connection,
                if_exists="append",
                index=False,
                chunksize=500,
                dtype=sentiment_dtype
            )
            
            print(f"Successfully inserted {rows_before} rows into '{TARGET_SENTIMENT_TABLE}'.")
            
    except Exception as e:
        print(f"Database load failed. Transaction rolled back: {e}")
        # Note: 'engine.begin()' handles the rollback automatically in case of exception
        raise 
    
    print("News source and market sentiment tables successfully updated.")

if __name__ == "__main__":
    from services.fetcher_news.helpers.transform_news import transform_news
    from sql.config.db_config import get_db_engine
    from services.fetcher_news.helpers.fetch_news import fetch_from_coinstats
    df_for_analysis = fetch_from_coinstats()
    processed_df = transform_news(df_for_analysis)
    engine = get_db_engine()
    try:
        load_data_to_db(processed_df, engine=engine)
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading data to DB: {e}")
        raise

        
