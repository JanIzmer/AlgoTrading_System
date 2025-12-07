#Imports
import os
import pandas as pd
import requests
from typing import Dict, Any    
CRYPTOPANIC_API_TOKEN = os.getenv("CRYPTOPANIC_API_TOKEN", '')
CRYPTOPANIC_URL = "https://cryptopanic.com/api/developer/v2/posts/"
def fetch_from_cryptopanic(max_requests: int = 2000, per_page: int = 100, sleep_between: float = 0.5) -> pd.DataFrame:
    """
    Fetch news from Cryptopanic API and return a DataFrame with columns:
      ['title', 'published_at'(UTC datetimetz), 'url', 'source', 'sentiment']
    published_at will be timezone-aware UTC datetime (no tz info stripped).
    """
    if not CRYPTOPANIC_API_TOKEN:
        raise ValueError("CRYPTOPANIC_API_TOKEN environment variable is not set.")
    params: Dict[str, Any] = {
    "auth_token": CRYPTOPANIC_API_TOKEN, 
    "public": "true", 
    "per_page": per_page,
    "sort": "published_at",
    "filter": "all" # 👈 Явно запросить все новости, а не только важные/горячие
}
    try:
        resp = requests.get(CRYPTOPANIC_URL, params=params, timeout=15)
        resp.raise_for_status()
        j = resp.json()
        results = j.get("results")
        if results and isinstance(results, list):
            print("\nКлючи (столбцы) в первом элементе данных:")
            print(results[0].keys())
    except Exception as e:
        print(f"Error fetching initial data from Cryptopanic: {e}")
        return pd.DataFrame(columns=["title", "published_at", "url", "source", "sentiment"])
    df = pd.DataFrame(j.get("results", []))
    df["url"] = CRYPTOPANIC_URL
    df["ticker"] = "BTCUSDT"
    df["source"] = "Cryptopanic"
    if 'kind' in df.columns:
        df = df[df['kind'].isin(['news', 'media'])].copy()
    else:
        print("Warning: 'kind' column not found in Cryptopanic response; skipping kind filter.")
    
    # 3. Define the required columns
    REQUIRED_COLS = ['id', 'title', 'published_at', 'url', 'description', 'source', 'ticker']
    
    # 4. Select existing columns from the required list to avoid KeyError on missing API fields
    final_cols = [col for col in REQUIRED_COLS if col in df.columns]
    
    # 5. Return the resulting DataFrame
    return df[final_cols]

# main function for testing
if __name__ == "__main__":
    news_df = fetch_from_cryptopanic(max_requests=1000)
    print("number of news articles fetched:", len(news_df))
    print("columns:", news_df.columns.tolist())
    print(news_df["published_at"].head(20))

    