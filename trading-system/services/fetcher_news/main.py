import asyncio
from services.fetcher_news.helpers.fetch_news import fetch_from_coinstats
from services.fetcher_news.helpers.transform_news import transform_news
from services.fetcher_news.helpers.data_loader import load_data_to_db

async def main_loop():
    """"Main loop for the fetcher news service."""
    while True:
        print("Fetcher news service is running...")
        # Fetch from API
        raw_news = fetch_from_coinstats()
        # Add valder and finbert scoring
        preprocessed_news = transform_news(raw_news)
        # Load to DB
        load_data_to_db(preprocessed_news)

        await asyncio.sleep(60)  # Run every 60 seconds