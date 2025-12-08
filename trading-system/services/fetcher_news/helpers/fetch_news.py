import os
import pandas as pd
import requests
from typing import Dict, Any
from datetime import datetime, timezone
from langdetect import detect_langs, DetectorFactory

DetectorFactory.seed = 0

COINSTATS_API_TOKEN = os.getenv("COINSTATS_API_TOKEN", "")
COINSTATS_URL = "https://openapiv1.coinstats.app/news"


def has_btc_keywords(item: Dict[str, Any]) -> bool:
    btc_keywords = {"btc", "btcusdt", "bitcoin", "btc/usdt"}
    for kw in item.get("searchKeyWords", []) or []:
        if isinstance(kw, str) and kw.lower() in btc_keywords:
            return True
    for coin in item.get("coins", []) or []:
        for key in ["coinKeyWords", "coinIdKeyWords", "coinNameKeyWords", "coinTitleKeyWords"]:
            kw = coin.get(key, "")
            if kw and isinstance(kw, str) and kw.lower() in btc_keywords:
                return True
    title = item.get("title", "")
    if isinstance(title, str) and ("btc" in title.lower() or "bitcoin" in title.lower()):
        return True
    return False


def is_featured(item: Dict) -> bool:
    return item.get("featured") is True
def is_content(item: Dict) -> bool:
    return item.get("content") is True


def is_english_text(text: str) -> bool:
    txt = text.strip()
    if not txt:
        return False
    try:
        top = detect_langs(txt)[0]
        return top.lang == "en" and top.prob >= 0.2  # low threshold
    except Exception:
        return True  # fallback


def fetch_from_coinstats(max_pages: int = 5, per_page: int = 100) -> pd.DataFrame:
    if not COINSTATS_API_TOKEN:
        raise ValueError("Coinstats_API_TOKEN environment variable is not set.")

    headers = {"x-api-key": COINSTATS_API_TOKEN}
    all_rows = []
    total = kept = dropped = 0

    for page in range(1, max_pages + 1):
        params = {"page": page, "limit": per_page}
        try:
            resp = requests.get(COINSTATS_URL, headers=headers, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json().get("result", [])
        except Exception as e:
            print(f"Error fetching page {page}: {e}")
            continue

        if not data:
            break  # no more data

        for item in data:
            total += 1
            if not isinstance(item, dict):
                continue
            if is_featured(item):
                continue
            if not is_content(item):
                continue
            if not has_btc_keywords(item):
                continue
            title = (item.get("title") or "").strip()
            description = (item.get("description") or "").strip()
            if not is_english_text(title):
                dropped += 1
                continue
            feed_ts = item.get("feedDate")
            published_at = datetime.fromtimestamp(feed_ts / 1000, tz=timezone.utc) if feed_ts else pd.NaT
            all_rows.append({
                "id": item.get("id"),
                "title": title,
                "published_at": published_at,
                "url": item.get("link") or item.get("sourceLink") or item.get("shareURL"),
                "description": description,
                "source": item.get("source"),
                "ticker": "BTCUSDT"
            })
            kept += 1

    df = pd.DataFrame(all_rows, columns=["id", "title", "published_at", "url", "description", "source", "ticker"])
    print(f"CoinStats fetched {total} items, kept {kept}, dropped {dropped} (non-English or not BTC).")
    return df


if __name__ == "__main__":
    news_df = fetch_from_coinstats(max_pages=5, per_page=100)
    print("Fetched:", len(news_df), "BTC-related English articles")
    print(news_df['published_at'].head(20))


    