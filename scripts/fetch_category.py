import sys
import time
import json
import requests
import logging
import os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

TOKEN = sys.argv[1]
CATEGORY = sys.argv[2]

URL = "https://rest.synatic.openeasy.io/easyequities/investnow/search"
HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json",
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36"
}

def fetch_category_page(page: int):
    payload = {
        "searchValue": "",
        "account_filter": "ALL",
        "category": CATEGORY,
        "page": page
    }
    response = requests.post(URL, headers=HEADERS, json=payload)
    if response.status_code == 200:
        return response.json().get("instruments", [])
    else:
        logger.warning(f"Page {page} failed with status {response.status_code}")
        return []

def fetch_all_pages():
    page = 1
    all_items = []
    while True:
        logger.info(f"Fetching page {page} for {CATEGORY}...")
        items = fetch_category_page(page)
        if not items:
            break
        all_items.extend(items)
        page += 1
        time.sleep(1)  # polite delay between pages
    return all_items

if __name__ == "__main__":
    data = fetch_all_pages()
    filename = f"{CATEGORY}.json"
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved {len(data)} items to {filename}")
