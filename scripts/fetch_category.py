import sys
import os
import json
import asyncio
import logging
import httpx

from app.database import connect_to_mongo, close_mongo_connection
from app.models.database import Stock

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_URL = "https://rest.synatic.openeasy.io/easyequities/investnow/search"

async def fetch_category(token: str, category: str, page: int = 1):
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    payload = {
        "searchValue": "",
        "account_filter": "ALL",
        "category": category,
        "page": page
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(API_URL, headers=headers, json=payload)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 204:
            logger.warning(f"No data for {category}, page {page}")
            return None
        else:
            logger.error(f"Request failed: {resp.status_code} {resp.text}")
            return None

async def save_and_db(token: str, category: str):
    await connect_to_mongo()
    
    all_items = []
    page = 1
    
    while True:
        logger.info(f"Fetching {category}, page {page}...")
        data = await fetch_category(token, category, page)
        if not data or "instruments" not in data or len(data["instruments"]) == 0:
            break
        all_items.extend(data["instruments"])
        page += 1

    file_name = f"{category}.json"
    with open(file_name, "w") as f:
        json.dump(all_items, f, indent=4)
    logger.info(f"Saved {len(all_items)} items to {file_name}")
    
    #TODO: Will update it , i might have to convert it into an array first so that i can see it as a Document , will also need to create the collection too for the categories
    for item in all_items:
        await Stock.find_one(Stock.id == item["id"]).upsert(
            dict(item)
        )
    logger.info(f"Saved {len(all_items)} items to MongoDB")
    
    await close_mongo_connection()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python fetch_category.py <TOKEN> <CATEGORY>")
        sys.exit(1)

    token = sys.argv[1]
    category = sys.argv[2]

    asyncio.run(save_and_db(token, category))
