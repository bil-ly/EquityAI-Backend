import os
import json
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import logging
from dotenv import load_dotenv
logger = logging.getLogger(__name__)

load_dotenv()
MONGODB_URL = os.getenv("MONGODB_URL")
MONGODB_DB= os.getenv("MONGODB_DB")
DB_NAME = MONGODB_DB
COLLECTION_NAME = "etfs"


async def insert_json_into_db(json_file_path: str, category: str):
    client = AsyncIOMotorClient(MONGODB_URL)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    with open(json_file_path, "r") as f:
        data = json.load(f)

    instruments = data if isinstance(data, list) else data.get("instruments", [])

    if not instruments:
        logger.warning(f"No instruments found in {json_file_path}.")
        return

    for instrument in instruments:
        instrument["category"] = category

    result = await collection.insert_many(instruments)
    logger.info(f"Inserted {len(result.inserted_ids)} documents into '{COLLECTION_NAME}'.")

    client.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    json_file_path = "etfs.json"
    category = "etfs" 

    asyncio.run(insert_json_into_db(json_file_path, category))
