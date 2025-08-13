"""
Script to sync stocks from EasyEquities to MongoDB
"""
import asyncio
import os
import sys
from dotenv import load_dotenv
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.stock_sync import StockSyncService
from app.database import connect_to_mongo, close_mongo_connection

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


async def sync_from_curl_response():
    """Sync stocks from the curl response you provided"""
    
    # Your curl response data
    stocks_data = [
        {"symbol": "AOV", "name": "Amotiv Limited", "contract_code": "EQU.AU.AOV", "sector": "Consumer Cyclical", "market": "ASX", "currency": "AUD", "last_price": 9.41, "market_cap": 1186.942, "pe_ratio": None, "dividend_yield": 4.63, "volume": 247116, "price_change_52w": -15.33, "returns_1m": 9.93, "returns_3m": 15.25, "returns_6m": -3.25, "logo_url": "https://resources.easyequities.co.za/logos/EQU.AU.AOV.png", "is_tradeable": True},
        {"symbol": "APE", "name": "AP Eagers Ltd", "contract_code": "EQU.AU.APE", "sector": "Consumer Cyclical", "market": "ASX", "currency": "AUD", "last_price": 21.33, "market_cap": 5153.741, "pe_ratio": None, "dividend_yield": 3.71, "volume": 363620, "price_change_52w": 99.41, "returns_1m": 10.18, "returns_3m": 14.35, "returns_6m": 65.09, "logo_url": "https://resources.easyequities.co.za/logos/EQU.AU.APE.png", "is_tradeable": True},
        # Add all 48 stocks here...
    ]
    
    try:
        # Connect to MongoDB
        await connect_to_mongo()
        logger.info("Connected to MongoDB")
        
        # Initialize sync service
        sync_service = StockSyncService()
        
        # Sync stocks
        stats = await sync_service.sync_stocks_from_api_response(stocks_data)
        
        logger.info(f"Sync completed: {stats}")
        
    except Exception as e:
        logger.error(f"Sync failed: {e}")
    finally:
        await close_mongo_connection()


async def sync_from_synatic_api(bearer_token: str):
    """Sync all stocks from Synatic API"""
    
    try:
        # Connect to MongoDB
        await connect_to_mongo()
        logger.info("Connected to MongoDB")
        
        # Initialize sync service
        sync_service = StockSyncService()
        
        # Sync all categories
        stats = await sync_service.sync_all_stocks_from_synatic(bearer_token)
        
        logger.info(f"Full sync completed: {stats}")
        
    except Exception as e:
        logger.error(f"Sync failed: {e}")
    finally:
        await close_mongo_connection()


async def main():
    """Main function"""
    
    # Option 1: Sync from your curl response
    # await sync_from_curl_response()
    
    # Option 2: Sync from Synatic API (need bearer token)
    bearer_token = os.getenv("SYNATIC_BEARER_TOKEN")
    if not bearer_token:
        logger.error("No bearer token found. Set SYNATIC_BEARER_TOKEN environment variable")
        return
    
    await sync_from_synatic_api(bearer_token)


if __name__ == "__main__":
    asyncio.run(main())
