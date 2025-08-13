import asyncio
import httpx
from typing import List, Dict, Any
from datetime import datetime
from beanie import BulkWriter
import logging

from app.models.database import Stock
from app.database import connect_to_mongo, close_mongo_connection
from app.synatic_client import EasyEquitiesSynaticClient

logger = logging.getLogger(__name__)


class StockSyncService:
    """Service to sync stocks from EasyEquities to MongoDB"""
    
    def __init__(self, bearer_token: str = None):
        self.synatic_client = EasyEquitiesSynaticClient()
        if bearer_token:
            self.synatic_client.set_bearer_token(bearer_token)
        self.stats = {
            "inserted": 0,
            "updated": 0,
            "failed": 0,
            "total": 0
        }
    
    async def sync_stocks_from_api_response(self, stocks_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Sync stocks from API response to MongoDB
        
        Args:
            stocks_data: List of stock dictionaries from API
        
        Returns:
            Sync statistics
        """
        self.stats = {"inserted": 0, "updated": 0, "failed": 0, "total": 0}
        
        for stock_data in stocks_data:
            try:
                await self._upsert_stock(stock_data)
                self.stats["total"] += 1
            except Exception as e:
                logger.error(f"Failed to sync stock {stock_data.get('symbol')}: {e}")
                self.stats["failed"] += 1
        
        return self.stats
    
    async def _upsert_stock(self, stock_data: Dict[str, Any]):
        """Insert or update a single stock"""
        symbol = stock_data.get("symbol")
        
        if not symbol:
            logger.warning(f"Stock data missing symbol: {stock_data}")
            return
        
        # Check if stock exists
        existing_stock = await Stock.find_one(Stock.symbol == symbol)
        
        # Prepare stock document
        stock_doc = {
            "symbol": symbol,
            "name": stock_data.get("name", ""),
            "contract_code": stock_data.get("contract_code", ""),
            "sector": stock_data.get("sector", "Unknown"),
            "market": stock_data.get("market", "Unknown"),
            "currency": stock_data.get("currency", "USD"),
            "last_price": stock_data.get("last_price"),
            "market_cap": stock_data.get("market_cap"),
            "pe_ratio": stock_data.get("pe_ratio"),
            "dividend_yield": stock_data.get("dividend_yield"),
            "volume": stock_data.get("volume"),
            "price_change_52w": stock_data.get("price_change_52w"),
            "returns_1m": stock_data.get("returns_1m"),
            "returns_3m": stock_data.get("returns_3m"),
            "returns_6m": stock_data.get("returns_6m"),
            "logo_url": stock_data.get("logo_url"),
            "is_tradeable": stock_data.get("is_tradeable", True),
            "last_updated": datetime.now(),
            "data_source": "synatic"
        }
        
        if existing_stock:
            # Update existing stock
            for key, value in stock_doc.items():
                setattr(existing_stock, key, value)
            await existing_stock.save()
            self.stats["updated"] += 1
            logger.debug(f"Updated stock: {symbol}")
        else:
            # Insert new stock
            new_stock = Stock(**stock_doc)
            await new_stock.insert()
            self.stats["inserted"] += 1
            logger.debug(f"Inserted new stock: {symbol}")
    
    async def sync_all_stocks_from_synatic(self, bearer_token: str) -> Dict[str, Any]:
        """
        Fetch and sync all stocks from Synatic API to MongoDB
        
        Args:
            bearer_token: Bearer token for Synatic API
        
        Returns:
            Sync statistics
        """
        self.synatic_client.set_bearer_token(bearer_token)
        
        total_stats = {
            "categories": {},
            "total_inserted": 0,
            "total_updated": 0,
            "total_failed": 0,
            "total_stocks": 0
        }
        
        categories = ["equitiesexpanded", "equities", "etfs", "bundles", "crypto"]
        
        for category in categories:
            logger.info(f"Syncing {category}...")
            
            try:
                # Fetch from Synatic
                instruments = await self.synatic_client.search_instruments(category=category)
                
                if not instruments:
                    logger.warning(f"No instruments found for {category}")
                    continue
                
                # Transform and sync
                stocks_to_sync = []
                for inst in instruments:
                    stock_data = self.synatic_client.transform_to_standard_format(inst)
                    stocks_to_sync.append(stock_data)
                
                # Sync to MongoDB
                category_stats = await self.sync_stocks_from_api_response(stocks_to_sync)
                
                total_stats["categories"][category] = category_stats
                total_stats["total_inserted"] += category_stats["inserted"]
                total_stats["total_updated"] += category_stats["updated"]
                total_stats["total_failed"] += category_stats["failed"]
                total_stats["total_stocks"] += category_stats["total"]
                
                logger.info(f"Synced {category}: {category_stats}")
                
                # Small delay between categories
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Failed to sync {category}: {e}")
                total_stats["categories"][category] = {"error": str(e)}
        
        return total_stats
    
    async def bulk_upsert_stocks(self, stocks_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Bulk upsert stocks for better performance
        
        Args:
            stocks_data: List of stock dictionaries
        
        Returns:
            Sync statistics
        """
        stats = {"inserted": 0, "updated": 0, "failed": 0, "total": 0}
        
        # Get all existing symbols
        existing_stocks = await Stock.find_all().to_list()
        existing_symbols = {stock.symbol: stock for stock in existing_stocks}
        
        stocks_to_insert = []
        stocks_to_update = []
        
        for stock_data in stocks_data:
            symbol = stock_data.get("symbol")
            if not symbol:
                stats["failed"] += 1
                continue
            
            stock_doc = {
                "symbol": symbol,
                "name": stock_data.get("name", ""),
                "contract_code": stock_data.get("contract_code", ""),
                "sector": stock_data.get("sector", "Unknown"),
                "market": stock_data.get("market", "Unknown"),
                "currency": stock_data.get("currency", "USD"),
                "last_price": stock_data.get("last_price"),
                "market_cap": stock_data.get("market_cap"),
                "pe_ratio": stock_data.get("pe_ratio"),
                "dividend_yield": stock_data.get("dividend_yield"),
                "volume": stock_data.get("volume"),
                "price_change_52w": stock_data.get("price_change_52w"),
                "returns_1m": stock_data.get("returns_1m"),
                "returns_3m": stock_data.get("returns_3m"),
                "returns_6m": stock_data.get("returns_6m"),
                "logo_url": stock_data.get("logo_url"),
                "is_tradeable": stock_data.get("is_tradeable", True),
                "last_updated": datetime.now(),
                "data_source": "synatic"
            }
            
            if symbol in existing_symbols:
                # Update existing
                existing = existing_symbols[symbol]
                for key, value in stock_doc.items():
                    setattr(existing, key, value)
                stocks_to_update.append(existing)
            else:
                # Insert new
                stocks_to_insert.append(Stock(**stock_doc))
        
        # Bulk operations
        if stocks_to_insert:
            await Stock.insert_many(stocks_to_insert)
            stats["inserted"] = len(stocks_to_insert)
        
        if stocks_to_update:
            # Update in batches
            for stock in stocks_to_update:
                await stock.save()
            stats["updated"] = len(stocks_to_update)
        
        stats["total"] = stats["inserted"] + stats["updated"]
        
        return stats

