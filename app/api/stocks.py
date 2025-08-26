from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from datetime import datetime
import os

router = APIRouter(tags=["stocks"])

# Database configuration
MONGODB_URL = os.getenv("MONGODB_URL")
DATABASE_NAME = os.getenv("MONGODB_DATABASE") or os.getenv("MONGODB_DB")
MONGODB_USERNAME = os.getenv("MONGODB_USERNAME")
MONGODB_PASSWORD = os.getenv("MONGODB_PASSWORD")

# Global database connection
db_client: Optional[AsyncIOMotorClient] = None
database: Optional[AsyncIOMotorDatabase] = None

def build_mongodb_url() -> str:
    """Build MongoDB connection URL with authentication if provided"""
    if not MONGODB_URL:
        raise ValueError("MONGODB_URL environment variable is not set")
    
    if MONGODB_USERNAME and MONGODB_PASSWORD:
        if "://" in MONGODB_URL:
            protocol, rest = MONGODB_URL.split("://", 1)
            if "@" not in rest:
                return f"{protocol}://{MONGODB_USERNAME}:{MONGODB_PASSWORD}@{rest}"
        return MONGODB_URL
    
    return MONGODB_URL

async def get_database() -> AsyncIOMotorDatabase:
    """Get database connection"""
    global db_client, database
    if database is None:
        # Check if environment variables are set
        if not MONGODB_URL:
            raise ValueError("MONGODB_URL environment variable is not set")
        if not DATABASE_NAME:
            raise ValueError("MONGODB_DB environment variable is not set")
        
        # Build connection URL with authentication
        connection_url = build_mongodb_url()
        
        try:
            db_client = AsyncIOMotorClient(
                connection_url,
                serverSelectionTimeoutMS=5000,  
                connectTimeoutMS=10000,         
                socketTimeoutMS=0,             
                maxPoolSize=10,                 
                retryWrites=True               
            )
            
            await db_client.admin.command('ping')
            print(f"Successfully connected to MongoDB")
            
            database = db_client[DATABASE_NAME]
            
        except Exception as e:
            print(f"Failed to connect to MongoDB: {str(e)}")
            raise ValueError(f"Database connection failed: {str(e)}")
    
    return database

# Response Models
class InvestmentInfo(BaseModel):
    id: Optional[int] = None
    symbol: str
    ticker: str
    name: str
    contract_code: Optional[str] = None
    exchange: Optional[str] = None
    category: Optional[str] = None  # equities, etfs, commodities
    flag_code: Optional[str] = None
    logo_url: Optional[str] = None
    description: Optional[str] = None
    
    # Price information
    last_price: Optional[float] = None
    current_price: Optional[float] = None
    price_52w_high: Optional[float] = None
    price_52w_low: Optional[float] = None
    price_change_52w: Optional[float] = None
    volume: Optional[int] = None
    beta: Optional[float] = None
    
    # General company info
    sector: Optional[str] = None
    industry: Optional[str] = None
    country: Optional[str] = None
    currency: Optional[str] = None
    market_cap: Optional[float] = None
    
    # Financial ratios
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    ps_ratio: Optional[float] = None
    peg_ratio: Optional[float] = None
    
    # Dividend information
    dividend_yield: Optional[float] = None
    forward_dividend_yield: Optional[float] = None
    ttm_dividend: Optional[float] = None
    next_dividend_date: Optional[str] = None
    dividend_payout: Optional[float] = None
    
    # Growth metrics
    earning_growth_1y: Optional[float] = None
    earning_growth_5y: Optional[float] = None
    revenue_growth_1y: Optional[float] = None
    revenue_growth_5y: Optional[float] = None
    
    # Performance returns
    return_1mo: Optional[float] = None
    return_3mo: Optional[float] = None
    return_6mo: Optional[float] = None
    
    # ETF/Fund specific
    investment_style: Optional[str] = None
    fund_manager: Optional[str] = None
    asset_sub_group: Optional[str] = None

class InvestmentListResponse(BaseModel):
    investments: List[InvestmentInfo]
    total_count: int
    returned_count: int

def safe_float_convert(value: Any) -> Optional[float]:
    """Safely convert value to float"""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def safe_int_convert(value: Any) -> Optional[int]:
    """Safely convert value to int"""
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None

def safe_get_nested(doc: Dict[str, Any], *keys: str) -> Any:
    """Safely get nested dictionary value"""
    current = doc
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None
    return current

def map_db_document_to_investment_info(doc: Dict[str, Any]) -> InvestmentInfo:
    """Map database document to InvestmentInfo model"""
    
    # Extract nested data safely
    general = doc.get("General") or {}
    price = doc.get("Price") or {}
    dividends = doc.get("Dividends") or {}
    valuation = doc.get("Valuation Ratio") or {}
    growth = doc.get("Growth") or {}
    returns = doc.get("returns") or {}
    fundamental = doc.get("Fundamental") or {}
    
    # Ensure all nested objects are dictionaries
    if not isinstance(general, dict):
        general = {}
    if not isinstance(price, dict):
        price = {}
    if not isinstance(dividends, dict):
        dividends = {}
    if not isinstance(valuation, dict):
        valuation = {}
    if not isinstance(growth, dict):
        growth = {}
    if not isinstance(returns, dict):
        returns = {}
    if not isinstance(fundamental, dict):
        fundamental = {}
    
    # Convert lastPrice string to float safely
    last_price = safe_float_convert(doc.get("lastPrice"))
    
    return InvestmentInfo(
        id=safe_int_convert(doc.get("id")),
        symbol=str(general.get("symbol") or doc.get("ticker") or ""),
        ticker=str(doc.get("ticker") or ""),
        name=str(doc.get("name") or ""),
        contract_code=doc.get("contractCode"),
        exchange=doc.get("Exchange") or general.get("exchange"),
        category=doc.get("category"),
        flag_code=doc.get("flagCode"),
        logo_url=doc.get("logoUrl"),
        description=doc.get("description"),
        
        # Price information
        last_price=last_price,
        current_price=safe_float_convert(price.get("price")),
        price_52w_high=safe_float_convert(price.get("price52whigh")),
        price_52w_low=safe_float_convert(price.get("price52wlow")),
        price_change_52w=safe_float_convert(price.get("pchange_52w")),
        volume=safe_int_convert(price.get("volume")),
        beta=safe_float_convert(price.get("beta")),
        
        # General company info
        sector=general.get("sector"),
        industry=general.get("industry"),
        country=general.get("country_iso"),
        currency=general.get("currency"),
        market_cap=safe_float_convert(fundamental.get("mktcap")),
        
        # Financial ratios
        pe_ratio=safe_float_convert(valuation.get("pe")),
        pb_ratio=safe_float_convert(valuation.get("pb")),
        ps_ratio=safe_float_convert(valuation.get("ps")),
        peg_ratio=safe_float_convert(valuation.get("peg")),
        
        # Dividend information
        dividend_yield=safe_float_convert(dividends.get("yield")),
        forward_dividend_yield=safe_float_convert(dividends.get("ForwardDividendYield")),
        ttm_dividend=safe_float_convert(dividends.get("ttm_dividend")),
        next_dividend_date=dividends.get("next_dividend_date"),
        dividend_payout=safe_float_convert(dividends.get("payout")),
        
        # Growth metrics
        earning_growth_1y=safe_float_convert(growth.get("earning_growth_1y")),
        earning_growth_5y=safe_float_convert(growth.get("earning_growth_5y")),
        revenue_growth_1y=safe_float_convert(growth.get("rvn_growth_1y")),
        revenue_growth_5y=safe_float_convert(growth.get("rvn_growth_5y")),
        
        # Performance returns
        return_1mo=safe_float_convert(safe_get_nested(returns, "1mo", "percentage")),
        return_3mo=safe_float_convert(safe_get_nested(returns, "3mo", "percentage")),
        return_6mo=safe_float_convert(safe_get_nested(returns, "6mo", "percentage")),
        
        # ETF/Fund specific
        investment_style=doc.get("investmentStyle"),
        fund_manager=doc.get("fundManager"),
        asset_sub_group=doc.get("assetSubGroup")
    )

async def search_investments_in_collection(
    collection_name: str,
    query_filter: Dict[str, Any],
    sort_field: str,
    sort_direction: int,
    limit: int,
    db: AsyncIOMotorDatabase
) -> List[InvestmentInfo]:
    """Search investments in a specific collection"""
    collection = getattr(db, collection_name)
    cursor = collection.find(query_filter).sort(sort_field, sort_direction).limit(limit)
    
    investments: List[InvestmentInfo] = []
    async for doc in cursor:
        investment_info = map_db_document_to_investment_info(doc)
        investments.append(investment_info)
    
    return investments

@router.get("/", response_model=InvestmentListResponse)
async def get_investments(
    limit: int = Query(20, ge=1, le=100, description="Number of investments to return"),
    search: Optional[str] = Query(None, description="Search by symbol, ticker or name"),
    category: Optional[str] = Query(None, description="Filter by category: equities, etfs, commodities"),
    sector: Optional[str] = Query(None, description="Filter by sector"),
    exchange: Optional[str] = Query(None, description="Filter by exchange"),
    currency: Optional[str] = Query(None, description="Filter by currency"),
    min_price: Optional[float] = Query(None, ge=0, description="Minimum price"),
    max_price: Optional[float] = Query(None, ge=0, description="Maximum price"),
    has_dividend: Optional[bool] = Query(None, description="Filter investments that pay dividends"),
    investment_style: Optional[str] = Query(None, description="Filter by investment style (for ETFs)"),
    fund_manager: Optional[str] = Query(None, description="Filter by fund manager"),
    sort_by: str = Query("ticker", description="Sort by: ticker, name, price, market_cap, dividend_yield"),
    sort_order: str = Query("asc", description="Sort order: asc, desc"),
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Get investments from database with filtering and pagination"""
    try:
        # Build MongoDB query filter
        query_filter: Dict[str, Any] = {}
        
        # Search filter - search across multiple fields
        if search:
            search_regex = {"$regex": search, "$options": "i"}
            query_filter["$or"] = [
                {"ticker": search_regex},
                {"name": search_regex},
                {"General.symbol": search_regex},
                {"contractCode": search_regex}
            ]
        
        # Sector filter (nested in General object)
        if sector:
            query_filter["General.sector"] = {"$regex": sector, "$options": "i"}
        
        # Exchange filter
        if exchange:
            query_filter["Exchange"] = {"$regex": exchange, "$options": "i"}
        
        # Currency filter - ETFs and commodities don't have General.currency, so we need to infer from exchange
        if currency:
            currency_conditions: List[Dict[str, Any]] = []
            
            # Direct currency match for equities
            currency_conditions.append({"General.currency": {"$regex": f"^{currency}$", "$options": "i"}})
            
            # For ZAR, also include South African exchanges (ETFs/commodities)
            if currency.upper() in ["ZAR", "RAND"]:
                currency_conditions.extend([
                    {"Exchange": {"$regex": "JSE", "$options": "i"}},
                    {"Exchange": {"$regex": "TFSA", "$options": "i"}},
                    {"flagCode": {"$regex": "south-african", "$options": "i"}}
                ])
            
            # For USD, include US exchanges
            elif currency.upper() == "USD":
                currency_conditions.extend([
                    {"Exchange": {"$regex": "USA", "$options": "i"}},
                    {"Exchange": {"$regex": "NASDAQ", "$options": "i"}},
                    {"Exchange": {"$regex": "NYSE", "$options": "i"}},
                    {"flagCode": {"$regex": "US", "$options": "i"}}
                ])
            
            # For EUR, include European exchanges
            elif currency.upper() == "EUR":
                currency_conditions.extend([
                    {"Exchange": {"$regex": "Germany", "$options": "i"}},
                    {"flagCode": {"$regex": "Germany", "$options": "i"}}
                ])
            
            if currency_conditions:
                # Merge with existing conditions
                if "$and" in query_filter:
                    query_filter["$and"].append({"$or": currency_conditions})
                elif "$or" in query_filter:
                    query_filter = {"$and": [{"$or": query_filter["$or"]}, {"$or": currency_conditions}]}
                else:
                    query_filter["$or"] = currency_conditions
        
        # Price filters (check both lastPrice string and Price.price number)
        if min_price is not None or max_price is not None:
            price_conditions: List[Dict[str, Any]] = []
            
            # For lastPrice (string field) - convert to number for comparison
            if min_price is not None and max_price is not None:
                price_conditions.append({
                    "$expr": {
                        "$and": [
                            {"$gte": [{"$toDouble": {"$ifNull": ["$lastPrice", "0"]}}, min_price]},
                            {"$lte": [{"$toDouble": {"$ifNull": ["$lastPrice", "0"]}}, max_price]}
                        ]
                    }
                })
            elif min_price is not None:
                price_conditions.append({
                    "$expr": {"$gte": [{"$toDouble": {"$ifNull": ["$lastPrice", "0"]}}, min_price]}
                })
            elif max_price is not None:
                price_conditions.append({
                    "$expr": {"$lte": [{"$toDouble": {"$ifNull": ["$lastPrice", "0"]}}, max_price]}
                })
            
            # For Price.price (number field)
            price_filter: Dict[str, float] = {}
            if min_price is not None:
                price_filter["$gte"] = min_price
            if max_price is not None:
                price_filter["$lte"] = max_price
            
            if price_filter:
                price_conditions.append({"Price.price": price_filter})
            
            if price_conditions:
                # Merge with existing $or conditions if they exist
                if "$or" in query_filter:
                    query_filter = {"$and": [{"$or": query_filter["$or"]}, {"$or": price_conditions}]}
                else:
                    query_filter["$or"] = price_conditions
        
        # Dividend filter
        if has_dividend is not None:
            dividend_conditions: List[Dict[str, Any]] = []
            if has_dividend:
                dividend_conditions = [
                    {"Dividends.yield": {"$gt": 0}},
                    {"Dividends.ttm_dividend": {"$gt": 0}}
                ]
            else:
                dividend_conditions = [
                    {"Dividends.yield": {"$exists": False}},
                    {"Dividends.yield": {"$lte": 0}},
                    {"Dividends.yield": None}
                ]
            
            # Merge with existing conditions
            if "$and" in query_filter:
                query_filter["$and"].append({"$or": dividend_conditions})
            elif "$or" in query_filter:
                query_filter = {"$and": [{"$or": query_filter["$or"]}, {"$or": dividend_conditions}]}
            else:
                query_filter["$or"] = dividend_conditions
        
        # Investment style filter (for ETFs)
        if investment_style:
            query_filter["investmentStyle"] = {"$regex": investment_style, "$options": "i"}
        
        # Fund manager filter
        if fund_manager:
            query_filter["fundManager"] = {"$regex": fund_manager, "$options": "i"}
        
        # Determine collections to search
        collections_to_search: List[str] = []
        if category:
            category_lower = category.lower()
            if category_lower in ["equities", "equity", "stocks", "stock"]:
                collections_to_search = ["equities"]
            elif category_lower in ["etfs", "etf"]:
                collections_to_search = ["etfs"]
            elif category_lower in ["commodities", "commodity"]:
                collections_to_search = ["commodities"]
            else:
                collections_to_search = ["equities", "etfs", "commodities"]
        else:
            collections_to_search = ["equities", "etfs", "commodities"]
        
        # Build sort criteria
        sort_field = "ticker"
        if sort_by == "name":
            sort_field = "name"
        elif sort_by == "price":
            sort_field = "lastPrice"
        elif sort_by == "market_cap":
            sort_field = "Fundamental.mktcap"
        elif sort_by == "dividend_yield":
            sort_field = "Dividends.yield"
        
        sort_direction = -1 if sort_order.lower() == "desc" else 1
        
        # Search across collections and combine results
        all_investments: List[InvestmentInfo] = []
        total_count = 0
        
        for collection_name in collections_to_search:
            collection = getattr(db, collection_name)
            
            # Get count for this collection
            count = await collection.count_documents(query_filter)
            total_count += count
            
            # Get investments from this collection
            investments = await search_investments_in_collection(
                collection_name, query_filter, sort_field, sort_direction, limit, db
            )
            all_investments.extend(investments)
        
        # Sort combined results (since we're combining from multiple collections)
        if sort_by == "ticker":
            all_investments.sort(key=lambda x: x.ticker or "", reverse=(sort_order.lower() == "desc"))
        elif sort_by == "name":
            all_investments.sort(key=lambda x: x.name or "", reverse=(sort_order.lower() == "desc"))
        elif sort_by == "price":
            all_investments.sort(key=lambda x: x.last_price or 0, reverse=(sort_order.lower() == "desc"))
        elif sort_by == "market_cap":
            all_investments.sort(key=lambda x: x.market_cap or 0, reverse=(sort_order.lower() == "desc"))
        elif sort_by == "dividend_yield":
            all_investments.sort(key=lambda x: x.dividend_yield or 0, reverse=(sort_order.lower() == "desc"))
        
        # Apply final limit
        limited_investments = all_investments[:limit]
        
        return InvestmentListResponse(
            investments=limited_investments,
            total_count=total_count,
            returned_count=len(limited_investments)
        )
        
    except Exception as e:
        print(f"Error in get_investments: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching investments: {str(e)}")

@router.get("/sectors")
async def get_available_sectors(db: AsyncIOMotorDatabase = Depends(get_database)):
    """Get all available sectors from the database"""
    try:
        sectors_set = set()
        
        # Get sectors from equities
        equity_sectors = await db.equities.distinct("General.sector", {"General.sector": {"$ne": None, "$ne": ""}})
        sectors_set.update(equity_sectors)
        
        # Filter out empty strings and None values
        sectors_list = [sector for sector in sectors_set if sector and isinstance(sector, str) and sector.strip()]
        
        return {"sectors": sorted(sectors_list)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching sectors: {str(e)}")

@router.get("/exchanges")
async def get_available_exchanges(db: AsyncIOMotorDatabase = Depends(get_database)):
    """Get all available exchanges from the database"""
    try:
        exchanges_set = set()
        
        # Get exchanges from all collections
        for collection_name in ["equities", "etfs", "commodities"]:
            collection = getattr(db, collection_name)
            collection_exchanges = await collection.distinct("Exchange", {"Exchange": {"$ne": None, "$ne": ""}})
            exchanges_set.update(collection_exchanges)
        
        # Filter out empty strings and None values
        exchanges_list = [exchange for exchange in exchanges_set if exchange and isinstance(exchange, str) and exchange.strip()]
        
        return {"exchanges": sorted(exchanges_list)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching exchanges: {str(e)}")

@router.get("/currencies")
async def get_available_currencies(db: AsyncIOMotorDatabase = Depends(get_database)):
    """Get all available currencies from the database"""
    try:
        currencies_set = set()
        
        # Get currencies from equities (nested in General object)
        equity_currencies = await db.equities.distinct("General.currency", {"General.currency": {"$ne": None, "$ne": ""}})
        currencies_set.update(equity_currencies)
        
        # Filter out empty strings and None values
        currencies_list = [currency for currency in currencies_set if currency and isinstance(currency, str) and currency.strip()]
        
        return {"currencies": sorted(currencies_list)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching currencies: {str(e)}")

@router.get("/categories")
async def get_available_categories(db: AsyncIOMotorDatabase = Depends(get_database)):
    """Get all available investment categories"""
    try:
        categories: List[Dict[str, Union[str, int]]] = []
        
        # Check which collections have data
        for collection_name in ["equities", "etfs", "commodities"]:
            collection = getattr(db, collection_name)
            count = await collection.count_documents({})
            if count > 0:
                categories.append({
                    "category": collection_name,
                    "count": count
                })
        
        return {"categories": categories}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching categories: {str(e)}")

@router.get("/fund-managers")
async def get_available_fund_managers(db: AsyncIOMotorDatabase = Depends(get_database)):
    """Get all available fund managers (mainly for ETFs)"""
    try:
        fund_managers_set = set()
        
        # Get fund managers from ETFs and other collections
        for collection_name in ["etfs", "commodities"]:
            collection = getattr(db, collection_name)
            managers = await collection.distinct("fundManager", {"fundManager": {"$ne": None, "$ne": ""}})
            fund_managers_set.update(managers)
        
        fund_managers_list = [manager for manager in fund_managers_set if manager and isinstance(manager, str)]
        
        return {"fund_managers": sorted(fund_managers_list)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching fund managers: {str(e)}")

@router.get("/{ticker}")
async def get_investment_by_ticker(
    ticker: str,
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Get a specific investment by ticker from any collection"""
    try:
        # Search in all collections
        for collection_name in ["equities", "etfs", "commodities"]:
            collection = getattr(db, collection_name)
            
            # Try exact match first
            doc = await collection.find_one({"ticker": ticker})
            if not doc:
                # Try case-insensitive match
                doc = await collection.find_one({"ticker": {"$regex": f"^{ticker}$", "$options": "i"}})
            
            if doc:
                investment_info = map_db_document_to_investment_info(doc)
                return investment_info
        
        raise HTTPException(status_code=404, detail=f"Investment with ticker '{ticker}' not found")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching investment: {str(e)}")

@router.get("/test/{ticker}")
async def test_investment(
    ticker: str,
    db: AsyncIOMotorDatabase = Depends(get_database)
):
    """Test endpoint to debug a single investment from database"""
    try:
        results: Dict[str, Dict[str, Any]] = {}
        
        # Search in all collections
        for collection_name in ["equities", "etfs", "commodities"]:
            collection = getattr(db, collection_name)
            
            # Try to find the document
            doc = await collection.find_one({"ticker": ticker})
            if not doc:
                doc = await collection.find_one({"ticker": {"$regex": f"^{ticker}$", "$options": "i"}})
            
            if doc:
                # Convert ObjectId to string for JSON serialization
                if "_id" in doc:
                    doc["_id"] = str(doc["_id"])
                
                # Get processed investment info
                processed_investment = map_db_document_to_investment_info(doc)
                
                results[collection_name] = {
                    "found": True,
                    "raw_document": doc,
                    "processed": processed_investment.dict()
                }
                break
            else:
                results[collection_name] = {"found": False}
        
        if not any(result.get("found", False) for result in results.values()):
            return {"error": f"Investment '{ticker}' not found in any collection", "searched_collections": list(results.keys())}
        
        return {
            "ticker": ticker,
            "results": results
        }
        
    except Exception as e:
        return {"error": str(e), "ticker": ticker}

# Health check endpoint
@router.get("/health")
async def health_check(db: AsyncIOMotorDatabase = Depends(get_database)):
    """Health check endpoint to test database connection"""
    try:
        # Simple database ping
        await db.admin.command('ping')
        
        # Count documents in each collection
        counts: Dict[str, int] = {}
        for collection_name in ["equities", "etfs", "commodities"]:
            collection = getattr(db, collection_name)
            counts[collection_name] = await collection.count_documents({})
        
        return {
            "status": "healthy",
            "database": "connected",
            "collection_counts": counts,
            "timestamp": datetime.utcnow()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database health check failed: {str(e)}")

# Cleanup function (call this when shutting down the application)
async def close_database_connection():
    """Close database connection"""
    global db_client
    if db_client:
        db_client.close()