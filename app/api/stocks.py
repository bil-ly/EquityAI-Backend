from fastapi import APIRouter, HTTPException, Query, Depends, Header
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime
import asyncio
from functools import lru_cache

# Import your Synatic client
from app.synatic_client import EasyEquitiesSynaticClient
from app.services.stock_sync import StockSyncService

router = APIRouter(tags=["stocks"])

# Initialize Synatic client globally
synatic_client = EasyEquitiesSynaticClient()

# Response Models
class StockInfo(BaseModel):
    symbol: str
    name: str
    contract_code: str
    sector: str
    market: str
    currency: str
    last_price: Optional[float] = None
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    volume: Optional[int] = None
    price_change_52w: Optional[float] = None
    returns_1m: Optional[float] = None
    returns_3m: Optional[float] = None
    returns_6m: Optional[float] = None
    logo_url: Optional[str] = None
    is_tradeable: bool = True
    
class StockSearchResponse(BaseModel):
    stocks: List[StockInfo]
    total_count: int
    page: int
    per_page: int
    from_cache: bool = False
    
class MarketSummary(BaseModel):
    market: str
    total_stocks: int
    sectors: List[str]
    
class StockDetail(BaseModel):
    symbol: str
    name: str
    contract_code: str
    sector: str
    industry: str
    market: str
    currency: str
    description: Optional[str] = None
    last_price: Optional[float] = None
    previous_close: Optional[float] = None
    change: Optional[float] = None
    change_percent: Optional[float] = None
    volume: Optional[int] = None
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    eps: Optional[float] = None
    dividend_yield: Optional[float] = None
    price_52w_high: Optional[float] = None
    price_52w_low: Optional[float] = None
    returns_1m: Optional[float] = None
    returns_3m: Optional[float] = None
    returns_6m: Optional[float] = None
    logo_url: Optional[str] = None
    company_website: Optional[str] = None
    is_tradeable: bool = True
    updated_at: datetime

# Dependency to get bearer token from header
async def get_bearer_token(authorization: Optional[str] = Header(None)) -> Optional[str]:
    """Extract bearer token from authorization header"""
    if authorization and authorization.startswith("Bearer "):
        return authorization.replace("Bearer ", "")
    return None

# Helper function to transform Synatic data to StockInfo
def transform_to_stock_info(instrument: Dict[str, Any]) -> StockInfo:
    """Transform Synatic instrument data to StockInfo model"""
    general = instrument.get("General", {})
    fundamental = instrument.get("Fundamental", {})
    price_data = instrument.get("Price", {})
    dividends = instrument.get("Dividends", {})
    returns = instrument.get("returns", {})
    
    # Determine market from exchange or flag code
    exchange = instrument.get("Exchange", "")
    flag_code = instrument.get("flagCode", "")
    if flag_code == "ZA" or exchange == "JSE":
        market = "JSE"
    elif exchange in ["USA", "NYSE", "NASDAQ"]:
        market = exchange if exchange != "USA" else "NYSE"
    else:
        market = exchange or "Unknown"
    
    return StockInfo(
        symbol=instrument.get("ticker", ""),
        name=instrument.get("name", ""),
        contract_code=instrument.get("contractCode", ""),
        sector=general.get("sector", instrument.get("category", "Unknown")),
        market=market,
        currency=general.get("currency", "ZAR" if flag_code == "ZA" else "USD"),
        last_price=float(instrument.get("lastPrice", 0) or 0) if instrument.get("lastPrice") else None,
        market_cap=fundamental.get("mktcap"),
        pe_ratio=fundamental.get("pe"),
        dividend_yield=dividends.get("yield"),
        volume=price_data.get("volume"),
        price_change_52w=price_data.get("pchange_52w"),
        returns_1m=returns.get("1mo", {}).get("percentage"),
        returns_3m=returns.get("3mo", {}).get("percentage"),
        returns_6m=returns.get("6mo", {}).get("percentage"),
        logo_url=instrument.get("logoUrl"),
        is_tradeable=True
    )
@router.post("/sync-to-mongodb")
async def sync_stocks_to_mongodb(
    token: Optional[str] = Depends(get_bearer_token)
):
    """Sync all stocks from Synatic to MongoDB"""
    
    if not token:
        raise HTTPException(status_code=401, detail="Authorization token required")
    
    try:
        # Initialize sync service
        sync_service = StockSyncService()
        
        # Sync all stocks
        stats = await sync_service.sync_all_stocks_from_synatic(token)
        
        return {
            "status": "success",
            "statistics": stats,
            "message": f"Synced {stats['total_stocks']} stocks to MongoDB"
        }
        
    except Exception as e:
        logger.error(f"Failed to sync stocks to MongoDB: {e}")
        raise HTTPException(status_code=500, detail=f"Sync failed: {str(e)}")

@router.get("/", response_model=StockSearchResponse)
async def get_all_stocks(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(50, ge=1, le=500, description="Items per page"),
    search: Optional[str] = Query(None, description="Search stocks by symbol or name"),
    category: str = Query("all", description="Category: all, equitiesexpanded, equities, etfs, bundles, crypto"),
    sector: Optional[str] = Query(None, description="Filter by sector"),
    market: Optional[str] = Query(None, description="Filter by market (JSE, NYSE, NASDAQ, etc.)"),
    currency: Optional[str] = Query(None, description="Filter by currency"),
    min_price: Optional[float] = Query(None, ge=0, description="Minimum stock price"),
    max_price: Optional[float] = Query(None, ge=0, description="Maximum stock price"),
    tradeable_only: bool = Query(True, description="Show only tradeable stocks"),
    sort_by: str = Query("symbol", description="Sort by: symbol, name, price, market_cap, returns_1m"),
    sort_order: str = Query("asc", description="Sort order: asc, desc"),
    token: Optional[str] = Depends(get_bearer_token)
):
    """Get all available stocks from EasyEquities with filtering and pagination"""
    
    if not token:
        raise HTTPException(status_code=401, detail="Authorization token required. Please include Bearer token in Authorization header")
    
    try:
        # Set token in client
        synatic_client.set_bearer_token(token)
        
        # Fetch instruments based on category
        all_instruments = []
        from_cache = False
        
        if category == "all":
            # Fetch from multiple categories
            sa_stocks = await synatic_client.get_all_sa_equities()
            us_stocks = await synatic_client.get_all_us_equities()
            etfs = await synatic_client.get_all_etfs()
            
            all_instruments.extend(sa_stocks)
            all_instruments.extend(us_stocks)
            all_instruments.extend(etfs)
        elif category == "equitiesexpanded":
            all_instruments = await synatic_client.get_all_sa_equities()
        elif category == "equities":
            all_instruments = await synatic_client.get_all_us_equities()
        elif category == "etfs":
            all_instruments = await synatic_client.get_all_etfs()
        elif category == "bundles":
            all_instruments = await synatic_client.get_all_bundles()
        elif category == "crypto":
            all_instruments = await synatic_client.get_all_crypto()
        else:
            all_instruments = await synatic_client.search_instruments(category=category)
        
        # Transform to StockInfo objects
        stocks = [transform_to_stock_info(inst) for inst in all_instruments]
        
        # Apply search filter
        if search:
            search_lower = search.lower()
            stocks = [
                stock for stock in stocks
                if search_lower in stock.symbol.lower() or 
                   search_lower in stock.name.lower() or
                   search_lower in stock.contract_code.lower()
            ]
        
        # Apply other filters
        if sector:
            stocks = [stock for stock in stocks if stock.sector.lower() == sector.lower()]
        
        if market:
            stocks = [stock for stock in stocks if stock.market.upper() == market.upper()]
        
        if currency:
            stocks = [stock for stock in stocks if stock.currency.upper() == currency.upper()]
        
        if min_price is not None:
            stocks = [stock for stock in stocks if stock.last_price and stock.last_price >= min_price]
        
        if max_price is not None:
            stocks = [stock for stock in stocks if stock.last_price and stock.last_price <= max_price]
        
        if tradeable_only:
            stocks = [stock for stock in stocks if stock.is_tradeable]
        
        # Sort stocks
        reverse = sort_order == "desc"
        if sort_by == "symbol":
            stocks.sort(key=lambda x: x.symbol, reverse=reverse)
        elif sort_by == "name":
            stocks.sort(key=lambda x: x.name, reverse=reverse)
        elif sort_by == "price":
            stocks.sort(key=lambda x: x.last_price or 0, reverse=reverse)
        elif sort_by == "market_cap":
            stocks.sort(key=lambda x: x.market_cap or 0, reverse=reverse)
        elif sort_by == "returns_1m":
            stocks.sort(key=lambda x: x.returns_1m or -999, reverse=reverse)
        
        # Apply pagination
        total_count = len(stocks)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_stocks = stocks[start_idx:end_idx]
        
        # Check if data came from cache
        cache_key = f"{category}__ALL_1"
        if cache_key in synatic_client._instruments_cache:
            from_cache = True
        
        return StockSearchResponse(
            stocks=paginated_stocks,
            total_count=total_count,
            page=page,
            per_page=per_page,
            from_cache=from_cache
        )
        
    except Exception as e:
        print(f"Error in get_all_stocks: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching stocks: {str(e)}")

@router.get("/markets", response_model=List[MarketSummary])
async def get_markets_summary(
    token: Optional[str] = Depends(get_bearer_token)
):
    """Get summary of all available markets"""
    
    if not token:
        raise HTTPException(status_code=401, detail="Authorization token required")
    
    try:
        synatic_client.set_bearer_token(token)
        
        # Fetch all categories
        all_data = await synatic_client.get_all_instruments()
        
        # Group by market
        markets_dict = {}
        for category, instruments in all_data.items():
            for inst in instruments:
                general = inst.get("General", {})
                exchange = inst.get("Exchange", "")
                flag_code = inst.get("flagCode", "")
                
                # Determine market
                if flag_code == "ZA" or exchange == "JSE":
                    market = "JSE"
                elif exchange in ["USA", "NYSE", "NASDAQ"]:
                    market = exchange if exchange != "USA" else "NYSE"
                else:
                    market = exchange or "Unknown"
                
                sector = general.get("sector", inst.get("category", "Unknown"))
                
                if market not in markets_dict:
                    markets_dict[market] = {
                        "count": 0,
                        "sectors": set()
                    }
                
                markets_dict[market]["count"] += 1
                markets_dict[market]["sectors"].add(sector)
        
        # Create summaries
        market_summaries = []
        for market_name, info in markets_dict.items():
            market_summaries.append(
                MarketSummary(
                    market=market_name,
                    total_stocks=info["count"],
                    sectors=sorted(list(info["sectors"]))
                )
            )
        
        return market_summaries
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching markets: {str(e)}")

@router.get("/sectors")
async def get_sectors(
    market: Optional[str] = Query(None, description="Filter sectors by market"),
    category: str = Query("all", description="Category to fetch sectors from"),
    token: Optional[str] = Depends(get_bearer_token)
):
    """Get all available sectors"""
    
    if not token:
        raise HTTPException(status_code=401, detail="Authorization token required")
    
    try:
        synatic_client.set_bearer_token(token)
        
        # Fetch instruments
        if category == "all":
            all_data = await synatic_client.get_all_instruments()
            all_instruments = []
            for instruments in all_data.values():
                all_instruments.extend(instruments)
        else:
            all_instruments = await synatic_client.search_instruments(category=category)
        
        # Extract unique sectors
        sectors = set()
        for inst in all_instruments:
            if market:
                inst_market = inst.get("Exchange", "")
                flag_code = inst.get("flagCode", "")
                if flag_code == "ZA":
                    inst_market = "JSE"
                if inst_market.upper() != market.upper():
                    continue
            
            general = inst.get("General", {})
            sector = general.get("sector", inst.get("category", "Unknown"))
            sectors.add(sector)
        
        return {"sectors": sorted(list(sectors))}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching sectors: {str(e)}")

@router.get("/search")
async def search_stocks(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(10, ge=1, le=50, description="Maximum number of results"),
    category: str = Query("all", description="Category to search in"),
    token: Optional[str] = Depends(get_bearer_token)
):
    """Search stocks by symbol or name"""
    
    if not token:
        raise HTTPException(status_code=401, detail="Authorization token required")
    
    try:
        synatic_client.set_bearer_token(token)
        
        # Search in specified category
        if category == "all":
            # Search in multiple categories
            sa_results = await synatic_client.search_instruments("equitiesexpanded", search_value=q)
            us_results = await synatic_client.search_instruments("equities", search_value=q)
            etf_results = await synatic_client.search_instruments("etfs", search_value=q)
            
            all_results = sa_results + us_results + etf_results
        else:
            all_results = await synatic_client.search_instruments(category, search_value=q)
        
        # Transform and limit results
        results = []
        for inst in all_results[:limit]:
            stock_info = transform_to_stock_info(inst)
            results.append({
                "symbol": stock_info.symbol,
                "name": stock_info.name,
                "contract_code": stock_info.contract_code,
                "market": stock_info.market,
                "last_price": stock_info.last_price,
                "currency": stock_info.currency,
                "logo_url": stock_info.logo_url
            })
        
        return {"results": results, "count": len(results)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching stocks: {str(e)}")

@router.get("/{symbol}", response_model=StockDetail)
async def get_stock_detail(
    symbol: str,
    token: Optional[str] = Depends(get_bearer_token)
):
    """Get detailed information for a specific stock"""
    
    if not token:
        raise HTTPException(status_code=401, detail="Authorization token required")
    
    try:
        synatic_client.set_bearer_token(token)
        
        # Search for the stock across categories
        found_instrument = None
        
        # Try searching by symbol
        for category in ["equitiesexpanded", "equities", "etfs"]:
            results = await synatic_client.search_instruments(category, search_value=symbol)
            for inst in results:
                if inst.get("ticker", "").upper() == symbol.upper() or \
                   inst.get("contractCode", "").upper() == symbol.upper():
                    found_instrument = inst
                    break
            if found_instrument:
                break
        
        if not found_instrument:
            raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")
        
        # Extract detailed information
        general = found_instrument.get("General", {})
        fundamental = found_instrument.get("Fundamental", {})
        price_data = found_instrument.get("Price", {})
        dividends = found_instrument.get("Dividends", {})
        returns = found_instrument.get("returns", {})
        
        # Determine market
        exchange = found_instrument.get("Exchange", "")
        flag_code = found_instrument.get("flagCode", "")
        if flag_code == "ZA" or exchange == "JSE":
            market = "JSE"
        elif exchange in ["USA", "NYSE", "NASDAQ"]:
            market = exchange if exchange != "USA" else "NYSE"
        else:
            market = exchange or "Unknown"
        
        return StockDetail(
            symbol=found_instrument.get("ticker", symbol),
            name=found_instrument.get("name", ""),
            contract_code=found_instrument.get("contractCode", ""),
            sector=general.get("sector", "Unknown"),
            industry=general.get("industry", "Unknown"),
            market=market,
            currency=general.get("currency", "ZAR" if flag_code == "ZA" else "USD"),
            description=found_instrument.get("description") or general.get("business_description"),
            last_price=float(found_instrument.get("lastPrice", 0) or 0) if found_instrument.get("lastPrice") else None,
            previous_close=None,  # Not provided by Synatic
            change=None,  # Would need to calculate
            change_percent=None,  # Would need to calculate
            volume=price_data.get("volume"),
            market_cap=fundamental.get("mktcap"),
            pe_ratio=fundamental.get("pe"),
            eps=fundamental.get("eps"),
            dividend_yield=dividends.get("yield"),
            price_52w_high=price_data.get("price52whigh"),
            price_52w_low=price_data.get("price52wlow"),
            returns_1m=returns.get("1mo", {}).get("percentage"),
            returns_3m=returns.get("3mo", {}).get("percentage"),
            returns_6m=returns.get("6mo", {}).get("percentage"),
            logo_url=found_instrument.get("logoUrl"),
            company_website=general.get("company_website"),
            is_tradeable=True,
            updated_at=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching stock details: {str(e)}")

@router.get("/{symbol}/similar")
async def get_similar_stocks(
    symbol: str,
    limit: int = Query(5, ge=1, le=20, description="Maximum number of similar stocks"),
    token: Optional[str] = Depends(get_bearer_token)
):
    """Get stocks similar to the specified stock (same sector/industry)"""
    
    if not token:
        raise HTTPException(status_code=401, detail="Authorization token required")
    
    try:
        synatic_client.set_bearer_token(token)
        
        # Find the target stock
        target_stock = None
        for category in ["equitiesexpanded", "equities", "etfs"]:
            results = await synatic_client.search_instruments(category, search_value=symbol)
            for inst in results:
                if inst.get("ticker", "").upper() == symbol.upper():
                    target_stock = inst
                    break
            if target_stock:
                break
        
        if not target_stock:
            raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")
        
        target_general = target_stock.get("General", {})
        target_sector = target_general.get("sector", "")
        target_industry = target_general.get("industry", "")
        target_market = target_stock.get("Exchange", "")
        
        # Get all stocks in the same category
        category = target_stock.get("category", "equities")
        all_stocks = await synatic_client.search_instruments(category)
        
        # Find similar stocks
        similar_stocks = []
        for stock in all_stocks:
            if stock.get("ticker") == target_stock.get("ticker"):
                continue
            
            stock_general = stock.get("General", {})
            similarity_score = 0
            
            # Calculate similarity
            if stock_general.get("sector") == target_sector:
                similarity_score += 0.5
            if stock_general.get("industry") == target_industry:
                similarity_score += 0.3
            if stock.get("Exchange") == target_market:
                similarity_score += 0.2
            
            if similarity_score > 0:
                similar_stocks.append({
                    "symbol": stock.get("ticker", ""),
                    "name": stock.get("name", ""),
                    "sector": stock_general.get("sector", ""),
                    "last_price": float(stock.get("lastPrice", 0) or 0),
                    "similarity_score": similarity_score
                })
        
        # Sort by similarity and limit
        similar_stocks.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        return {
            "symbol": symbol.upper(),
            "similar_stocks": similar_stocks[:limit]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching similar stocks: {str(e)}")

@router.post("/sync-all")
async def sync_all_instruments(
    token: Optional[str] = Depends(get_bearer_token)
):
    """Sync all instruments from EasyEquities (admin endpoint)"""
    
    if not token:
        raise HTTPException(status_code=401, detail="Authorization token required")
    
    try:
        synatic_client.set_bearer_token(token)
        
        # Fetch all categories
        all_data = await synatic_client.get_all_instruments()
        
        summary = {}
        total_instruments = 0
        
        for category, instruments in all_data.items():
            count = len(instruments)
            summary[category] = count
            total_instruments += count
        
        return {
            "status": "success",
            "total_instruments": total_instruments,
            "breakdown": summary,
            "cached": True,
            "cache_ttl_seconds": synatic_client._cache_ttl
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error syncing instruments: {str(e)}")