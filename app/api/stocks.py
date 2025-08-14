from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
from datetime import datetime

router = APIRouter(tags=["stocks"])

# Response Models
class StockInfo(BaseModel):
    symbol: str
    name: str
    sector: Optional[str] = None
    industry: Optional[str] = None
    market: Optional[str] = None
    currency: Optional[str] = None
    current_price: Optional[float] = None
    market_cap: Optional[float] = None
    volume: Optional[int] = None
    pe_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    
class StockListResponse(BaseModel):
    stocks: List[StockInfo]
    total_count: int
    returned_count: int

# JSE stocks (South African)
JSE_STOCKS = [
    "NPN.JO",   # Naspers
    "PRX.JO",   # Prosus  
    "SHP.JO",   # Shoprite
    "ABG.JO",   # Absa Group
    "FSR.JO",   # FirstRand
    "SOL.JO",   # Sasol
    "NED.JO",   # Nedbank
    "AGL.JO",   # Anglo American
    "BVT.JO",   # Bidvest
    "MTN.JO",   # MTN Group
    "VOD.JO",   # Vodacom
    "TKG.JO",   # Telkom
    "IMP.JO",   # Impala Platinum
    "BHP.JO",   # BHP Billiton
    "APN.JO",   # Aspen Pharmacare
    "KIO.JO",   # Kumba Iron Ore
    "MRP.JO",   # Mr Price Group
    "TBS.JO",   # Tiger Brands
    "PIK.JO",   # Pick n Pay
    "WHL.JO"    # Woolworths
]

# US stocks
US_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", 
    "META", "NVDA", "NFLX", "JPM", "JNJ"
]

# Combine all stocks
POPULAR_STOCKS = JSE_STOCKS + US_STOCKS

def get_stock_info(symbol: str) -> Optional[StockInfo]:
    """Get stock information from yfinance"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Debug: Print some info to see what we're getting
        print(f"Fetching {symbol}...")
        
        # Get current price - try multiple fields
        current_price = (info.get('currentPrice') or 
                        info.get('regularMarketPrice') or 
                        info.get('ask') or 
                        info.get('bid') or
                        info.get('previousClose'))
        
        # Determine market and currency based on symbol
        if symbol.endswith(".JO"):
            market = "JSE"
            default_currency = "ZAR"
        elif symbol.endswith(".SW"):
            market = "SWX"  # Swiss Exchange
            default_currency = "CHF"
        elif symbol.endswith(".PA"):
            market = "EPA"  # Euronext Paris
            default_currency = "EUR"
        elif "." not in symbol:
            market = "NASDAQ"
            default_currency = "USD"
        else:
            market = "Other"
            default_currency = "USD"
        
        # Get currency, fallback to default based on market
        raw_currency = info.get('currency') or default_currency
        
        # Normalize South African currency and convert cents to rands
        if raw_currency == "ZAc":  # South African cents
            currency = "ZAR"   # Convert to ZAR for consistency
            # Convert price from cents to rands
            if current_price:
                current_price = current_price / 100
        else:
            currency = raw_currency
        
        # Get name
        name = info.get('longName') or info.get('shortName') or symbol
        
        stock_info = StockInfo(
            symbol=symbol,
            name=name,
            sector=info.get('sector'),
            industry=info.get('industry'),
            market=market,
            currency=currency,
            current_price=current_price,
            market_cap=info.get('marketCap'),
            volume=info.get('volume') or info.get('regularMarketVolume'),
            pe_ratio=info.get('trailingPE') or info.get('forwardPE'),
            dividend_yield=info.get('dividendYield')
        )
        
        print(f"✓ {symbol}: {name}, {currency}, ${current_price}")
        return stock_info
        
    except Exception as e:
        print(f"✗ Error fetching data for {symbol}: {str(e)}")
        return None

@router.get("/", response_model=StockListResponse)
async def get_stocks(
    limit: int = Query(20, ge=1, le=100, description="Number of stocks to return"),
    search: Optional[str] = Query(None, description="Search stocks by symbol or name"),
    sector: Optional[str] = Query(None, description="Filter by sector"),
    market: Optional[str] = Query(None, description="Filter by market"),
    currency: Optional[str] = Query(None, description="Filter by currency"),
    min_price: Optional[float] = Query(None, ge=0, description="Minimum stock price"),
    max_price: Optional[float] = Query(None, ge=0, description="Maximum stock price"),
    has_dividend: Optional[bool] = Query(None, description="Filter stocks that pay dividends"),
    sort_by: str = Query("symbol", description="Sort by: symbol, name, price, market_cap"),
    sort_order: str = Query("asc", description="Sort order: asc, desc")
):
    """Get stocks from yfinance with filtering and simple limit"""
    try:
        print(f"Fetching stocks with filters: market={market}, currency={currency}, limit={limit}")
        
        # Get stock data
        stocks_data = []
        
        # Determine which stocks to fetch based on filters
        symbols_to_fetch = POPULAR_STOCKS
        
        # If filtering by market or currency, pre-filter symbols
        if market:
            if market.upper() == "JSE":
                symbols_to_fetch = JSE_STOCKS
            elif market.upper() in ["NASDAQ", "NYSE"]:
                symbols_to_fetch = US_STOCKS
        
        print(f"Fetching {len(symbols_to_fetch)} symbols...")
        
        # Fetch data for selected symbols
        for i, symbol in enumerate(symbols_to_fetch):
            print(f"Progress: {i+1}/{len(symbols_to_fetch)} - {symbol}")
            stock_info = get_stock_info(symbol)
            if stock_info:
                stocks_data.append(stock_info)
            
            # Limit fetching for performance (remove this in production)
            if len(stocks_data) >= 50:  # Fetch more stocks to allow better filtering
                break
        
        print(f"Successfully fetched {len(stocks_data)} stocks")
        
        # Apply filters
        filtered_stocks = stocks_data.copy()
        
        if search:
            search_lower = search.lower()
            filtered_stocks = [
                stock for stock in filtered_stocks
                if (search_lower in stock.symbol.lower() or 
                    (stock.name and search_lower in stock.name.lower()))
            ]
        
        if sector:
            filtered_stocks = [
                stock for stock in filtered_stocks 
                if stock.sector and sector.lower() in stock.sector.lower()
            ]
        
        if market:
            filtered_stocks = [
                stock for stock in filtered_stocks 
                if stock.market and stock.market.upper() == market.upper()
            ]
        
        if currency:
            print(f"Filtering by currency: {currency}")
            before_count = len(filtered_stocks)
            
            # Handle both ZAR and ZAc (South African cents)
            if currency.upper() == "ZAR":
                filtered_stocks = [
                    stock for stock in filtered_stocks 
                    if stock.currency and stock.currency.upper() in ["ZAR", "ZAC"]
                ]
            else:
                filtered_stocks = [
                    stock for stock in filtered_stocks 
                    if stock.currency and stock.currency.upper() == currency.upper()
                ]
            print(f"Currency filter: {before_count} -> {len(filtered_stocks)} stocks")
        
        if min_price is not None:
            filtered_stocks = [
                stock for stock in filtered_stocks 
                if stock.current_price and stock.current_price >= min_price
            ]
        
        if max_price is not None:
            filtered_stocks = [
                stock for stock in filtered_stocks 
                if stock.current_price and stock.current_price <= max_price
            ]
        
        if has_dividend is not None:
            if has_dividend:
                filtered_stocks = [
                    stock for stock in filtered_stocks 
                    if stock.dividend_yield and stock.dividend_yield > 0
                ]
            else:
                filtered_stocks = [
                    stock for stock in filtered_stocks 
                    if not stock.dividend_yield or stock.dividend_yield == 0
                ]
        
        # Sort stocks
        reverse = sort_order.lower() == "desc"
        
        if sort_by == "symbol":
            filtered_stocks.sort(key=lambda x: x.symbol, reverse=reverse)
        elif sort_by == "name":
            filtered_stocks.sort(key=lambda x: x.name or "", reverse=reverse)
        elif sort_by == "price":
            filtered_stocks.sort(key=lambda x: x.current_price or 0, reverse=reverse)
        elif sort_by == "market_cap":
            filtered_stocks.sort(key=lambda x: x.market_cap or 0, reverse=reverse)
        
        # Apply limit
        total_count = len(filtered_stocks)
        limited_stocks = filtered_stocks[:limit]
        
        print(f"Final result: {len(limited_stocks)} stocks returned (total available: {total_count})")
        
        return StockListResponse(
            stocks=limited_stocks,
            total_count=total_count,
            returned_count=len(limited_stocks)
        )
        
    except Exception as e:
        print(f"Error in get_stocks: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching stocks: {str(e)}")

@router.get("/sectors")
async def get_available_sectors():
    """Get all available sectors from the current stock list"""
    try:
        print("Fetching sectors from yfinance...")
        sectors = set()
        
        # Get unique sectors from a sample of stocks
        sample_stocks = POPULAR_STOCKS[:20]  # Just check first 20 for performance
        for symbol in sample_stocks:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                sector = info.get('sector')
                if sector:
                    sectors.add(sector)
            except:
                continue
        
        return {"sectors": sorted(list(sectors))}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching sectors: {str(e)}")

@router.get("/markets")
async def get_available_markets():
    """Get all available markets"""
    return {
        "markets": ["JSE", "NASDAQ", "NYSE", "Other"],
        "description": {
            "JSE": "Johannesburg Stock Exchange (South African stocks ending in .JO)",
            "NASDAQ": "NASDAQ Global Select Market",
            "NYSE": "New York Stock Exchange", 
            "Other": "Other international exchanges"
        }
    }

@router.get("/test/{symbol}")
async def test_stock(symbol: str):
    """Test endpoint to debug a single stock"""
    try:
        print(f"Testing stock: {symbol}")
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Return raw info for debugging
        return {
            "symbol": symbol,
            "raw_info": {
                "longName": info.get('longName'),
                "shortName": info.get('shortName'),
                "currency": info.get('currency'),
                "currentPrice": info.get('currentPrice'),
                "regularMarketPrice": info.get('regularMarketPrice'),
                "sector": info.get('sector'),
                "industry": info.get('industry'),
                "marketCap": info.get('marketCap'),
                "volume": info.get('volume')
            },
            "processed": get_stock_info(symbol)
        }
    except Exception as e:
        return {"error": str(e), "symbol": symbol}

@router.get("/currencies")
async def get_available_currencies():
    """Get all available currencies"""
    return {
        "currencies": ["USD", "ZAR", "EUR", "GBP", "JPY"],
        "description": {
            "USD": "US Dollar",
            "ZAR": "South African Rand",
            "EUR": "Euro",
            "GBP": "British Pound",
            "JPY": "Japanese Yen"
        }
    }