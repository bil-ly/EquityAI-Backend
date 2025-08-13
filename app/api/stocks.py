from fastapi import APIRouter, HTTPException, Query, Depends, Header
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime, timedelta
import httpx
import asyncio
from functools import lru_cache
import os
import json

router = APIRouter(tags=["stocks"])

# Response Models
class StockInfo(BaseModel):
    symbol: str
    name: str
    sector: str
    market: str
    currency: str
    last_price: Optional[float] = None
    market_cap: Optional[float] = None
    is_tradeable: bool = True
    
class StockSearchResponse(BaseModel):
    stocks: List[StockInfo]
    total_count: int
    page: int
    per_page: int
    
class MarketSummary(BaseModel):
    market: str
    total_stocks: int
    sectors: List[str]
    
class StockDetail(BaseModel):
    symbol: str
    name: str
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
    dividend_yield: Optional[float] = None
    is_tradeable: bool = True
    updated_at: datetime

# Real Market Data Client
class MarketDataClient:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        self._instruments_cache = None
        self._prices_cache = {}
        self._cache_timestamp = None
        self._price_cache_timestamps = {}
        self._cache_ttl = 3600  # Cache instruments for 1 hour
        self._price_cache_ttl = 60  # Cache prices for 1 minute
        
        # API keys (you can get free keys from these services)
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY", "demo")
        self.twelve_data_key = os.getenv("TWELVE_DATA_API_KEY", "demo")
        self.finnhub_key = os.getenv("FINNHUB_API_KEY", "")
    
    async def close(self):
        await self.client.aclose()
    
    async def fetch_jse_instruments(self) -> List[Dict[str, Any]]:
        """Fetch JSE instruments from various sources"""
        instruments = []
        
        # JSE Top 40 companies with real data
        jse_stocks = [
            {"symbol": "NPN.JO", "ticker": "NPN", "name": "Naspers Ltd", "sector": "Technology", "industry": "Internet & Digital Media"},
            {"symbol": "PRX.JO", "ticker": "PRX", "name": "Prosus NV", "sector": "Technology", "industry": "Internet & Digital Media"},
            {"symbol": "AGL.JO", "ticker": "AGL", "name": "Anglo American Plc", "sector": "Materials", "industry": "Mining"},
            {"symbol": "BHP.JO", "ticker": "BHP", "name": "BHP Group Plc", "sector": "Materials", "industry": "Mining"},
            {"symbol": "CFR.JO", "ticker": "CFR", "name": "Compagnie Fin Richemont", "sector": "Consumer Discretionary", "industry": "Luxury Goods"},
            {"symbol": "SHP.JO", "ticker": "SHP", "name": "Shoprite Holdings Ltd", "sector": "Consumer Staples", "industry": "Food Retail"},
            {"symbol": "ABG.JO", "ticker": "ABG", "name": "Absa Group Ltd", "sector": "Financial Services", "industry": "Banks"},
            {"symbol": "CPI.JO", "ticker": "CPI", "name": "Capitec Bank Holdings Ltd", "sector": "Financial Services", "industry": "Banks"},
            {"symbol": "DSY.JO", "ticker": "DSY", "name": "Discovery Ltd", "sector": "Financial Services", "industry": "Insurance"},
            {"symbol": "FSR.JO", "ticker": "FSR", "name": "Firstrand Ltd", "sector": "Financial Services", "industry": "Banks"},
            {"symbol": "GLD.JO", "ticker": "GLD", "name": "Gold Fields Ltd", "sector": "Materials", "industry": "Gold Mining"},
            {"symbol": "IMP.JO", "ticker": "IMP", "name": "Impala Platinum Holdings Ltd", "sector": "Materials", "industry": "Platinum Mining"},
            {"symbol": "MTN.JO", "ticker": "MTN", "name": "MTN Group Ltd", "sector": "Communication Services", "industry": "Telecommunications"},
            {"symbol": "NED.JO", "ticker": "NED", "name": "Nedbank Group Ltd", "sector": "Financial Services", "industry": "Banks"},
            {"symbol": "NRP.JO", "ticker": "NRP", "name": "NEPI Rockcastle Plc", "sector": "Real Estate", "industry": "REITs"},
            {"symbol": "OMU.JO", "ticker": "OMU", "name": "Old Mutual Ltd", "sector": "Financial Services", "industry": "Insurance"},
            {"symbol": "REM.JO", "ticker": "REM", "name": "Remgro Ltd", "sector": "Industrials", "industry": "Conglomerates"},
            {"symbol": "SBK.JO", "ticker": "SBK", "name": "Standard Bank Group Ltd", "sector": "Financial Services", "industry": "Banks"},
            {"symbol": "SLM.JO", "ticker": "SLM", "name": "Sanlam Ltd", "sector": "Financial Services", "industry": "Insurance"},
            {"symbol": "SOL.JO", "ticker": "SOL", "name": "Sasol Ltd", "sector": "Energy", "industry": "Oil & Gas"},
            {"symbol": "TBS.JO", "ticker": "TBS", "name": "Tiger Brands Ltd", "sector": "Consumer Staples", "industry": "Food Products"},
            {"symbol": "TFG.JO", "ticker": "TFG", "name": "The Foschini Group Ltd", "sector": "Consumer Discretionary", "industry": "Apparel Retail"},
            {"symbol": "VOD.JO", "ticker": "VOD", "name": "Vodacom Group Ltd", "sector": "Communication Services", "industry": "Telecommunications"},
            {"symbol": "WHL.JO", "ticker": "WHL", "name": "Woolworths Holdings Ltd", "sector": "Consumer Discretionary", "industry": "Department Stores"},
            {"symbol": "APN.JO", "ticker": "APN", "name": "Aspen Pharmacare Holdings Ltd", "sector": "Healthcare", "industry": "Pharmaceuticals"},
            {"symbol": "BTI.JO", "ticker": "BTI", "name": "British American Tobacco Plc", "sector": "Consumer Staples", "industry": "Tobacco"},
            {"symbol": "GLN.JO", "ticker": "GLN", "name": "Glencore Plc", "sector": "Materials", "industry": "Diversified Mining"},
            {"symbol": "AMS.JO", "ticker": "AMS", "name": "Anglo American Platinum Ltd", "sector": "Materials", "industry": "Platinum Mining"},
            {"symbol": "BID.JO", "ticker": "BID", "name": "Bid Corporation Ltd", "sector": "Consumer Staples", "industry": "Food Distribution"},
            {"symbol": "CLS.JO", "ticker": "CLS", "name": "Clicks Group Ltd", "sector": "Consumer Staples", "industry": "Drug Retail"},
        ]
        
        for stock in jse_stocks:
            instruments.append({
                "symbol": stock["ticker"],
                "name": stock["name"],
                "exchange": "JSE",
                "market": "JSE",
                "currency": "ZAR",
                "type": "equity",
                "sector": stock["sector"],
                "industry": stock["industry"],
                "isTradeable": True,
                "yahoo_symbol": stock["symbol"]  # Store Yahoo Finance symbol for price data
            })
        
        # Add JSE ETFs
        jse_etfs = [
            {"symbol": "STX40", "name": "Satrix 40 ETF", "yahoo": "STX40.JO"},
            {"symbol": "STXNDQ", "name": "Satrix Nasdaq 100 ETF", "yahoo": "STXNDQ.JO"},
            {"symbol": "STXWDM", "name": "Satrix MSCI World ETF", "yahoo": "STXWDM.JO"},
            {"symbol": "ASHGEQ", "name": "Ashburton Global 1200 Equity ETF", "yahoo": "ASHGEQ.JO"},
            {"symbol": "STXEMG", "name": "Satrix Emerging Markets ETF", "yahoo": "STXEMG.JO"},
            {"symbol": "DIVTRX", "name": "CoreShares Dividend Aristocrats ETF", "yahoo": "DIVTRX.JO"},
            {"symbol": "SMART", "name": "CoreShares Smart Beta ETF", "yahoo": "SMART.JO"},
        ]
        
        for etf in jse_etfs:
            instruments.append({
                "symbol": etf["symbol"],
                "name": etf["name"],
                "exchange": "JSE",
                "market": "JSE",
                "currency": "ZAR",
                "type": "etf",
                "sector": "ETFs",
                "industry": "Exchange Traded Funds",
                "isTradeable": True,
                "yahoo_symbol": etf["yahoo"]
            })
        
        return instruments
    
    async def fetch_us_instruments(self) -> List[Dict[str, Any]]:
        """Fetch popular US instruments available on EasyEquities"""
        instruments = []
        
        # Popular US stocks on EasyEquities
        us_stocks = [
            {"symbol": "AAPL", "name": "Apple Inc", "sector": "Technology", "exchange": "NASDAQ"},
            {"symbol": "MSFT", "name": "Microsoft Corp", "sector": "Technology", "exchange": "NASDAQ"},
            {"symbol": "GOOGL", "name": "Alphabet Inc Class A", "sector": "Technology", "exchange": "NASDAQ"},
            {"symbol": "AMZN", "name": "Amazon.com Inc", "sector": "Consumer Discretionary", "exchange": "NASDAQ"},
            {"symbol": "TSLA", "name": "Tesla Inc", "sector": "Consumer Discretionary", "exchange": "NASDAQ"},
            {"symbol": "META", "name": "Meta Platforms Inc", "sector": "Technology", "exchange": "NASDAQ"},
            {"symbol": "NVDA", "name": "NVIDIA Corp", "sector": "Technology", "exchange": "NASDAQ"},
            {"symbol": "BRK.B", "name": "Berkshire Hathaway Inc Class B", "sector": "Financial Services", "exchange": "NYSE"},
            {"symbol": "JPM", "name": "JPMorgan Chase & Co", "sector": "Financial Services", "exchange": "NYSE"},
            {"symbol": "V", "name": "Visa Inc", "sector": "Financial Services", "exchange": "NYSE"},
            {"symbol": "MA", "name": "Mastercard Inc", "sector": "Financial Services", "exchange": "NYSE"},
            {"symbol": "DIS", "name": "Walt Disney Co", "sector": "Communication Services", "exchange": "NYSE"},
            {"symbol": "NFLX", "name": "Netflix Inc", "sector": "Communication Services", "exchange": "NASDAQ"},
            {"symbol": "PFE", "name": "Pfizer Inc", "sector": "Healthcare", "exchange": "NYSE"},
            {"symbol": "KO", "name": "Coca-Cola Co", "sector": "Consumer Staples", "exchange": "NYSE"},
            {"symbol": "PEP", "name": "PepsiCo Inc", "sector": "Consumer Staples", "exchange": "NASDAQ"},
            {"symbol": "WMT", "name": "Walmart Inc", "sector": "Consumer Staples", "exchange": "NYSE"},
            {"symbol": "NKE", "name": "Nike Inc", "sector": "Consumer Discretionary", "exchange": "NYSE"},
            {"symbol": "BA", "name": "Boeing Co", "sector": "Industrials", "exchange": "NYSE"},
            {"symbol": "AMD", "name": "Advanced Micro Devices Inc", "sector": "Technology", "exchange": "NASDAQ"},
        ]
        
        for stock in us_stocks:
            instruments.append({
                "symbol": stock["symbol"],
                "name": stock["name"],
                "exchange": stock["exchange"],
                "market": stock["exchange"],
                "currency": "USD",
                "type": "equity",
                "sector": stock["sector"],
                "industry": stock.get("industry", stock["sector"]),
                "isTradeable": True,
                "yahoo_symbol": stock["symbol"]
            })
        
        # Popular US ETFs
        us_etfs = [
            {"symbol": "SPY", "name": "SPDR S&P 500 ETF", "exchange": "NYSE"},
            {"symbol": "VOO", "name": "Vanguard S&P 500 ETF", "exchange": "NYSE"},
            {"symbol": "QQQ", "name": "Invesco QQQ Trust", "exchange": "NASDAQ"},
            {"symbol": "VTI", "name": "Vanguard Total Stock Market ETF", "exchange": "NYSE"},
            {"symbol": "IWM", "name": "iShares Russell 2000 ETF", "exchange": "NYSE"},
        ]
        
        for etf in us_etfs:
            instruments.append({
                "symbol": etf["symbol"],
                "name": etf["name"],
                "exchange": etf["exchange"],
                "market": etf["exchange"],
                "currency": "USD",
                "type": "etf",
                "sector": "ETFs",
                "industry": "Exchange Traded Funds",
                "isTradeable": True,
                "yahoo_symbol": etf["symbol"]
            })
        
        return instruments
    
    async def fetch_all_instruments(self) -> List[Dict[str, Any]]:
        """Fetch all available instruments"""
        # Check cache first
        if self._instruments_cache and self._cache_timestamp:
            if (datetime.now() - self._cache_timestamp).seconds < self._cache_ttl:
                return self._instruments_cache
        
        try:
            # Fetch from different markets
            jse_instruments = await self.fetch_jse_instruments()
            us_instruments = await self.fetch_us_instruments()
            
            # Combine all instruments
            all_instruments = jse_instruments + us_instruments
            
            # Cache the result
            self._instruments_cache = all_instruments
            self._cache_timestamp = datetime.now()
            
            print(f"Loaded {len(all_instruments)} instruments")
            return all_instruments
            
        except Exception as e:
            print(f"Error fetching instruments: {str(e)}")
            return self._instruments_cache or []
    
    async def fetch_yahoo_quote(self, symbol: str) -> Dict[str, Any]:
        """Fetch quote from Yahoo Finance"""
        try:
            # Yahoo Finance API endpoint (unofficial but works)
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            
            response = await self.client.get(url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            })
            
            if response.status_code == 200:
                data = response.json()
                result = data.get("chart", {}).get("result", [{}])[0]
                meta = result.get("meta", {})
                
                price = meta.get("regularMarketPrice")
                currency = meta.get("currency", "")
                
                # Convert JSE prices from cents to Rands if needed
                if currency == "ZAc" and price:  # Yahoo returns JSE prices in cents
                    price = price / 100
                    currency = "ZAR"
                
                return {
                    "price": price,
                    "previousClose": meta.get("previousClose"),
                    "volume": meta.get("regularMarketVolume"),
                    "marketCap": meta.get("marketCap"),
                    "currency": currency,
                    "exchange": meta.get("exchangeName"),
                }
            
        except Exception as e:
            print(f"Error fetching Yahoo quote for {symbol}: {str(e)}")
        
        return {}
    
    async def fetch_batch_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch quotes for multiple symbols"""
        quotes = {}
        
        # Check cache first
        symbols_to_fetch = []
        for symbol in symbols:
            if symbol in self._prices_cache:
                cache_time = self._price_cache_timestamps.get(symbol)
                if cache_time and (datetime.now() - cache_time).seconds < self._price_cache_ttl:
                    quotes[symbol] = self._prices_cache[symbol]
                    continue
            symbols_to_fetch.append(symbol)
        
        if not symbols_to_fetch:
            return quotes
        
        # Fetch quotes in parallel
        tasks = []
        for symbol in symbols_to_fetch[:20]:  # Limit to 20 concurrent requests
            tasks.append(self.fetch_yahoo_quote(symbol))
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for symbol, result in zip(symbols_to_fetch[:20], results):
                if isinstance(result, dict) and result.get("price"):
                    quotes[symbol] = result
                    # Cache the result
                    self._prices_cache[symbol] = result
                    self._price_cache_timestamps[symbol] = datetime.now()
        
        return quotes

# Dependency to get market data client
@lru_cache()
def get_market_client():
    return MarketDataClient()

# Helper function to transform to StockInfo
def transform_to_stock_info(instrument: Dict[str, Any], quote: Optional[Dict[str, Any]] = None) -> StockInfo:
    """Transform instrument and quote data to StockInfo model"""
    price = None
    market_cap = None
    
    if quote:
        price = quote.get("price")
        market_cap = quote.get("marketCap")
        
        # Handle JSE prices in cents
        currency = quote.get("currency", instrument.get("currency", ""))
        if currency == "ZAc" and price:
            price = price / 100  # Convert cents to Rands
    
    return StockInfo(
        symbol=instrument.get("symbol", ""),
        name=instrument.get("name", ""),
        sector=instrument.get("sector", "Other"),
        market=instrument.get("market", instrument.get("exchange", "")),
        currency=instrument.get("currency", "USD"),
        last_price=round(float(price), 2) if price else None,
        market_cap=float(market_cap) if market_cap else None,
        is_tradeable=instrument.get("isTradeable", True)
    )

@router.get("/", response_model=StockSearchResponse)
async def get_all_stocks(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(50, ge=1, le=500, description="Items per page"),
    search: Optional[str] = Query(None, description="Search stocks by symbol or name"),
    sector: Optional[str] = Query(None, description="Filter by sector"),
    market: Optional[str] = Query(None, description="Filter by market (JSE, NYSE, NASDAQ, etc.)"),
    currency: Optional[str] = Query(None, description="Filter by currency"),
    min_price: Optional[float] = Query(None, ge=0, description="Minimum stock price"),
    max_price: Optional[float] = Query(None, ge=0, description="Maximum stock price"),
    tradeable_only: bool = Query(True, description="Show only tradeable stocks"),
    sort_by: str = Query("symbol", description="Sort by: symbol, name, price, market_cap"),
    sort_order: str = Query("asc", description="Sort order: asc, desc"),
    include_prices: bool = Query(True, description="Include real-time prices"),
    client: MarketDataClient = Depends(get_market_client)
):
    """Get all available stocks with real market data"""
    try:
        # Fetch all instruments
        instruments = await client.fetch_all_instruments()
        
        # Filter based on search
        if search:
            search_lower = search.lower()
            instruments = [
                inst for inst in instruments
                if search_lower in inst.get("symbol", "").lower() or 
                   search_lower in inst.get("name", "").lower()
            ]
        
        # Apply other filters
        if sector:
            instruments = [inst for inst in instruments if inst.get("sector", "").lower() == sector.lower()]
        
        if market:
            instruments = [inst for inst in instruments if inst.get("market", "").upper() == market.upper()]
        
        if currency:
            instruments = [inst for inst in instruments if inst.get("currency", "").upper() == currency.upper()]
        
        # Calculate pagination indices
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        
        # Get quotes for ALL instruments if we need to filter by price
        # Otherwise only get quotes for visible page
        quotes = {}
        if include_prices:
            if min_price is not None or max_price is not None:
                # Need all quotes for price filtering
                yahoo_symbols = [inst.get("yahoo_symbol", inst.get("symbol")) for inst in instruments]
                quotes = await client.fetch_batch_quotes(yahoo_symbols)
            else:
                # Only get quotes for visible page
                visible_instruments = instruments[start_idx:end_idx]
                yahoo_symbols = [inst.get("yahoo_symbol", inst.get("symbol")) for inst in visible_instruments]
                quotes = await client.fetch_batch_quotes(yahoo_symbols)
        
        # Transform to StockInfo objects with prices
        all_stocks = []
        for instrument in instruments:
            yahoo_symbol = instrument.get("yahoo_symbol", instrument.get("symbol"))
            quote = quotes.get(yahoo_symbol)
            stock_info = transform_to_stock_info(instrument, quote)
            all_stocks.append(stock_info)
        
        # Apply price filters after getting quotes
        if min_price is not None:
            all_stocks = [s for s in all_stocks if s.last_price and s.last_price >= min_price]
        
        if max_price is not None:
            all_stocks = [s for s in all_stocks if s.last_price and s.last_price <= max_price]
        
        if tradeable_only:
            all_stocks = [s for s in all_stocks if s.is_tradeable]
        
        # Sort stocks
        reverse = sort_order == "desc"
        if sort_by == "symbol":
            all_stocks.sort(key=lambda x: x.symbol, reverse=reverse)
        elif sort_by == "name":
            all_stocks.sort(key=lambda x: x.name, reverse=reverse)
        elif sort_by == "price":
            all_stocks.sort(key=lambda x: x.last_price or 0, reverse=reverse)
        elif sort_by == "market_cap":
            all_stocks.sort(key=lambda x: x.market_cap or 0, reverse=reverse)
        
        # Apply pagination (if we haven't already fetched quotes for just the page)
        total_count = len(all_stocks)
        if min_price is not None or max_price is not None:
            # Re-apply pagination after filtering
            paginated_stocks = all_stocks[start_idx:end_idx]
        else:
            # Already have the right page
            paginated_stocks = all_stocks[start_idx:end_idx]
        
        # If we didn't get quotes yet (because no price filter and include_prices=True after pagination)
        if include_prices and not quotes and paginated_stocks:
            yahoo_symbols = [
                inst.get("yahoo_symbol", inst.get("symbol")) 
                for inst in instruments[start_idx:end_idx]
            ]
            page_quotes = await client.fetch_batch_quotes(yahoo_symbols)
            
            # Update paginated stocks with quotes
            updated_stocks = []
            for i, stock in enumerate(paginated_stocks):
                if i < len(instruments[start_idx:end_idx]):
                    instrument = instruments[start_idx + i]
                    yahoo_symbol = instrument.get("yahoo_symbol", instrument.get("symbol"))
                    quote = page_quotes.get(yahoo_symbol)
                    if quote:
                        updated_stock = transform_to_stock_info(instrument, quote)
                        updated_stocks.append(updated_stock)
                    else:
                        updated_stocks.append(stock)
                else:
                    updated_stocks.append(stock)
            paginated_stocks = updated_stocks
        
        return StockSearchResponse(
            stocks=paginated_stocks,
            total_count=total_count,
            page=page,
            per_page=per_page
        )
        
    except Exception as e:
        print(f"Error in get_all_stocks: {str(e)}")
        # Return empty result instead of error
        return StockSearchResponse(
            stocks=[],
            total_count=0,
            page=page,
            per_page=per_page
        )

@router.get("/markets", response_model=List[MarketSummary])
async def get_markets_summary(
    client: MarketDataClient = Depends(get_market_client)
):
    """Get summary of all available markets"""
    try:
        instruments = await client.fetch_all_instruments()
        
        # Group by market
        markets_dict = {}
        for instrument in instruments:
            market = instrument.get("market", "Unknown")
            sector = instrument.get("sector", "Unknown")
            
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
        print(f"Error fetching markets: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching markets: {str(e)}")

@router.get("/sectors")
async def get_sectors(
    market: Optional[str] = Query(None, description="Filter sectors by market"),
    client: MarketDataClient = Depends(get_market_client)
):
    """Get all available sectors"""
    try:
        instruments = await client.fetch_all_instruments()
        
        # Extract unique sectors
        sectors = set()
        for instrument in instruments:
            if market:
                if instrument.get("market", "").upper() != market.upper():
                    continue
            
            sector = instrument.get("sector", "Unknown")
            sectors.add(sector)
        
        return {"sectors": sorted(list(sectors))}
        
    except Exception as e:
        print(f"Error fetching sectors: {str(e)}")
        return {"sectors": []}

@router.get("/search")
async def search_stocks(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(10, ge=1, le=50, description="Maximum number of results"),
    client: MarketDataClient = Depends(get_market_client)
):
    """Search stocks by symbol or name"""
    try:
        instruments = await client.fetch_all_instruments()
        
        # Search
        search_lower = q.lower()
        results = []
        
        for instrument in instruments:
            if (search_lower in instrument.get("symbol", "").lower() or 
                search_lower in instrument.get("name", "").lower()):
                
                # Get quote for the result
                yahoo_symbol = instrument.get("yahoo_symbol", instrument.get("symbol"))
                quote = await client.fetch_yahoo_quote(yahoo_symbol)
                
                stock_info = transform_to_stock_info(instrument, quote)
                results.append({
                    "symbol": stock_info.symbol,
                    "name": stock_info.name,
                    "market": stock_info.market,
                    "last_price": stock_info.last_price,
                    "currency": stock_info.currency
                })
                
                if len(results) >= limit:
                    break
        
        return {"results": results, "count": len(results)}
        
    except Exception as e:
        print(f"Error searching stocks: {str(e)}")
        return {"results": [], "count": 0}

@router.get("/{symbol}", response_model=StockDetail)
async def get_stock_detail(
    symbol: str,
    client: MarketDataClient = Depends(get_market_client)
):
    """Get detailed information for a specific stock"""
    try:
        instruments = await client.fetch_all_instruments()
        
        # Find the instrument
        instrument = None
        for inst in instruments:
            if inst.get("symbol", "").upper() == symbol.upper():
                instrument = inst
                break
        
        if not instrument:
            raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")
        
        # Get real-time quote
        yahoo_symbol = instrument.get("yahoo_symbol", instrument.get("symbol"))
        quote = await client.fetch_yahoo_quote(yahoo_symbol)
        
        # Calculate changes
        price = quote.get("price", 0)
        prev_close = quote.get("previousClose", price)
        change = price - prev_close if price and prev_close else None
        change_percent = (change / prev_close * 100) if change and prev_close else None
        
        return StockDetail(
            symbol=instrument.get("symbol"),
            name=instrument.get("name"),
            sector=instrument.get("sector", "Unknown"),
            industry=instrument.get("industry", "Unknown"),
            market=instrument.get("market"),
            currency=quote.get("currency") or instrument.get("currency"),
            description=f"{instrument.get('name')} operates in the {instrument.get('sector')} sector",
            last_price=price,
            previous_close=prev_close,
            change=change,
            change_percent=change_percent,
            volume=quote.get("volume"),
            market_cap=quote.get("marketCap"),
            pe_ratio=None,  # Would need additional API for this
            dividend_yield=None,  # Would need additional API for this
            is_tradeable=instrument.get("isTradeable", True),
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
    client: MarketDataClient = Depends(get_market_client)
):
    """Get stocks similar to the specified stock"""
    try:
        instruments = await client.fetch_all_instruments()
        
        # Find target instrument
        target = None
        for inst in instruments:
            if inst.get("symbol", "").upper() == symbol.upper():
                target = inst
                break
        
        if not target:
            raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")
        
        # Find similar stocks
        similar = []
        for inst in instruments:
            if inst.get("symbol") == target.get("symbol"):
                continue
            
            score = 0
            if inst.get("sector") == target.get("sector"):
                score += 0.5
            if inst.get("market") == target.get("market"):
                score += 0.3
            if inst.get("currency") == target.get("currency"):
                score += 0.2
            
            if score > 0:
                similar.append({
                    "symbol": inst.get("symbol"),
                    "name": inst.get("name"),
                    "sector": inst.get("sector"),
                    "similarity_score": score
                })
        
        # Sort and limit
        similar.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        return {
            "symbol": symbol.upper(),
            "similar_stocks": similar[:limit]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@router.get("/debug/test-quotes")
async def test_quotes(
    symbols: str = Query("AAPL,NPN,SHP", description="Comma-separated symbols to test"),
    client: MarketDataClient = Depends(get_market_client)
):
    """Test quote fetching for specific symbols"""
    try:
        symbol_list = [s.strip() for s in symbols.split(",")]
        results = {}
        
        instruments = await client.fetch_all_instruments()
        
        for symbol in symbol_list:
            # Find the instrument
            instrument = None
            for inst in instruments:
                if inst.get("symbol", "").upper() == symbol.upper():
                    instrument = inst
                    break
            
            if instrument:
                yahoo_symbol = instrument.get("yahoo_symbol", instrument.get("symbol"))
                quote = await client.fetch_yahoo_quote(yahoo_symbol)
                results[symbol] = {
                    "instrument": instrument,
                    "yahoo_symbol": yahoo_symbol,
                    "quote": quote,
                    "has_price": quote.get("price") is not None
                }
            else:
                results[symbol] = {"error": "Instrument not found"}
        
        return results
        
    except Exception as e:
        return {"error": str(e)}

@router.get("/debug/cache-status")
async def cache_status(
    client: MarketDataClient = Depends(get_market_client)
):
    """Check cache status and statistics"""
    try:
        instruments = await client.fetch_all_instruments()
        
        # Count by market and sector
        market_counts = {}
        sector_counts = {}
        currency_counts = {}
        
        for inst in instruments:
            market = inst.get("market", "Unknown")
            sector = inst.get("sector", "Unknown")
            currency = inst.get("currency", "Unknown")
            
            market_counts[market] = market_counts.get(market, 0) + 1
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
            currency_counts[currency] = currency_counts.get(currency, 0) + 1
        
        return {
            "total_instruments": len(instruments),
            "instruments_cached": client._instruments_cache is not None,
            "cache_age_seconds": (datetime.now() - client._cache_timestamp).seconds if client._cache_timestamp else None,
            "price_cache_size": len(client._prices_cache),
            "markets": market_counts,
            "sectors": sector_counts,
            "currencies": currency_counts
        }
        
    except Exception as e:
        return {"error": str(e)}