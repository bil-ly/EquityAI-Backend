from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import os
import httpx
import logging
import yfinance as yf
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic models
class StockAnalysisRequest(BaseModel):
    symbol: str
    analysis_type: str = "general"  # general, dividend, risk, technical
    user_context: Optional[str] = None

class PortfolioAnalysisRequest(BaseModel):
    account_id: str
    focus: str = "optimization"  # optimization, diversification, risk, income

class AnalysisResponse(BaseModel):
    symbol: Optional[str]
    analysis_type: str
    recommendation: str  #TODO: Chnage this into ENUM{BUY, SELL, HOLD}
    confidence_score: float
    reasoning: str
    key_points: List[str]
    risks: List[str]
    opportunities: List[str]
    target_price: Optional[float] = None
    real_time_data: Optional[Dict] = None
    analysis_date: str

class MarketScanRequest(BaseModel):
    focus: str = "dividend_growth"  # dividend_growth, value, growth, momentum
    max_results: int = 10

# Claude API configuration
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"

class EnhancedStockAnalyzer:
    """Enhanced analyzer with real market data"""
    
    async def get_jse_stock_data(self, symbol: str) -> Dict:
        """Get real-time JSE stock data"""
        try:
            ticker = yf.Ticker(f"{symbol}.JO")
            
            info = ticker.info
            
            dividends = ticker.dividends
            
            current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
            
            currency = info.get('currency', 'ZAR')
            if currency == 'ZAc':
                current_price = current_price / 100 
                currency = 'ZAR'
            
            dividend_yield = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
            
            annual_dividend = 0
            if len(dividends) > 0:
                recent_dividends = dividends.last('12M')
                annual_dividend = recent_dividends.sum()
                
                if currency == 'ZAR' and annual_dividend > 1000:
                    annual_dividend = annual_dividend / 100
            
            if current_price > 0 and annual_dividend > 0:
                dividend_yield = (annual_dividend / current_price) * 100
            
            dividend_growth = self._calculate_dividend_growth(dividends, currency_converted=(currency == 'ZAR'))
            
            book_value = info.get('bookValue', 0)
            week_high = info.get('fiftyTwoWeekHigh', 0)
            week_low = info.get('fiftyTwoWeekLow', 0)
            eps = info.get('trailingEps', 0)
            
            if currency == 'ZAR':
                book_value = book_value / 100 if book_value > 1000 else book_value
                week_high = week_high / 100 if week_high > 1000 else week_high
                week_low = week_low / 100 if week_low > 1000 else week_low
                eps = eps / 100 if eps > 100 else eps
            
            stock_data = {
                "symbol": symbol,
                "company_name": info.get('longName', f"{symbol} Ltd"),
                "current_price": round(current_price, 2),
                "currency": currency,
                "dividend_yield": round(dividend_yield, 2),
                "annual_dividend": round(annual_dividend, 2),
                "dividend_growth_5yr": dividend_growth,
                "pe_ratio": round(info.get('trailingPE', 0), 1),
                "forward_pe": round(info.get('forwardPE', 0), 1),
                "market_cap": info.get('marketCap', 0),
                "sector": info.get('sector', 'Unknown'),
                "industry": info.get('industry', 'Unknown'),
                "book_value": round(book_value, 2),
                "debt_to_equity": round(info.get('debtToEquity', 0), 1),
                "roe": round(info.get('returnOnEquity', 0) * 100, 1) if info.get('returnOnEquity') else 0,
                "profit_margins": round(info.get('profitMargins', 0) * 100, 1) if info.get('profitMargins') else 0,
                "52_week_high": round(week_high, 2),
                "52_week_low": round(week_low, 2),
                "beta": round(info.get('beta', 1.0), 2),
                "volume": info.get('volume', 0),
                "avg_volume": info.get('averageVolume', 0),
                "payout_ratio": round(info.get('payoutRatio', 0) * 100, 1) if info.get('payoutRatio') else 0,
                "eps": round(eps, 2),
                "revenue_growth": round(info.get('revenueGrowth', 0) * 100, 1) if info.get('revenueGrowth') else 0,
                "data_source": f"Yahoo Finance ({currency})",
                "note": "Prices converted from cents to rands" if currency == 'ZAR' else None
            }
            
            logger.info(f"Retrieved real-time data for {symbol} (converted from {info.get('currency', 'unknown')} to {currency})")
            return stock_data
            
        except Exception as e:
            logger.error(f"Failed to get stock data for {symbol}: {e}")
            return {
                "symbol": symbol, 
                "error": f"Real-time data unavailable: {str(e)}",
                "company_name": f"{symbol} Ltd"
            }
    
    def _calculate_dividend_growth(self, dividends, currency_converted=False) -> Optional[float]:
        """Calculate dividend growth rate"""
        try:
            if len(dividends) < 5:
                return None
            
            annual_divs = dividends.resample('Y').sum().tail(5)
            
            if len(annual_divs) < 2:
                return None
            
            if currency_converted:
                annual_divs = annual_divs / 100
            
            start_div = annual_divs.iloc[0]
            end_div = annual_divs.iloc[-1]
            years = len(annual_divs) - 1
            
            if start_div <= 0:
                return None
                
            growth_rate = ((end_div / start_div) ** (1/years) - 1) * 100
            return round(growth_rate, 2)
            
        except Exception as e:
            logger.error(f"Dividend growth calculation failed: {e}")
            return None
    
    def create_enhanced_prompt(self, symbol: str, stock_data: Dict, analysis_type: str, user_context: str = None) -> str:
        """Create enhanced prompt with real market data"""
        
        if stock_data.get("error"):
            return f"""
            Analyze {symbol} (JSE-listed stock) based on your general knowledge.
            Note: Current market data is unavailable ({stock_data.get('error')}).
            Provide analysis based on historical knowledge of this company and sector.
            Context: {user_context or 'South African investor'}
            """
        
        market_cap = stock_data.get('market_cap', 0)
        market_cap_str = f"R{market_cap/1_000_000_000:.1f}B" if market_cap > 1_000_000_000 else f"R{market_cap/1_000_000:.0f}M"
        
        base_data = f"""
        REAL-TIME MARKET DATA for {symbol} ({stock_data.get('company_name')}):
        
        CURRENT VALUATION:
        - Stock Price: R{stock_data.get('current_price', 0):.2f}
        - Market Cap: {market_cap_str}
        - P/E Ratio: {stock_data.get('pe_ratio', 0):.1f}x
        - Forward P/E: {stock_data.get('forward_pe', 0):.1f}x
        - Book Value: R{stock_data.get('book_value', 0):.2f}
        - Price Range (52w): R{stock_data.get('52_week_low', 0):.2f} - R{stock_data.get('52_week_high', 0):.2f}
        
        DIVIDEND METRICS:
        - Current Dividend Yield: {stock_data.get('dividend_yield', 0):.2f}%
        - Annual Dividend: R{stock_data.get('annual_dividend', 0):.2f}
        - Dividend Growth (5yr): {stock_data.get('dividend_growth_5yr') or 'N/A'}%
        - Payout Ratio: {stock_data.get('payout_ratio', 0):.1f}%
        
        FINANCIAL HEALTH:
        - ROE: {stock_data.get('roe', 0):.1f}%
        - Profit Margin: {stock_data.get('profit_margins', 0):.1f}%
        - Debt/Equity: {stock_data.get('debt_to_equity', 0):.1f}
        - Revenue Growth: {stock_data.get('revenue_growth', 0):.1f}%
        - EPS: R{stock_data.get('eps', 0):.2f}
        
        MARKET DATA:
        - Sector: {stock_data.get('sector', 'Unknown')}
        - Beta: {stock_data.get('beta', 1.0):.2f}
        - Volume: {stock_data.get('volume', 0):,}
        """
        
        if analysis_type == "dividend":
            return f"""
            As a dividend investing expert, analyze {symbol} using this LIVE market data:

            {base_data}
            
            DIVIDEND ANALYSIS FOCUS:
            1. Dividend Sustainability: Analyze the {stock_data.get('payout_ratio', 0):.1f}% payout ratio and {stock_data.get('roe', 0):.1f}% ROE
            2. Yield Attractiveness: Compare {stock_data.get('dividend_yield', 0):.2f}% yield to SA 10-year bonds (~11%)
            3. Growth Prospects: Evaluate {stock_data.get('dividend_growth_5yr') or 'unknown'} historical growth rate
            4. Coverage Analysis: Assess sustainability with current profit margins of {stock_data.get('profit_margins', 0):.1f}%
            5. Sector Comparison: How does this yield compare to {stock_data.get('sector')} sector average?
            
            PROVIDE SPECIFIC ANALYSIS:
            - BUY/HOLD/SELL recommendation with confidence score (0-1)
            - Dividend sustainability score (1-10) based on current financials
            - Expected dividend per share for next 12 months
            - Key risks to dividend payments
            - Fair value price target using dividend discount model
            
            Context: {user_context or 'Income-focused South African investor'}
            
            Base your analysis on the LIVE data provided above - these are current market metrics, not estimates.
            """
            
        elif analysis_type == "risk":
            return f"""
            Perform a comprehensive risk analysis for {symbol} using current market data:

            {base_data}
            
            RISK ASSESSMENT AREAS:
            1. Financial Risk: Debt/Equity of {stock_data.get('debt_to_equity', 0):.1f} and profit margins
            2. Valuation Risk: P/E of {stock_data.get('pe_ratio', 0):.1f}x vs sector average
            3. Volatility Risk: Beta of {stock_data.get('beta', 1.0):.2f}
            4. Liquidity Risk: Trading volume analysis
            5. Sector Risk: {stock_data.get('sector')} specific risks in SA context
            6. Currency Risk: ZAR exposure and hedging
            
            PROVIDE:
            - Overall risk rating (LOW/MEDIUM/HIGH)
            - Top 3 specific risks with probability assessment
            - Risk-adjusted position sizing recommendation
            - Key metrics to monitor for early warning signs
            - Stress test scenarios (recession, rate hikes, currency crisis)
            
            Consider South African economic environment and emerging market risks.
            """
            
        else: 
            return f"""
            Provide a comprehensive investment analysis for {symbol}:

            {base_data}
            
            INVESTMENT ANALYSIS:
            1. Valuation: Is P/E of {stock_data.get('pe_ratio', 0):.1f}x attractive for this quality of business?
            2. Growth: Assess {stock_data.get('revenue_growth', 0):.1f}% revenue growth sustainability
            3. Quality: Evaluate {stock_data.get('roe', 0):.1f}% ROE and {stock_data.get('profit_margins', 0):.1f}% margins
            4. Dividend Policy: {stock_data.get('dividend_yield', 0):.2f}% yield with {stock_data.get('payout_ratio', 0):.1f}% payout ratio
            5. Market Position: Competitive advantages in {stock_data.get('sector')} sector
            
            PROVIDE:
            - Clear BUY/HOLD/SELL recommendation with confidence
            - 12-month price target with upside/downside scenarios
            - Key catalysts that could drive performance
            - Main risks and how to mitigate them
            - Suitability for different investor types
            
            Context: {user_context or 'Long-term South African investor'}
            """

analyzer = EnhancedStockAnalyzer()

async def call_claude_api(prompt: str, max_tokens: int = 1500) -> str:
    """Call Claude API for analysis"""
    if not CLAUDE_API_KEY:
        raise HTTPException(status_code=500, detail="Claude API key not configured")
    
    headers = {
        "x-api-key": CLAUDE_API_KEY,
        "content-type": "application/json",
        "anthropic-version": "2023-06-01"
    }
    
    model = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022")
    
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(CLAUDE_API_URL, headers=headers, json=payload, timeout=30.0)
            
            logger.info(f"Claude API Response Status: {response.status_code}")
            if response.status_code != 200:
                logger.error(f"Claude API Error Response: {response.text}")
            
            response.raise_for_status()
            result = response.json()
            return result["content"][0]["text"]
        except httpx.HTTPStatusError as e:
            logger.error(f"Claude API HTTP error: {e.response.status_code} - {e.response.text}")
            raise HTTPException(status_code=500, detail=f"AI analysis failed: {e.response.status_code} {e.response.reason_phrase}")
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            raise HTTPException(status_code=500, detail=f"AI analysis failed: {str(e)}")

@router.post("/stock", response_model=AnalysisResponse)
async def analyze_stock(request: StockAnalysisRequest):
    """
    Get AI analysis for a specific stock with real market data
    """
    try:
        stock_data = await analyzer.get_jse_stock_data(request.symbol)
        
        prompt = analyzer.create_enhanced_prompt(
            request.symbol, 
            stock_data, 
            request.analysis_type, 
            request.user_context
        )
        
        analysis_text = await call_claude_api(prompt)
        
        lines = analysis_text.split('\n')
        
        recommendation = "HOLD"
        confidence_score = 0.7
        
        for line in lines:
            line_upper = line.upper()
            if "BUY" in line_upper and "SELL" not in line_upper:
                recommendation = "BUY"
            elif "SELL" in line_upper:
                recommendation = "SELL"
            
            if "confidence" in line.lower():
                try:
                    import re
                    confidence_match = re.search(r'(\d+\.?\d*)%?', line)
                    if confidence_match:
                        confidence_val = float(confidence_match.group(1))
                        confidence_score = confidence_val / 100 if confidence_val > 1 else confidence_val
                except:
                    pass

        key_points = []
        risks = []
        opportunities = []
        
        for line in lines:
            line = line.strip()
            if any(line.startswith(prefix) for prefix in ['1.', '2.', '3.', '•', '-', '*']):
                if 'risk' in line.lower() or 'concern' in line.lower():
                    risks.append(line)
                elif 'opportunity' in line.lower() or 'growth' in line.lower() or 'catalyst' in line.lower():
                    opportunities.append(line)
                else:
                    key_points.append(line)

        target_price = None
        try:
            import re
            price_matches = re.findall(r'R(\d+\.?\d*)', analysis_text)
            if price_matches:
                prices = [float(p) for p in price_matches]
                target_price = max(prices) if max(prices) > stock_data.get('current_price', 0) else None
        except:
            pass

        logger.info(f"✅ Generated enhanced {request.analysis_type} analysis for {request.symbol} with real market data")

        return AnalysisResponse(
            symbol=request.symbol,
            analysis_type=request.analysis_type,
            recommendation=recommendation,
            confidence_score=confidence_score,
            reasoning=analysis_text[:800] + "..." if len(analysis_text) > 800 else analysis_text,
            key_points=key_points[:5],
            risks=risks[:3],
            opportunities=opportunities[:3],
            target_price=target_price,
            real_time_data=stock_data,
            analysis_date=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Enhanced stock analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.get("/stock-data/{symbol}")
async def get_stock_data(symbol: str):
    """
    Get real-time stock data without analysis
    """
    try:
        stock_data = await analyzer.get_jse_stock_data(symbol)
        return {
            "symbol": symbol,
            "data": stock_data,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get stock data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stock data: {str(e)}")

@router.post("/portfolio")
async def analyze_portfolio(request: PortfolioAnalysisRequest):
    """
    Analyze entire portfolio for optimization opportunities
    """
    try:
        from easy_equities_client.clients import EasyEquitiesClient
        
        username = os.getenv("EASYEQUITIES_USERNAME")
        password = os.getenv("EASYEQUITIES_PASSWORD")
        
        if not username or not password:
            raise HTTPException(status_code=500, detail="EasyEquities credentials not configured")
        
        client = EasyEquitiesClient()
        client.login(username=username, password=password)
        
        holdings = client.accounts.holdings(request.account_id, include_shares=True)
        
        if not holdings:
            return {
                "message": "No holdings found for portfolio analysis",
                "recommendations": []
            }
        holdings_summary = "\n".join([
            f"- {holding.get('name', '')}: {holding.get('current_value', 'R0')}"
            for holding in holdings[:10]
        ])
        
        prompt = f"""
        As a portfolio analyst, analyze this South African investment portfolio for {request.focus}:

        Current Holdings:
        {holdings_summary}

        Portfolio Size: {len(holdings)} holdings
        Focus: {request.focus}

        Provide:
        1. Overall portfolio assessment
        2. Diversification analysis (sectors, market caps, geographies)
        3. Risk concentration analysis
        4. Income generation potential
        5. Specific recommendations for improvement
        6. Suggested position adjustments or new investments

        Consider South African market conditions, JSE sectors, and rand hedging opportunities.
        Be specific and actionable.
        """
        
        analysis = await call_claude_api(prompt, max_tokens=2000)
        
        logger.info(f"Generated portfolio analysis for account {request.account_id}")
        
        return {
            "account_id": request.account_id,
            "analysis_type": "portfolio_" + request.focus,
            "analysis": analysis,
            "holdings_analyzed": len(holdings),
            "focus": request.focus,
            "analysis_date": "2024-08-07T10:00:00Z"
        }
        
    except Exception as e:
        logger.error(f"Portfolio analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Portfolio analysis failed: {str(e)}")

@router.post("/market-scan")
async def scan_market_opportunities(request: MarketScanRequest):
    """
    Scan JSE market for investment opportunities
    """
    try:
        prompt = f"""
        As a market analyst, scan the JSE (Johannesburg Stock Exchange) for the best {request.focus} opportunities:

        Focus: {request.focus}
        Target: Top {request.max_results} recommendations

        Analyze based on:
        1. Current market conditions in South Africa
        2. Sector rotation trends
        3. Valuation metrics and technical indicators
        4. Dividend sustainability and growth (if relevant)
        5. Currency and emerging market factors

        For each recommendation, provide:
        - Stock symbol and name
        - Key investment thesis (2-3 sentences)
        - Entry price range
        - Main risk factors
        - Expected timeframe for returns

        Focus on liquid, reputable JSE companies suitable for individual investors.
        Consider both large caps and quality mid caps.
        """
        
        analysis = await call_claude_api(prompt, max_tokens=2000)
        
        logger.info(f"Generated market scan for {request.focus}")
        
        return {
            "scan_type": request.focus,
            "market": "JSE",
            "analysis": analysis,
            "max_results": request.max_results,
            "scan_date": "2024-08-07T10:00:00Z"
        }
        
    except Exception as e:
        logger.error(f"Market scan failed: {e}")
        raise HTTPException(status_code=500, detail=f"Market scan failed: {str(e)}")

@router.get("/test-claude")
async def test_claude_connection():
    """
    Test Claude API connection
    """
    try:
        test_prompt = "Briefly explain why diversification is important in investing. Keep it under 100 words."
        
        response = await call_claude_api(test_prompt, max_tokens=200)
        
        return {
            "status": "success",
            "message": "Claude API connection working",
            "test_response": response
        }
        
    except Exception as e:
        logger.error(f"Claude test failed: {e}")
        raise HTTPException(status_code=500, detail=f"Claude test failed: {str(e)}")