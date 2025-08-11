from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import os
import httpx
import logging
import yfinance as yf
from datetime import datetime
import asyncio

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

class BeginnerInvestmentRequest(BaseModel):
    deposit_amount: float
    risk_tolerance: str = "moderate"  # conservative, moderate, aggressive
    investment_goal: str = "growth"   # growth, income, balanced
    time_horizon: str = "long_term"   # short_term (1-2y), medium_term (3-5y), long_term (5y+)
    user_context: Optional[str] = None

class StockRecommendation(BaseModel):
    symbol: str
    company_name: str
    allocation_percentage: float
    rand_amount: float
    shares_to_buy: int
    current_price: float
    rationale: str
    risk_level: str
    expected_dividend_yield: Optional[float] = None
    sector: Optional[str] = None

class BeginnerPortfolioResponse(BaseModel):
    total_investment: float
    cash_reserve: float
    total_allocated: float
    recommendations: List[StockRecommendation]
    portfolio_strategy: str
    risk_assessment: str
    key_principles: List[str]
    next_steps: List[str]
    estimated_annual_yield: Optional[float] = None
    diversification_score: str

class DynamicStockScreener:
    """Dynamically screen JSE stocks based on real market data"""
    
    def __init__(self):
        # Common JSE symbols to screen - this is just a starting point for data gathering
        self.jse_symbols = [
            # Major banks
            "SBK", "FSR", "NED", "ABG", "CPI",
            # Retailers  
            "SHP", "PIK", "WHL", "TRU", "MRP",
            # Telecoms
            "MTN", "VOD", "TEL",
            # Mining
            "AGL", "BIL", "AMS", "GFI", "HAR", "IMP",
            # REITs
            "GRT", "RDF", "HYP", "ATT", "FFB",
            # Industrials
            "NPN", "PRX", "SAP", "APN", "BVT", "RMH",
            # Insurance
            "SLM", "OML", "MMI", "DSY",
            # Consumer
            "ABI", "CFR", "RFG", "TBS", "AVI",
            # Healthcare
            "NTC", "ADH", "LHC",
            # Technology
            "EOH", "ALT", "OCE"
        ]
    
    async def screen_stocks_for_beginners(self, risk_tolerance: str, investment_goal: str, max_stocks: int = 15) -> List[Dict]:
        """Screen stocks dynamically based on real market data"""
        
        logger.info(f"Screening {len(self.jse_symbols)} JSE stocks for beginner portfolio...")
        
        # Get real-time data for all stocks
        stock_data_tasks = []
        for symbol in self.jse_symbols:
            stock_data_tasks.append(analyzer.get_jse_stock_data(symbol))
        
        # Execute all API calls concurrently for speed
        stock_data_results = await asyncio.gather(*stock_data_tasks, return_exceptions=True)
        
        valid_stocks = []
        for i, result in enumerate(stock_data_results):
            if isinstance(result, dict) and not result.get("error"):
                symbol = self.jse_symbols[i]
                data = result
                
                # Basic quality filters
                if (data.get("current_price", 0) > 0 and 
                    data.get("market_cap", 0) > 1_000_000_000 and  # R1B+ market cap
                    data.get("volume", 0) > 10000):  # Reasonable liquidity
                    
                    # Calculate quality score
                    quality_score = self.calculate_quality_score(data, risk_tolerance, investment_goal)
                    data["quality_score"] = quality_score
                    data["symbol"] = symbol
                    valid_stocks.append(data)
        
        # Sort by quality score and return top stocks
        valid_stocks.sort(key=lambda x: x["quality_score"], reverse=True)
        
        logger.info(f"Found {len(valid_stocks)} suitable stocks after screening")
        return valid_stocks[:max_stocks]
    
    def calculate_quality_score(self, stock_data: Dict, risk_tolerance: str, investment_goal: str) -> float:
        """Calculate a quality score for the stock based on multiple factors"""
        
        score = 0.0
        
        # Dividend yield scoring (0-25 points)
        div_yield = stock_data.get("dividend_yield", 0)
        if div_yield > 8:
            score += 25
        elif div_yield > 5:
            score += 20
        elif div_yield > 3:
            score += 15
        elif div_yield > 1:
            score += 10
        
        # P/E ratio scoring (0-20 points) - prefer reasonable valuations
        pe_ratio = stock_data.get("pe_ratio", 0)
        if 8 <= pe_ratio <= 18:
            score += 20
        elif 5 <= pe_ratio <= 25:
            score += 15
        elif pe_ratio > 0:
            score += 5
        
        # ROE scoring (0-20 points)
        roe = stock_data.get("roe", 0)
        if roe > 20:
            score += 20
        elif roe > 15:
            score += 15
        elif roe > 10:
            score += 10
        elif roe > 5:
            score += 5
        
        # Debt/Equity scoring (0-15 points) - prefer lower debt
        debt_equity = stock_data.get("debt_to_equity", 0)
        if debt_equity < 30:
            score += 15
        elif debt_equity < 50:
            score += 10
        elif debt_equity < 100:
            score += 5
        
        # Market cap scoring (0-10 points) - prefer larger, stable companies for beginners
        market_cap = stock_data.get("market_cap", 0)
        if market_cap > 50_000_000_000:  # R50B+
            score += 10
        elif market_cap > 10_000_000_000:  # R10B+
            score += 8
        elif market_cap > 5_000_000_000:   # R5B+
            score += 6
        
        # Adjust for risk tolerance
        if risk_tolerance == "conservative":
            # Bonus for stable sectors and lower volatility
            if stock_data.get("beta", 1.0) < 0.8:
                score += 10
            if stock_data.get("sector") in ["Utilities", "Consumer Staples", "Healthcare"]:
                score += 5
        elif risk_tolerance == "aggressive":
            # Bonus for growth metrics
            revenue_growth = stock_data.get("revenue_growth", 0)
            if revenue_growth > 15:
                score += 10
            elif revenue_growth > 5:
                score += 5
        
        # Adjust for investment goal
        if investment_goal == "income":
            # Extra weight on dividend yield
            score += min(div_yield * 2, 15)
        elif investment_goal == "growth":
            # Extra weight on revenue growth
            revenue_growth = stock_data.get("revenue_growth", 0)
            score += min(revenue_growth, 15)
        
        return score

screener = DynamicStockScreener()

@router.post("/beginner-portfolio", response_model=BeginnerPortfolioResponse)
async def create_beginner_portfolio(request: BeginnerInvestmentRequest):
    """
    Create a beginner-friendly portfolio recommendation based on deposit amount and real market data
    """
    try:
        # Validate minimum investment
        if request.deposit_amount < 100:
            raise HTTPException(
                status_code=400, 
                detail="Minimum investment amount is R100"
            )
        
        # Reserve cash for brokerage fees (EasyEquities charges ~R7-20 per transaction)
        # Reserve 3-5% for fees, minimum R50, maximum R200
        brokerage_reserve = max(50, min(request.deposit_amount * 0.04, 200))
        investable_amount = request.deposit_amount - brokerage_reserve
        
        logger.info(f"Creating beginner portfolio for R{request.deposit_amount} (R{investable_amount} investable)")
        
        # Screen stocks dynamically
        suitable_stocks = await screener.screen_stocks_for_beginners(
            request.risk_tolerance,
            request.investment_goal,
            max_stocks=12
        )
        
        if not suitable_stocks:
            raise HTTPException(
                status_code=500,
                detail="Could not find suitable stocks in current market conditions"
            )
        
        # Create AI prompt for intelligent portfolio allocation
        stocks_summary = "\n".join([
            f"- {stock['symbol']} ({stock.get('company_name', 'Unknown')}): "
            f"Price R{stock.get('current_price', 0):.2f}, "
            f"Div Yield {stock.get('dividend_yield', 0):.1f}%, "
            f"P/E {stock.get('pe_ratio', 0):.1f}, "
            f"ROE {stock.get('roe', 0):.1f}%, "
            f"Sector: {stock.get('sector', 'Unknown')}, "
            f"Quality Score: {stock.get('quality_score', 0):.1f}/100"
            for stock in suitable_stocks[:8]
        ])
        
        allocation_prompt = f"""
        You are an expert investment advisor creating a beginner portfolio for a South African investor.

        INVESTOR PROFILE:
        - Total Deposit: R{request.deposit_amount:.2f}
        - Available for Investment: R{investable_amount:.2f} (after R{brokerage_reserve:.2f} for fees)
        - Risk Tolerance: {request.risk_tolerance}
        - Investment Goal: {request.investment_goal}
        - Time Horizon: {request.time_horizon}
        - Experience Level: Complete beginner

        TOP QUALITY JSE STOCKS (Real Market Data):
        {stocks_summary}

        PORTFOLIO REQUIREMENTS:
        1. Select 3-5 stocks maximum for simplicity and diversification
        2. No single stock should exceed 35% of portfolio (concentration risk)
        3. Minimum position size should be at least R100 (meaningful allocation)
        4. Must be able to buy whole shares only
        5. Prioritize different sectors for diversification
        6. Consider dividend income potential for {request.investment_goal} goal

        For EACH recommended stock, provide:
        - Symbol and company name
        - Exact percentage allocation (total must be 95-100%)
        - Specific Rand amount to invest
        - Number of whole shares to purchase
        - Investment rationale (2-3 sentences)
        - Risk level (Low/Moderate/High)

        Also provide:
        - Overall portfolio strategy (3-4 sentences)
        - 5 key investment principles for beginners
        - 5 practical next steps after investing

        Format your response clearly with sections for each stock recommendation.
        Be specific with numbers - this person will use your exact recommendations to invest.
        """
        
        # Get AI allocation recommendation
        allocation_analysis = await call_claude_api(allocation_prompt, max_tokens=2500)
        
        # Parse the AI response to extract structured recommendations
        recommendations = parse_ai_allocation_response(
            allocation_analysis,
            suitable_stocks,
            investable_amount
        )
        
        # If parsing failed, create a fallback allocation
        if not recommendations:
            recommendations = create_intelligent_fallback_allocation(
                suitable_stocks,
                investable_amount,
                request.risk_tolerance,
                request.investment_goal
            )
        
        # Calculate portfolio metrics
        total_allocated = sum(rec.rand_amount for rec in recommendations)
        cash_remaining = investable_amount - total_allocated
        
        # Calculate estimated annual dividend yield
        estimated_yield = 0
        if total_allocated > 0:
            weighted_yield = sum(
                rec.rand_amount * (rec.expected_dividend_yield or 0) / 100
                for rec in recommendations
            )
            estimated_yield = (weighted_yield / total_allocated) * 100
        
        # Calculate diversification score
        sectors = set(rec.sector for rec in recommendations if rec.sector)
        diversification_score = f"Good ({len(sectors)} sectors)" if len(sectors) >= 3 else f"Moderate ({len(sectors)} sectors)"
        
        # Extract guidance from AI response
        strategy, principles, next_steps = extract_guidance_from_ai_response(allocation_analysis)
        
        logger.info(f"✅ Created beginner portfolio: {len(recommendations)} stocks, R{total_allocated:.2f} allocated")
        
        return BeginnerPortfolioResponse(
            total_investment=request.deposit_amount,
            cash_reserve=brokerage_reserve + cash_remaining,
            total_allocated=total_allocated,
            recommendations=recommendations,
            portfolio_strategy=strategy,
            risk_assessment=f"{request.risk_tolerance.title()} risk portfolio optimized for {request.investment_goal}",
            key_principles=principles,
            next_steps=next_steps,
            estimated_annual_yield=estimated_yield if estimated_yield > 0 else None,
            diversification_score=diversification_score
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Beginner portfolio creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Portfolio creation failed: {str(e)}")

def parse_ai_allocation_response(ai_response: str, stock_data_list: List[Dict], investable_amount: float) -> List[StockRecommendation]:
    """Parse AI response to extract specific stock recommendations"""
    
    recommendations = []
    stock_data_map = {stock['symbol']: stock for stock in stock_data_list}
    
    lines = ai_response.split('\n')
    
    for line in lines:
        line = line.strip()
        
        # Look for lines containing stock symbols and allocation info
        for symbol in stock_data_map.keys():
            if symbol in line and any(char in line for char in ['R', '%', 'rand', 'Rand']):
                try:
                    import re
                    
                    # Extract percentage
                    percentage_match = re.search(r'(\d+(?:\.\d+)?)%', line)
                    percentage = float(percentage_match.group(1)) if percentage_match else 0
                    
                    # Extract Rand amount - handle various formats
                    amount_patterns = [
                        r'R\s*(\d+(?:[,\s]\d+)*(?:\.\d+)?)',
                        r'(\d+(?:[,\s]\d+)*(?:\.\d+)?)\s*rand',
                        r'invest\s+R?(\d+(?:[,\s]\d+)*(?:\.\d+)?)'
                    ]
                    
                    rand_amount = 0
                    for pattern in amount_patterns:
                        amount_match = re.search(pattern, line, re.IGNORECASE)
                        if amount_match:
                            amount_str = amount_match.group(1).replace(',', '').replace(' ', '')
                            rand_amount = float(amount_str)
                            break
                    
                    # If no amount found but percentage exists, calculate from percentage
                    if rand_amount == 0 and percentage > 0:
                        rand_amount = (percentage / 100) * investable_amount
                    
                    stock_data = stock_data_map[symbol]
                    current_price = stock_data.get('current_price', 0)
                    
                    if current_price > 0 and rand_amount >= 50:  # Minimum R50 position
                        shares_to_buy = int(rand_amount / current_price)
                        if shares_to_buy > 0:
                            actual_amount = shares_to_buy * current_price
                            actual_percentage = (actual_amount / investable_amount) * 100
                            
                            # Extract rationale from nearby lines
                            rationale = extract_rationale_for_stock(ai_response, symbol)
                            
                            # Determine risk level based on stock characteristics
                            risk_level = determine_risk_level(stock_data)
                            
                            recommendations.append(StockRecommendation(
                                symbol=symbol,
                                company_name=stock_data.get('company_name', f"{symbol} Ltd"),
                                allocation_percentage=actual_percentage,
                                rand_amount=actual_amount,
                                shares_to_buy=shares_to_buy,
                                current_price=current_price,
                                rationale=rationale,
                                risk_level=risk_level,
                                expected_dividend_yield=stock_data.get('dividend_yield', 0),
                                sector=stock_data.get('sector', 'Unknown')
                            ))
                            break
                            
                except Exception as e:
                    logger.warning(f"Could not parse recommendation for {symbol}: {e}")
                    continue
    
    # Remove duplicates and sort by allocation
    seen_symbols = set()
    unique_recommendations = []
    for rec in recommendations:
        if rec.symbol not in seen_symbols:
            unique_recommendations.append(rec)
            seen_symbols.add(rec.symbol)
    
    return sorted(unique_recommendations, key=lambda x: x.allocation_percentage, reverse=True)[:5]

def create_intelligent_fallback_allocation(stock_data_list: List[Dict], investable_amount: float, 
                                         risk_tolerance: str, investment_goal: str) -> List[StockRecommendation]:
    """Create intelligent fallback allocation if AI parsing fails"""
    
    logger.info("Creating fallback allocation using rule-based approach")
    
    # Sort stocks by quality score
    sorted_stocks = sorted(stock_data_list, key=lambda x: x.get('quality_score', 0), reverse=True)
    
    recommendations = []
    sectors_used = set()
    remaining_amount = investable_amount
    
    # Target 3-4 stocks for simplicity
    target_stocks = 4 if investable_amount > 2000 else 3
    
    for i, stock_data in enumerate(sorted_stocks):
        if len(recommendations) >= target_stocks or remaining_amount < 100:
            break
        
        sector = stock_data.get('sector', 'Unknown')
        current_price = stock_data.get('current_price', 0)
        
        # Skip if same sector already represented (unless it's a much better stock)
        if sector in sectors_used and i > 0:
            continue
        
        if current_price > 0 and current_price <= remaining_amount:
            # Calculate allocation based on position in ranking and remaining amount
            if i == 0:  # Top stock gets larger allocation
                allocation_percent = 40 if risk_tolerance == "conservative" else 35
            else:
                allocation_percent = 25
            
            target_amount = min(
                (allocation_percent / 100) * investable_amount,
                remaining_amount * 0.6  # Don't use more than 60% of remaining on one stock
            )
            
            shares_to_buy = int(target_amount / current_price)
            if shares_to_buy > 0:
                actual_amount = shares_to_buy * current_price
                actual_percentage = (actual_amount / investable_amount) * 100
                
                recommendations.append(StockRecommendation(
                    symbol=stock_data['symbol'],
                    company_name=stock_data.get('company_name', f"{stock_data['symbol']} Ltd"),
                    allocation_percentage=actual_percentage,
                    rand_amount=actual_amount,
                    shares_to_buy=shares_to_buy,
                    current_price=current_price,
                    rationale=f"High-quality {sector} stock with {stock_data.get('dividend_yield', 0):.1f}% dividend yield",
                    risk_level=determine_risk_level(stock_data),
                    expected_dividend_yield=stock_data.get('dividend_yield', 0),
                    sector=sector
                ))
                
                sectors_used.add(sector)
                remaining_amount -= actual_amount
    
    return recommendations

def extract_rationale_for_stock(ai_response: str, symbol: str) -> str:
    """Extract investment rationale for a specific stock from AI response"""
    
    lines = ai_response.split('\n')
    rationale_lines = []
    
    # Look for lines near the symbol mention
    for i, line in enumerate(lines):
        if symbol in line:
            # Check surrounding lines for rationale
            for j in range(max(0, i-2), min(len(lines), i+4)):
                check_line = lines[j].strip()
                if (len(check_line) > 20 and 
                    any(word in check_line.lower() for word in ['because', 'due to', 'offers', 'strong', 'stable', 'growth', 'dividend'])):
                    rationale_lines.append(check_line)
    
    if rationale_lines:
        return '. '.join(rationale_lines[:2])  # Max 2 sentences
    else:
        return f"Selected for portfolio diversification and dividend income potential"

def determine_risk_level(stock_data: Dict) -> str:
    """Determine risk level based on stock characteristics"""
    
    beta = stock_data.get('beta', 1.0)
    debt_equity = stock_data.get('debt_to_equity', 0)
    sector = stock_data.get('sector', '').lower()
    
    risk_score = 0
    
    # Beta scoring
    if beta > 1.3:
        risk_score += 2
    elif beta > 1.1:
        risk_score += 1
    
    # Debt scoring
    if debt_equity > 100:
        risk_score += 2
    elif debt_equity > 50:
        risk_score += 1
    
    # Sector scoring
    if any(sector_name in sector for sector_name in ['mining', 'technology', 'biotech']):
        risk_score += 2
    elif any(sector_name in sector for sector_name in ['banking', 'retail', 'industrial']):
        risk_score += 1
    
    if risk_score >= 4:
        return "High"
    elif risk_score >= 2:
        return "Moderate"
    else:
        return "Low"

def extract_guidance_from_ai_response(ai_response: str) -> tuple:
    """Extract strategy, principles and next steps from AI response"""
    
    # Default fallbacks
    strategy = "Diversified beginner portfolio focusing on quality JSE companies with sustainable dividends and growth potential."
    
    principles = [
        "Invest only in companies you understand",
        "Diversify across different sectors",
        "Focus on dividend-paying stocks for passive income",
        "Reinvest dividends for compound growth",
        "Review portfolio quarterly, not daily"
    ]
    
    next_steps = [
        "Set up automatic dividend reinvestment",
        "Learn about each company you own",
        "Add R200-500 monthly if possible",
        "Monitor quarterly results, not daily prices",
        "Consider tax-free savings account for future investments"
    ]
    
    # Try to extract better content from AI response
    lines = ai_response.split('\n')
    current_section = None
    extracted_principles = []
    extracted_steps = []
    
    for line in lines:
        line_clean = line.strip()
        line_lower = line_clean.lower()
        
        # Identify sections
        if 'strategy' in line_lower and len(line_clean) > 30:
            strategy = line_clean
        elif any(word in line_lower for word in ['principle', 'key point', 'important']):
            current_section = 'principles'
        elif any(word in line_lower for word in ['next step', 'action', 'should']):
            current_section = 'next_steps'
        elif line_clean.startswith(('-', '•', '*', '1.', '2.', '3.', '4.', '5.')):
            content = line_clean.lstrip('-•*1234567890. ')
            if current_section == 'principles' and len(content) > 10:
                extracted_principles.append(content)
            elif current_section == 'next_steps' and len(content) > 10:
                extracted_steps.append(content)
    
    # Use extracted content if found
    if extracted_principles:
        principles = extracted_principles[:5]
    if extracted_steps:
        next_steps = extracted_steps[:5]
    
    return strategy, principles, next_steps

@router.get("/market-overview")
async def get_market_overview_for_beginners():
    """Get current market overview suitable for beginner investors"""
    
    try:
        # Screen a subset of stocks to get market overview
        market_stocks = await screener.screen_stocks_for_beginners("moderate", "balanced", max_stocks=20)
        
        if not market_stocks:
            return {"message": "Unable to retrieve market data at this time"}
        
        # Calculate market metrics
        avg_dividend_yield = sum(stock.get('dividend_yield', 0) for stock in market_stocks) / len(market_stocks)
        avg_pe_ratio = sum(stock.get('pe_ratio', 0) for stock in market_stocks if stock.get('pe_ratio', 0) > 0) / max(1, len([s for s in market_stocks if s.get('pe_ratio', 0) > 0]))
        
        # Categorize by sectors
        sector_counts = {}
        for stock in market_stocks:
            sector = stock.get('sector', 'Unknown')
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        
        top_opportunities = market_stocks[:5]
        
        return {
            "market_summary": {
                "total_stocks_analyzed": len(market_stocks),
                "average_dividend_yield": round(avg_dividend_yield, 2),
                "average_pe_ratio": round(avg_pe_ratio, 1),
                "sectors_available": len(sector_counts),
                "sector_distribution": sector_counts
            },
            "top_beginner_opportunities": [
                {
                    "symbol": stock['symbol'],
                    "name": stock.get('company_name', 'Unknown'),
                    "price": stock.get('current_price', 0),
                    "dividend_yield": stock.get('dividend_yield', 0),
                    "sector": stock.get('sector', 'Unknown'),
                    "quality_score": round(stock.get('quality_score', 0), 1)
                }
                for stock in top_opportunities
            ],
            "market_note": "Data refreshed in real-time from JSE. Suitable for beginner investors.",
            "timestamp": "2024-08-07T10:00:00Z"
        }
        
    except Exception as e:
        logger.error(f"Market overview failed: {e}")
        raise HTTPException(status_code=500, detail=f"Market overview failed: {str(e)}")

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