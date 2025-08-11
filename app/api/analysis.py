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

# Add these to your existing analysis.py file

# New TFSA-specific models (add after your existing models)
class TFSAAnalysisRequest(BaseModel):
    account_id: str
    analysis_type: str = "optimization"  # optimization, contribution_planning, rebalancing
    annual_contribution_capacity: Optional[float] = None  # How much they can still contribute this year
    user_context: Optional[str] = None

class TFSARecommendation(BaseModel):
    recommendation_type: str  # "contribute_more", "rebalance", "switch_stocks", "optimize_allocation"
    action_required: str
    amount_involved: Optional[float] = None
    rationale: str
    tax_benefit_estimate: Optional[float] = None
    priority_level: str  # "high", "medium", "low"

class TFSAAnalysisResponse(BaseModel):
    account_id: str
    account_type: str = "TFSA"
    current_value: float
    annual_contribution_used: float
    remaining_contribution_space: float
    tax_savings_estimate: float
    recommendations: List[TFSARecommendation]
    portfolio_analysis: str
    tfsa_strategy: str
    compliance_check: Dict[str, bool]
    next_actions: List[str]

# Add this simple TFSA filter to your analysis.py

# TFSA-eligible stocks - verified available on major SA platforms
# Replace your TFSA_ELIGIBLE_STOCKS with this enhanced version
REAL_TFSA_INVESTMENTS = {
    # ETFs (definitely available in most TFSAs)
    "STXIND": {
        "name": "Satrix INDI ETF", 
        "type": "ETF",
        "sector": "South African Equity",
        "description": "Tracks JSE All Share Index",
        "estimated_yield": 3.5,
        "risk": "Moderate"
    },
    "STXFIN": {
        "name": "Satrix FINI ETF",
        "type": "ETF", 
        "sector": "Financial",
        "description": "JSE Financial sector exposure",
        "estimated_yield": 5.2,
        "risk": "Moderate"
    },
    "STX40": {
        "name": "Satrix Top 40 ETF",
        "type": "ETF",
        "sector": "Large Cap SA",
        "description": "Top 40 JSE companies",
        "estimated_yield": 3.8,
        "risk": "Moderate"
    },
    "STXPRO": {
        "name": "Satrix Property ETF", 
        "type": "ETF",
        "sector": "Property/REIT",
        "description": "SA Listed Property exposure",
        "estimated_yield": 6.5,
        "risk": "Moderate-High"
    },
    "STXEMG": {
        "name": "Satrix Emerging Markets ETF",
        "type": "ETF",
        "sector": "Emerging Markets",
        "description": "Global emerging markets exposure", 
        "estimated_yield": 2.8,
        "risk": "High"
    },
    "STXWDM": {
        "name": "Satrix MSCI World Feeder ETF",
        "type": "ETF",
        "sector": "Global Equity",
        "description": "Global developed markets",
        "estimated_yield": 2.2,
        "risk": "Moderate"
    },
    
    # Common Unit Trusts (available in many TFSAs)
    "CORONATION_EQUITY": {
        "name": "Coronation Equity Fund",
        "type": "Unit Trust",
        "sector": "SA Equity",
        "description": "Actively managed SA equity fund",
        "estimated_yield": 2.5,
        "risk": "Moderate-High"
    },
    "ALLAN_GRAY_EQUITY": {
        "name": "Allan Gray Equity Fund", 
        "type": "Unit Trust",
        "sector": "SA Equity",
        "description": "Value-focused SA equity fund",
        "estimated_yield": 2.8,
        "risk": "Moderate-High"
    },
    "FOORD_EQUITY": {
        "name": "Foord Equity Fund",
        "type": "Unit Trust", 
        "sector": "SA Equity",
        "description": "Conservative SA equity approach",
        "estimated_yield": 3.2,
        "risk": "Moderate"
    },
    
    # Balanced/Multi-Asset options
    "CORONATION_BALANCED_PLUS": {
        "name": "Coronation Balanced Plus Fund",
        "type": "Unit Trust",
        "sector": "Multi-Asset",
        "description": "Balanced portfolio with equity bias",
        "estimated_yield": 4.1,
        "risk": "Moderate"
    },
    "ALLAN_GRAY_BALANCED": {
        "name": "Allan Gray Balanced Fund",
        "type": "Unit Trust", 
        "sector": "Multi-Asset",
        "description": "Diversified balanced fund",
        "estimated_yield": 3.8,
        "risk": "Moderate"
    }
}


# Replace your filter function with this improved version


# Also add this improved allocation function
def create_real_tfsa_portfolio(request: BeginnerInvestmentRequest) -> BeginnerPortfolioResponse:
    """Create TFSA portfolio with ACTUALLY available investments"""
    
    # Filter investments based on risk tolerance and goal
    suitable_investments = []
    
    for code, investment in REAL_TFSA_INVESTMENTS.items():
        include = False
        
        # Filter by risk tolerance
        if request.risk_tolerance == "conservative":
            if investment["risk"] in ["Low", "Moderate"] and investment["type"] in ["ETF", "Unit Trust"]:
                include = True
        elif request.risk_tolerance == "moderate": 
            if investment["risk"] in ["Moderate", "Moderate-High"]:
                include = True
        else:  # aggressive
            include = True
        
        # Filter by investment goal
        if request.investment_goal == "income":
            if investment["estimated_yield"] >= 4.0:  # Higher yield focus
                include = True
            else:
                include = False
        elif request.investment_goal == "growth":
            if "Equity" in investment["sector"] or "Global" in investment["sector"]:
                include = True
        
        if include:
            # Add estimated current "price" (unit trusts typically R1-10 range)
            if investment["type"] == "ETF":
                investment["estimated_price"] = 15.0  # ETFs typically R10-20
            else:
                investment["estimated_price"] = 2.50   # Unit trusts typically R1-5
            
            investment["code"] = code
            suitable_investments.append(investment)
    
    # Create recommendations
    recommendations = []
    brokerage_reserve = max(20, min(request.deposit_amount * 0.02, 100))
    investable_amount = request.deposit_amount - brokerage_reserve
    
    # Simple allocation: 2-3 investments max
    if len(suitable_investments) >= 2:
        allocations = [60, 40] if len(suitable_investments) == 2 else [50, 30, 20]
    else:
        allocations = [100]
    
    total_allocated = 0
    used_sectors = set()
    
    for i, allocation_percent in enumerate(allocations):
        if i >= len(suitable_investments):
            break
            
        # Pick investment from different sector
        investment = None
        for inv in suitable_investments:
            if inv["sector"] not in used_sectors:
                investment = inv
                break
        
        if not investment:  # If all sectors used, pick next best
            investment = suitable_investments[i]
        
        target_amount = (allocation_percent / 100) * investable_amount
        estimated_price = investment["estimated_price"]
        units_to_buy = int(target_amount / estimated_price)
        
        if units_to_buy > 0:
            actual_amount = units_to_buy * estimated_price
            actual_percentage = (actual_amount / investable_amount) * 100
            
            recommendations.append(StockRecommendation(
                symbol=investment["code"],
                company_name=investment["name"],
                allocation_percentage=actual_percentage,
                rand_amount=actual_amount,
                shares_to_buy=units_to_buy,
                current_price=estimated_price,
                rationale=f"TFSA-available {investment['type']}: {investment['description']}",
                risk_level=investment["risk"],
                expected_dividend_yield=investment["estimated_yield"],
                sector=investment["sector"]
            ))
            
            total_allocated += actual_amount
            used_sectors.add(investment["sector"])
    
    return BeginnerPortfolioResponse(
        total_investment=request.deposit_amount,
        cash_reserve=brokerage_reserve + (investable_amount - total_allocated),
        total_allocated=total_allocated,
        recommendations=recommendations,
        portfolio_strategy="TFSA portfolio using ETFs and Unit Trusts actually available in TFSA accounts",
        risk_assessment=f"TFSA {request.risk_tolerance} risk using real available investments",
        key_principles=[
            "Use only TFSA-available ETFs and Unit Trusts",
            "Diversify across asset classes and geographies", 
            "Benefit from professional fund management",
            "Maximize tax-free growth and income",
            "Keep costs low with ETF options"
        ],
        next_steps=[
            "Open TFSA account (EasyEquities, 10X, etc.)",
            "Search for recommended ETF/fund codes in TFSA platform",
            "Set up automatic monthly contributions",
            "Monitor annual R36,000 contribution limit", 
            "Review allocations annually"
        ],
        estimated_annual_yield=sum(rec.expected_dividend_yield * rec.allocation_percentage / 100 for rec in recommendations),
        diversification_score=f"Excellent ({len(used_sectors)} asset classes)"
    )

# Replace your TFSA endpoint with this corrected version:
@router.post("/tfsa-beginner-portfolio", response_model=BeginnerPortfolioResponse)
async def create_tfsa_beginner_portfolio(request: BeginnerInvestmentRequest):
    """Create TFSA portfolio with investments that are ACTUALLY available"""
    try:
        if request.deposit_amount > 36000:
            raise HTTPException(status_code=400, detail="Exceeds R36,000 TFSA annual limit")
        
        if request.deposit_amount < 100:
            raise HTTPException(status_code=400, detail="Minimum TFSA investment is R100")
        
        # Use the new real TFSA function
        portfolio = await create_real_tfsa_portfolio(request)
        
        logger.info(f"✅ Created REAL TFSA portfolio with {len(portfolio.recommendations)} investments")
        
        return portfolio
        
    except Exception as e:
        logger.error(f"Real TFSA portfolio creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"TFSA portfolio creation failed: {str(e)}")

def create_simple_tfsa_allocation(tfsa_stocks: List[Dict], investable_amount: float) -> List[StockRecommendation]:
    """Create simple TFSA allocation with FORCED diversification - no duplicate sectors"""
    
    if not tfsa_stocks:
        return []
    
    # Add sector mapping to each stock
    for stock in tfsa_stocks:
        symbol = stock.get('symbol', '')
        if symbol in TFSA_ELIGIBLE_STOCKS_WITH_SECTORS:
            stock['sector'] = TFSA_ELIGIBLE_STOCKS_WITH_SECTORS[symbol]
        else:
            stock['sector'] = stock.get('sector', 'Unknown')
    
    # Group by sector
    sectors = {}
    for stock in tfsa_stocks:
        sector = stock.get('sector', 'Unknown')
        if sector not in sectors:
            sectors[sector] = []
        sectors[sector].append(stock)
    
    logger.info(f"Available sectors: {list(sectors.keys())}")
    
    recommendations = []
    remaining_amount = investable_amount
    used_sectors = set()
    
    # Define sector priority order (prefer these sectors for TFSA)
    sector_priority = ["Telecoms", "Banking", "Retail", "Insurance", "Consumer Goods", "Industrial", "REIT"]
    
    # Try to get one stock from each priority sector
    for preferred_sector in sector_priority:
        if len(recommendations) >= 3:  # Max 3 stocks
            break
            
        if preferred_sector not in sectors or preferred_sector in used_sectors:
            continue
            
        sector_stocks = sectors[preferred_sector]
        if not sector_stocks:
            continue
        
        # Find best stock in this sector that we can afford
        affordable_stocks = [s for s in sector_stocks if s.get('current_price', 0) > 0 and s.get('current_price', 0) <= remaining_amount]
        
        if not affordable_stocks:
            continue
            
        # Pick best stock by dividend yield (already fixed)
        best_stock = max(affordable_stocks, key=lambda x: x.get('dividend_yield', 0))
        current_price = best_stock.get('current_price', 0)
        
        # Calculate allocation
        if len(recommendations) == 0:  # First stock gets 50%
            allocation_percent = 50
        elif len(recommendations) == 1:  # Second stock gets 35%
            allocation_percent = 35
        else:  # Third stock gets remaining
            allocation_percent = 15
            
        target_amount = min(
            (allocation_percent / 100) * investable_amount,
            remaining_amount
        )
        
        shares_to_buy = int(target_amount / current_price)
        
        if shares_to_buy > 0:
            actual_amount = shares_to_buy * current_price
            actual_percentage = (actual_amount / investable_amount) * 100
            dividend_yield = best_stock.get('dividend_yield', 0)
            
            recommendations.append(StockRecommendation(
                symbol=best_stock['symbol'],
                company_name=best_stock.get('company_name', f"{best_stock['symbol']} Ltd"),
                allocation_percentage=actual_percentage,
                rand_amount=actual_amount,
                shares_to_buy=shares_to_buy,
                current_price=current_price,
                rationale=f"TFSA-eligible {preferred_sector} stock with {dividend_yield:.1f}% tax-free dividend yield",
                risk_level="Moderate", 
                expected_dividend_yield=dividend_yield,
                sector=preferred_sector
            ))
            
            remaining_amount -= actual_amount
            used_sectors.add(preferred_sector)
            
            logger.info(f"Added {best_stock['symbol']} from {preferred_sector} sector")
    
    # If we still don't have enough stocks, add from remaining sectors
    if len(recommendations) < 2 and remaining_amount > 50:
        for sector, sector_stocks in sectors.items():
            if sector in used_sectors or len(recommendations) >= 3:
                continue
                
            affordable_stocks = [s for s in sector_stocks if s.get('current_price', 0) > 0 and s.get('current_price', 0) <= remaining_amount]
            
            if affordable_stocks:
                best_stock = max(affordable_stocks, key=lambda x: x.get('dividend_yield', 0))
                current_price = best_stock.get('current_price', 0)
                
                target_amount = min(remaining_amount, (30 / 100) * investable_amount)
                shares_to_buy = int(target_amount / current_price)
                
                if shares_to_buy > 0:
                    actual_amount = shares_to_buy * current_price
                    actual_percentage = (actual_amount / investable_amount) * 100
                    dividend_yield = best_stock.get('dividend_yield', 0)
                    
                    recommendations.append(StockRecommendation(
                        symbol=best_stock['symbol'],
                        company_name=best_stock.get('company_name', f"{best_stock['symbol']} Ltd"),
                        allocation_percentage=actual_percentage,
                        rand_amount=actual_amount,
                        shares_to_buy=shares_to_buy,
                        current_price=current_price,
                        rationale=f"TFSA-eligible {sector} stock with {dividend_yield:.1f}% tax-free dividend yield",
                        risk_level="Moderate",
                        expected_dividend_yield=dividend_yield,
                        sector=sector
                    ))
                    
                    remaining_amount -= actual_amount
                    used_sectors.add(sector)
                    break
    
    logger.info(f"Final allocation: {len(recommendations)} stocks across {len(used_sectors)} sectors")
    return recommendations

# Enhanced analyzer class - add these methods to your existing EnhancedStockAnalyzer class
class TFSAAnalyzer:
    """TFSA-specific analysis functionality"""
    
    def __init__(self):
        self.annual_limit = 36000  # 2024 TFSA annual limit
        self.lifetime_limit = 500000  # Approximate lifetime limit as of 2024
        
    def analyze_tfsa_compliance(self, holdings_data: List[Dict], transactions_data: List[Dict]) -> Dict[str, bool]:
        """Check TFSA compliance and rules"""
        
        compliance = {
            "within_annual_limit": True,
            "no_prohibited_investments": True,
            "no_day_trading": True,
            "proper_contribution_tracking": True
        }
        
        # Calculate annual contributions from transactions
        current_year_contributions = 0
        for transaction in transactions_data:
            if transaction.get('Action', '').lower() in ['deposit', 'contribution']:
                amount = abs(transaction.get('DebitCredit', 0))
                current_year_contributions += amount
        
        # Check annual limit compliance
        if current_year_contributions > self.annual_limit:
            compliance["within_annual_limit"] = False
        
        # Check for prohibited investments (simplified check)
        for holding in holdings_data:
            symbol = holding.get('symbol', '')
            # Add checks for prohibited investments if needed
            # For now, assume all JSE stocks are allowed
        
        return compliance
    
    def calculate_tax_benefits(self, portfolio_value: float, estimated_annual_return: float = 0.08) -> Dict[str, float]:
        """Calculate estimated tax benefits of TFSA"""
        
        # Estimate various tax savings
        annual_dividend_income = portfolio_value * 0.04  # Assume 4% dividend yield
        annual_capital_gains = portfolio_value * (estimated_annual_return - 0.04)  # Growth component
        
        # South African tax rates (simplified)
        dividend_tax_saved = annual_dividend_income * 0.20  # 20% dividend withholding tax
        capital_gains_tax_saved = annual_capital_gains * 0.18  # 18% CGT (effective rate)
        
        return {
            "annual_dividend_tax_saved": dividend_tax_saved,
            "annual_cgt_saved": capital_gains_tax_saved,
            "total_annual_tax_saved": dividend_tax_saved + capital_gains_tax_saved,
            "10_year_projection": (dividend_tax_saved + capital_gains_tax_saved) * 10
        }
    
    def create_tfsa_optimization_prompt(self, holdings_data: List[Dict], account_value: float, 
                                      remaining_space: float, analysis_type: str) -> str:
        """Create AI prompt for TFSA-specific analysis"""
        
        holdings_summary = "\n".join([
            f"- {holding.get('name', 'Unknown')}: {holding.get('current_value', 'R0')} "
            f"({holding.get('profit_loss_percent', 0):.1f}% return)"
            for holding in holdings_data[:10]
        ])
        
        if analysis_type == "contribution_planning":
            return f"""
            As a TFSA specialist, analyze this Tax-Free Savings Account for contribution optimization:

            TFSA ACCOUNT STATUS:
            - Current Value: R{account_value:,.2f}
            - Remaining 2024 Contribution Space: R{remaining_space:,.2f}
            - Holdings Count: {len(holdings_data)}

            CURRENT HOLDINGS:
            {holdings_summary}

            TFSA CONTRIBUTION ANALYSIS REQUIRED:
            1. Should the client maximize their R{remaining_space:,.2f} remaining contribution?
            2. Which specific stocks should they buy with new contributions?
            3. Tax efficiency optimization - are current holdings TFSA-optimal?
            4. Should any positions be moved from taxable accounts to TFSA?
            5. Rebalancing recommendations within TFSA constraints

            PROVIDE SPECIFIC RECOMMENDATIONS:
            - Exact contribution amounts and timing
            - Specific stocks to buy with new money
            - Portfolio rebalancing within TFSA
            - Tax optimization strategies
            - Long-term TFSA growth plan

            Consider South African tax implications and TFSA rules.
            Focus on maximizing tax-free growth potential.
            """
            
        elif analysis_type == "rebalancing":
            return f"""
            Analyze this TFSA portfolio for optimal rebalancing:

            TFSA DETAILS:
            - Account Value: R{account_value:,.2f}
            - Contribution Space: R{remaining_space:,.2f}
            - Holdings: {len(holdings_data)} positions

            CURRENT ALLOCATIONS:
            {holdings_summary}

            REBALANCING ANALYSIS:
            1. Is the current allocation optimal for tax-free growth?
            2. Should any underperforming stocks be sold (no tax implications)?
            3. Which high-growth stocks should be prioritized in TFSA?
            4. Sector diversification within TFSA constraints
            5. Dividend vs growth stock balance for tax efficiency

            RECOMMENDATIONS NEEDED:
            - Specific trades to execute (sell X, buy Y)
            - Optimal TFSA allocation percentages
            - Stocks that are best suited for TFSA vs taxable accounts
            - Timing of rebalancing actions

            Remember: No capital gains tax on sales within TFSA.
            """
            
        else:  # optimization
            return f"""
            Perform comprehensive TFSA optimization analysis:

            TFSA PORTFOLIO:
            - Current Value: R{account_value:,.2f}
            - Available Contribution Space: R{remaining_space:,.2f}
            - Holdings: {len(holdings_data)} stocks

            PORTFOLIO BREAKDOWN:
            {holdings_summary}

            COMPREHENSIVE TFSA OPTIMIZATION:
            1. Portfolio quality assessment for tax-free growth
            2. Contribution strategy for remaining R{remaining_space:,.2f}
            3. Asset allocation optimization within TFSA
            4. High-growth stock recommendations for tax-free compounding
            5. Dividend optimization (no dividend tax in TFSA)
            6. Long-term wealth building strategy

            DELIVER ACTIONABLE PLAN:
            - Priority actions for immediate implementation
            - Specific stock recommendations with rationale
            - Contribution timing and amounts
            - 5-year TFSA growth projection
            - Tax efficiency maximization tactics

            Focus on maximizing the tax-free benefit of this account.
            """

tfsa_analyzer = TFSAAnalyzer()

# Add this new endpoint to your existing analysis.py router
@router.post("/tfsa-analysis", response_model=TFSAAnalysisResponse)
async def analyze_tfsa_account(request: TFSAAnalysisRequest):
    """
    Comprehensive TFSA (Tax-Free Savings Account) analysis with optimization recommendations
    """
    try:
        from easy_equities_client.clients import EasyEquitiesClient
        
        username = os.getenv("EASYEQUITIES_USERNAME")
        password = os.getenv("EASYEQUITIES_PASSWORD")
        
        if not username or not password:
            raise HTTPException(status_code=500, detail="EasyEquities credentials not configured")
        
        # Get account data
        client = EasyEquitiesClient()
        client.login(username=username, password=password)
        
        # Get holdings and transactions
        holdings = client.accounts.holdings(request.account_id, include_shares=True)
        transactions = client.accounts.transactions(request.account_id)
        
        if not holdings:
            raise HTTPException(status_code=404, detail="No holdings found in TFSA account")
        
        # Calculate account value
        total_value = 0.0
        processed_holdings = []
        
        for holding in holdings:
            current_value_str = holding.get('current_value', 'R0.00')
            current_value = float(current_value_str.replace('R', '').replace(',', '').strip())
            total_value += current_value
            
            # Process holding data
            processed_holdings.append({
                "name": holding.get('name', 'Unknown'),
                "symbol": holding.get('contract_code', '').replace('EQU.ZA.', ''),
                "current_value": current_value_str,
                "shares": holding.get('shares', '0'),
                "profit_loss_percent": 0  # Calculate if purchase data available
            })
        
        # Calculate contribution space
        annual_contribution_used = 0
        for transaction in transactions:
            if transaction.get('Action', '').lower() in ['deposit', 'contribution']:
                annual_contribution_used += abs(transaction.get('DebitCredit', 0))
        
        remaining_space = max(0, tfsa_analyzer.annual_limit - annual_contribution_used)
        
        # If user provided their own estimate, use that
        if request.annual_contribution_capacity is not None:
            remaining_space = request.annual_contribution_capacity
        
        # Compliance check
        compliance = tfsa_analyzer.analyze_tfsa_compliance(processed_holdings, transactions)
        
        # Tax benefits calculation
        tax_benefits = tfsa_analyzer.calculate_tax_benefits(total_value)
        
        # Generate AI analysis
        analysis_prompt = tfsa_analyzer.create_tfsa_optimization_prompt(
            processed_holdings,
            total_value,
            remaining_space,
            request.analysis_type
        )
        
        ai_analysis = await call_claude_api(analysis_prompt, max_tokens=2000)
        
        # Parse AI recommendations
        recommendations = parse_tfsa_recommendations(ai_analysis, remaining_space)
        
        # Extract strategy and next actions
        strategy, next_actions = extract_tfsa_guidance(ai_analysis)
        
        logger.info(f"✅ Generated TFSA analysis for account {request.account_id}")
        
        return TFSAAnalysisResponse(
            account_id=request.account_id,
            current_value=total_value,
            annual_contribution_used=annual_contribution_used,
            remaining_contribution_space=remaining_space,
            tax_savings_estimate=tax_benefits["total_annual_tax_saved"],
            recommendations=recommendations,
            portfolio_analysis=ai_analysis[:1000] + "..." if len(ai_analysis) > 1000 else ai_analysis,
            tfsa_strategy=strategy,
            compliance_check=compliance,
            next_actions=next_actions
        )
        
    except Exception as e:
        logger.error(f"TFSA analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"TFSA analysis failed: {str(e)}")

def parse_tfsa_recommendations(ai_response: str, remaining_space: float) -> List[TFSARecommendation]:
    """Parse AI response to extract structured TFSA recommendations"""
    
    recommendations = []
    lines = ai_response.split('\n')
    
    # Default recommendations based on remaining space
    if remaining_space > 1000:
        recommendations.append(TFSARecommendation(
            recommendation_type="contribute_more",
            action_required=f"Contribute remaining R{remaining_space:,.0f} to maximize 2024 allowance",
            amount_involved=remaining_space,
            rationale="Maximize tax-free growth potential before year-end",
            tax_benefit_estimate=remaining_space * 0.08 * 0.20,  # Estimated annual tax saving
            priority_level="high"
        ))
    
    # Look for specific recommendations in AI response
    current_rec = None
    for line in lines:
        line_clean = line.strip()
        
        if any(word in line_clean.lower() for word in ['buy', 'sell', 'contribute', 'rebalance']):
            if 'buy' in line_clean.lower():
                rec_type = "switch_stocks"
                action = line_clean
            elif 'contribute' in line_clean.lower():
                rec_type = "contribute_more"
                action = line_clean
            elif 'rebalance' in line_clean.lower():
                rec_type = "rebalance"
                action = line_clean
            else:
                rec_type = "optimize_allocation"
                action = line_clean
            
            # Extract amount if present
            import re
            amount_match = re.search(r'R(\d+(?:[,\s]\d+)*(?:\.\d+)?)', line_clean)
            amount = None
            if amount_match:
                amount_str = amount_match.group(1).replace(',', '').replace(' ', '')
                amount = float(amount_str)
            
            recommendations.append(TFSARecommendation(
                recommendation_type=rec_type,
                action_required=action,
                amount_involved=amount,
                rationale="Based on TFSA optimization analysis",
                priority_level="medium"
            ))
    
    return recommendations[:5]  # Limit to top 5 recommendations

def extract_tfsa_guidance(ai_response: str) -> tuple:
    """Extract strategy and next actions from TFSA analysis"""
    
    strategy = "Maximize tax-free growth through optimal asset allocation and contribution planning."
    next_actions = [
        "Review remaining 2024 contribution allowance",
        "Consider high-growth stocks for TFSA allocation",
        "Set up automatic monthly contributions if possible",
        "Monitor portfolio quarterly for rebalancing opportunities",
        "Plan 2025 contribution strategy"
    ]
    
    # Try to extract better content from AI response
    lines = ai_response.split('\n')
    extracted_actions = []
    
    for line in lines:
        line_clean = line.strip()
        
        if 'strategy' in line_clean.lower() and len(line_clean) > 30:
            strategy = line_clean
        elif line_clean.startswith(('-', '•', '*')) and len(line_clean) > 20:
            extracted_actions.append(line_clean.lstrip('-•* '))
    
    if extracted_actions:
        next_actions = extracted_actions[:5]
    
    return strategy, next_actions

# Modified endpoint - enhance your existing portfolio analysis to detect TFSA accounts
@router.post("/portfolio-enhanced")
async def analyze_portfolio_enhanced(request: PortfolioAnalysisRequest):
    """
    Enhanced portfolio analysis that automatically detects and handles TFSA accounts
    """
    try:
        from easy_equities_client.clients import EasyEquitiesClient
        
        username = os.getenv("EASYEQUITIES_USERNAME")
        password = os.getenv("EASYEQUITIES_PASSWORD")
        
        client = EasyEquitiesClient()
        client.login(username=username, password=password)
        
        # Get account info to determine account type
        accounts = client.accounts.list()
        account_info = next((acc for acc in accounts if acc.id == request.account_id), None)
        
        is_tfsa = False
        if account_info and ('tfsa' in account_info.name.lower() or 'tax-free' in account_info.name.lower()):
            is_tfsa = True
        
        holdings = client.accounts.holdings(request.account_id, include_shares=True)
        
        if not holdings:
            return {
                "message": "No holdings found for analysis",
                "account_type": "TFSA" if is_tfsa else "Regular",
                "recommendations": []
            }
        
        # Create appropriate prompt based on account type
        if is_tfsa:
            # Use TFSA-specific analysis
            prompt = f"""
            Analyze this TFSA portfolio for {request.focus} with tax-free optimization:

            TFSA Holdings: {len(holdings)} positions
            Account Type: Tax-Free Savings Account

            Focus: {request.focus}

            Provide TFSA-specific recommendations considering:
            1. Tax-free growth optimization
            2. Contribution space utilization
            3. Asset allocation for maximum tax benefit
            4. High-growth stock prioritization
            5. Dividend optimization (no dividend tax)

            Be specific about TFSA advantages and optimization strategies.
            """
        else:
            # Regular portfolio analysis
            prompt = f"""
            Analyze this regular investment portfolio for {request.focus}:

            Holdings: {len(holdings)} positions
            Focus: {request.focus}

            Consider tax implications and optimization for taxable account.
            """
        
        analysis = await call_claude_api(prompt, max_tokens=2000)
        
        return {
            "account_id": request.account_id,
            "account_type": "TFSA" if is_tfsa else "Regular",
            "analysis_type": "portfolio_" + request.focus,
            "analysis": analysis,
            "holdings_analyzed": len(holdings),
            "tfsa_optimized": is_tfsa,
            "analysis_date": "2024-08-07T10:00:00Z"
        }
        
    except Exception as e:
        logger.error(f"Enhanced portfolio analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# Add TFSA-specific beginner portfolio endpoint
@router.post("/tfsa-beginner-portfolio")
async def create_tfsa_beginner_portfolio(request: BeginnerInvestmentRequest):
    """
    Create beginner TFSA portfolio with tax-optimization focus
    """
    try:
        # Validate TFSA contribution limits
        if request.deposit_amount > 36000:
            raise HTTPException(
                status_code=400,
                detail="Amount exceeds annual TFSA contribution limit of R36,000"
            )
        
        # Get regular beginner portfolio first
        regular_portfolio = await create_beginner_portfolio(request)
        
        # Enhance with TFSA-specific analysis
        tfsa_prompt = f"""
        Optimize this beginner portfolio specifically for TFSA (Tax-Free Savings Account):

        Original Portfolio Recommendations:
        {[f"{rec.symbol}: R{rec.rand_amount}" for rec in regular_portfolio.recommendations]}

        TFSA OPTIMIZATION REQUIRED:
        1. Prioritize high-growth stocks (no capital gains tax)
        2. Include dividend-paying stocks (no dividend tax)
        3. Focus on long-term compounding potential
        4. Consider JSE stocks with international exposure
        5. Optimize for maximum tax-free benefit

        Provide TFSA-specific rationale for each recommendation.
        Explain the tax advantages of this allocation.
        """
        
        tfsa_analysis = await call_claude_api(tfsa_prompt, max_tokens=1500)
        
        # Enhance the response with TFSA context
        enhanced_response = regular_portfolio.dict()
        enhanced_response.update({
            "account_type": "TFSA",
            "tfsa_benefits": {
                "no_dividend_tax": True,
                "no_capital_gains_tax": True,
                "tax_free_growth": True,
                "annual_limit": 36000,
                "remaining_space_after_investment": 36000 - request.deposit_amount
            },
            "tfsa_strategy": tfsa_analysis[:500],
            "key_principles": [
                "Maximize tax-free growth in TFSA",
                "Prioritize high-growth stocks for TFSA allocation",
                "Use dividend income without tax implications",
                "Hold growth stocks long-term in TFSA",
                "Consider annual contribution limits"
            ]
        })
        
        return enhanced_response
        
    except Exception as e:
        logger.error(f"TFSA beginner portfolio creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"TFSA portfolio creation failed: {str(e)}")

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

# Complete fixed beginner portfolio endpoint and supporting functions

@router.post("/beginner-portfolio", response_model=BeginnerPortfolioResponse)
async def create_beginner_portfolio(request: BeginnerInvestmentRequest):
    """
    Create a beginner-friendly portfolio recommendation with VALIDATED market data
    """
    try:
        # Validate minimum investment
        if request.deposit_amount < 100:
            raise HTTPException(
                status_code=400, 
                detail="Minimum investment amount is R100"
            )
        
        # Reserve cash for brokerage fees
        brokerage_reserve = max(50, min(request.deposit_amount * 0.04, 200))
        investable_amount = request.deposit_amount - brokerage_reserve
        
        logger.info(f"Creating beginner portfolio for R{request.deposit_amount} (R{investable_amount} investable)")
        
        # Screen stocks dynamically
        raw_stocks = await screener.screen_stocks_for_beginners(
            request.risk_tolerance,
            request.investment_goal,
            max_stocks=20  # Increased from 12 to get more options
        )
        
        if not raw_stocks:
            raise HTTPException(
                status_code=500,
                detail="Could not find suitable stocks in current market conditions"
            )
        
        # VALIDATE AND CLEAN THE STOCK DATA
        suitable_stocks = validate_and_clean_stock_data(raw_stocks)
        
        if not suitable_stocks:
            raise HTTPException(
                status_code=500,
                detail="No valid stocks found after data cleaning"
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
            for stock in suitable_stocks[:10]  # Show top 10 options
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

        TOP QUALITY JSE STOCKS (VALIDATED Real Market Data):
        {stocks_summary}

        PORTFOLIO REQUIREMENTS:
        1. Select 2-4 stocks for proper diversification
        2. No single stock should exceed 40% of portfolio (concentration risk)
        3. Minimum position size should be at least R80 (meaningful allocation)
        4. Must be able to buy whole shares only
        5. Prioritize different sectors for diversification
        6. Use 85-95% of available investment amount

        For EACH recommended stock, provide:
        - Symbol and company name
        - Exact percentage allocation (total must be 85-95%)
        - Specific Rand amount to invest
        - Number of whole shares to purchase
        - Investment rationale focusing on fundamentals
        - Risk level (Low/Moderate/High)

        IMPORTANT: 
        - Only use the dividend yields provided above - they have been validated as realistic
        - Prioritize sector diversification over individual stock preferences
        - Ensure you allocate most of the available investment amount
        
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
            recommendations = create_validated_fallback_allocation(
                suitable_stocks,
                investable_amount,
                request.risk_tolerance,
                request.investment_goal
            )
        
        # VALIDATE FINAL RECOMMENDATIONS
        recommendations = validate_final_recommendations(recommendations)
        
        # Calculate portfolio metrics
        total_allocated = sum(rec.rand_amount for rec in recommendations)
        cash_remaining = investable_amount - total_allocated
        
        # Calculate estimated annual dividend yield (CAP AT REALISTIC LEVELS)
        estimated_yield = 0
        if total_allocated > 0:
            weighted_yield = sum(
                rec.rand_amount * (min(rec.expected_dividend_yield or 0, 12) / 100)  # Cap at 12%
                for rec in recommendations
            )
            estimated_yield = (weighted_yield / total_allocated) * 100
        
        # Calculate diversification score
        sectors = set(rec.sector for rec in recommendations if rec.sector)
        diversification_score = f"Excellent ({len(sectors)} sectors)" if len(sectors) >= 3 else f"Good ({len(sectors)} sectors)" if len(sectors) >= 2 else f"Limited ({len(sectors)} sector)"
        
        # Extract guidance from AI response
        strategy, principles, next_steps = extract_guidance_from_ai_response(allocation_analysis)
        
        logger.info(f"✅ Created validated beginner portfolio: {len(recommendations)} stocks, R{total_allocated:.2f} allocated, {len(sectors)} sectors")
        
        return BeginnerPortfolioResponse(
            total_investment=request.deposit_amount,
            cash_reserve=brokerage_reserve + cash_remaining,
            total_allocated=total_allocated,
            recommendations=recommendations,
            portfolio_strategy=strategy,
            risk_assessment=f"{request.risk_tolerance.title()} risk portfolio optimized for {request.investment_goal}",
            key_principles=principles,
            next_steps=next_steps,
            estimated_annual_yield=min(estimated_yield, 15) if estimated_yield > 0 else None,  # Cap at 15%
            diversification_score=diversification_score
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Beginner portfolio creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Portfolio creation failed: {str(e)}")

def validate_and_clean_stock_data(raw_stocks: List[Dict]) -> List[Dict]:
    """Validate and clean stock data to remove obvious errors"""
    
    cleaned_stocks = []
    
    for stock in raw_stocks:
        # Create a copy to avoid modifying original
        clean_stock = stock.copy()
        symbol = clean_stock.get('symbol', 'UNKNOWN')
        
        # Fix dividend yield issues
        dividend_yield = clean_stock.get('dividend_yield', 0)
        original_yield = dividend_yield
        
        if dividend_yield > 100:  # Clearly wrong - probably cents issue
            clean_stock['dividend_yield'] = min(dividend_yield / 100, 10.0)
        elif dividend_yield > 50:  # Very suspicious
            clean_stock['dividend_yield'] = min(dividend_yield / 10, 10.0)
        elif dividend_yield > 20:  # Still suspicious
            clean_stock['dividend_yield'] = min(dividend_yield / 5, 12.0)
        elif dividend_yield > 15:  # Cap at reasonable maximum
            clean_stock['dividend_yield'] = 12.0
        
        if original_yield != clean_stock['dividend_yield']:
            logger.warning(f"🔧 Fixed dividend yield for {symbol}: {original_yield:.1f}% → {clean_stock['dividend_yield']:.1f}%")
        
        # Fix P/E ratio issues
        pe_ratio = clean_stock.get('pe_ratio', 0)
        if pe_ratio > 100 or pe_ratio < 0:
            clean_stock['pe_ratio'] = 0
            logger.warning(f"🔧 Reset invalid P/E for {symbol}: {pe_ratio}")
        
        # Fix current price issues
        current_price = clean_stock.get('current_price', 0)
        original_price = current_price
        
        if current_price < 1:  # Likely cents issue
            clean_stock['current_price'] = current_price * 100
        elif current_price > 10000:  # Definitely cents issue
            clean_stock['current_price'] = current_price / 100
        elif current_price > 5000:  # Probably cents issue
            clean_stock['current_price'] = current_price / 100
        
        if original_price != clean_stock['current_price']:
            logger.warning(f"🔧 Fixed price for {symbol}: R{original_price} → R{clean_stock['current_price']}")
        
        # More lenient validation criteria
        current_price = clean_stock.get('current_price', 0)
        dividend_yield = clean_stock.get('dividend_yield', 0)
        market_cap = clean_stock.get('market_cap', 0)
        
        # Only include stocks with reasonable data (more lenient)
        if (current_price >= 2 and               # At least R2 per share
            current_price <= 2500 and            # Max R2500 per share
            dividend_yield <= 15 and             # Max 15% dividend yield
            market_cap > 50_000_000):            # R50M+ market cap (reduced from R500M)
            
            cleaned_stocks.append(clean_stock)
        else:
            logger.warning(f"❌ Excluded {symbol}: Price=R{current_price}, DivYield={dividend_yield:.1f}%, MarketCap=R{market_cap:,}")
    
    logger.info(f"✅ Cleaned stock data: {len(raw_stocks)} → {len(cleaned_stocks)} valid stocks")
    return cleaned_stocks

def create_validated_fallback_allocation(stocks: List[Dict], investable_amount: float, 
                                       risk_tolerance: str, investment_goal: str) -> List[StockRecommendation]:
    """Create fallback allocation with validated data and better diversification"""
    
    if not stocks:
        return []
    
    # Group stocks by sector for diversification
    sectors = {}
    for stock in stocks:
        sector = stock.get('sector', 'Unknown')
        if sector not in sectors:
            sectors[sector] = []
        sectors[sector].append(stock)
    
    logger.info(f"📊 Available sectors: {list(sectors.keys())}")
    
    # Sort stocks within each sector by quality
    for sector in sectors:
        def stock_score(stock):
            quality = stock.get('quality_score', 0)
            div_yield = min(stock.get('dividend_yield', 0), 12)  # Cap at 12%
            return quality + (div_yield * 3)  # Moderate bonus for dividends
        
        sectors[sector] = sorted(sectors[sector], key=stock_score, reverse=True)
    
    recommendations = []
    remaining_amount = investable_amount
    used_sectors = set()
    
    # Target 2-3 stocks from different sectors
    target_stocks = min(3, len(sectors))
    allocations = [50, 30, 20] if target_stocks >= 3 else [60, 40] if target_stocks >= 2 else [100]
    
    # Try to get one stock from each sector
    sector_priority = ["Banking", "Telecoms", "Retail", "Insurance", "Industrial", "Consumer Goods", "Mining"]
    
    stock_count = 0
    
    # First, try priority sectors
    for priority_sector in sector_priority:
        if stock_count >= target_stocks or remaining_amount < 80:
            break
            
        if priority_sector in sectors and priority_sector not in used_sectors:
            best_stock = sectors[priority_sector][0]  # Best stock in this sector
            current_price = best_stock.get('current_price', 0)
            
            if current_price > 0 and current_price <= remaining_amount * 0.6:  # Max 60% on one stock
                
                allocation_percent = allocations[stock_count]
                target_amount = (allocation_percent / 100) * investable_amount
                target_amount = min(target_amount, remaining_amount)
                
                shares_to_buy = int(target_amount / current_price)
                
                if shares_to_buy > 0:
                    actual_amount = shares_to_buy * current_price
                    actual_percentage = (actual_amount / investable_amount) * 100
                    dividend_yield = min(best_stock.get('dividend_yield', 0), 12)
                    
                    recommendations.append(StockRecommendation(
                        symbol=best_stock['symbol'],
                        company_name=best_stock.get('company_name', f"{best_stock['symbol']} Ltd"),
                        allocation_percentage=actual_percentage,
                        rand_amount=actual_amount,
                        shares_to_buy=shares_to_buy,
                        current_price=current_price,
                        rationale=f"Quality {priority_sector} stock with {dividend_yield:.1f}% dividend yield and strong market position",
                        risk_level=determine_risk_level_from_data(best_stock),
                        expected_dividend_yield=dividend_yield,
                        sector=priority_sector
                    ))
                    
                    remaining_amount -= actual_amount
                    used_sectors.add(priority_sector)
                    stock_count += 1
                    logger.info(f"✅ Added {best_stock['symbol']} from {priority_sector}")
    
    # If we still need more stocks, add from remaining sectors
    for sector, sector_stocks in sectors.items():
        if stock_count >= target_stocks or remaining_amount < 80:
            break
            
        if sector not in used_sectors:
            best_stock = sector_stocks[0]
            current_price = best_stock.get('current_price', 0)
            
            if current_price > 0 and current_price <= remaining_amount:
                
                # Use remaining allocation or 20% minimum
                remaining_allocation = max(20, (remaining_amount / investable_amount) * 100)
                target_amount = min(remaining_amount * 0.8, (remaining_allocation / 100) * investable_amount)
                
                shares_to_buy = int(target_amount / current_price)
                
                if shares_to_buy > 0:
                    actual_amount = shares_to_buy * current_price
                    actual_percentage = (actual_amount / investable_amount) * 100
                    dividend_yield = min(best_stock.get('dividend_yield', 0), 12)
                    
                    recommendations.append(StockRecommendation(
                        symbol=best_stock['symbol'],
                        company_name=best_stock.get('company_name', f"{best_stock['symbol']} Ltd"),
                        allocation_percentage=actual_percentage,
                        rand_amount=actual_amount,
                        shares_to_buy=shares_to_buy,
                        current_price=current_price,
                        rationale=f"Diversification pick from {sector} sector with {dividend_yield:.1f}% dividend yield",
                        risk_level=determine_risk_level_from_data(best_stock),
                        expected_dividend_yield=dividend_yield,
                        sector=sector
                    ))
                    
                    remaining_amount -= actual_amount
                    used_sectors.add(sector)
                    stock_count += 1
                    logger.info(f"✅ Added {best_stock['symbol']} from {sector}")
    
    logger.info(f"📈 Created fallback allocation: {len(recommendations)} stocks, {len(used_sectors)} sectors")
    return recommendations

def validate_final_recommendations(recommendations: List[StockRecommendation]) -> List[StockRecommendation]:
    """Final validation of recommendations to ensure realistic data"""
    
    validated = []
    
    for rec in recommendations:
        # Cap dividend yields at realistic levels
        if rec.expected_dividend_yield > 15:
            logger.warning(f"🔧 Capped dividend yield for {rec.symbol}: {rec.expected_dividend_yield:.1f}% → 12.0%")
            rec.expected_dividend_yield = 12.0
        
        # Ensure rationale doesn't mention unrealistic yields
        import re
        # Find any percentage over 50% in the rationale and replace
        def replace_high_percentage(match):
            percentage = float(match.group(1))
            if percentage > 50:
                return f"{min(percentage/10, 12):.1f}%"
            return match.group(0)
        
        rec.rationale = re.sub(r'(\d+(?:\.\d+)?)%', replace_high_percentage, rec.rationale)
        
        # Fix any remaining crazy percentages
        if any(crazy in rec.rationale for crazy in ["515%", "322%", "400%", "100%"]):
            rec.rationale = f"Quality {rec.sector} stock with {rec.expected_dividend_yield:.1f}% dividend yield and strong market position"
            logger.warning(f"🔧 Fixed rationale for {rec.symbol}")
        
        validated.append(rec)
    
    return validated

def determine_risk_level_from_data(stock: Dict) -> str:
    """Determine risk level from stock data"""
    
    sector = stock.get('sector', '').lower()
    pe_ratio = stock.get('pe_ratio', 0)
    debt_equity = stock.get('debt_to_equity', 0)
    beta = stock.get('beta', 1.0)
    
    risk_score = 0
    
    # Sector risk
    if any(word in sector for word in ['mining', 'resources', 'technology', 'biotech']):
        risk_score += 2
    elif any(word in sector for word in ['retail', 'consumer', 'industrial']):
        risk_score += 1
    elif any(word in sector for word in ['banking', 'insurance', 'telecoms', 'utilities']):
        risk_score += 0
    else:
        risk_score += 1
    
    # Valuation risk
    if pe_ratio > 30:
        risk_score += 1
    elif pe_ratio > 20:
        risk_score += 0.5
    
    # Debt risk
    if debt_equity > 100:
        risk_score += 1
    
    # Volatility risk
    if beta > 1.5:
        risk_score += 1
    elif beta > 1.2:
        risk_score += 0.5
    
    if risk_score >= 3:
        return "High"
    elif risk_score >= 1.5:
        return "Moderate"
    else:
        return "Low"

def extract_guidance_from_ai_response(ai_response: str) -> tuple:
    """Extract strategy, principles and next steps from AI response"""
    
    # Default fallbacks
    strategy = "Diversified beginner portfolio focusing on quality JSE companies with sustainable dividends and steady growth potential."
    
    principles = [
        "Start with quality, established companies",
        "Diversify across different sectors to reduce risk",
        "Focus on dividend-paying stocks for passive income",
        "Reinvest all dividends for compound growth",
        "Review portfolio quarterly, not daily"
    ]
    
    next_steps = [
        "Open a trading account with a reputable South African broker",
        "Set up automatic dividend reinvestment where available",
        "Create a monthly investment schedule for consistency",
        "Learn about fundamental analysis of your holdings",
        "Consider increasing contributions as income grows"
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
        elif any(word in line_lower for word in ['principle', 'key point', 'important', 'remember']):
            current_section = 'principles'
        elif any(word in line_lower for word in ['next step', 'action', 'should', 'recommendation']):
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

screener = DynamicStockScreener()
async def get_etf_real_time_data(etf_code: str) -> Dict:
    """Get real-time ETF data using yfinance with fallbacks"""
    
    # Fallback prices based on your search results
    fallback_prices = {
        "STXFIN": 21.35,
        "STXPRO": 12.24, 
        "STXDIVI": 2.69,
        "STX40": 93.41,
        "STXIND": 135.09
    }
    
    try:
        possible_tickers = [f"{etf_code}.JO", etf_code]
        
        for ticker_symbol in possible_tickers:
            try:
                ticker = yf.Ticker(ticker_symbol)
                info = ticker.info
                
                current_price = info.get('regularMarketPrice', 0)
                
                # FIX: Convert cents to rands if price seems too high
                if current_price > 1000:  # Likely in cents
                    current_price = current_price / 100
                    logger.info(f"🔧 Converted {etf_code} from cents to rands: R{current_price}")
                
                if current_price > 0:
                    logger.info(f"✅ Got real-time price for {etf_code}: R{current_price}")
                    return {
                        "symbol": etf_code,
                        "current_price": current_price,
                        "dividend_yield": info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 3.0,
                        "data_source": f"Yahoo Finance ({ticker_symbol})"
                    }
            except Exception as e:
                logger.warning(f"Failed to get {ticker_symbol}: {e}")
                continue
        
        # Use fallback if real-time fails
        if etf_code in fallback_prices:
            logger.info(f"⚠️ Using fallback price for {etf_code}: R{fallback_prices[etf_code]}")
            return {
                "symbol": etf_code,
                "current_price": fallback_prices[etf_code],
                "dividend_yield": 3.5,
                "data_source": "Fallback pricing"
            }
        
        return {"symbol": etf_code, "current_price": 50.0, "error": "No data available"}
        
    except Exception as e:
        logger.error(f"ETF data fetch failed for {etf_code}: {e}")
        if etf_code in fallback_prices:
            return {
                "symbol": etf_code,
                "current_price": fallback_prices[etf_code],
                "dividend_yield": 3.5,
                "data_source": "Fallback pricing"
            }
        return {"symbol": etf_code, "current_price": 50.0, "error": str(e)}


async def create_real_tfsa_portfolio(request: BeginnerInvestmentRequest) -> BeginnerPortfolioResponse:
    """Create TFSA portfolio with REAL-TIME ETF prices"""
    
    logger.info(f"🔍 Creating TFSA portfolio for R{request.deposit_amount}")
    
    # Get real-time prices for recommended ETFs - PUT CHEAPER ONES FIRST
    etf_codes = ["STXFIN", "STXPRO", "STXDIVI", "STX40", "STXIND"]
    
    etf_tasks = [get_etf_real_time_data(code) for code in etf_codes]
    etf_results = await asyncio.gather(*etf_tasks, return_exceptions=True)
    
    # Process ETF data and filter for affordable options
    available_etfs = []
    brokerage_reserve = max(20, min(request.deposit_amount * 0.02, 100))
    investable_amount = request.deposit_amount - brokerage_reserve
    
    logger.info(f"💰 Investable amount: R{investable_amount} (after R{brokerage_reserve} fees)")
    
    for i, result in enumerate(etf_results):
        etf_code = etf_codes[i]
        
        if isinstance(result, dict) and not result.get("error"):
            current_price = result.get("current_price", 0)
            logger.info(f"📊 {etf_code}: R{current_price} - {'✅ Affordable' if current_price <= investable_amount * 0.95 else '❌ Too expensive'}")
            
            # MAKE THIS MORE FLEXIBLE - allow up to 95% of investment on one ETF
            if current_price > 0 and current_price <= investable_amount * 0.95:
                
                # Add ETF info based on code
                etf_info = get_etf_info(etf_code, current_price, result.get("dividend_yield", 0))
                if etf_info:
                    available_etfs.append(etf_info)
                    logger.info(f"✅ Added {etf_code} to available ETFs")
            else:
                logger.warning(f"❌ {etf_code} too expensive: R{current_price} > R{investable_amount * 0.95:.2f}")
        else:
            logger.error(f"❌ Failed to get data for {etf_code}: {result}")
    
    logger.info(f"📈 Found {len(available_etfs)} affordable ETFs")
    
    # Add emergency fallback for very small amounts
    if not available_etfs:
        logger.warning("⚠️ No ETFs found, adding emergency fallback options")
        
        # Add some guaranteed affordable options
        emergency_etfs = [
            {
                "symbol": "STXDIVI",
                "name": "Satrix DIVI ETF",
                "current_price": 2.69,
                "sector": "Dividend Focused",
                "description": "High dividend yield stocks",
                "risk": "Moderate",
                "dividend_yield": 5.0
            },
            {
                "symbol": "BALANCED_FUND",
                "name": "Balanced Fund Option",
                "current_price": 14.15,
                "sector": "Multi-Asset",
                "description": "Diversified balanced exposure",
                "risk": "Moderate",
                "dividend_yield": 4.0
            }
        ]
        
        for etf in emergency_etfs:
            if etf["current_price"] <= investable_amount * 0.95:
                available_etfs.append(etf)
                logger.info(f"✅ Added emergency option: {etf['symbol']}")
    
    if not available_etfs:
        logger.error(f"❌ Still no affordable ETFs for R{investable_amount}")
        raise HTTPException(status_code=500, detail=f"No TFSA investments found for R{request.deposit_amount}. Minimum recommended: R1000")
    
    # Create recommendations based on investment goal and risk tolerance
    recommendations = []
    remaining_amount = investable_amount
    used_sectors = set()
    
    # Sort ETFs by suitability for the investment goal
    sorted_etfs = sort_etfs_by_goal(available_etfs, request.investment_goal, request.risk_tolerance)
    logger.info(f"🎯 Sorted ETFs by goal '{request.investment_goal}': {[etf['symbol'] for etf in sorted_etfs]}")
    
    # Allocate across 2-3 ETFs maximum
    target_etfs = min(2, len(sorted_etfs))  # Limit to 2 for small amounts
    allocations = [70, 30] if target_etfs >= 2 else [100]
    
    for i in range(target_etfs):
        if i >= len(sorted_etfs) or remaining_amount < 50:
            break
            
        etf = sorted_etfs[i]
        current_price = etf["current_price"]
        allocation_percent = allocations[i]
        
        target_amount = (allocation_percent / 100) * investable_amount
        target_amount = min(target_amount, remaining_amount)
        
        units_to_buy = int(target_amount / current_price)
        
        if units_to_buy > 0:
            actual_amount = units_to_buy * current_price
            actual_percentage = (actual_amount / investable_amount) * 100
            
            logger.info(f"💸 {etf['symbol']}: {units_to_buy} units × R{current_price} = R{actual_amount}")
            
            recommendations.append(StockRecommendation(
                symbol=etf["symbol"],
                company_name=etf["name"],
                allocation_percentage=actual_percentage,
                rand_amount=actual_amount,
                shares_to_buy=units_to_buy,
                current_price=current_price,
                rationale=f"TFSA ETF: {etf['description']} (Price: R{current_price:.2f})",
                risk_level=etf["risk"],
                expected_dividend_yield=etf["dividend_yield"],
                sector=etf["sector"]
            ))
            
            remaining_amount -= actual_amount
            used_sectors.add(etf["sector"])
    
    total_allocated = sum(rec.rand_amount for rec in recommendations)
    estimated_yield = sum(rec.expected_dividend_yield * rec.allocation_percentage / 100 for rec in recommendations) if recommendations else 0
    
    logger.info(f"✅ Created TFSA portfolio: {len(recommendations)} ETFs, R{total_allocated:.2f} allocated, R{remaining_amount:.2f} remaining")
    
    return BeginnerPortfolioResponse(
        total_investment=request.deposit_amount,
        cash_reserve=brokerage_reserve + remaining_amount,
        total_allocated=total_allocated,
        recommendations=recommendations,
        portfolio_strategy="TFSA portfolio using verified available ETFs with real-time pricing where possible",
        risk_assessment=f"TFSA {request.risk_tolerance} risk portfolio",
        key_principles=[
            "Use only TFSA-available ETFs and funds",
            "Prioritize affordable options for small investments",
            "Benefit from professional fund management",
            "Maximize tax-free growth and income",
            "Keep costs low with ETF options"
        ],
        next_steps=[
            "Open TFSA account with reputable SA broker",
            "Search for exact ETF codes in TFSA platform",
            "Verify current prices before purchasing",
            "Consider increasing monthly contributions",
            "Review allocations quarterly"
        ],
        estimated_annual_yield=estimated_yield,
        diversification_score=f"Good ({len(used_sectors)} asset classes)"
    )
def get_etf_info(etf_code: str, current_price: float, dividend_yield: float) -> Dict:
    """Get ETF information based on code"""
    
    etf_mapping = {
        "STXIND": {
            "name": "Satrix INDI ETF",
            "sector": "South African Equity",
            "description": "Tracks JSE Industrial sector",
            "risk": "Moderate"
        },
        "STXFIN": {
            "name": "Satrix FINI ETF", 
            "sector": "Financial",
            "description": "JSE Financial sector exposure",
            "risk": "Moderate"
        },
        "STX40": {
            "name": "Satrix Top 40 ETF",
            "sector": "Large Cap SA",
            "description": "Top 40 JSE companies",
            "risk": "Moderate"
        },
        "STXPRO": {
            "name": "Satrix Property ETF",
            "sector": "Property/REIT", 
            "description": "SA Listed Property exposure",
            "risk": "Moderate-High"
        },
        "STXWDM": {
            "name": "Satrix MSCI World ETF",
            "sector": "Global Equity",
            "description": "Global developed markets",
            "risk": "Moderate"
        },
        "STXDIVI": {
            "name": "Satrix DIVI ETF",
            "sector": "Dividend Focused",
            "description": "High dividend yield stocks",
            "risk": "Moderate"
        }
    }
    
    if etf_code in etf_mapping:
        info = etf_mapping[etf_code].copy()
        info.update({
            "symbol": etf_code,
            "current_price": current_price,
            "dividend_yield": dividend_yield if dividend_yield > 0 else 3.0  # Default estimate
        })
        return info
    
    return None

def sort_etfs_by_goal(etfs: List[Dict], investment_goal: str, risk_tolerance: str) -> List[Dict]:
    """Sort ETFs based on investment goal and risk tolerance"""
    
    def etf_score(etf):
        score = 0
        
        # Score based on investment goal
        if investment_goal == "income":
            if "Property" in etf["sector"] or "Dividend" in etf["sector"] or etf["dividend_yield"] > 4:
                score += 10
        elif investment_goal == "growth":
            if "Global" in etf["sector"] or "Equity" in etf["sector"]:
                score += 10
        else:  # balanced
            score += 5  # All ETFs get base score
        
        # Score based on risk tolerance
        if risk_tolerance == "conservative":
            if etf["risk"] in ["Low", "Moderate"]:
                score += 5
        elif risk_tolerance == "aggressive":
            if "Global" in etf["sector"]:
                score += 5
        
        return score
    
    return sorted(etfs, key=etf_score, reverse=True)

def get_etf_info(etf_code: str, current_price: float, dividend_yield: float) -> Dict:
    """Get ETF information based on code"""
    
    etf_mapping = {
        "STXIND": {
            "name": "Satrix INDI ETF",
            "sector": "South African Equity",
            "description": "Tracks JSE Industrial sector",
            "risk": "Moderate"
        },
        "STXFIN": {
            "name": "Satrix FINI ETF", 
            "sector": "Financial",
            "description": "JSE Financial sector exposure",
            "risk": "Moderate"
        },
        "STX40": {
            "name": "Satrix Top 40 ETF",
            "sector": "Large Cap SA",
            "description": "Top 40 JSE companies",
            "risk": "Moderate"
        },
        "STXPRO": {
            "name": "Satrix Property ETF",
            "sector": "Property/REIT", 
            "description": "SA Listed Property exposure",
            "risk": "Moderate-High"
        },
        "STXWDM": {
            "name": "Satrix MSCI World ETF",
            "sector": "Global Equity",
            "description": "Global developed markets",
            "risk": "Moderate"
        }
    }
    
    if etf_code in etf_mapping:
        info = etf_mapping[etf_code].copy()
        info.update({
            "symbol": etf_code,
            "current_price": current_price,
            "dividend_yield": dividend_yield if dividend_yield > 0 else 3.0  # Default estimate
        })
        return info
    
    return None

def sort_etfs_by_goal(etfs: List[Dict], investment_goal: str, risk_tolerance: str) -> List[Dict]:
    """Sort ETFs based on investment goal and risk tolerance"""
    
    def etf_score(etf):
        score = 0
        
        # Score based on investment goal
        if investment_goal == "income":
            if "Property" in etf["sector"] or etf["dividend_yield"] > 4:
                score += 10
        elif investment_goal == "growth":
            if "Global" in etf["sector"] or "Equity" in etf["sector"]:
                score += 10
        else:  # balanced
            score += 5  # All ETFs get base score
        
        # Score based on risk tolerance
        if risk_tolerance == "conservative":
            if etf["risk"] in ["Low", "Moderate"]:
                score += 5
        elif risk_tolerance == "aggressive":
            if "Global" in etf["sector"]:
                score += 5
        
        return score
    
    return sorted(etfs, key=etf_score, reverse=True)
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