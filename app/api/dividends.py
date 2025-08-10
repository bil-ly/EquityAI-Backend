from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime, date
import os
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

try:
    from app.services.dividend_calendar_service import DividendCalendarService, DividendEvent
    dividend_service = DividendCalendarService()
except ImportError:
    logger.warning("Dividend calendar service not available - using basic functionality")
    dividend_service = None

# Pydantic models
class DividendPayment(BaseModel):
    transaction_id: int
    symbol: str
    company_name: str
    amount: float
    payment_date: str
    comment: str
    shares_at_time: Optional[float] = None

class DividendSummary(BaseModel):
    account_id: str
    total_dividends_received: float
    dividend_payments_count: int
    top_dividend_payers: List[dict]
    annual_dividend_income: float
    dividend_payments: List[DividendPayment]

class DividendCalendarEvent(BaseModel):
    symbol: str
    company_name: str
    ex_dividend_date: Optional[date] = None
    payment_date: Optional[date] = None
    dividend_amount: Optional[float] = None
    estimated: bool = True

class DividendAnalysisRequest(BaseModel):
    symbol: str
    analysis_focus: str = "sustainability"  # sustainability, growth, yield

# Dependency for EasyEquities credentials
def get_ee_credentials():
    username = os.getenv("EASYEQUITIES_USERNAME")
    password = os.getenv("EASYEQUITIES_PASSWORD")
    
    if not username or not password:
        raise HTTPException(
            status_code=500, 
            detail="EasyEquities credentials not configured"
        )
    
    return {"username": username, "password": password}

@router.get("/history/{account_id}", response_model=DividendSummary)
async def get_dividend_history(account_id: str, credentials: dict = Depends(get_ee_credentials)):
    """
    Get dividend payment history for an account
    """
    try:
        from easy_equities_client.clients import EasyEquitiesClient
        
        client = EasyEquitiesClient()
        client.login(username=credentials["username"], password=credentials["password"])
        
        # Get all transactions
        transactions = client.accounts.transactions(account_id)
        
        # Filter dividend transactions
        dividend_payments = []
        dividend_by_symbol = {}
        total_dividends = 0.0
        
        for transaction in transactions:
            action = transaction.get('Action', '').lower()
            comment = transaction.get('Comment', '').lower()
            
            if 'dividend' in action or 'dividend' in comment:
                amount = transaction.get('DebitCredit', 0)
                if amount > 0:  # Only positive amounts (received dividends)
                    # Extract symbol from contract code or comment
                    contract_code = transaction.get('ContractCode', '')
                    symbol = contract_code.replace('EQU.ZA.', '') if contract_code.startswith('EQU.ZA.') else ''
                    
                    # If no symbol from contract code, try to extract from comment
                    if not symbol:
                        comment_text = transaction.get('Comment', '')
                        # Try to extract symbol from comment patterns
                        import re
                        symbol_match = re.search(r'([A-Z]{2,5})', comment_text)
                        symbol = symbol_match.group(1) if symbol_match else 'UNKNOWN'
                    
                    company_name = transaction.get('Comment', '').split('-')[0].strip() if '-' in transaction.get('Comment', '') else symbol
                    
                    dividend_payment = DividendPayment(
                        transaction_id=transaction.get('TransactionId', 0),
                        symbol=symbol,
                        company_name=company_name,
                        amount=amount,
                        payment_date=transaction.get('TransactionDate', ''),
                        comment=transaction.get('Comment', '')
                    )
                    
                    dividend_payments.append(dividend_payment)
                    total_dividends += amount
                    
                    # Track by symbol
                    if symbol in dividend_by_symbol:
                        dividend_by_symbol[symbol]['total'] += amount
                        dividend_by_symbol[symbol]['count'] += 1
                    else:
                        dividend_by_symbol[symbol] = {
                            'symbol': symbol,
                            'company_name': company_name,
                            'total': amount,
                            'count': 1
                        }
        
        # Get top dividend payers
        top_payers = sorted(
            dividend_by_symbol.values(), 
            key=lambda x: x['total'], 
            reverse=True
        )[:5]
        
        # Estimate annual dividend income (simple approach)
        # This is rough - in production, you'd want more sophisticated calculation
        annual_estimate = total_dividends * 2 if len(dividend_payments) > 0 else 0
        
        logger.info(f"✅ Retrieved {len(dividend_payments)} dividend payments for account {account_id}")
        
        return DividendSummary(
            account_id=account_id,
            total_dividends_received=round(total_dividends, 2),
            dividend_payments_count=len(dividend_payments),
            top_dividend_payers=top_payers,
            annual_dividend_income=round(annual_estimate, 2),
            dividend_payments=dividend_payments
        )
        
    except Exception as e:
        logger.error(f"Failed to get dividend history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get dividend history: {str(e)}")

@router.get("/calendar")
async def get_dividend_calendar():
    """
    Get upcoming dividend calendar (mock data for now)
    In production, this would scrape JSE or financial data providers
    """
    try:
        # Mock dividend calendar data
        # In production, you'd scrape this from JSE, Sharenet, or other sources
        mock_calendar = [
            DividendCalendarEvent(
                symbol="SBK",
                company_name="Standard Bank Group Ltd",
                ex_dividend_date=date(2024, 9, 15),
                payment_date=date(2024, 9, 30),
                dividend_amount=11.50,
                estimated=True
            ),
            DividendCalendarEvent(
                symbol="MTN",
                company_name="MTN Group Ltd",
                ex_dividend_date=date(2024, 8, 20),
                payment_date=date(2024, 9, 5),
                dividend_amount=7.80,
                estimated=True
            ),
            DividendCalendarEvent(
                symbol="SHP",
                company_name="Shoprite Holdings Ltd",
                ex_dividend_date=date(2024, 10, 5),
                payment_date=date(2024, 10, 20),
                dividend_amount=6.45,
                estimated=True
            ),
            DividendCalendarEvent(
                symbol="NPN",
                company_name="Naspers Ltd",
                ex_dividend_date=date(2024, 9, 25),
                payment_date=date(2024, 10, 10),
                dividend_amount=25.00,
                estimated=True
            )
        ]
        
        logger.info("Retrieved dividend calendar (mock data)")
        
        return {
            "calendar_events": mock_calendar,
            "total_events": len(mock_calendar),
            "data_source": "mock",
            "note": "This is sample data. Production version would fetch real dividend calendar."
        }
        
    except Exception as e:
        logger.error(f"Failed to get dividend calendar: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get calendar: {str(e)}")

@router.post("/analyze")
async def analyze_dividend_stock(request: DividendAnalysisRequest):
    """
    Get AI analysis of a stock's dividend prospects
    """
    try:
        # Import Claude function from analysis router
        from app.api.analysis import call_claude_api
        
        # Create dividend-specific analysis prompt
        prompt = f"""
        As a dividend investing specialist, analyze {request.symbol} (JSE-listed stock) for dividend investment:

        Focus on {request.analysis_focus}:

        1. DIVIDEND SUSTAINABILITY:
           - Payout ratio and dividend coverage
           - Free cash flow vs dividend payments
           - Debt levels and financial stability
           - Earnings consistency and predictability

        2. DIVIDEND GROWTH PROSPECTS:
           - Historical dividend growth rate (5+ years)
           - Management's dividend policy and commitment
           - Business growth drivers supporting dividend increases
           - Capital allocation priorities

        3. YIELD ATTRACTIVENESS:
           - Current dividend yield vs sector average
           - Yield vs 10-year government bonds
           - Risk-adjusted yield assessment
           - Comparison to dividend aristocrats

        4. RISKS TO DIVIDENDS:
           - Cyclical business model risks
           - Regulatory or sector headwinds
           - Capital intensity requirements
           - Economic sensitivity

        Provide:
        - Dividend sustainability score (1-10)
        - Expected dividend growth rate (next 3 years)
        - Fair dividend yield range
        - Key catalysts and risks
        - Investment recommendation for dividend investors

        Consider South African economic conditions, rand strength, and JSE sector dynamics.
        """
        
        analysis = await call_claude_api(prompt, max_tokens=1500)
        
        logger.info(f"Generated dividend analysis for {request.symbol}")
        
        return {
            "symbol": request.symbol,
            "analysis_type": f"dividend_{request.analysis_focus}",
            "analysis": analysis,
            "focus": request.analysis_focus,
            "analysis_date": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Dividend analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Dividend analysis failed: {str(e)}")

@router.get("/yield-scanner")
async def scan_high_yield_stocks():
    """
    Scan JSE for high-yield dividend opportunities
    """
    try:
        from app.api.analysis import call_claude_api
        
        prompt = """
        As a dividend yield specialist, identify the best high-yield dividend opportunities on the JSE:

        Criteria:
        - Dividend yield > 6%
        - Sustainable business models
        - Reasonable payout ratios (<80%)
        - Positive free cash flow
        - Market cap > R5 billion (liquidity)

        For each recommendation:
        1. Stock symbol and current yield
        2. Why the yield is sustainable
        3. Key business strengths
        4. Main dividend risks
        5. Entry price recommendation

        Focus on:
        - REITs with strong property portfolios
        - Banks with solid capital ratios
        - Utilities with regulated income
        - Resource companies with long-life assets
        - Consumer staples with pricing power

        Avoid dividend traps and distressed situations.
        Rank by risk-adjusted yield attractiveness.
        """
        
        analysis = await call_claude_api(prompt, max_tokens=2000)
        
        logger.info("Generated high-yield dividend scan")
        
        return {
            "scan_type": "high_yield_dividends",
            "market": "JSE",
            "minimum_yield": 6.0,
            "analysis": analysis,
            "scan_date": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Yield scanner failed: {e}")
        raise HTTPException(status_code=500, detail=f"Yield scanner failed: {str(e)}")

@router.get("/income-forecast/{account_id}")
async def forecast_dividend_income(account_id: str, credentials: dict = Depends(get_ee_credentials)):
    """
    Forecast future dividend income based on current holdings
    """
    try:
        from easy_equities_client.clients import EasyEquitiesClient
        
        client = EasyEquitiesClient()
        client.login(username=credentials["username"], password=credentials["password"])
        
        # Get current holdings
        holdings = client.accounts.holdings(account_id, include_shares=True)
        
        if not holdings:
            return {
                "message": "No holdings found for income forecast",
                "forecast": {}
            }
        
        # Mock dividend yield data (in production, fetch real yield data)
        mock_yields = {
            "SBK": 6.2,
            "MTN": 8.1,
            "SHP": 4.5,
            "NPN": 2.1,
            "AGL": 5.8,
            "BTI": 7.2
        }
        
        total_annual_income = 0.0
        holdings_forecast = []
        
        for holding in holdings:
            contract_code = holding.get('contract_code', '')
            symbol = contract_code.replace('EQU.ZA.', '') if contract_code.startswith('EQU.ZA.') else contract_code
            
            current_value_str = holding.get('current_value', 'R0.00')
            current_value = float(current_value_str.replace('R', '').replace(',', '').strip())
            
            # Use mock yield or default 4%
            dividend_yield = mock_yields.get(symbol, 4.0)
            annual_income = current_value * (dividend_yield / 100)
            total_annual_income += annual_income
            
            holdings_forecast.append({
                "symbol": symbol,
                "company_name": holding.get('name', ''),
                "current_value": current_value,
                "dividend_yield": dividend_yield,
                "estimated_annual_income": round(annual_income, 2)
            })
        
        # Sort by income contribution
        holdings_forecast.sort(key=lambda x: x['estimated_annual_income'], reverse=True)
        
        logger.info(f"✅ Generated dividend income forecast for account {account_id}")
        
        return {
            "account_id": account_id,
            "total_estimated_annual_income": round(total_annual_income, 2),
            "monthly_income_estimate": round(total_annual_income / 12, 2),
            "holdings_forecast": holdings_forecast,
            "forecast_date": datetime.now().isoformat(),
            "note": "Estimates based on current yields and holdings. Actual income may vary."
        }
        
    except Exception as e:
        logger.error(f"❌ Income forecast failed: {e}")
        raise HTTPException(status_code=500, detail=f"Income forecast failed: {str(e)}")