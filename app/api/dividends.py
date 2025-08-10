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

class DividendCalendarResponse(BaseModel):
    symbol: str
    company_name: str
    dividend_amount: Optional[float]
    dividend_type: str
    announcement_date: Optional[str]
    last_day_to_register: Optional[str]
    ex_dividend_date: Optional[str]
    record_date: Optional[str]
    payment_date: Optional[str]
    days_until_ldr: Optional[int]
    days_until_payment: Optional[int]
    can_still_buy: bool
    trading_window: str
    urgency: str

class DividendTimingAnalysis(BaseModel):
    symbol: str
    recommendation: str  # BUY_NOW, LAST_CHANCE, TOO_LATE, WAIT
    days_remaining: Optional[int]
    urgency: str  # HIGH, MEDIUM, LOW
    action_required: str

class DividendAnalysisRequest(BaseModel):
    symbol: str
    analysis_focus: str = "sustainability"  # sustainability, growth, yield

class HoldingForecast(BaseModel):
    symbol: str
    company_name: str
    current_value: float
    shares_held: float
    dividend_yield: float
    annual_dividend_per_share: float
    estimated_annual_income: float
    data_source: str
    last_updated: str

class IncomeForecastResponse(BaseModel):
    account_id: str
    total_estimated_annual_income: float
    monthly_income_estimate: float
    quarterly_income_estimate: float
    portfolio_average_yield: float
    holdings_forecast: List[HoldingForecast]
    forecast_date: str
    data_quality: Dict
    note: str

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
        annual_estimate = total_dividends * 2 if len(dividend_payments) > 0 else 0
        
        logger.info(f"Retrieved {len(dividend_payments)} dividend payments for account {account_id}")
        
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
async def get_enhanced_dividend_calendar(months_ahead: int = Query(6, ge=1, le=12)):
    """
    Get comprehensive dividend calendar with buy/sell timing
    """
    try:
        if not dividend_service:
            # Fallback to basic mock data if service not available
            return {
                "message": "Enhanced dividend calendar service not available",
                "basic_events": [
                    {
                        "symbol": "SBK",
                        "company_name": "Standard Bank Group Ltd",
                        "ex_dividend_date": "2024-09-16",
                        "payment_date": "2024-09-30",
                        "dividend_amount": 12.50,
                        "trading_window": "Check manually"
                    }
                ]
            }
        
        calendar_events = await dividend_service.get_comprehensive_dividend_calendar(months_ahead)
        
        # Format for API response
        formatted_events = []
        for event in calendar_events:
            urgency = "LOW"
            if event.days_until_ldr is not None:
                if event.days_until_ldr <= 1:
                    urgency = "HIGH"
                elif event.days_until_ldr <= 5:
                    urgency = "MEDIUM"
            
            formatted_event = DividendCalendarResponse(
                symbol=event.symbol,
                company_name=event.company_name,
                dividend_amount=event.dividend_amount,
                dividend_type=event.dividend_type,
                announcement_date=event.announcement_date.isoformat() if event.announcement_date else None,
                last_day_to_register=event.last_day_to_register.isoformat() if event.last_day_to_register else None,
                ex_dividend_date=event.ex_dividend_date.isoformat() if event.ex_dividend_date else None,
                record_date=event.record_date.isoformat() if event.record_date else None,
                payment_date=event.payment_date.isoformat() if event.payment_date else None,
                days_until_ldr=event.days_until_ldr,
                days_until_payment=event.days_until_payment,
                can_still_buy=event.can_still_buy,
                trading_window=event.trading_window,
                urgency=urgency
            )
            formatted_events.append(formatted_event)
        
        # Categorize events
        buy_now = [e for e in formatted_events if e.can_still_buy and e.days_until_ldr and e.days_until_ldr > 0]
        last_chance = [e for e in formatted_events if e.can_still_buy and e.days_until_ldr == 0]
        upcoming_payments = [e for e in formatted_events if e.days_until_payment and 0 <= e.days_until_payment <= 30]
        
        logger.info(f"Retrieved {len(formatted_events)} dividend events")
        
        return {
            "all_events": formatted_events,
            "buy_opportunities": buy_now,
            "last_chance_today": last_chance,
            "upcoming_payments": upcoming_payments,
            "summary": {
                "total_events": len(formatted_events),
                "buy_opportunities": len(buy_now),
                "urgent_actions": len(last_chance),
                "upcoming_payments": len(upcoming_payments)
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get dividend calendar: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get calendar: {str(e)}")

@router.get("/timing/{symbol}")
async def get_dividend_timing_analysis(symbol: str):
    """
    Get detailed dividend timing analysis for specific stock
    """
    try:
        if not dividend_service:
            return {"error": "Dividend timing service not available"}
        
        timing_analysis = await dividend_service.analyze_dividend_timing(symbol.upper())
        
        if "error" in timing_analysis:
            raise HTTPException(status_code=404, detail=timing_analysis["error"])
        
        logger.info(f"Generated dividend timing analysis for {symbol}")
        return timing_analysis
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dividend timing analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.get("/opportunities")
async def get_dividend_opportunities(holdings: Optional[str] = Query(None, description="Comma-separated list of stock symbols")):
    """
    Get dividend opportunities based on user holdings
    """
    try:
        if not dividend_service:
            return {"error": "Dividend opportunities service not available"}
        
        user_holdings = holdings.split(",") if holdings else None
        
        opportunities = await dividend_service.get_dividend_opportunities(user_holdings)
        
        logger.info(f"Retrieved dividend opportunities")
        return opportunities
        
    except Exception as e:
        logger.error(f"Failed to get opportunities: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get opportunities: {str(e)}")

@router.get("/alerts")
async def get_dividend_alerts():
    """
    Get urgent dividend alerts (last chance to buy, payment due, etc.)
    """
    try:
        if not dividend_service:
            return {"alerts": [], "message": "Alert service not available"}
        
        calendar_events = await dividend_service.get_comprehensive_dividend_calendar(3)  # Next 3 months
        
        # Find urgent alerts
        alerts = []
        
        for event in calendar_events:
            if event.days_until_ldr is not None:
                if event.days_until_ldr == 0:
                    alerts.append({
                        "type": "LAST_CHANCE",
                        "symbol": event.symbol,
                        "company_name": event.company_name,
                        "message": f"Last day to buy {event.symbol} for R{event.dividend_amount} dividend",
                        "urgency": "HIGH",
                        "dividend_amount": event.dividend_amount,
                        "ex_date": event.ex_dividend_date.isoformat() if event.ex_dividend_date else None
                    })
                elif 1 <= event.days_until_ldr <= 3:
                    alerts.append({
                        "type": "BUY_SOON",
                        "symbol": event.symbol,
                        "company_name": event.company_name,
                        "message": f"Only {event.days_until_ldr} days left to buy {event.symbol} for dividend",
                        "urgency": "MEDIUM",
                        "dividend_amount": event.dividend_amount,
                        "days_remaining": event.days_until_ldr
                    })
            
            if event.days_until_payment is not None and 0 <= event.days_until_payment <= 7:
                alerts.append({
                    "type": "PAYMENT_DUE",
                    "symbol": event.symbol,
                    "company_name": event.company_name,
                    "message": f"Dividend payment of R{event.dividend_amount} due in {event.days_until_payment} days",
                    "urgency": "LOW",
                    "dividend_amount": event.dividend_amount,
                    "days_until_payment": event.days_until_payment
                })
        
        # Sort by urgency
        urgency_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        alerts.sort(key=lambda x: urgency_order.get(x["urgency"], 3))
        
        logger.info(f"Generated {len(alerts)} dividend alerts")
        
        return {
            "alerts": alerts,
            "alert_count": len(alerts),
            "high_priority": len([a for a in alerts if a["urgency"] == "HIGH"]),
            "medium_priority": len([a for a in alerts if a["urgency"] == "MEDIUM"])
        }
        
    except Exception as e:
        logger.error(f"Failed to get alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {str(e)}")

@router.get("/income-forecast/{account_id}", response_model=IncomeForecastResponse)
async def forecast_dividend_income(account_id: str, credentials: dict = Depends(get_ee_credentials)):
    """
    Forecast future dividend income based on current holdings with REAL market data
    """
    try:
        from easy_equities_client.clients import EasyEquitiesClient
        from app.api.analysis import analyzer  # Import your enhanced analyzer
        
        client = EasyEquitiesClient()
        client.login(username=credentials["username"], password=credentials["password"])
        
        # Get current holdings
        holdings = client.accounts.holdings(account_id, include_shares=True)
        
        if not holdings:
            return IncomeForecastResponse(
                account_id=account_id,
                total_estimated_annual_income=0.0,
                monthly_income_estimate=0.0,
                quarterly_income_estimate=0.0,
                portfolio_average_yield=0.0,
                holdings_forecast=[],
                forecast_date=datetime.now().isoformat(),
                data_quality={"live_data_count": 0, "estimated_count": 0, "accuracy": "no_data"},
                note="No holdings found for income forecast"
            )
        
        total_annual_income = 0.0
        holdings_forecast = []
        live_data_count = 0
        estimated_count = 0
        
        for holding in holdings:
            contract_code = holding.get('contract_code', '')
            symbol = contract_code.replace('EQU.ZA.', '') if contract_code.startswith('EQU.ZA.') else contract_code
            
            current_value_str = holding.get('current_value', 'R0.00')
            current_value = float(current_value_str.replace('R', '').replace(',', '').strip())
            
            # GET REAL DIVIDEND YIELD DATA
            try:
                # Use your enhanced stock analyzer to get real yield
                stock_data = await analyzer.get_jse_stock_data(symbol)
                dividend_yield = stock_data.get('dividend_yield', 0)
                annual_dividend = stock_data.get('annual_dividend', 0)
                
                # Calculate income two ways for validation
                income_from_yield = current_value * (dividend_yield / 100)
                
                # If we have shares data, calculate more precisely
                shares_held = float(holding.get('shares', '0') or '0')
                income_from_shares = shares_held * annual_dividend if shares_held > 0 else income_from_yield
                
                # Use the more accurate calculation
                estimated_annual_income = income_from_shares if shares_held > 0 else income_from_yield
                
                data_source = "live_market_data"
                live_data_count += 1
                
            except Exception as e:
                logger.warning(f"Failed to get real yield for {symbol}: {e}")
                # Fallback to conservative estimate
                dividend_yield = 4.0  # Conservative default
                annual_dividend = 0
                shares_held = float(holding.get('shares', '0') or '0')
                estimated_annual_income = current_value * 0.04
                data_source = "estimated"
                estimated_count += 1
            
            total_annual_income += estimated_annual_income
            
            holdings_forecast.append(HoldingForecast(
                symbol=symbol,
                company_name=holding.get('name', ''),
                current_value=current_value,
                shares_held=shares_held,
                dividend_yield=dividend_yield,
                annual_dividend_per_share=annual_dividend,
                estimated_annual_income=round(estimated_annual_income, 2),
                data_source=data_source,
                last_updated=datetime.now().isoformat()
            ))
        
        # Sort by income contribution
        holdings_forecast.sort(key=lambda x: x.estimated_annual_income, reverse=True)
        
        # Calculate additional metrics
        total_portfolio_value = sum(h.current_value for h in holdings_forecast)
        average_yield = (total_annual_income / total_portfolio_value) * 100 if total_portfolio_value > 0 else 0
        monthly_income = total_annual_income / 12
        quarterly_income = total_annual_income / 4
        
        # Data quality assessment
        total_holdings = len(holdings_forecast)
        accuracy = "high" if live_data_count > total_holdings * 0.8 else "medium" if live_data_count > total_holdings * 0.5 else "low"
        
        logger.info(f"Generated REAL dividend income forecast for account {account_id}")
        
        return IncomeForecastResponse(
            account_id=account_id,
            total_estimated_annual_income=round(total_annual_income, 2),
            monthly_income_estimate=round(monthly_income, 2),
            quarterly_income_estimate=round(quarterly_income, 2),
            portfolio_average_yield=round(average_yield, 2),
            holdings_forecast=holdings_forecast,
            forecast_date=datetime.now().isoformat(),
            data_quality={
                "live_data_count": live_data_count,
                "estimated_count": estimated_count,
                "accuracy": accuracy
            },
            note="Estimates based on current market yields and actual share holdings. Dividends may vary based on company performance."
        )
        
    except Exception as e:
        logger.error(f"Income forecast failed: {e}")
        raise HTTPException(status_code=500, detail=f"Income forecast failed: {str(e)}")

@router.get("/real-yield/{symbol}")
async def get_real_dividend_yield(symbol: str):
    """
    Get current real dividend yield for any JSE stock
    """
    try:
        from app.api.analysis import analyzer
        
        stock_data = await analyzer.get_jse_stock_data(symbol.upper())
        
        return {
            "symbol": symbol.upper(),
            "company_name": stock_data.get('company_name'),
            "current_price": stock_data.get('current_price'),
            "dividend_yield": stock_data.get('dividend_yield'),
            "annual_dividend": stock_data.get('annual_dividend'),
            "dividend_growth_5yr": stock_data.get('dividend_growth_5yr'),
            "payout_ratio": stock_data.get('payout_ratio'),
            "last_updated": datetime.now().isoformat(),
            "data_source": "yahoo_finance_live"
        }
        
    except Exception as e:
        logger.error(f"Failed to get real yield for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get yield data: {str(e)}")

@router.post("/bulk-yields")
async def get_bulk_dividend_yields(symbols: List[str]):
    """
    Get real dividend yields for multiple stocks at once
    """
    try:
        from app.api.analysis import analyzer
        
        results = []
        
        for symbol in symbols:
            try:
                stock_data = await analyzer.get_jse_stock_data(symbol.upper())
                results.append({
                    "symbol": symbol.upper(),
                    "dividend_yield": stock_data.get('dividend_yield', 0),
                    "annual_dividend": stock_data.get('annual_dividend', 0),
                    "current_price": stock_data.get('current_price', 0),
                    "success": True
                })
            except Exception as e:
                results.append({
                    "symbol": symbol.upper(),
                    "dividend_yield": 0,
                    "annual_dividend": 0,
                    "current_price": 0,
                    "success": False,
                    "error": str(e)
                })
        
        return {
            "results": results,
            "total_requested": len(symbols),
            "successful": len([r for r in results if r['success']]),
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Bulk yield lookup failed: {e}")
        raise HTTPException(status_code=500, detail=f"Bulk lookup failed: {str(e)}")

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