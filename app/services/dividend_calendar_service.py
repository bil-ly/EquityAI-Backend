from datetime import datetime, date, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class DividendEvent:
    symbol: str
    company_name: str
    announcement_date: Optional[date] = None
    last_day_to_register: Optional[date] = None
    ex_dividend_date: Optional[date] = None
    record_date: Optional[date] = None
    payment_date: Optional[date] = None
    dividend_amount: Optional[float] = None
    dividend_type: str = "interim"
    currency: str = "ZAR"
    
    # Calculated fields
    days_until_ldr: Optional[int] = None
    days_until_ex_date: Optional[int] = None
    days_until_payment: Optional[int] = None
    can_still_buy: bool = False
    trading_window: str = ""
    
    def calculate_timing(self):
        """Calculate buy/sell timing based on current date"""
        today = date.today()
        
        if self.last_day_to_register:
            self.days_until_ldr = (self.last_day_to_register - today).days
            self.can_still_buy = self.days_until_ldr >= 0
            
            if self.days_until_ldr > 0:
                self.trading_window = f"BUY NOW ({self.days_until_ldr} days left)"
            elif self.days_until_ldr == 0:
                self.trading_window = "LAST DAY TO BUY"
            else:
                self.trading_window = "TOO LATE - Already ex-dividend"
        
        if self.ex_dividend_date:
            self.days_until_ex_date = (self.ex_dividend_date - today).days
            
        if self.payment_date:
            self.days_until_payment = (self.payment_date - today).days

class DividendCalendarService:
    """Simple dividend calendar with mock JSE data"""
    
    async def get_comprehensive_dividend_calendar(self, months_ahead: int = 6) -> List[DividendEvent]:
        """Get mock dividend calendar data"""
        
        # Creating realistic future dates
        today = date.today()
        
        mock_events = [
            DividendEvent(
                symbol="SBK",
                company_name="Standard Bank Group Ltd",
                announcement_date=today + timedelta(days=5),
                last_day_to_register=today + timedelta(days=25),
                ex_dividend_date=today + timedelta(days=26),
                record_date=today + timedelta(days=28),
                payment_date=today + timedelta(days=40),
                dividend_amount=12.50,
                dividend_type="interim"
            ),
            DividendEvent(
                symbol="MTN",
                company_name="MTN Group Ltd",
                announcement_date=today + timedelta(days=10),
                last_day_to_register=today + timedelta(days=30),
                ex_dividend_date=today + timedelta(days=31),
                record_date=today + timedelta(days=33),
                payment_date=today + timedelta(days=45),
                dividend_amount=8.00,
                dividend_type="interim"
            ),
            DividendEvent(
                symbol="SHP",
                company_name="Shoprite Holdings Ltd",
                announcement_date=today + timedelta(days=15),
                last_day_to_register=today + timedelta(days=35),
                ex_dividend_date=today + timedelta(days=36),
                record_date=today + timedelta(days=38),
                payment_date=today + timedelta(days=50),
                dividend_amount=6.75,
                dividend_type="final"
            ),
            DividendEvent(
                symbol="NPN",
                company_name="Naspers Ltd",
                announcement_date=today + timedelta(days=20),
                last_day_to_register=today + timedelta(days=40),
                ex_dividend_date=today + timedelta(days=41),
                record_date=today + timedelta(days=43),
                payment_date=today + timedelta(days=55),
                dividend_amount=25.00,
                dividend_type="interim"
            )
        ]
        
        # Calculate timing for each event
        for event in mock_events:
            event.calculate_timing()
        
        # Filter by months ahead
        cutoff_date = today + timedelta(days=months_ahead * 30)
        filtered_events = [
            event for event in mock_events 
            if event.ex_dividend_date and event.ex_dividend_date <= cutoff_date
        ]
        
        return filtered_events
    
    async def analyze_dividend_timing(self, symbol: str) -> Dict:
        """Analyze timing for specific stock"""
        
        calendar = await self.get_comprehensive_dividend_calendar()
        stock_events = [event for event in calendar if event.symbol == symbol.upper()]
        
        if not stock_events:
            return {"error": f"No dividend events found for {symbol}"}
        
        next_event = stock_events[0]
        
        return {
            "symbol": symbol,
            "company_name": next_event.company_name,
            "next_dividend": {
                "amount": next_event.dividend_amount,
                "type": next_event.dividend_type,
                "ex_date": next_event.ex_dividend_date.isoformat() if next_event.ex_dividend_date else None,
                "payment_date": next_event.payment_date.isoformat() if next_event.payment_date else None
            },
            "trading_action": {
                "recommendation": next_event.trading_window,
                "can_still_buy": next_event.can_still_buy,
                "days_remaining": next_event.days_until_ldr,
                "urgency": "HIGH" if next_event.days_until_ldr and next_event.days_until_ldr <= 2 else "MEDIUM" if next_event.days_until_ldr and next_event.days_until_ldr <= 7 else "LOW"
            },
            "important_dates": {
                "last_day_to_register": next_event.last_day_to_register.isoformat() if next_event.last_day_to_register else None,
                "ex_dividend_date": next_event.ex_dividend_date.isoformat() if next_event.ex_dividend_date else None,
                "record_date": next_event.record_date.isoformat() if next_event.record_date else None,
                "payment_date": next_event.payment_date.isoformat() if next_event.payment_date else None
            }
        }
    
    async def get_dividend_opportunities(self, user_holdings: List[str] = None) -> Dict:
        """Get dividend opportunities"""
        
        calendar = await self.get_comprehensive_dividend_calendar()
        
        if user_holdings:
            holding_events = [event for event in calendar if event.symbol in user_holdings]
        else:
            holding_events = []
        
        buy_now = [event for event in calendar if event.can_still_buy and event.days_until_ldr and event.days_until_ldr > 0]
        last_chance = [event for event in calendar if event.can_still_buy and event.days_until_ldr == 0]
        upcoming_payments = [event for event in calendar if event.days_until_payment and 0 <= event.days_until_payment <= 30]
        
        return {
            "your_holdings": holding_events,
            "buy_opportunities": buy_now[:10],
            "last_chance_today": last_chance,
            "upcoming_payments": upcoming_payments,
            "total_events": len(calendar)
        }