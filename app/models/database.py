from beanie import Document, Indexed
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime, date
from enum import Enum

class RecommendationType(str, Enum):
    BUY = "BUY"
    SELL = "SELL" 
    HOLD = "HOLD"

class AnalysisType(str, Enum):
    STOCK = "stock"
    PORTFOLIO = "portfolio"
    DIVIDEND = "dividend"

class User(Document):
    username: str = Indexed(unique=True)
    email: Optional[str] = None
    easyequities_username: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Settings:
        name = "users"

class Stock(Document):
    symbol: str = Indexed(unique=True)
    name: str
    sector: Optional[str] = None
    exchange: str = "JSE"
    current_price: Optional[float] = None
    dividend_yield: Optional[float] = None
    last_dividend: Optional[float] = None
    ex_dividend_date: Optional[date] = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Settings:
        name = "stocks"

class Portfolio(Document):
    user_id: str = Indexed()
    account_id: str = Indexed()
    account_name: str
    currency: str = "ZAR"
    total_value: float = 0.0
    total_cost: float = 0.0
    profit_loss: float = 0.0
    profit_loss_percent: float = 0.0
    annual_dividend_income: float = 0.0
    holdings: List[dict] = []
    last_synced: datetime = Field(default_factory=datetime.utcnow)
    
    class Settings:
        name = "portfolios"

class Transaction(Document):
    user_id: str = Indexed()
    account_id: str = Indexed()
    transaction_id: str
    symbol: Optional[str] = None
    action: str 
    quantity: Optional[float] = None
    price: Optional[float] = None
    amount: float
    comment: str
    date: datetime = Indexed()
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Settings:
        name = "transactions"

class Dividend(Document):
    user_id: str = Indexed()
    symbol: str = Indexed()
    company_name: str
    dividend_amount: float
    shares_held: float
    total_received: float
    ex_dividend_date: date = Indexed()
    payment_date: Optional[date] = None
    announcement_date: Optional[date] = None
    dividend_yield: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Settings:
        name = "dividends"

class Analysis(Document):
    user_id: str = Indexed()
    symbol: Optional[str] = None
    analysis_type: AnalysisType
    recommendation: Optional[RecommendationType] = None
    confidence_score: Optional[float] = None
    reasoning: str
    key_points: List[str] = []
    target_price: Optional[float] = None
    risk_level: Optional[str] = None
    prompt_used: str
    response_tokens: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    
    class Settings:
        name = "analyses"

class PortfolioSummary(BaseModel):
    total_value: float
    total_cost: float
    profit_loss: float
    profit_loss_percent: float
    annual_dividend_income: float
    holdings_count: int
    top_performers: List[dict]
    dividend_yield: float

class StockAnalysisResponse(BaseModel):
    symbol: str
    recommendation: RecommendationType
    confidence_score: float
    reasoning: str
    key_points: List[str]
    target_price: Optional[float]
    risk_level: str
    current_price: float
    analysis_date: datetime

class DividendInsight(BaseModel):
    symbol: str
    sustainability_score: float
    growth_trend: str
    next_expected_date: Optional[date]
    yield_attractiveness: str
    risk_factors: List[str]



class Stock(Document):
    """Stock document model for MongoDB"""
    symbol: Indexed(str, unique=True)  # Unique index on symbol
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
    
    # Additional fields from Synatic
    industry: Optional[str] = None
    description: Optional[str] = None
    eps: Optional[float] = None
    price_52w_high: Optional[float] = None
    price_52w_low: Optional[float] = None
    beta: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    debt_to_equity: Optional[float] = None
    gross_margin: Optional[float] = None
    net_margin: Optional[float] = None
    
    # Metadata
    last_updated: datetime = Field(default_factory=datetime.now)
    data_source: str = "synatic"  # Track where data came from
    
    class Settings:
        name = "stocks"
        indexes = [
            "symbol",
            "market",
            "sector",
            "currency",
            [("symbol", 1), ("market", 1)]  # Compound index
        ]

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "NPN",
                "name": "Naspers Ltd",
                "contract_code": "EQU.ZA.NPN",
                "sector": "Technology",
                "market": "JSE",
                "currency": "ZAR",
                "last_price": 2850.00,
                "market_cap": 1200000000000
            }
        }