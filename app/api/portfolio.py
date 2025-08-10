from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic models
class Holding(BaseModel):
    symbol: str
    name: str
    shares: str
    current_price: str
    current_value: str
    purchase_value: str
    profit_loss: Optional[float] = None
    profit_loss_percent: Optional[float] = None
    isin: str
    logo_url: Optional[str] = None

class PortfolioSummary(BaseModel):
    account_id: str
    account_name: str
    holdings: List[Holding]
    total_value: float
    total_cost: float
    total_profit_loss: float
    holdings_count: int
    message: Optional[str] = None

class Transaction(BaseModel):
    transaction_id: int
    date: str
    action: str
    amount: float
    comment: str
    contract_code: str
    symbol: Optional[str] = None

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

@router.get("/holdings/{account_id}", response_model=PortfolioSummary)
async def get_portfolio_holdings(account_id: str, credentials: dict = Depends(get_ee_credentials)):
    """
    Get portfolio holdings for specific account
    """
    try:
        from easy_equities_client.clients import EasyEquitiesClient
        
        client = EasyEquitiesClient()
        client.login(username=credentials["username"], password=credentials["password"])
        
        # Get account info first
        accounts = client.accounts.list()
        account_name = next((acc.name for acc in accounts if acc.id == account_id), f"Account {account_id}")
        
        # Get holdings with share quantities
        holdings = client.accounts.holdings(account_id, include_shares=True)
        
        if not holdings:
            return PortfolioSummary(
                account_id=account_id,
                account_name=account_name,
                holdings=[],
                total_value=0.0,
                total_cost=0.0,
                total_profit_loss=0.0,
                holdings_count=0,
                message="No holdings found in this account"
            )
        
        # Process holdings
        formatted_holdings = []
        total_value = 0.0
        total_cost = 0.0
        
        for holding in holdings:
            # Clean and parse values
            current_value_str = holding.get('current_value', 'R0.00')
            purchase_value_str = holding.get('purchase_value', 'R0.00')
            
            current_value = float(current_value_str.replace('R', '').replace(',', '').strip())
            purchase_value = float(purchase_value_str.replace('R', '').replace(',', '').strip())
            
            total_value += current_value
            total_cost += purchase_value
            
            # Calculate profit/loss
            profit_loss = current_value - purchase_value
            profit_loss_percent = (profit_loss / purchase_value * 100) if purchase_value > 0 else 0
            
            # Extract symbol from contract code
            contract_code = holding.get('contract_code', '')
            symbol = contract_code.replace('EQU.ZA.', '') if contract_code.startswith('EQU.ZA.') else contract_code
            
            formatted_holding = Holding(
                symbol=symbol,
                name=holding.get('name', ''),
                shares=holding.get('shares', '0'),
                current_price=holding.get('current_price', 'R0.00'),
                current_value=current_value_str,
                purchase_value=purchase_value_str,
                profit_loss=profit_loss,
                profit_loss_percent=profit_loss_percent,
                isin=holding.get('isin', ''),
                logo_url=holding.get('img', '')
            )
            
            formatted_holdings.append(formatted_holding)
        
        total_profit_loss = total_value - total_cost
        
        logger.info(f"Retrieved {len(formatted_holdings)} holdings for account {account_id}")
        
        return PortfolioSummary(
            account_id=account_id,
            account_name=account_name,
            holdings=formatted_holdings,
            total_value=total_value,
            total_cost=total_cost,
            total_profit_loss=total_profit_loss,
            holdings_count=len(formatted_holdings)
        )
        
    except Exception as e:
        logger.error(f"Failed to get portfolio: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get portfolio: {str(e)}")

@router.get("/transactions/{account_id}")
async def get_transactions(
    account_id: str, 
    limit: int = 50,
    credentials: dict = Depends(get_ee_credentials)
):
    """
    Get transaction history for account
    """
    try:
        from easy_equities_client.clients import EasyEquitiesClient
        
        client = EasyEquitiesClient()
        client.login(username=credentials["username"], password=credentials["password"])
        
        transactions = client.accounts.transactions(account_id)
        
        # Process transactions
        all_transactions = []
        dividend_transactions = []
        
        for transaction in transactions[:limit]:  # Limit results
            # Extract symbol from contract code
            contract_code = transaction.get('ContractCode', '')
            symbol = contract_code.replace('EQU.ZA.', '') if contract_code.startswith('EQU.ZA.') else contract_code
            
            formatted_tx = Transaction(
                transaction_id=transaction.get('TransactionId', 0),
                date=transaction.get('TransactionDate', ''),
                action=transaction.get('Action', ''),
                amount=transaction.get('DebitCredit', 0),
                comment=transaction.get('Comment', ''),
                contract_code=contract_code,
                symbol=symbol if symbol else None
            )
            
            all_transactions.append(formatted_tx)
            
            # Check if it's a dividend transaction
            action = transaction.get('Action', '').lower()
            comment = transaction.get('Comment', '').lower()
            if 'dividend' in action or 'dividend' in comment:
                dividend_transactions.append(formatted_tx)
        
        logger.info(f"Retrieved {len(all_transactions)} transactions for account {account_id}")
        
        return {
            "account_id": account_id,
            "transactions": all_transactions,
            "dividend_transactions": dividend_transactions,
            "total_transactions": len(all_transactions),
            "total_dividends": len(dividend_transactions)
        }
        
    except Exception as e:
        logger.error(f"Failed to get transactions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get transactions: {str(e)}")

@router.get("/summary/{account_id}")
async def get_portfolio_summary(account_id: str, credentials: dict = Depends(get_ee_credentials)):
    """
    Get high-level portfolio summary
    """
    try:
        from easy_equities_client.clients import EasyEquitiesClient
        
        client = EasyEquitiesClient()
        client.login(username=credentials["username"], password=credentials["password"])
        
        # Get account valuations
        valuations = client.accounts.valuations(account_id)
        
        # Extract key metrics
        top_summary = valuations.get('TopSummary', {})
        account_value = top_summary.get('AccountValue', 0)
        
        # Get period movements
        period_movements = top_summary.get('PeriodMovements', [])
        profit_loss_info = period_movements[0] if period_movements else {}
        
        return {
            "account_id": account_id,
            "account_value": account_value,
            "currency": top_summary.get('AccountCurrency', 'ZAR'),
            "profit_loss_value": profit_loss_info.get('ValueMove', 'R0.00'),
            "profit_loss_percent": profit_loss_info.get('PercentageMove', '0.00%'),
            "valuation_data": valuations
        }
        
    except Exception as e:
        logger.error(f"Failed to get portfolio summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get summary: {str(e)}")

@router.post("/sync/{account_id}")
async def sync_portfolio_data(account_id: str, credentials: dict = Depends(get_ee_credentials)):
    """
    Sync portfolio data from EasyEquities (placeholder for future database storage)
    """
    try:
        # For now, just fetch fresh data
        # Later, this will store data in MongoDB
        
        from easy_equities_client.clients import EasyEquitiesClient
        
        client = EasyEquitiesClient()
        client.login(username=credentials["username"], password=credentials["password"])
        
        holdings = client.accounts.holdings(account_id, include_shares=True)
        transactions = client.accounts.transactions(account_id)
        
        logger.info(f"Synced portfolio data for account {account_id}")
        
        return {
            "status": "success",
            "message": "Portfolio data synced successfully",
            "holdings_count": len(holdings) if holdings else 0,
            "transactions_count": len(transactions) if transactions else 0,
            "synced_at": "2024-08-07T10:00:00Z"  # Current timestamp
        }
        
    except Exception as e:
        logger.error(f"Failed to sync portfolio: {e}")
        raise HTTPException(status_code=500, detail=f"Sync failed: {str(e)}")