from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic models
class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    status: str
    message: str
    accounts_count: int

class AccountResponse(BaseModel):
    id: str
    name: str
    currency: str
    type: str

def get_ee_credentials():
    """Get EasyEquities credentials from environment"""
    username = os.getenv("EASYEQUITIES_USERNAME")
    password = os.getenv("EASYEQUITIES_PASSWORD")
    
    if not username or not password:
        raise HTTPException(
            status_code=500, 
            detail="EasyEquities credentials not configured in environment"
        )
    
    return {"username": username, "password": password}

@router.post("/login", response_model=LoginResponse)
async def login_easyequities(credentials: dict = Depends(get_ee_credentials)):
    """
    Test EasyEquities login with environment credentials
    """
    try:
        from easy_equities_client.clients import EasyEquitiesClient
        
        client = EasyEquitiesClient()
        client.login(username=credentials["username"], password=credentials["password"])
        
        accounts = client.accounts.list()
        
        logger.info(f"✅ EasyEquities login successful for user: {credentials['username']}")
        
        return LoginResponse(
            status="success",
            message="Login successful",
            accounts_count=len(accounts)
        )
        
    except Exception as e:
        logger.error(f"❌ EasyEquities login failed: {e}")
        raise HTTPException(status_code=401, detail=f"Login failed: {str(e)}")

@router.post("/login-custom")
async def login_with_custom_credentials(login_request: LoginRequest):
    """
    Login with custom credentials (for testing different accounts)
    """
    try:
        from easy_equities_client.clients import EasyEquitiesClient
        
        client = EasyEquitiesClient()
        client.login(username=login_request.username, password=login_request.password)
        
        accounts = client.accounts.list()
        
        logger.info(f"✅ Custom login successful for user: {login_request.username}")
        
        return {
            "status": "success",
            "message": "Login successful",
            "accounts_count": len(accounts),
            "username": login_request.username
        }
        
    except Exception as e:
        logger.error(f"❌ Custom login failed: {e}")
        raise HTTPException(status_code=401, detail=f"Login failed: {str(e)}")

@router.get("/accounts")
async def get_accounts(credentials: dict = Depends(get_ee_credentials)):
    """
    Get all EasyEquities accounts for the authenticated user
    """
    try:
        from easy_equities_client.clients import EasyEquitiesClient
        
        client = EasyEquitiesClient()
        client.login(username=credentials["username"], password=credentials["password"])
        
        accounts = client.accounts.list()
        
        formatted_accounts = []
        for account in accounts:
            account_type = "TFSA" if "TFSA" in account.name else "Standard"
            if "Demo" in account.name:
                account_type = "Demo"
            elif "USD" in account.name:
                account_type += " USD"
            
            formatted_accounts.append(AccountResponse(
                id=account.id,
                name=account.name,
                currency=account.trading_currency_id,
                type=account_type
            ))
        
        logger.info(f"✅ Retrieved {len(formatted_accounts)} accounts")
        
        return {
            "accounts": formatted_accounts,
            "total_count": len(formatted_accounts)
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get accounts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get accounts: {str(e)}")

@router.get("/validate")
async def validate_session(credentials: dict = Depends(get_ee_credentials)):
    """
    Validate that EasyEquities session is working
    """
    try:
        from easy_equities_client.clients import EasyEquitiesClient
        
        client = EasyEquitiesClient()
        client.login(username=credentials["username"], password=credentials["password"])
        
        accounts = client.accounts.list()
        
        return {
            "valid": True,
            "message": "Session is valid",
            "accounts_found": len(accounts)
        }
        
    except Exception as e:
        logger.error(f"❌ Session validation failed: {e}")
        return {
            "valid": False,
            "message": f"Session invalid: {str(e)}"
        }