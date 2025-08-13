from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import logging
import httpx
import json
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import jwt

load_dotenv()
logger = logging.getLogger(__name__)

router = APIRouter()

# Global token storage (in production, use Redis or database)
token_storage = {
    "bearer_token": None,
    "token_expiry": None,
    "ee_client": None
}

# Pydantic models
class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    status: str
    message: str
    accounts_count: int
    access_token: str
    token_type: str = "Bearer"
    expires_in: int = 1800  # 30 minutes

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

async def get_synatic_token(ee_client) -> Optional[str]:
    """
    Get the Synatic Bearer token after EasyEquities login.
    This requires getting a token for the invest-now platform specifically.
    """
    try:
        # Extract cookies from the EasyEquities client session
        cookies_dict = {}
        
        # Try different ways to get the session cookies
        if hasattr(ee_client, '_session'):
            session = ee_client._session
        elif hasattr(ee_client, 'session'):
            session = ee_client.session
        else:
            logger.warning("Could not find session in EasyEquities client")
            return None
        
        # Extract cookies
        if hasattr(session, 'cookies'):
            for key, value in session.cookies.items():
                cookies_dict[key] = value
        
        async with httpx.AsyncClient() as client:
            # Step 1: Get the main platform token first (like you showed)
            platform_token_response = await client.post(
                "https://identity.openeasy.io/connect/token",
                data={
                    "grant_type": "password",
                    "client_id": "fa4d2622bc1e45a7be79395d941e2548",  # Platform client ID
                    "scope": "openid platform profile api_gateway user_profile_api static_data_api",
                    "username": os.getenv("EASYEQUITIES_USERNAME"),
                    "password": os.getenv("EASYEQUITIES_PASSWORD")
                },
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Origin": "https://platform.easyequities.io",
                    "Referer": "https://platform.easyequities.io/"
                }
            )
            
            if platform_token_response.status_code != 200:
                logger.error(f"Failed to get platform token: {platform_token_response.text}")
                return None
            
            platform_token_data = platform_token_response.json()
            platform_access_token = platform_token_data.get("access_token")
            
            # Step 2: Use platform token to get invest-now token
            # The invest-now platform uses a different client_id
            invest_token_response = await client.post(
                "https://identity.openeasy.io/connect/token",
                data={
                    "grant_type": "client_credentials",
                    "client_id": "58af25d07a934c67b65aa8c159f1c1c2",  # Invest-now client ID
                    "scope": "openid platform profile api_gateway invest_now_api user_profile_api static_data_api"
                },
                headers={
                    "Authorization": f"Bearer {platform_access_token}",
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Origin": "https://invest-now.apps.easyequities.io",
                    "Referer": "https://invest-now.apps.easyequities.io/"
                }
            )
            
            if invest_token_response.status_code == 200:
                invest_token_data = invest_token_response.json()
                return invest_token_data.get("access_token")
            
            # Alternative approach: Try authorization code flow
            # This is what the browser actually does
            auth_response = await client.get(
                "https://identity.openeasy.io/connect/authorize",
                params={
                    "client_id": "58af25d07a934c67b65aa8c159f1c1c2",
                    "redirect_uri": "https://invest-now.apps.easyequities.io/signin-oidc",
                    "response_type": "code id_token",
                    "scope": "openid platform profile api_gateway invest_now_api",
                    "response_mode": "form_post",
                    "nonce": "random_nonce_here"
                },
                headers={
                    "Authorization": f"Bearer {platform_access_token}",
                    "Cookie": "; ".join([f"{k}={v}" for k, v in cookies_dict.items()])
                },
                follow_redirects=False
            )
            
            if auth_response.status_code == 302:
                # Extract code from redirect
                location = auth_response.headers.get("location", "")
                if "code=" in location:
                    import re
                    code_match = re.search(r'code=([^&]+)', location)
                    if code_match:
                        auth_code = code_match.group(1)
                        
                        # Exchange code for token
                        token_exchange_response = await client.post(
                            "https://identity.openeasy.io/connect/token",
                            data={
                                "grant_type": "authorization_code",
                                "code": auth_code,
                                "redirect_uri": "https://invest-now.apps.easyequities.io/signin-oidc",
                                "client_id": "58af25d07a934c67b65aa8c159f1c1c2"
                            }
                        )
                        
                        if token_exchange_response.status_code == 200:
                            final_token_data = token_exchange_response.json()
                            return final_token_data.get("access_token")
        
        logger.warning("Could not automatically get Synatic token - manual entry required")
        return None
        
    except Exception as e:
        logger.error(f"Failed to get Synatic token: {e}")
        return None

def create_mock_token() -> str:
    """Create a mock JWT token for testing when real token is not available"""
    payload = {
        "iss": "https://identity.openeasy.io",
        "exp": int((datetime.now() + timedelta(minutes=30)).timestamp()),
        "iat": int(datetime.now().timestamp()),
        "client_id": "58af25d07a934c67b65aa8c159f1c1c2",
        "scope": ["openid", "platform", "profile", "api_gateway", "invest_now_api"],
        "sub": "mock_user"
    }
    
    # This creates an unsigned token for testing
    # In production, you need a real token from the auth server
    header = {"alg": "none", "typ": "JWT"}
    header_encoded = json.dumps(header).encode()
    payload_encoded = json.dumps(payload).encode()
    
    import base64
    header_b64 = base64.urlsafe_b64encode(header_encoded).decode().rstrip("=")
    payload_b64 = base64.urlsafe_b64encode(payload_encoded).decode().rstrip("=")
    
    return f"{header_b64}.{payload_b64}."

@router.post("/login", response_model=LoginResponse)
async def login_easyequities(credentials: dict = Depends(get_ee_credentials)):
    """
    Login to EasyEquities and get Synatic Bearer token
    """
    try:
        from easy_equities_client.clients import EasyEquitiesClient
        
        client = EasyEquitiesClient()
        client.login(username=credentials["username"], password=credentials["password"])
        
        accounts = client.accounts.list()
        
        # Store the client for later use
        token_storage["ee_client"] = client
        
        # Try to get the Synatic token
        synatic_token = await get_synatic_token(client)
        
        if not synatic_token:
            # For development: provide instructions to get token manually
            logger.warning("Could not automatically get Synatic token")
            synatic_token = "MANUAL_TOKEN_REQUIRED"
            
            return LoginResponse(
                status="success",
                message="Login successful but token extraction failed. Please get token manually from browser.",
                accounts_count=len(accounts),
                access_token=synatic_token,
                token_type="Bearer",
                expires_in=1800
            )
        
        # Store token for reuse
        token_storage["bearer_token"] = synatic_token
        token_storage["token_expiry"] = datetime.now() + timedelta(minutes=30)
        
        logger.info(f"EasyEquities login successful for user: {credentials['username']}")
        
        return LoginResponse(
            status="success",
            message="Login successful",
            accounts_count=len(accounts),
            access_token=synatic_token,
            token_type="Bearer",
            expires_in=1800
        )
        
    except Exception as e:
        logger.error(f"EasyEquities login failed: {e}")
        raise HTTPException(status_code=401, detail=f"Login failed: {str(e)}")

@router.post("/login-custom")
async def login_with_custom_credentials(login_request: LoginRequest):
    """
    Login with custom credentials and get token
    """
    try:
        from easy_equities_client.clients import EasyEquitiesClient
        
        client = EasyEquitiesClient()
        client.login(username=login_request.username, password=login_request.password)
        
        accounts = client.accounts.list()
        
        
        logger.info(f"Custom login successful for user: {login_request.username}")
        
        return {
            "status": "success",
            "message": "Login successful",
            "accounts_count": len(accounts),
            "username": login_request.username,
        }
        
    except Exception as e:
        logger.error(f"Custom login failed: {e}")
        raise HTTPException(status_code=401, detail=f"Login failed: {str(e)}")

@router.post("/set-token")
async def set_manual_token(token: str):
    """
    Manually set the Synatic Bearer token (for development)
    Use this if automatic token extraction fails
    """
    try:
        # Validate token format
        if not token:
            raise HTTPException(status_code=400, detail="Token cannot be empty")
        
        # Store the token
        token_storage["bearer_token"] = token
        token_storage["token_expiry"] = datetime.now() + timedelta(minutes=30)
        
        # Try to decode to get actual expiry
        try:
            decoded = jwt.decode(token, options={"verify_signature": False})
            exp = decoded.get('exp')
            if exp:
                token_storage["token_expiry"] = datetime.fromtimestamp(exp)
        except:
            pass
        
        return {
            "status": "success",
            "message": "Token set successfully",
            "expires_at": token_storage["token_expiry"].isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to set token: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to set token: {str(e)}")

@router.get("/token")
async def get_current_token():
    """
    Get the current Bearer token (for internal use or debugging)
    """
    if not token_storage["bearer_token"]:
        raise HTTPException(
            status_code=404, 
            detail="No token available. Please login first or set token manually."
        )
    
    is_expired = False
    if token_storage["token_expiry"]:
        is_expired = datetime.now() > token_storage["token_expiry"]
    
    return {
        "token": token_storage["bearer_token"],
        "expires_at": token_storage["token_expiry"].isoformat() if token_storage["token_expiry"] else None,
        "is_expired": is_expired,
        "token_type": "Bearer"
    }

@router.get("/accounts")
async def get_accounts(credentials: dict = Depends(get_ee_credentials)):
    """
    Get all EasyEquities accounts for the authenticated user
    """
    try:
        from easy_equities_client.clients import EasyEquitiesClient
        
        # Use stored client if available
        client = token_storage.get("ee_client")
        if not client:
            client = EasyEquitiesClient()
            client.login(username=credentials["username"], password=credentials["password"])
            token_storage["ee_client"] = client
        
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
        
        logger.info(f"Retrieved {len(formatted_accounts)} accounts")
        
        return {
            "accounts": formatted_accounts,
            "total_count": len(formatted_accounts),
            "bearer_token": token_storage.get("bearer_token")  # Include token if available
        }
        
    except Exception as e:
        logger.error(f"Failed to get accounts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get accounts: {str(e)}")

@router.get("/validate")
async def validate_session(credentials: dict = Depends(get_ee_credentials)):
    """
    Validate that EasyEquities session is working
    """
    try:
        from easy_equities_client.clients import EasyEquitiesClient
        
        client = token_storage.get("ee_client")
        if not client:
            client = EasyEquitiesClient()
            client.login(username=credentials["username"], password=credentials["password"])
            token_storage["ee_client"] = client
        
        accounts = client.accounts.list()
        
        # Check token status
        token_valid = False
        if token_storage["bearer_token"] and token_storage["token_expiry"]:
            token_valid = datetime.now() < token_storage["token_expiry"]
        
        return {
            "valid": True,
            "message": "Session is valid",
            "accounts_found": len(accounts),
            "token_available": token_storage["bearer_token"] is not None,
            "token_valid": token_valid
        }
        
    except Exception as e:
        logger.error(f"Session validation failed: {e}")
        return {
            "valid": False,
            "message": f"Session invalid: {str(e)}",
            "token_available": token_storage["bearer_token"] is not None
        }

@router.get("/instructions")
async def get_token_instructions():
    """
    Get instructions for manually obtaining the Synatic Bearer token
    """
    return {
        "instructions": [
            "1. Open your browser and go to: https://invest-now.apps.easyequities.io",
            "2. Login with your EasyEquities credentials",
            "3. Open Developer Tools (F12)",
            "4. Go to the Network tab",
            "5. Look for any request to 'rest.synatic.openeasy.io'",
            "6. Click on the request and go to Headers tab",
            "7. Find the 'Authorization' header",
            "8. Copy the entire value (starting with 'Bearer ')",
            "9. Use the /api/v1/auth/set-token endpoint to set it",
            "10. Or include it directly in your API calls"
        ],
        "example_token_format": "Bearer eyJhbGciOiJSUzI1NiIs...",
        "set_token_endpoint": "/api/v1/auth/set-token",
        "current_token_status": {
            "has_token": token_storage["bearer_token"] is not None,
            "is_expired": token_storage["token_expiry"] and datetime.now() > token_storage["token_expiry"] if token_storage["token_expiry"] else None
        }
    }