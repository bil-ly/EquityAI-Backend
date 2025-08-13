import httpx
import asyncio
from typing import List, Dict, Any
from datetime import datetime, timedelta
import jwt

class EasyEquitiesSynaticClient:
    """
    Client for fetching real instrument data from EasyEquities via Synatic API
    """
    
    def __init__(self, ee_client=None):
        self.ee_client = ee_client
        self.synatic_base = "https://rest.synatic.openeasy.io/easyequities/investnow"
        self.client = httpx.AsyncClient(timeout=30.0)
        self._bearer_token = None
        self._token_expiry = None
        self._instruments_cache = {}
        self._cache_timestamps = {}
        self._cache_ttl = 3600  # 1 hour cache for instruments
        
    async def close(self):
        await self.client.aclose()
    
    def set_bearer_token(self, token: str):
        self._bearer_token = token
        try:
            decoded = jwt.decode(token, options={"verify_signature": False})
            self._token_expiry = datetime.fromtimestamp(decoded.get('exp', 0))
            print(f"Token set, expires at: {self._token_expiry}")
        except:
            self._token_expiry = datetime.now() + timedelta(minutes=30)
    
    def is_token_valid(self) -> bool:
        if not self._bearer_token or not self._token_expiry:
            return False
        return datetime.now() < self._token_expiry - timedelta(minutes=2)
    
    async def get_bearer_token(self) -> str:
        if self.is_token_valid():
            return self._bearer_token
        
        if self.ee_client:
            if hasattr(self.ee_client, 'token'):
                self.set_bearer_token(self.ee_client.token)
            elif hasattr(self.ee_client, 'bearer_token'):
                self.set_bearer_token(self.ee_client.bearer_token)
            elif hasattr(self.ee_client, 'get_token'):
                token = await self.ee_client.get_token()
                self.set_bearer_token(token)
            else:
                raise Exception("Cannot extract token from EasyEquities client")
        
        if not self._bearer_token:
            raise Exception("No valid bearer token available")
        
        return self._bearer_token
    
    async def search_instruments(
        self,
        category: str = "equitiesexpanded",
        search_value: str = "",
        account_filter: str = "ALL",
        page: int = 1
    ) -> List[Dict[str, Any]]:
        cache_key = f"{category}_{search_value}_{account_filter}_{page}"
        if cache_key in self._instruments_cache:
            cache_time = self._cache_timestamps.get(cache_key)
            if cache_time and (datetime.now() - cache_time).seconds < self._cache_ttl:
                print(f"Returning cached data for {category}")
                return self._instruments_cache[cache_key]
        
        token = await self.get_bearer_token()
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/plain, */*",
            "Origin": "https://invest-now.apps.easyequities.io",
            "Referer": "https://invest-now.apps.easyequities.io/",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        payload = {
            "searchValue": search_value,
            "account_filter": account_filter,
            "category": category,
            "page": page
        }
        
        try:
            response = await self.client.post(
                f"{self.synatic_base}/search",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()

                if isinstance(data, dict):
                    print(f"API returned dict keys for {category}: {list(data.keys())}")
                    instruments = data.get("instruments", [])
                elif isinstance(data, list):
                    instruments = data
                else:
                    instruments = []

                self._instruments_cache[cache_key] = instruments
                self._cache_timestamps[cache_key] = datetime.now()
                
                print(f"Fetched {len(instruments)} instruments for category: {category}")
                return instruments
            else:
                print(f"Error fetching instruments: {response.status_code}")
                print(f"Response: {response.text}")
                return []
                
        except Exception as e:
            print(f"Error in search_instruments: {str(e)}")
            return []
    
    async def get_all_sa_equities(self) -> List[Dict[str, Any]]:
        return await self.search_instruments(category="equitiesexpanded")
    
    async def get_all_us_equities(self) -> List[Dict[str, Any]]:
        return await self.search_instruments(category="equities")
    
    async def get_all_etfs(self) -> List[Dict[str, Any]]:
        return await self.search_instruments(category="etfs")
    
    async def get_all_bundles(self) -> List[Dict[str, Any]]:
        return await self.search_instruments(category="bundles")
    
    async def get_all_crypto(self) -> List[Dict[str, Any]]:
        return await self.search_instruments(category="crypto")
    
    async def get_all_instruments(self) -> Dict[str, List[Dict[str, Any]]]:
        categories = [
            "equitiesexpanded",
            "equities",
            "etfs",
            "bundles",
            "crypto",
            "commodities"
        ]
        
        all_instruments = {}
        
        for category in categories:
            try:
                instruments = await self.search_instruments(category=category)
                all_instruments[category] = instruments
                await asyncio.sleep(0.5)
            except Exception as e:
                print(f"Error fetching {category}: {str(e)}")
                all_instruments[category] = []
        
        return all_instruments
    
    def transform_to_standard_format(self, instrument: Dict[str, Any]) -> Dict[str, Any]:
        general = instrument.get("General", {})
        price_data = instrument.get("Price", {})
        fundamental = instrument.get("Fundamental", {})
        
        return {
            "symbol": instrument.get("ticker", ""),
            "contract_code": instrument.get("contractCode", ""),
            "name": instrument.get("name", ""),
            "exchange": instrument.get("Exchange", ""),
            "category": instrument.get("category", ""),
            "asset_sub_group": instrument.get("assetSubGroup", ""),
            "flag_code": instrument.get("flagCode", ""),
            "logo_url": instrument.get("logoUrl", ""),
            "description": instrument.get("description", ""),
            "last_price": float(instrument.get("lastPrice", 0) or 0),
            "currency": general.get("currency", "ZAR" if instrument.get("flagCode") == "ZA" else "USD"),
            "sector": general.get("sector", ""),
            "industry": general.get("industry", ""),
            "market_cap": fundamental.get("mktcap"),
            "pe_ratio": fundamental.get("pe"),
            "eps": fundamental.get("eps"),
            "dividend_yield": instrument.get("Dividends", {}).get("yield"),
            "volume": price_data.get("volume"),
            "price_change_52w": price_data.get("pchange_52w"),
            "price_52w_high": price_data.get("price52whigh"),
            "price_52w_low": price_data.get("price52wlow"),
            "returns_1m": instrument.get("returns", {}).get("1mo", {}).get("percentage"),
            "returns_3m": instrument.get("returns", {}).get("3mo", {}).get("percentage"),
            "returns_6m": instrument.get("returns", {}).get("6mo", {}).get("percentage"),
            "sparkline_1m": instrument.get("returns", {}).get("1mo", {}).get("sparkline"),
            "company_website": general.get("company_website", ""),
            "is_tradeable": True
        }


# Example usage
async def main():
    client = EasyEquitiesSynaticClient()
    
    sa_stocks = await client.get_all_sa_equities()
    print(f"Found {len(sa_stocks)} SA stocks")
    
    if sa_stocks:
        first_stock = client.transform_to_standard_format(sa_stocks[0])
        print(f"First stock: {first_stock['name']} ({first_stock['symbol']}) - R{first_stock['last_price']}")
    
    us_stocks = await client.get_all_us_equities()
    print(f"Found {len(us_stocks)} US stocks")
    
    all_instruments = await client.get_all_instruments()
    for category, instruments in all_instruments.items():
        print(f"{category}: {len(instruments)} instruments")
    
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
