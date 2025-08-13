import asyncio
from synatic_client import EasyEquitiesSynaticClient

async def test():
    client = EasyEquitiesSynaticClient()
    
    token = "eyJhbGciOiJSUzI1NiIsImtpZCI6Ijc4RDg1MDk0OTZCMkM5OTcyQ0EwQkQ5NDU4NzMxQUJEIiwidHlwIjoiYXQrand0In0.eyJpc3MiOiJodHRwczovL2lkZW50aXR5Lm9wZW5lYXN5LmlvIiwibmJmIjoxNzU1MDY0ODQ2LCJpYXQiOjE3NTUwNjQ4NDYsImV4cCI6MTc1NTA2NjY0NiwiYXVkIjpbImFwaV9nYXRld2F5IiwidXNlcl9wcm9maWxlX2FwaSIsInN0YXRpY19kYXRhX2FwaSIsImludmVzdF9ub3dfYXBpIiwiYXV0b19yZWZpY2FfYXBpIl0sInNjb3BlIjpbIm9wZW5pZCIsInBsYXRmb3JtIiwicHJvZmlsZSIsImFwaV9nYXRld2F5IiwidXNlcl9wcm9maWxlX2FwaSIsInN0YXRpY19kYXRhX2FwaSIsImludmVzdF9ub3dfYXBpIiwiYXV0b19yZWZpY2FfYXBpIl0sImFtciI6WyJwd2QiXSwiY2xpZW50X2lkIjoiNThhZjI1ZDA3YTkzNGM2N2I2NWFhOGMxNTlmMWMxYzIiLCJzdWIiOiJiYmYxYzkwNS1kZjZiLTQxZjktODkwOC04MTMwZWYyMjcxNDUiLCJhdXRoX3RpbWUiOjE3NTUwNjQ4NDYsImlkcCI6ImxvY2FsIiwic3Vic3lzdGVtaWQiOiIxIiwidGVuYW50aWQiOiIxIiwidXNlcmlkIjoiMjgyNjU2MCIsInNpZCI6IkIzMUI4QTFFNzUxQUIwQkE4N0EzQjAwMkYwMTlCRUE3IiwianRpIjoiRkU2NURGMzNDNUJFMTI2NzhBQ0M3QjY0QTBBODNGNTkifQ.beH_h_FsoG9n9jlSre5jeu7nt8m_UZj0U59OhSTuBFStMh6ZG8IqRzF_eR2hj14nASxqMcIrkZNhinKKr8iyyh9yTJtkRQqqy4InopTZdWsy3bIYqUtgzN-9_1EdpjD9ylVaDbK-Sb0rG6icG6SC0SN3WKJK64VYvGC1AeQQYWG7oakU2vzGBpUxX-JUVs4S81rhlTpx0uo5Dz4kOqJ19tAmK9QqBtNeDNqqwPSVIDjIiW9CM6t0j8UM_mMuc__YMFb9ms_1mPGG8JmVBNg0DiYpIHlVDN9nrW3AtOcM8EkhKvFE7Jj0-eRPFx3Esl2-j6oIR8b_16Bdo5gP4HD5sw"  # Your full token here
    client.set_bearer_token(token)
    
    # Fetch SA stocks
    sa_stocks = await client.get_all_sa_equities()
    print(f"Found {len(sa_stocks)} SA stocks")
    
    for stock in sa_stocks[:]:
        print(f"- {stock['name']} ({stock['ticker']}) - {stock['lastPrice']}")
    
    await client.close()

# Run it
asyncio.run(test())