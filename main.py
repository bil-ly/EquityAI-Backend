from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import auth, portfolio, analysis, dividends, stocks

app = FastAPI(
    title="EquityAI Backend",
    description="AI-powered investment analysis API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api/v1/auth", tags=["authentication"])
app.include_router(portfolio.router, prefix="/api/v1/portfolio", tags=["portfolio"])
app.include_router(analysis.router, prefix="/api/v1/analysis", tags=["analysis"])
app.include_router(dividends.router, prefix="/api/v1/dividends", tags=["dividends"])
app.include_router(stocks.router, prefix="/api/v1/stocks", tags=["stocks"])

@app.get("/")
async def root():
    return {"message": "EquityAI Backend API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": "2024-08-07T00:00:00Z"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
