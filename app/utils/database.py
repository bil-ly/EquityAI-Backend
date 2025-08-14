import os
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
import logging
from dotenv import load_dotenv

load_dotenv()

from app.models.database import User, Stock, Portfolio, Transaction, Dividend, Analysis

logger = logging.getLogger(__name__)
MONGODB_URL = os.getenv("MONGODB_URL")
MONGODB_DB = os.getenv("MONGODB_DB")
class Database:
    client: AsyncIOMotorClient = None
    database = None

db = Database()

async def connect_to_mongo():
    """Create database connection"""
    try:
        mongodb_url = MONGODB_URL 
        
        logger.info(f"Connecting to MongoDB...")
        
        db.client = AsyncIOMotorClient(mongodb_url)
        
        database_name = MONGODB_DB
        db.database = db.client[database_name]
        
        await db.client.admin.command('ping')
        logger.info("Connected to MongoDB successfully")
        
        await init_beanie(
            database=db.database,
            document_models=[
                User,
                Stock, 
                Portfolio,
                Transaction,
                Dividend,
                Analysis
            ]
        )
        
        logger.info("Beanie ODM initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise e

async def close_mongo_connection():
    """Close database connection"""
    if db.client:
        db.client.close()
        logger.info("MongoDB connection closed")

async def get_database():
    """Get database instance"""
    return db.database

async def ping_database():
    """Check if database is responding"""
    try:
        await db.client.admin.command('ping')
        return True
    except Exception as e:
        logger.error(f"Database ping failed: {e}")
        return False