"""
Database configuration and session management
"""
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
# from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from app.config import settings

# Synchronous database
engine = create_engine(settings.database_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Asynchronous database (disabled for SQLite compatibility)
# async_engine = create_async_engine(settings.database_url_async)
# AsyncSessionLocal = async_sessionmaker(
#     async_engine, class_=AsyncSession, expire_on_commit=False
# )

# Base class for models
Base = declarative_base()

# Metadata for migrations
metadata = MetaData()


def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# async def get_async_db():
#     """Dependency to get async database session"""
#     async with AsyncSessionLocal() as session:
#         yield session
