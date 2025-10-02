"""
Database models for the Sales Analytics Platform
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, Index
from sqlalchemy.sql import func
from app.database import Base
from datetime import datetime
from typing import Optional


class Transaction(Base):
    """Transaction model for retail store transactions"""
    __tablename__ = "transactions"
    
    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(String(100), unique=True, index=True, nullable=False)
    customer_id = Column(String(50), index=True, nullable=False)
    product_id = Column(String(50), index=True, nullable=False)
    product_name = Column(String(255), nullable=False)
    category = Column(String(100), index=True)
    subcategory = Column(String(100))
    brand = Column(String(100))
    quantity = Column(Integer, nullable=False, default=1)
    unit_price = Column(Float, nullable=False)
    total_amount = Column(Float, nullable=False)
    discount_amount = Column(Float, default=0.0)
    tax_amount = Column(Float, default=0.0)
    transaction_date = Column(DateTime, index=True, nullable=False)
    store_id = Column(String(50), index=True)
    store_name = Column(String(255))
    sales_person = Column(String(100))
    payment_method = Column(String(50))
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Indexes for better query performance
    __table_args__ = (
        Index('idx_customer_date', 'customer_id', 'transaction_date'),
        Index('idx_product_date', 'product_id', 'transaction_date'),
        Index('idx_store_date', 'store_id', 'transaction_date'),
    )


class Customer(Base):
    """Customer model with RFM analysis data"""
    __tablename__ = "customers"
    
    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(String(50), unique=True, index=True, nullable=False)
    first_name = Column(String(100))
    last_name = Column(String(100))
    email = Column(String(255), unique=True, index=True)
    phone = Column(String(20))
    address = Column(Text)
    city = Column(String(100))
    state = Column(String(100))
    country = Column(String(100))
    postal_code = Column(String(20))
    
    # RFM Analysis fields
    recency_score = Column(Integer)
    frequency_score = Column(Integer)
    monetary_score = Column(Integer)
    rfm_segment = Column(String(50))
    customer_lifetime_value = Column(Float)
    total_orders = Column(Integer, default=0)
    total_spent = Column(Float, default=0.0)
    avg_order_value = Column(Float, default=0.0)
    last_purchase_date = Column(DateTime)
    first_purchase_date = Column(DateTime)
    
    # Customer status
    is_active = Column(Boolean, default=True)
    churn_probability = Column(Float)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class Product(Base):
    """Product model with sales analytics"""
    __tablename__ = "products"
    
    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(String(50), unique=True, index=True, nullable=False)
    product_name = Column(String(255), nullable=False)
    category = Column(String(100), index=True)
    subcategory = Column(String(100))
    brand = Column(String(100), index=True)
    description = Column(Text)
    unit_price = Column(Float, nullable=False)
    cost_price = Column(Float)
    profit_margin = Column(Float)
    
    # Sales analytics
    total_quantity_sold = Column(Integer, default=0)
    total_revenue = Column(Float, default=0.0)
    avg_rating = Column(Float)
    stock_quantity = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class Store(Base):
    """Store model for multi-location analytics"""
    __tablename__ = "stores"
    
    id = Column(Integer, primary_key=True, index=True)
    store_id = Column(String(50), unique=True, index=True, nullable=False)
    store_name = Column(String(255), nullable=False)
    address = Column(Text)
    city = Column(String(100))
    state = Column(String(100))
    country = Column(String(100))
    postal_code = Column(String(20))
    phone = Column(String(20))
    manager_name = Column(String(100))
    
    # Performance metrics
    total_sales = Column(Float, default=0.0)
    total_transactions = Column(Integer, default=0)
    avg_transaction_value = Column(Float, default=0.0)
    is_active = Column(Boolean, default=True)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class SalesForecast(Base):
    """Sales forecasting model"""
    __tablename__ = "sales_forecasts"
    
    id = Column(Integer, primary_key=True, index=True)
    forecast_date = Column(DateTime, index=True, nullable=False)
    product_id = Column(String(50), index=True)
    store_id = Column(String(50), index=True)
    category = Column(String(100), index=True)
    predicted_sales = Column(Float, nullable=False)
    confidence_interval_lower = Column(Float)
    confidence_interval_upper = Column(Float)
    model_used = Column(String(100))
    accuracy_score = Column(Float)
    created_at = Column(DateTime, default=func.now())


class AnalyticsCache(Base):
    """Cache for expensive analytics computations"""
    __tablename__ = "analytics_cache"
    
    id = Column(Integer, primary_key=True, index=True)
    cache_key = Column(String(255), unique=True, index=True, nullable=False)
    cache_data = Column(Text, nullable=False)  # JSON data
    cache_type = Column(String(50), nullable=False)  # rfm, forecast, etc.
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=func.now())
