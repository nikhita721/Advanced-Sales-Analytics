#!/usr/bin/env python3
"""
Script to set up the database for the Sales Analytics Platform
"""
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.database import engine, Base
from app.models import Transaction, Customer, Product, Store, SalesForecast, AnalyticsCache
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_tables():
    """Create all database tables"""
    try:
        print("Creating database tables...")
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables created successfully!")
        
        # Create additional indexes for performance
        print("Creating performance indexes...")
        from sqlalchemy import text
        
        with engine.connect() as conn:
            # Transaction indexes
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_transactions_customer_date ON transactions(customer_id, transaction_date)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_transactions_product_date ON transactions(product_id, transaction_date)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_transactions_store_date ON transactions(store_id, transaction_date)"))
            
            # Customer indexes
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_customers_rfm_segment ON customers(rfm_segment)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_customers_clv ON customers(customer_lifetime_value)"))
            
            # Product indexes
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_products_category ON products(category)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_products_brand ON products(brand)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_products_revenue ON products(total_revenue)"))
            
            conn.commit()
        
        print("✅ Performance indexes created successfully!")
        
    except Exception as e:
        logger.error(f"Error creating database tables: {str(e)}")
        print(f"❌ Error: {str(e)}")
        return False
    
    return True

def verify_tables():
    """Verify that all tables exist"""
    try:
        print("Verifying database tables...")
        from sqlalchemy import text
        
        with engine.connect() as conn:
            # Check if tables exist (SQLite version)
            tables = [
                'transactions', 'customers', 'products', 'stores', 
                'sales_forecasts', 'analytics_cache'
            ]
            
            for table in tables:
                result = conn.execute(text(f"""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='{table}';
                """))
                exists = result.fetchone() is not None
                
                if exists:
                    print(f"✅ Table '{table}' exists")
                else:
                    print(f"❌ Table '{table}' missing")
                    return False
        
        print("✅ All database tables verified!")
        return True
        
    except Exception as e:
        logger.error(f"Error verifying tables: {str(e)}")
        print(f"❌ Error: {str(e)}")
        return False

def main():
    """Main function to set up the database"""
    print("🗄️  Setting up Sales Analytics Database")
    print("="*50)
    
    # Create tables
    if not create_tables():
        print("❌ Failed to create database tables")
        sys.exit(1)
    
    # Verify tables
    if not verify_tables():
        print("❌ Database verification failed")
        sys.exit(1)
    
    print("\n" + "="*50)
    print("🎉 DATABASE SETUP COMPLETED SUCCESSFULLY!")
    print("="*50)
    print("Next steps:")
    print("1. Ingest your data: python scripts/ingest_data.py <excel_file>")
    print("2. Start the API: python run.py api")
    print("3. Start the dashboard: python run.py dashboard")
    print("4. Or use Docker: docker-compose up -d")

if __name__ == "__main__":
    main()
