"""
FastAPI endpoints for the Sales Analytics Platform
"""
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
import json
import logging

from app.database import get_db
from app.models import Transaction, Customer, Product, Store, SalesForecast, AnalyticsCache
from app.analytics import AdvancedAnalytics
from app.data_ingestion import DataIngestionPipeline
from app.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=settings.api_description,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Sales Analytics API",
        "version": settings.api_version,
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# Data Ingestion Endpoints
@app.post("/api/v1/ingest/excel")
async def ingest_excel_data(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db)
):
    """Ingest Excel data file"""
    try:
        # Validate file type
        if not file.filename.endswith(('.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="Only Excel files are supported")
        
        # Save uploaded file temporarily
        file_path = f"/tmp/{file.filename}"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process file
        pipeline = DataIngestionPipeline(db)
        results = pipeline.process_excel_file(file_path)
        
        return {
            "message": "Data ingestion completed",
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error ingesting Excel data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Analytics Endpoints
@app.get("/api/v1/analytics/overview")
async def get_analytics_overview(
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    store_id: Optional[str] = Query(None, description="Store ID filter"),
    db: Session = Depends(get_db)
):
    """Get analytics overview"""
    try:
        analytics = AdvancedAnalytics(db)
        insights = analytics.get_sales_insights()
        
        return {
            "insights": insights,
            "filters": {
                "start_date": start_date,
                "end_date": end_date,
                "store_id": store_id
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting analytics overview: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/analytics/rfm")
async def perform_rfm_analysis(
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db)
):
    """Perform RFM analysis"""
    try:
        analytics = AdvancedAnalytics(db)
        results = analytics.perform_rfm_analysis()
        
        if "error" in results:
            raise HTTPException(status_code=500, detail=results["error"])
        
        return {
            "rfm_analysis": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error performing RFM analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/analytics/clv")
async def get_customer_lifetime_value(
    db: Session = Depends(get_db)
):
    """Get customer lifetime value analysis"""
    try:
        analytics = AdvancedAnalytics(db)
        clv_results = analytics.calculate_customer_lifetime_value()
        
        if "error" in clv_results:
            raise HTTPException(status_code=500, detail=clv_results["error"])
        
        return {
            "clv_analysis": clv_results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error calculating CLV: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/analytics/forecast")
async def get_sales_forecast(
    days_ahead: int = Query(30, description="Number of days to forecast"),
    product_id: Optional[str] = Query(None, description="Product ID filter"),
    store_id: Optional[str] = Query(None, description="Store ID filter"),
    db: Session = Depends(get_db)
):
    """Get sales forecast"""
    try:
        analytics = AdvancedAnalytics(db)
        forecast = analytics.generate_sales_forecast(
            product_id=product_id,
            store_id=store_id,
            days_ahead=days_ahead
        )
        
        if "error" in forecast:
            raise HTTPException(status_code=500, detail=forecast["error"])
        
        return {
            "forecast": forecast,
            "parameters": {
                "days_ahead": days_ahead,
                "product_id": product_id,
                "store_id": store_id
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating forecast: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Transaction Endpoints
@app.get("/api/v1/transactions")
async def get_transactions(
    skip: int = Query(0, description="Number of records to skip"),
    limit: int = Query(100, description="Number of records to return"),
    start_date: Optional[str] = Query(None, description="Start date filter"),
    end_date: Optional[str] = Query(None, description="End date filter"),
    customer_id: Optional[str] = Query(None, description="Customer ID filter"),
    product_id: Optional[str] = Query(None, description="Product ID filter"),
    store_id: Optional[str] = Query(None, description="Store ID filter"),
    db: Session = Depends(get_db)
):
    """Get transactions with filters"""
    try:
        query = db.query(Transaction)
        
        # Apply filters
        if start_date:
            query = query.filter(Transaction.transaction_date >= start_date)
        if end_date:
            query = query.filter(Transaction.transaction_date <= end_date)
        if customer_id:
            query = query.filter(Transaction.customer_id == customer_id)
        if product_id:
            query = query.filter(Transaction.product_id == product_id)
        if store_id:
            query = query.filter(Transaction.store_id == store_id)
        
        # Get total count
        total_count = query.count()
        
        # Apply pagination
        transactions = query.offset(skip).limit(limit).all()
        
        return {
            "transactions": [
                {
                    "id": t.id,
                    "transaction_id": t.transaction_id,
                    "customer_id": t.customer_id,
                    "product_id": t.product_id,
                    "product_name": t.product_name,
                    "category": t.category,
                    "quantity": t.quantity,
                    "unit_price": t.unit_price,
                    "total_amount": t.total_amount,
                    "transaction_date": t.transaction_date.isoformat(),
                    "store_id": t.store_id,
                    "store_name": t.store_name,
                    "payment_method": t.payment_method
                }
                for t in transactions
            ],
            "pagination": {
                "skip": skip,
                "limit": limit,
                "total_count": total_count,
                "has_more": skip + limit < total_count
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting transactions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/transactions/{transaction_id}")
async def get_transaction(
    transaction_id: str,
    db: Session = Depends(get_db)
):
    """Get specific transaction"""
    try:
        transaction = db.query(Transaction).filter(
            Transaction.transaction_id == transaction_id
        ).first()
        
        if not transaction:
            raise HTTPException(status_code=404, detail="Transaction not found")
        
        return {
            "transaction": {
                "id": transaction.id,
                "transaction_id": transaction.transaction_id,
                "customer_id": transaction.customer_id,
                "product_id": transaction.product_id,
                "product_name": transaction.product_name,
                "category": transaction.category,
                "subcategory": transaction.subcategory,
                "brand": transaction.brand,
                "quantity": transaction.quantity,
                "unit_price": transaction.unit_price,
                "total_amount": transaction.total_amount,
                "discount_amount": transaction.discount_amount,
                "tax_amount": transaction.tax_amount,
                "transaction_date": transaction.transaction_date.isoformat(),
                "store_id": transaction.store_id,
                "store_name": transaction.store_name,
                "sales_person": transaction.sales_person,
                "payment_method": transaction.payment_method,
                "created_at": transaction.created_at.isoformat()
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting transaction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Customer Endpoints
@app.get("/api/v1/customers")
async def get_customers(
    skip: int = Query(0, description="Number of records to skip"),
    limit: int = Query(100, description="Number of records to return"),
    segment: Optional[str] = Query(None, description="RFM segment filter"),
    db: Session = Depends(get_db)
):
    """Get customers with filters"""
    try:
        query = db.query(Customer)
        
        if segment:
            query = query.filter(Customer.rfm_segment == segment)
        
        total_count = query.count()
        customers = query.offset(skip).limit(limit).all()
        
        return {
            "customers": [
                {
                    "id": c.id,
                    "customer_id": c.customer_id,
                    "first_name": c.first_name,
                    "last_name": c.last_name,
                    "email": c.email,
                    "phone": c.phone,
                    "city": c.city,
                    "state": c.state,
                    "country": c.country,
                    "recency_score": c.recency_score,
                    "frequency_score": c.frequency_score,
                    "monetary_score": c.monetary_score,
                    "rfm_segment": c.rfm_segment,
                    "customer_lifetime_value": c.customer_lifetime_value,
                    "total_orders": c.total_orders,
                    "total_spent": c.total_spent,
                    "avg_order_value": c.avg_order_value,
                    "last_purchase_date": c.last_purchase_date.isoformat() if c.last_purchase_date else None,
                    "first_purchase_date": c.first_purchase_date.isoformat() if c.first_purchase_date else None,
                    "is_active": c.is_active,
                    "churn_probability": c.churn_probability
                }
                for c in customers
            ],
            "pagination": {
                "skip": skip,
                "limit": limit,
                "total_count": total_count,
                "has_more": skip + limit < total_count
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting customers: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/customers/{customer_id}")
async def get_customer(
    customer_id: str,
    db: Session = Depends(get_db)
):
    """Get specific customer"""
    try:
        customer = db.query(Customer).filter(
            Customer.customer_id == customer_id
        ).first()
        
        if not customer:
            raise HTTPException(status_code=404, detail="Customer not found")
        
        return {
            "customer": {
                "id": customer.id,
                "customer_id": customer.customer_id,
                "first_name": customer.first_name,
                "last_name": customer.last_name,
                "email": customer.email,
                "phone": customer.phone,
                "address": customer.address,
                "city": customer.city,
                "state": customer.state,
                "country": customer.country,
                "postal_code": customer.postal_code,
                "recency_score": customer.recency_score,
                "frequency_score": customer.frequency_score,
                "monetary_score": customer.monetary_score,
                "rfm_segment": customer.rfm_segment,
                "customer_lifetime_value": customer.customer_lifetime_value,
                "total_orders": customer.total_orders,
                "total_spent": customer.total_spent,
                "avg_order_value": customer.avg_order_value,
                "last_purchase_date": customer.last_purchase_date.isoformat() if customer.last_purchase_date else None,
                "first_purchase_date": customer.first_purchase_date.isoformat() if customer.first_purchase_date else None,
                "is_active": customer.is_active,
                "churn_probability": customer.churn_probability,
                "created_at": customer.created_at.isoformat()
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting customer: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Product Endpoints
@app.get("/api/v1/products")
async def get_products(
    skip: int = Query(0, description="Number of records to skip"),
    limit: int = Query(100, description="Number of records to return"),
    category: Optional[str] = Query(None, description="Category filter"),
    brand: Optional[str] = Query(None, description="Brand filter"),
    db: Session = Depends(get_db)
):
    """Get products with filters"""
    try:
        query = db.query(Product)
        
        if category:
            query = query.filter(Product.category == category)
        if brand:
            query = query.filter(Product.brand == brand)
        
        total_count = query.count()
        products = query.offset(skip).limit(limit).all()
        
        return {
            "products": [
                {
                    "id": p.id,
                    "product_id": p.product_id,
                    "product_name": p.product_name,
                    "category": p.category,
                    "subcategory": p.subcategory,
                    "brand": p.brand,
                    "description": p.description,
                    "unit_price": p.unit_price,
                    "cost_price": p.cost_price,
                    "profit_margin": p.profit_margin,
                    "total_quantity_sold": p.total_quantity_sold,
                    "total_revenue": p.total_revenue,
                    "avg_rating": p.avg_rating,
                    "stock_quantity": p.stock_quantity,
                    "is_active": p.is_active,
                    "created_at": p.created_at.isoformat()
                }
                for p in products
            ],
            "pagination": {
                "skip": skip,
                "limit": limit,
                "total_count": total_count,
                "has_more": skip + limit < total_count
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting products: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Store Endpoints
@app.get("/api/v1/stores")
async def get_stores(
    skip: int = Query(0, description="Number of records to skip"),
    limit: int = Query(100, description="Number of records to return"),
    db: Session = Depends(get_db)
):
    """Get stores"""
    try:
        query = db.query(Store)
        total_count = query.count()
        stores = query.offset(skip).limit(limit).all()
        
        return {
            "stores": [
                {
                    "id": s.id,
                    "store_id": s.store_id,
                    "store_name": s.store_name,
                    "address": s.address,
                    "city": s.city,
                    "state": s.state,
                    "country": s.country,
                    "postal_code": s.postal_code,
                    "phone": s.phone,
                    "manager_name": s.manager_name,
                    "total_sales": s.total_sales,
                    "total_transactions": s.total_transactions,
                    "avg_transaction_value": s.avg_transaction_value,
                    "is_active": s.is_active,
                    "created_at": s.created_at.isoformat()
                }
                for s in stores
            ],
            "pagination": {
                "skip": skip,
                "limit": limit,
                "total_count": total_count,
                "has_more": skip + limit < total_count
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting stores: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Dashboard Data Endpoints
@app.get("/api/v1/dashboard/metrics")
async def get_dashboard_metrics(
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    db: Session = Depends(get_db)
):
    """Get dashboard metrics"""
    try:
        analytics = AdvancedAnalytics(db)
        insights = analytics.get_sales_insights()
        
        return {
            "metrics": insights.get("sales_metrics", {}),
            "top_products": insights.get("top_products", []),
            "top_customers": insights.get("top_customers", []),
            "sales_trends": insights.get("sales_trends", {}),
            "seasonal_patterns": insights.get("seasonal_patterns", {}),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting dashboard metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/dashboard/charts")
async def get_dashboard_charts(
    chart_type: str = Query("revenue", description="Chart type"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    db: Session = Depends(get_db)
):
    """Get dashboard chart data"""
    try:
        # This would generate chart data based on the chart_type
        # For now, returning sample data
        chart_data = {
            "revenue": {
                "labels": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
                "data": [100000, 120000, 110000, 130000, 140000, 150000]
            },
            "transactions": {
                "labels": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
                "data": [1000, 1200, 1100, 1300, 1400, 1500]
            },
            "customers": {
                "labels": ["Champions", "Loyal", "Potential", "New", "At Risk"],
                "data": [150, 300, 200, 100, 80]
            }
        }
        
        return {
            "chart_type": chart_type,
            "data": chart_data.get(chart_type, {}),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting chart data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Resource not found", "timestamp": datetime.now().isoformat()}
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "timestamp": datetime.now().isoformat()}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
