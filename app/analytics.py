"""
Advanced analytics engine for sales data
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_
from app.models import Transaction, Customer, Product, Store
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

logger = logging.getLogger(__name__)


class AdvancedAnalytics:
    """Advanced analytics engine for sales data"""
    
    def __init__(self, db_session: Session):
        self.db = db_session
        
    def perform_rfm_analysis(self) -> Dict[str, Any]:
        """
        Perform RFM (Recency, Frequency, Monetary) Analysis
        
        Returns:
            Dictionary with RFM analysis results
        """
        try:
            # Get customer transaction data
            customer_data = self._get_customer_transaction_data()
            
            if customer_data.empty:
                return {"error": "No customer data found"}
            
            # Calculate RFM metrics
            rfm_data = self._calculate_rfm_metrics(customer_data)
            
            # Perform customer segmentation
            segments = self._segment_customers(rfm_data)
            
            # Update customer records with RFM scores
            self._update_customer_rfm_scores(rfm_data, segments)
            
            return {
                "total_customers": len(rfm_data),
                "segments": segments,
                "rfm_summary": self._generate_rfm_summary(rfm_data, segments)
            }
            
        except Exception as e:
            logger.error(f"Error in RFM analysis: {str(e)}")
            return {"error": str(e)}
    
    def _get_customer_transaction_data(self) -> pd.DataFrame:
        """Get customer transaction data for RFM analysis"""
        query = """
        SELECT 
            t.customer_id,
            t.transaction_date,
            t.total_amount,
            c.first_purchase_date,
            c.last_purchase_date
        FROM transactions t
        LEFT JOIN customers c ON t.customer_id = c.customer_id
        ORDER BY t.customer_id, t.transaction_date
        """
        
        return pd.read_sql(query, self.db.bind)
    
    def _calculate_rfm_metrics(self, customer_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate RFM metrics for each customer"""
        # Convert dates
        customer_data['transaction_date'] = pd.to_datetime(customer_data['transaction_date'])
        
        # Calculate metrics
        rfm_data = customer_data.groupby('customer_id').agg({
            'transaction_date': ['max', 'count'],
            'total_amount': 'sum'
        }).reset_index()
        
        # Flatten column names
        rfm_data.columns = ['customer_id', 'last_purchase', 'frequency', 'monetary']
        
        # Calculate recency (days since last purchase)
        current_date = datetime.now()
        rfm_data['recency'] = (current_date - rfm_data['last_purchase']).dt.days
        
        # Remove customers with no recent activity (more than 2 years)
        rfm_data = rfm_data[rfm_data['recency'] <= 730]
        
        return rfm_data
    
    def _segment_customers(self, rfm_data: pd.DataFrame) -> Dict[str, Any]:
        """Segment customers using RFM analysis"""
        # Create RFM scores (1-5 scale)
        rfm_data['recency_score'] = pd.qcut(rfm_data['recency'], 5, labels=[5,4,3,2,1])
        rfm_data['frequency_score'] = pd.qcut(rfm_data['frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
        rfm_data['monetary_score'] = pd.qcut(rfm_data['monetary'], 5, labels=[1,2,3,4,5])
        
        # Convert to numeric
        rfm_data['recency_score'] = rfm_data['recency_score'].astype(int)
        rfm_data['frequency_score'] = rfm_data['frequency_score'].astype(int)
        rfm_data['monetary_score'] = rfm_data['monetary_score'].astype(int)
        
        # Create RFM segments
        rfm_data['rfm_segment'] = rfm_data.apply(self._assign_rfm_segment, axis=1)
        
        # Segment distribution
        segment_distribution = rfm_data['rfm_segment'].value_counts().to_dict()
        
        return {
            "segment_distribution": segment_distribution,
            "rfm_data": rfm_data.to_dict('records')
        }
    
    def _assign_rfm_segment(self, row) -> str:
        """Assign RFM segment based on scores"""
        r, f, m = row['recency_score'], row['frequency_score'], row['monetary_score']
        
        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"
        elif r >= 3 and f >= 3 and m >= 3:
            return "Loyal Customers"
        elif r >= 4 and f <= 2 and m <= 2:
            return "New Customers"
        elif r >= 3 and f <= 2 and m <= 2:
            return "Potential Loyalists"
        elif r <= 2 and f >= 3 and m >= 3:
            return "At Risk"
        elif r <= 2 and f >= 4 and m >= 4:
            return "Cannot Lose Them"
        elif r <= 2 and f <= 2 and m <= 2:
            return "Lost"
        else:
            return "Others"
    
    def _update_customer_rfm_scores(self, rfm_data: pd.DataFrame, segments: Dict[str, Any]):
        """Update customer records with RFM scores"""
        for _, row in rfm_data.iterrows():
            customer = self.db.query(Customer).filter(
                Customer.customer_id == row['customer_id']
            ).first()
            
            if customer:
                customer.recency_score = row['recency_score']
                customer.frequency_score = row['frequency_score']
                customer.monetary_score = row['monetary_score']
                customer.rfm_segment = row['rfm_segment']
                customer.total_orders = row['frequency']
                customer.total_spent = row['monetary']
                customer.avg_order_value = row['monetary'] / row['frequency']
                customer.last_purchase_date = row['last_purchase']
        
        self.db.commit()
    
    def _generate_rfm_summary(self, rfm_data: pd.DataFrame, segments: Dict[str, Any]) -> Dict[str, Any]:
        """Generate RFM analysis summary"""
        return {
            "avg_recency": rfm_data['recency'].mean(),
            "avg_frequency": rfm_data['frequency'].mean(),
            "avg_monetary": rfm_data['monetary'].mean(),
            "total_customers": len(rfm_data),
            "high_value_customers": len(rfm_data[rfm_data['monetary'] > rfm_data['monetary'].quantile(0.8)]),
            "at_risk_customers": len(rfm_data[rfm_data['rfm_segment'] == 'At Risk']),
            "champion_customers": len(rfm_data[rfm_data['rfm_segment'] == 'Champions'])
        }
    
    def calculate_customer_lifetime_value(self) -> Dict[str, Any]:
        """Calculate Customer Lifetime Value (CLV)"""
        try:
            # Get customer data with purchase history
            customer_data = self._get_customer_clv_data()
            
            if customer_data.empty:
                return {"error": "No customer data found"}
            
            # Calculate CLV using different methods
            clv_results = self._calculate_clv_metrics(customer_data)
            
            # Update customer records
            self._update_customer_clv(clv_results)
            
            return {
                "total_customers": len(clv_results),
                "avg_clv": clv_results['clv'].mean(),
                "high_clv_customers": len(clv_results[clv_results['clv'] > clv_results['clv'].quantile(0.8)]),
                "clv_distribution": clv_results['clv'].describe().to_dict()
            }
            
        except Exception as e:
            logger.error(f"Error calculating CLV: {str(e)}")
            return {"error": str(e)}
    
    def _get_customer_clv_data(self) -> pd.DataFrame:
        """Get customer data for CLV calculation"""
        query = """
        SELECT 
            customer_id,
            COUNT(*) as total_orders,
            SUM(total_amount) as total_spent,
            AVG(total_amount) as avg_order_value,
            MIN(transaction_date) as first_purchase,
            MAX(transaction_date) as last_purchase,
            DATEDIFF(MAX(transaction_date), MIN(transaction_date)) as customer_lifespan_days
        FROM transactions
        GROUP BY customer_id
        """
        
        return pd.read_sql(query, self.db.bind)
    
    def _calculate_clv_metrics(self, customer_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate CLV metrics"""
        # Simple CLV calculation: (Average Order Value × Purchase Frequency × Customer Lifespan)
        customer_data['purchase_frequency'] = customer_data['total_orders'] / (customer_data['customer_lifespan_days'] / 365)
        customer_data['clv'] = customer_data['avg_order_value'] * customer_data['purchase_frequency'] * 2  # 2 years projection
        
        # Advanced CLV with RFM
        customer_data['clv_rfm'] = customer_data.apply(self._calculate_rfm_clv, axis=1)
        
        return customer_data
    
    def _calculate_rfm_clv(self, row) -> float:
        """Calculate CLV based on RFM scores"""
        # Get RFM scores from database
        customer = self.db.query(Customer).filter(
            Customer.customer_id == row['customer_id']
        ).first()
        
        if customer and customer.recency_score and customer.frequency_score and customer.monetary_score:
            # Weighted CLV based on RFM
            rfm_score = (customer.recency_score + customer.frequency_score + customer.monetary_score) / 3
            return row['avg_order_value'] * row['purchase_frequency'] * (rfm_score / 5) * 2
        else:
            return row['clv']
    
    def _update_customer_clv(self, clv_results: pd.DataFrame):
        """Update customer records with CLV"""
        for _, row in clv_results.iterrows():
            customer = self.db.query(Customer).filter(
                Customer.customer_id == row['customer_id']
            ).first()
            
            if customer:
                customer.customer_lifetime_value = row['clv']
        
        self.db.commit()
    
    def generate_sales_forecast(self, product_id: Optional[str] = None, 
                              store_id: Optional[str] = None, 
                              days_ahead: int = 30) -> Dict[str, Any]:
        """Generate sales forecast using time series analysis"""
        try:
            # Get historical sales data
            sales_data = self._get_sales_forecast_data(product_id, store_id)
            
            if sales_data.empty:
                return {"error": "No sales data found"}
            
            # Prepare time series data
            ts_data = self._prepare_time_series_data(sales_data)
            
            # Generate forecast
            forecast = self._generate_time_series_forecast(ts_data, days_ahead)
            
            return {
                "forecast_period": days_ahead,
                "forecast_data": forecast,
                "model_accuracy": self._calculate_forecast_accuracy(ts_data)
            }
            
        except Exception as e:
            logger.error(f"Error generating sales forecast: {str(e)}")
            return {"error": str(e)}
    
    def _get_sales_forecast_data(self, product_id: Optional[str], store_id: Optional[str]) -> pd.DataFrame:
        """Get sales data for forecasting"""
        query = """
        SELECT 
            DATE(transaction_date) as date,
            SUM(total_amount) as daily_sales,
            COUNT(*) as daily_transactions
        FROM transactions
        WHERE transaction_date >= DATE_SUB(NOW(), INTERVAL 1 YEAR)
        """
        
        params = []
        if product_id:
            query += " AND product_id = %s"
            params.append(product_id)
        if store_id:
            query += " AND store_id = %s"
            params.append(store_id)
            
        query += " GROUP BY DATE(transaction_date) ORDER BY date"
        
        return pd.read_sql(query, self.db.bind, params=params)
    
    def _prepare_time_series_data(self, sales_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare time series data for forecasting"""
        sales_data['date'] = pd.to_datetime(sales_data['date'])
        sales_data = sales_data.set_index('date')
        
        # Fill missing dates with 0 sales
        date_range = pd.date_range(start=sales_data.index.min(), end=sales_data.index.max(), freq='D')
        sales_data = sales_data.reindex(date_range, fill_value=0)
        
        return sales_data
    
    def _generate_time_series_forecast(self, ts_data: pd.DataFrame, days_ahead: int) -> List[Dict[str, Any]]:
        """Generate time series forecast using simple moving average"""
        # Simple moving average forecast
        window_size = min(30, len(ts_data) // 4)
        moving_avg = ts_data['daily_sales'].rolling(window=window_size).mean()
        
        # Get the last known moving average
        last_ma = moving_avg.dropna().iloc[-1]
        
        # Generate forecast
        forecast_dates = pd.date_range(start=ts_data.index[-1] + timedelta(days=1), periods=days_ahead, freq='D')
        forecast_values = [last_ma] * days_ahead
        
        forecast_data = []
        for date, value in zip(forecast_dates, forecast_values):
            forecast_data.append({
                "date": date.strftime('%Y-%m-%d'),
                "predicted_sales": round(value, 2),
                "confidence_lower": round(value * 0.8, 2),
                "confidence_upper": round(value * 1.2, 2)
            })
        
        return forecast_data
    
    def _calculate_forecast_accuracy(self, ts_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate forecast accuracy using historical data"""
        # Simple accuracy calculation
        actual_sales = ts_data['daily_sales'].values
        mean_sales = np.mean(actual_sales)
        
        # Calculate MAPE (Mean Absolute Percentage Error) for naive forecast
        naive_forecast = np.full_like(actual_sales, mean_sales)
        mape = np.mean(np.abs((actual_sales - naive_forecast) / actual_sales)) * 100
        
        return {
            "mape": round(mape, 2),
            "rmse": round(np.sqrt(np.mean((actual_sales - naive_forecast) ** 2)), 2),
            "r2_score": round(1 - (np.sum((actual_sales - naive_forecast) ** 2) / np.sum((actual_sales - mean_sales) ** 2)), 2)
        }
    
    def get_sales_insights(self) -> Dict[str, Any]:
        """Get comprehensive sales insights"""
        try:
            insights = {}
            
            # Basic sales metrics
            insights['sales_metrics'] = self._get_sales_metrics()
            
            # Top products
            insights['top_products'] = self._get_top_products()
            
            # Top customers
            insights['top_customers'] = self._get_top_customers()
            
            # Sales trends
            insights['sales_trends'] = self._get_sales_trends()
            
            # Seasonal patterns
            insights['seasonal_patterns'] = self._get_seasonal_patterns()
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating sales insights: {str(e)}")
            return {"error": str(e)}
    
    def _get_sales_metrics(self) -> Dict[str, Any]:
        """Get basic sales metrics"""
        query = """
        SELECT 
            COUNT(*) as total_transactions,
            SUM(total_amount) as total_revenue,
            AVG(total_amount) as avg_transaction_value,
            COUNT(DISTINCT customer_id) as unique_customers,
            COUNT(DISTINCT product_id) as unique_products,
            COUNT(DISTINCT store_id) as unique_stores
        FROM transactions
        """
        
        result = self.db.execute(query).fetchone()
        
        return {
            "total_transactions": result[0],
            "total_revenue": float(result[1]) if result[1] else 0,
            "avg_transaction_value": float(result[2]) if result[2] else 0,
            "unique_customers": result[3],
            "unique_products": result[4],
            "unique_stores": result[5]
        }
    
    def _get_top_products(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top selling products"""
        query = """
        SELECT 
            product_id,
            product_name,
            category,
            brand,
            SUM(quantity) as total_quantity,
            SUM(total_amount) as total_revenue,
            COUNT(*) as total_orders
        FROM transactions
        GROUP BY product_id, product_name, category, brand
        ORDER BY total_revenue DESC
        LIMIT %s
        """
        
        results = self.db.execute(query, (limit,)).fetchall()
        
        return [
            {
                "product_id": row[0],
                "product_name": row[1],
                "category": row[2],
                "brand": row[3],
                "total_quantity": row[4],
                "total_revenue": float(row[5]),
                "total_orders": row[6]
            }
            for row in results
        ]
    
    def _get_top_customers(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top customers by revenue"""
        query = """
        SELECT 
            customer_id,
            SUM(total_amount) as total_spent,
            COUNT(*) as total_orders,
            AVG(total_amount) as avg_order_value,
            MAX(transaction_date) as last_purchase
        FROM transactions
        GROUP BY customer_id
        ORDER BY total_spent DESC
        LIMIT %s
        """
        
        results = self.db.execute(query, (limit,)).fetchall()
        
        return [
            {
                "customer_id": row[0],
                "total_spent": float(row[1]),
                "total_orders": row[2],
                "avg_order_value": float(row[3]),
                "last_purchase": row[4].strftime('%Y-%m-%d') if row[4] else None
            }
            for row in results
        ]
    
    def _get_sales_trends(self) -> Dict[str, Any]:
        """Get sales trends over time"""
        query = """
        SELECT 
            DATE(transaction_date) as date,
            SUM(total_amount) as daily_revenue,
            COUNT(*) as daily_transactions
        FROM transactions
        WHERE transaction_date >= DATE_SUB(NOW(), INTERVAL 30 DAY)
        GROUP BY DATE(transaction_date)
        ORDER BY date
        """
        
        results = self.db.execute(query).fetchall()
        
        return {
            "daily_sales": [
                {
                    "date": row[0].strftime('%Y-%m-%d'),
                    "revenue": float(row[1]),
                    "transactions": row[2]
                }
                for row in results
            ]
        }
    
    def _get_seasonal_patterns(self) -> Dict[str, Any]:
        """Get seasonal sales patterns"""
        query = """
        SELECT 
            MONTH(transaction_date) as month,
            DAYOFWEEK(transaction_date) as day_of_week,
            HOUR(transaction_date) as hour,
            SUM(total_amount) as revenue,
            COUNT(*) as transactions
        FROM transactions
        GROUP BY MONTH(transaction_date), DAYOFWEEK(transaction_date), HOUR(transaction_date)
        ORDER BY month, day_of_week, hour
        """
        
        results = self.db.execute(query).fetchall()
        
        return {
            "monthly_patterns": [
                {
                    "month": row[0],
                    "day_of_week": row[1],
                    "hour": row[2],
                    "revenue": float(row[3]),
                    "transactions": row[4]
                }
                for row in results
            ]
        }
