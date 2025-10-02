"""
Interactive Sales Analytics Dashboard using Streamlit and Plotly
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
from sqlalchemy import create_engine
from app.analytics import AdvancedAnalytics
from app.database import get_db
from app.models import Transaction, Customer, Product, Store
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Sales Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

class SalesDashboard:
    """Interactive Sales Analytics Dashboard"""
    
    def __init__(self):
        self.db = next(get_db())
        self.analytics = AdvancedAnalytics(self.db)
        
    def run(self):
        """Main dashboard function"""
        # Header
        st.markdown('<h1 class="main-header">📊 Sales Analytics Dashboard</h1>', unsafe_allow_html=True)
        
        # Sidebar
        self.create_sidebar()
        
        # Main content based on sidebar selection
        page = st.session_state.get('page', 'Overview')
        
        if page == 'Overview':
            self.show_overview()
        elif page == 'Customer Analytics':
            self.show_customer_analytics()
        elif page == 'Product Analytics':
            self.show_product_analytics()
        elif page == 'Sales Forecasting':
            self.show_sales_forecasting()
        elif page == 'RFM Analysis':
            self.show_rfm_analysis()
        elif page == 'Real-time Analytics':
            self.show_realtime_analytics()
    
    def create_sidebar(self):
        """Create sidebar navigation"""
        st.sidebar.title("📈 Navigation")
        
        pages = [
            "Overview",
            "Customer Analytics", 
            "Product Analytics",
            "Sales Forecasting",
            "RFM Analysis",
            "Real-time Analytics"
        ]
        
        # Initialize page if not exists
        if 'page' not in st.session_state:
            st.session_state.page = "Overview"
        
        selected_page = st.sidebar.selectbox("Select Page", pages, 
                                           index=pages.index(st.session_state.page) if st.session_state.page in pages else 0)
        st.session_state.page = selected_page
        
        # Date range filter
        st.sidebar.markdown("---")
        st.sidebar.subheader("📅 Date Range")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=30))
        with col2:
            end_date = st.date_input("End Date", value=datetime.now())
        
        st.session_state.date_range = (start_date, end_date)
        
        # Additional filters
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔍 Filters")
        
        # Store filter
        stores = self.get_stores()
        selected_stores = st.sidebar.multiselect("Select Stores", stores, default=stores)
        st.session_state.selected_stores = selected_stores
        
        # Category filter
        categories = self.get_categories()
        selected_categories = st.sidebar.multiselect("Select Categories", categories, default=categories)
        st.session_state.selected_categories = selected_categories
    
    def show_overview(self):
        """Show overview dashboard"""
        st.subheader("📊 Sales Overview")
        
        # Key metrics
        metrics = self.get_key_metrics()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Revenue",
                value=f"${metrics['total_revenue']:,.2f}",
                delta=f"{metrics['revenue_growth']:+.1f}%"
            )
        
        with col2:
            st.metric(
                label="Total Transactions",
                value=f"{metrics['total_transactions']:,}",
                delta=f"{metrics['transaction_growth']:+.1f}%"
            )
        
        with col3:
            st.metric(
                label="Average Order Value",
                value=f"${metrics['avg_order_value']:.2f}",
                delta=f"{metrics['aov_growth']:+.1f}%"
            )
        
        with col4:
            st.metric(
                label="Active Customers",
                value=f"{metrics['active_customers']:,}",
                delta=f"{metrics['customer_growth']:+.1f}%"
            )
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Revenue Trend")
            revenue_chart = self.create_revenue_trend_chart()
            st.plotly_chart(revenue_chart, use_container_width=True)
        
        with col2:
            st.subheader("🛒 Transaction Volume")
            transaction_chart = self.create_transaction_volume_chart()
            st.plotly_chart(transaction_chart, use_container_width=True)
        
        # Top products and customers
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Top Products")
            top_products = self.get_top_products(10)
            st.dataframe(top_products, use_container_width=True)
        
        with col2:
            st.subheader("👥 Top Customers")
            top_customers = self.get_top_customers(10)
            st.dataframe(top_customers, use_container_width=True)
    
    def show_customer_analytics(self):
        """Show customer analytics"""
        st.subheader("👥 Customer Analytics")
        
        # Customer segments
        st.subheader("🎯 Customer Segmentation")
        segment_chart = self.create_customer_segment_chart()
        st.plotly_chart(segment_chart, use_container_width=True)
        
        # Customer lifetime value
        st.subheader("💰 Customer Lifetime Value")
        clv_data = self.get_customer_lifetime_value()
        
        col1, col2 = st.columns(2)
        
        with col1:
            clv_chart = self.create_clv_distribution_chart(clv_data)
            st.plotly_chart(clv_chart, use_container_width=True)
        
        with col2:
            st.subheader("📊 CLV Statistics")
            st.metric("Average CLV", f"${clv_data['avg_clv']:,.2f}")
            st.metric("High Value Customers", f"{clv_data['high_clv_customers']:,}")
            st.metric("Total CLV", f"${clv_data['total_clv']:,.2f}")
        
        # Customer behavior analysis
        st.subheader("🔍 Customer Behavior Analysis")
        behavior_chart = self.create_customer_behavior_chart()
        st.plotly_chart(behavior_chart, use_container_width=True)
    
    def show_product_analytics(self):
        """Show product analytics"""
        st.subheader("🛍️ Product Analytics")
        
        # Product performance
        st.subheader("📊 Product Performance")
        product_performance = self.get_product_performance()
        
        col1, col2 = st.columns(2)
        
        with col1:
            performance_chart = self.create_product_performance_chart(product_performance)
            st.plotly_chart(performance_chart, use_container_width=True)
        
        with col2:
            category_chart = self.create_category_analysis_chart()
            st.plotly_chart(category_chart, use_container_width=True)
        
        # Product trends
        st.subheader("📈 Product Trends")
        trend_chart = self.create_product_trend_chart()
        st.plotly_chart(trend_chart, use_container_width=True)
        
        # Product recommendations
        st.subheader("💡 Product Recommendations")
        recommendations = self.get_product_recommendations()
        st.dataframe(recommendations, use_container_width=True)
    
    def show_sales_forecasting(self):
        """Show sales forecasting"""
        st.subheader("🔮 Sales Forecasting")
        
        # Forecast parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            forecast_days = st.selectbox("Forecast Period", [7, 14, 30, 60, 90], index=2)
        
        with col2:
            products = self.get_products()
            selected_product = st.selectbox("Product (Optional)", ["All"] + products)
        
        with col3:
            stores = self.get_stores()
            selected_store = st.selectbox("Store (Optional)", ["All"] + stores)
        
        # Generate forecast
        if st.button("Generate Forecast", type="primary"):
            with st.spinner("Generating forecast..."):
                forecast_data = self.generate_sales_forecast(
                    forecast_days, 
                    selected_product if selected_product != "All" else None,
                    selected_store if selected_store != "All" else None
                )
                
                if "error" not in forecast_data:
                    # Display forecast
                    st.subheader("📊 Sales Forecast")
                    forecast_chart = self.create_forecast_chart(forecast_data)
                    st.plotly_chart(forecast_chart, use_container_width=True)
                    
                    # Forecast metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Predicted Revenue", f"${forecast_data['total_predicted_revenue']:,.2f}")
                    with col2:
                        st.metric("Confidence Level", f"{forecast_data['confidence_level']:.1f}%")
                    with col3:
                        st.metric("Model Accuracy", f"{forecast_data['model_accuracy']:.1f}%")
                else:
                    st.error(f"Error generating forecast: {forecast_data['error']}")
    
    def show_rfm_analysis(self):
        """Show RFM analysis"""
        st.subheader("🎯 RFM Analysis")
        
        # Perform RFM analysis
        if st.button("Run RFM Analysis", type="primary"):
            with st.spinner("Running RFM analysis..."):
                rfm_results = self.analytics.perform_rfm_analysis()
                
                if "error" not in rfm_results:
                    # RFM segments
                    st.subheader("📊 Customer Segments")
                    segment_chart = self.create_rfm_segment_chart(rfm_results)
                    st.plotly_chart(segment_chart, use_container_width=True)
                    
                    # RFM matrix
                    st.subheader("🎯 RFM Matrix")
                    rfm_matrix = self.create_rfm_matrix(rfm_results)
                    st.plotly_chart(rfm_matrix, use_container_width=True)
                    
                    # Segment details
                    st.subheader("📋 Segment Details")
                    segment_details = self.get_rfm_segment_details(rfm_results)
                    st.dataframe(segment_details, use_container_width=True)
                else:
                    st.error(f"Error in RFM analysis: {rfm_results['error']}")
    
    def show_realtime_analytics(self):
        """Show real-time analytics"""
        st.subheader("⚡ Real-time Analytics")
        
        # Auto-refresh
        auto_refresh = st.checkbox("Auto-refresh (30 seconds)", value=False)
        
        if auto_refresh:
            st.rerun()
        
        # Real-time metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Live Transactions", "1,234", delta="+12")
        with col2:
            st.metric("Live Revenue", "$45,678", delta="+$1,234")
        with col3:
            st.metric("Active Users", "89", delta="+5")
        with col4:
            st.metric("Conversion Rate", "3.2%", delta="+0.1%")
        
        # Real-time charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Live Sales")
            live_sales_chart = self.create_live_sales_chart()
            st.plotly_chart(live_sales_chart, use_container_width=True)
        
        with col2:
            st.subheader("🛒 Live Transactions")
            live_transactions_chart = self.create_live_transactions_chart()
            st.plotly_chart(live_transactions_chart, use_container_width=True)
    
    # Helper methods for data retrieval and chart creation
    def get_key_metrics(self):
        """Get key performance metrics"""
        # This would typically query the database
        return {
            "total_revenue": 1250000.00,
            "revenue_growth": 12.5,
            "total_transactions": 15420,
            "transaction_growth": 8.3,
            "avg_order_value": 81.05,
            "aov_growth": 3.9,
            "active_customers": 2840,
            "customer_growth": 15.2
        }
    
    def create_revenue_trend_chart(self):
        """Create revenue trend chart"""
        # Sample data - replace with actual database query
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        revenue = np.random.normal(40000, 5000, len(dates)).cumsum()
        
        fig = px.line(
            x=dates, 
            y=revenue,
            title="Daily Revenue Trend",
            labels={'x': 'Date', 'y': 'Revenue ($)'}
        )
        fig.update_layout(showlegend=False)
        return fig
    
    def create_transaction_volume_chart(self):
        """Create transaction volume chart"""
        # Sample data
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        transactions = np.random.poisson(500, len(dates))
        
        fig = px.bar(
            x=dates,
            y=transactions,
            title="Daily Transaction Volume",
            labels={'x': 'Date', 'y': 'Transactions'}
        )
        return fig
    
    def get_top_products(self, limit=10):
        """Get top products"""
        # Sample data - replace with actual database query
        return pd.DataFrame({
            'Product': [f'Product {i}' for i in range(1, limit+1)],
            'Revenue': np.random.uniform(10000, 50000, limit),
            'Quantity': np.random.randint(100, 1000, limit)
        })
    
    def get_top_customers(self, limit=10):
        """Get top customers"""
        # Sample data
        return pd.DataFrame({
            'Customer': [f'Customer {i}' for i in range(1, limit+1)],
            'Total Spent': np.random.uniform(5000, 25000, limit),
            'Orders': np.random.randint(10, 100, limit)
        })
    
    def get_stores(self):
        """Get list of stores"""
        return ["Store A", "Store B", "Store C", "Store D"]
    
    def get_categories(self):
        """Get list of categories"""
        return ["Electronics", "Clothing", "Home & Garden", "Sports", "Books"]
    
    def get_products(self):
        """Get list of products"""
        return [f"Product {i}" for i in range(1, 21)]
    
    def create_customer_segment_chart(self):
        """Create customer segment chart"""
        segments = ['Champions', 'Loyal Customers', 'Potential Loyalists', 'New Customers', 'At Risk', 'Cannot Lose Them', 'Lost']
        counts = [150, 300, 200, 100, 80, 50, 20]
        
        fig = px.pie(
            values=counts,
            names=segments,
            title="Customer Segmentation"
        )
        return fig
    
    def get_customer_lifetime_value(self):
        """Get customer lifetime value data"""
        return {
            "avg_clv": 1250.50,
            "high_clv_customers": 150,
            "total_clv": 3550000.00
        }
    
    def create_clv_distribution_chart(self, clv_data):
        """Create CLV distribution chart"""
        # Sample CLV distribution
        clv_values = np.random.lognormal(6, 1, 1000)
        
        fig = px.histogram(
            x=clv_values,
            nbins=50,
            title="Customer Lifetime Value Distribution",
            labels={'x': 'CLV ($)', 'y': 'Count'}
        )
        return fig
    
    def create_customer_behavior_chart(self):
        """Create customer behavior chart"""
        behaviors = ['High Frequency', 'High Value', 'Recent', 'At Risk']
        percentages = [25, 30, 35, 10]
        
        fig = px.bar(
            x=behaviors,
            y=percentages,
            title="Customer Behavior Analysis",
            labels={'x': 'Behavior Type', 'y': 'Percentage (%)'}
        )
        return fig
    
    def get_product_performance(self):
        """Get product performance data"""
        return pd.DataFrame({
            'Product': [f'Product {i}' for i in range(1, 11)],
            'Revenue': np.random.uniform(10000, 50000, 10),
            'Units Sold': np.random.randint(100, 1000, 10),
            'Profit Margin': np.random.uniform(0.1, 0.4, 10)
        })
    
    def create_product_performance_chart(self, data):
        """Create product performance chart"""
        fig = px.scatter(
            data,
            x='Units Sold',
            y='Revenue',
            size='Profit Margin',
            hover_data=['Product'],
            title="Product Performance Analysis"
        )
        return fig
    
    def create_category_analysis_chart(self):
        """Create category analysis chart"""
        categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books']
        revenue = [150000, 120000, 80000, 60000, 40000]
        
        fig = px.bar(
            x=categories,
            y=revenue,
            title="Revenue by Category",
            labels={'x': 'Category', 'y': 'Revenue ($)'}
        )
        return fig
    
    def create_product_trend_chart(self):
        """Create product trend chart"""
        # Sample trend data
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        products = ['Product A', 'Product B', 'Product C']
        
        fig = go.Figure()
        for product in products:
            sales = np.random.normal(100, 20, len(dates))
            fig.add_trace(go.Scatter(x=dates, y=sales, name=product, mode='lines'))
        
        fig.update_layout(title="Product Sales Trends", xaxis_title="Date", yaxis_title="Sales")
        return fig
    
    def get_product_recommendations(self):
        """Get product recommendations"""
        return pd.DataFrame({
            'Product': [f'Recommended Product {i}' for i in range(1, 6)],
            'Reason': ['High demand', 'Low stock', 'Seasonal', 'Trending', 'Cross-sell'],
            'Potential Revenue': np.random.uniform(5000, 25000, 5),
            'Confidence': np.random.uniform(0.7, 0.95, 5)
        })
    
    def generate_sales_forecast(self, days, product=None, store=None):
        """Generate sales forecast"""
        # Sample forecast data
        forecast_dates = pd.date_range(start=datetime.now(), periods=days, freq='D')
        predicted_sales = np.random.normal(40000, 5000, days)
        
        return {
            "forecast_data": [
                {
                    "date": date.strftime('%Y-%m-%d'),
                    "predicted_sales": round(sales, 2),
                    "confidence_lower": round(sales * 0.8, 2),
                    "confidence_upper": round(sales * 1.2, 2)
                }
                for date, sales in zip(forecast_dates, predicted_sales)
            ],
            "total_predicted_revenue": sum(predicted_sales),
            "confidence_level": 85.5,
            "model_accuracy": 78.2
        }
    
    def create_forecast_chart(self, forecast_data):
        """Create forecast chart"""
        df = pd.DataFrame(forecast_data['forecast_data'])
        df['date'] = pd.to_datetime(df['date'])
        
        fig = go.Figure()
        
        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['confidence_upper'],
            fill=None,
            mode='lines',
            line_color='rgba(0,100,80,0)',
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['confidence_lower'],
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,100,80,0)',
            name='Confidence Interval'
        ))
        
        # Add forecast line
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['predicted_sales'],
            mode='lines',
            name='Forecast',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title="Sales Forecast",
            xaxis_title="Date",
            yaxis_title="Predicted Sales ($)"
        )
        
        return fig
    
    def create_rfm_segment_chart(self, rfm_results):
        """Create RFM segment chart"""
        segments = list(rfm_results['segments']['segment_distribution'].keys())
        counts = list(rfm_results['segments']['segment_distribution'].values())
        
        fig = px.pie(
            values=counts,
            names=segments,
            title="RFM Customer Segments"
        )
        return fig
    
    def create_rfm_matrix(self, rfm_results):
        """Create RFM matrix heatmap"""
        # Sample RFM matrix data
        rfm_matrix = np.random.randint(1, 6, (5, 5))
        
        fig = px.imshow(
            rfm_matrix,
            labels=dict(x="Frequency", y="Recency"),
            title="RFM Matrix",
            color_continuous_scale="RdYlBu_r"
        )
        return fig
    
    def get_rfm_segment_details(self, rfm_results):
        """Get RFM segment details"""
        return pd.DataFrame({
            'Segment': ['Champions', 'Loyal Customers', 'Potential Loyalists', 'New Customers', 'At Risk'],
            'Count': [150, 300, 200, 100, 80],
            'Percentage': [18.5, 37.0, 24.7, 12.3, 9.9],
            'Avg CLV': [2500, 1800, 1200, 800, 600],
            'Action': ['Retain', 'Upsell', 'Engage', 'Onboard', 'Win Back']
        })
    
    def create_live_sales_chart(self):
        """Create live sales chart"""
        # Sample live data
        times = pd.date_range(start=datetime.now() - timedelta(hours=1), end=datetime.now(), freq='5min')
        sales = np.random.normal(1000, 200, len(times))
        
        fig = px.line(
            x=times,
            y=sales,
            title="Live Sales (Last Hour)",
            labels={'x': 'Time', 'y': 'Sales ($)'}
        )
        return fig
    
    def create_live_transactions_chart(self):
        """Create live transactions chart"""
        # Sample live data
        times = pd.date_range(start=datetime.now() - timedelta(hours=1), end=datetime.now(), freq='5min')
        transactions = np.random.poisson(10, len(times))
        
        fig = px.bar(
            x=times,
            y=transactions,
            title="Live Transactions (Last Hour)",
            labels={'x': 'Time', 'y': 'Transactions'}
        )
        return fig


def main():
    """Main function to run the dashboard"""
    try:
        dashboard = SalesDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"Error running dashboard: {str(e)}")
        logger.error(f"Dashboard error: {str(e)}")


if __name__ == "__main__":
    main()
