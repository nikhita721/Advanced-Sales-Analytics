# 🚀 Advanced Sales Analytics Platform

A comprehensive retail analytics platform built with FastAPI, Streamlit, and advanced machine learning capabilities for sales data analysis and business intelligence.

## 🚀 **Quick Access**

After setup, access your analytics platform at:
- **📊 Dashboard**: [http://localhost:8501](http://localhost:8501)
- **🔌 API**: [http://localhost:8000](http://localhost:8000)
- **📚 API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

## ✨ Features

### 📊 Analytics Dashboard
- Interactive Streamlit dashboard with real-time visualizations
- Sales performance metrics and KPIs
- Customer behavior analysis
- Product performance tracking

### 🤖 Machine Learning Analytics
- **RFM Analysis**: Customer segmentation based on Recency, Frequency, and Monetary value
- **Customer Lifetime Value (CLV)**: Predictive customer value modeling
- **Sales Forecasting**: Time-series forecasting with confidence intervals
- **Customer Segmentation**: Advanced clustering algorithms

### 🔌 REST API
- FastAPI backend with 20+ endpoints
- Real-time data processing
- Automated data ingestion pipeline
- Performance optimized with SQLAlchemy

### 📈 Business Intelligence
- Revenue trend analysis
- Customer retention metrics
- Product category performance
- Geographic sales analysis

## 🛠 Tech Stack

- **Backend**: FastAPI, SQLAlchemy, SQLite
- **Frontend**: Streamlit, Plotly
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn
- **Database**: SQLite with optimized indexing
- **Deployment**: Docker, Docker Compose

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/nikhita721/Advanced-Sales-Analytics.git
cd Advanced-Sales-Analytics
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Setup database**
```bash
python scripts/setup_database.py
```

4. **Ingest sample data** (place your Excel file in the project root)
```bash
python scripts/ingest_data.py path/to/your/data.xlsx
```

5. **Run the application**

**Start API server:**
```bash
python run.py api
# API will be available at http://localhost:8000
# API documentation at http://localhost:8000/docs
```

**Start Dashboard:**
```bash
streamlit run dashboard.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true
# Dashboard will be available at http://localhost:8501
```

**Run both services:**
```bash
python run.py both
```

## 🌐 **Access URLs**

Once the services are running, access them at:

- **📊 Interactive Dashboard**: http://localhost:8501
- **🔌 REST API**: http://localhost:8000
- **📚 API Documentation**: http://localhost:8000/docs
- **❤️ Health Check**: http://localhost:8000/health

## 🐳 Docker Deployment

1. **Build and run with Docker Compose**
```bash
docker-compose up --build
```

**Docker Services URLs:**
- **📊 Interactive Dashboard**: http://localhost:8501
- **🔌 REST API**: http://localhost:8000
- **📚 API Documentation**: http://localhost:8000/docs

## 📁 Project Structure

```
sales-analytics/
├── app/                    # Core application
│   ├── __init__.py
│   ├── api.py             # FastAPI application
│   ├── models.py          # Database models
│   ├── database.py        # Database configuration
│   ├── analytics.py       # ML analytics engine
│   ├── data_ingestion.py  # Data processing pipeline
│   └── config.py          # Configuration settings
├── scripts/               # Utility scripts
│   ├── setup_database.py  # Database initialization
│   └── ingest_data.py     # Data ingestion script
├── dashboard.py           # Streamlit dashboard
├── run.py                # Application runner
├── requirements.txt       # Python dependencies
├── docker-compose.yml     # Docker configuration
├── Dockerfile            # Docker image definition
└── README.md             # This file
```

## 📊 API Endpoints

### Analytics Endpoints
- `GET /analytics/overview` - Business metrics overview
- `GET /analytics/rfm` - RFM analysis results
- `GET /analytics/clv` - Customer lifetime value analysis
- `GET /analytics/forecast` - Sales forecasting

### Data Endpoints
- `GET /customers` - Customer data with pagination
- `GET /products` - Product information
- `GET /transactions` - Transaction history
- `GET /sales/trends` - Sales trend analysis

### Management Endpoints
- `POST /ingest` - Upload and process new data
- `GET /health` - Health check
- `GET /` - API information

## 📈 Analytics Features

### RFM Analysis
Segments customers based on:
- **Recency**: How recently they made a purchase
- **Frequency**: How often they make purchases
- **Monetary**: How much they spend

### Customer Lifetime Value (CLV)
Predicts the total value a customer will bring over their lifetime using:
- Purchase frequency
- Average order value
- Customer lifespan estimation

### Sales Forecasting
Time-series forecasting with:
- Trend analysis
- Seasonal decomposition
- Confidence intervals
- Multiple forecasting horizons

## 🔧 Configuration

Environment variables can be set in `.env` file:

```env
DATABASE_URL=sqlite:///./sales_analytics.db
REDIS_URL=redis://localhost:6379
DEBUG=False
```

## 📝 Data Format

The platform supports Excel/CSV files with columns:
- Transaction ID
- Customer ID
- Product ID
- Date
- Quantity
- Unit Price
- Total Price
- Store ID (optional)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For support and questions:
- Create an issue on GitHub
- Check the API documentation at `/docs`
- Review the example data formats in the repository

---

**Built with ❤️ for retail analytics and business intelligence**