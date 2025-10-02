"""
Data ingestion pipeline for processing retail transaction data
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from app.models import Transaction, Customer, Product, Store
from app.database import get_db
import logging
import hashlib
import uuid

logger = logging.getLogger(__name__)


class DataIngestionPipeline:
    """Advanced data ingestion pipeline for retail transactions"""
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self.processed_count = 0
        self.error_count = 0
        
    def process_excel_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process Excel file and extract transaction data
        
        Args:
            file_path: Path to the Excel file
            
        Returns:
            Dictionary with processing results
        """
        try:
            # Read Excel file with multiple sheets support
            excel_data = pd.read_excel(file_path, sheet_name=None)
            
            results = {
                "total_sheets": len(excel_data),
                "processed_transactions": 0,
                "processed_customers": 0,
                "processed_products": 0,
                "processed_stores": 0,
                "errors": []
            }
            
            # Process each sheet
            for sheet_name, df in excel_data.items():
                logger.info(f"Processing sheet: {sheet_name}")
                sheet_results = self._process_dataframe(df, sheet_name)
                
                # Aggregate results
                for key in ["processed_transactions", "processed_customers", 
                           "processed_products", "processed_stores"]:
                    results[key] += sheet_results.get(key, 0)
                results["errors"].extend(sheet_results.get("errors", []))
            
            self.db.commit()
            return results
            
        except Exception as e:
            logger.error(f"Error processing Excel file: {str(e)}")
            self.db.rollback()
            return {"error": str(e)}
    
    def _process_dataframe(self, df: pd.DataFrame, sheet_name: str) -> Dict[str, Any]:
        """Process individual dataframe"""
        results = {
            "processed_transactions": 0,
            "processed_customers": 0,
            "processed_products": 0,
            "processed_stores": 0,
            "errors": []
        }
        
        try:
            # Clean and standardize column names
            df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
            
            # Detect transaction structure
            transaction_columns = self._detect_transaction_columns(df.columns)
            
            if not transaction_columns:
                results["errors"].append(f"No transaction columns found in sheet: {sheet_name}")
                return results
            
            # Process each row
            for index, row in df.iterrows():
                try:
                    # Extract transaction data
                    transaction_data = self._extract_transaction_data(row, transaction_columns)
                    
                    if transaction_data:
                        # Create/update transaction
                        transaction = self._create_transaction(transaction_data)
                        if transaction:
                            results["processed_transactions"] += 1
                        
                        # Create/update customer
                        customer = self._create_customer(transaction_data)
                        if customer:
                            results["processed_customers"] += 1
                        
                        # Create/update product
                        product = self._create_product(transaction_data)
                        if product:
                            results["processed_products"] += 1
                        
                        # Create/update store
                        store = self._create_store(transaction_data)
                        if store:
                            results["processed_stores"] += 1
                            
                except Exception as e:
                    results["errors"].append(f"Error processing row {index}: {str(e)}")
                    self.error_count += 1
                    
        except Exception as e:
            results["errors"].append(f"Error processing dataframe: {str(e)}")
            
        return results
    
    def _detect_transaction_columns(self, columns: List[str]) -> Dict[str, str]:
        """Detect transaction columns from available columns"""
        column_mapping = {}
        
        # Common column mappings
        mappings = {
            'transaction_id': ['transaction_id', 'id', 'order_id', 'invoice_id', 'transactionid'],
            'customer_id': ['customer_id', 'customer', 'client_id', 'user_id'],
            'product_id': ['product_id', 'product', 'item_id', 'sku'],
            'product_name': ['product_name', 'product', 'item_name', 'description'],
            'category': ['category', 'cat', 'product_category'],
            'subcategory': ['subcategory', 'sub_cat', 'product_subcategory'],
            'brand': ['brand', 'manufacturer', 'company'],
            'quantity': ['quantity', 'qty', 'amount', 'units'],
            'unit_price': ['unit_price', 'price', 'unit_cost', 'cost', 'unitprice'],
            'total_amount': ['total_amount', 'total', 'amount', 'value', 'revenue', 'totalprice'],
            'discount_amount': ['discount', 'discount_amount', 'discount_value'],
            'tax_amount': ['tax', 'tax_amount', 'tax_value', 'vat'],
            'transaction_date': ['date', 'transaction_date', 'order_date', 'purchase_date'],
            'store_id': ['store_id', 'store', 'location_id', 'branch_id', 'storeid'],
            'store_name': ['store_name', 'store', 'location', 'branch'],
            'sales_person': ['sales_person', 'salesperson', 'staff', 'employee', 'cashier'],
            'payment_method': ['payment_method', 'payment', 'pay_method', 'card_type', 'paymenttype']
        }
        
        for target_col, possible_names in mappings.items():
            for col in columns:
                if any(name.lower() in col.lower() for name in possible_names):
                    column_mapping[target_col] = col
                    break
        
        return column_mapping
    
    def _extract_transaction_data(self, row: pd.Series, column_mapping: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Extract transaction data from row"""
        try:
            data = {}
            
            # Required fields
            required_fields = ['product_name', 'quantity', 'unit_price']
            for field in required_fields:
                if field in column_mapping and pd.notna(row.get(column_mapping[field])):
                    data[field] = row[column_mapping[field]]
                else:
                    return None  # Skip if required field is missing
            
            # Generate customer_id if not present
            if 'customer_id' not in data:
                data['customer_id'] = f"CUST_{hash(str(row.get('TransactionID', row.get('transactionid', 'unknown')))) % 10000:04d}"
            
            # Generate product_id if not present
            if 'product_id' not in data:
                data['product_id'] = f"PROD_{hash(str(data['product_name'])) % 10000:04d}"
            
            # Optional fields
            optional_fields = ['transaction_id', 'category', 'subcategory', 'brand', 
                             'total_amount', 'discount_amount', 'tax_amount', 
                             'transaction_date', 'store_id', 'store_name', 
                             'sales_person', 'payment_method']
            
            for field in optional_fields:
                if field in column_mapping and pd.notna(row.get(column_mapping[field])):
                    data[field] = row[column_mapping[field]]
            
            # Generate transaction_id if not present
            if 'transaction_id' not in data:
                data['transaction_id'] = self._generate_transaction_id(data)
            
            # Calculate total_amount if not present
            if 'total_amount' not in data:
                data['total_amount'] = data['quantity'] * data['unit_price']
            
            # Parse transaction_date
            if 'transaction_date' in data:
                data['transaction_date'] = self._parse_date(data['transaction_date'])
            else:
                data['transaction_date'] = datetime.now()
            
            # Set default values
            data.setdefault('discount_amount', 0.0)
            data.setdefault('tax_amount', 0.0)
            data.setdefault('payment_method', 'Unknown')
            
            return data
            
        except Exception as e:
            logger.error(f"Error extracting transaction data: {str(e)}")
            return None
    
    def _generate_transaction_id(self, data: Dict[str, Any]) -> str:
        """Generate unique transaction ID"""
        # Create hash from customer_id, product_id, and timestamp
        content = f"{data['customer_id']}_{data['product_id']}_{datetime.now().timestamp()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _parse_date(self, date_value: Any) -> datetime:
        """Parse various date formats"""
        if isinstance(date_value, datetime):
            return date_value
        elif isinstance(date_value, str):
            # Try common date formats
            formats = [
                '%Y-%m-%d',
                '%Y-%m-%d %H:%M:%S',
                '%d/%m/%Y',
                '%m/%d/%Y',
                '%Y-%m-%d %H:%M:%S.%f'
            ]
            for fmt in formats:
                try:
                    return datetime.strptime(str(date_value), fmt)
                except ValueError:
                    continue
        elif pd.notna(date_value):
            return pd.to_datetime(date_value)
        
        return datetime.now()
    
    def _create_transaction(self, data: Dict[str, Any]) -> Optional[Transaction]:
        """Create transaction record"""
        try:
            # Check if transaction already exists
            existing = self.db.query(Transaction).filter(
                Transaction.transaction_id == data['transaction_id']
            ).first()
            
            if existing:
                return None
            
            transaction = Transaction(
                transaction_id=data['transaction_id'],
                customer_id=str(data['customer_id']),
                product_id=str(data['product_id']),
                product_name=str(data['product_name']),
                category=data.get('category'),
                subcategory=data.get('subcategory'),
                brand=data.get('brand'),
                quantity=int(data['quantity']),
                unit_price=float(data['unit_price']),
                total_amount=float(data['total_amount']),
                discount_amount=float(data.get('discount_amount', 0)),
                tax_amount=float(data.get('tax_amount', 0)),
                transaction_date=data['transaction_date'],
                store_id=data.get('store_id'),
                store_name=data.get('store_name'),
                sales_person=data.get('sales_person'),
                payment_method=data.get('payment_method')
            )
            
            self.db.add(transaction)
            return transaction
            
        except Exception as e:
            logger.error(f"Error creating transaction: {str(e)}")
            return None
    
    def _create_customer(self, data: Dict[str, Any]) -> Optional[Customer]:
        """Create or update customer record"""
        try:
            customer_id = str(data['customer_id'])
            
            # Check if customer exists
            customer = self.db.query(Customer).filter(
                Customer.customer_id == customer_id
            ).first()
            
            if not customer:
                customer = Customer(
                    customer_id=customer_id,
                    first_purchase_date=data['transaction_date'],
                    last_purchase_date=data['transaction_date']
                )
                self.db.add(customer)
                self.db.flush()  # Flush to get the ID
                return customer
            else:
                # Update last purchase date
                if customer.last_purchase_date is None or data['transaction_date'] > customer.last_purchase_date:
                    customer.last_purchase_date = data['transaction_date']
                if customer.first_purchase_date is None or data['transaction_date'] < customer.first_purchase_date:
                    customer.first_purchase_date = data['transaction_date']
                return customer
                
        except Exception as e:
            logger.error(f"Error creating customer: {str(e)}")
            return None
    
    def _create_product(self, data: Dict[str, Any]) -> Optional[Product]:
        """Create or update product record"""
        try:
            product_id = str(data['product_id'])
            
            # Check if product exists
            product = self.db.query(Product).filter(
                Product.product_id == product_id
            ).first()
            
            if not product:
                product = Product(
                    product_id=product_id,
                    product_name=str(data['product_name']),
                    category=data.get('category'),
                    subcategory=data.get('subcategory'),
                    brand=data.get('brand'),
                    unit_price=float(data['unit_price'])
                )
                self.db.add(product)
                self.db.flush()  # Flush to get the ID
                return product
            else:
                # Update product info if needed
                if not product.category and data.get('category'):
                    product.category = data['category']
                if not product.brand and data.get('brand'):
                    product.brand = data['brand']
                return product
                
        except Exception as e:
            logger.error(f"Error creating product: {str(e)}")
            return None
    
    def _create_store(self, data: Dict[str, Any]) -> Optional[Store]:
        """Create or update store record"""
        try:
            store_id = data.get('store_id')
            if not store_id:
                return None
                
            store_id = str(store_id)
            
            # Check if store exists
            store = self.db.query(Store).filter(
                Store.store_id == store_id
            ).first()
            
            if not store:
                store = Store(
                    store_id=store_id,
                    store_name=data.get('store_name', f"Store {store_id}")
                )
                self.db.add(store)
                self.db.flush()  # Flush to get the ID
                return store
            else:
                return store
                
        except Exception as e:
            logger.error(f"Error creating store: {str(e)}")
            return None


def ingest_excel_data(file_path: str) -> Dict[str, Any]:
    """
    Main function to ingest Excel data
    
    Args:
        file_path: Path to Excel file
        
    Returns:
        Processing results
    """
    db = next(get_db())
    pipeline = DataIngestionPipeline(db)
    
    try:
        results = pipeline.process_excel_file(file_path)
        return results
    finally:
        db.close()
