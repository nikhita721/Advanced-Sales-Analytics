#!/usr/bin/env python3
"""
Script to ingest Excel data into the Sales Analytics Platform
"""
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.data_ingestion import ingest_excel_data
from app.database import get_db
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main function to ingest data"""
    if len(sys.argv) < 2:
        print("Usage: python scripts/ingest_data.py <path_to_excel_file>")
        print("Example: python scripts/ingest_data.py /path/to/Retail-Store-Transactions.xlsx")
        sys.exit(1)
    
    excel_file_path = sys.argv[1]
    
    if not os.path.exists(excel_file_path):
        print(f"Error: File {excel_file_path} does not exist")
        sys.exit(1)
    
    if not excel_file_path.endswith(('.xlsx', '.xls')):
        print("Error: Only Excel files (.xlsx, .xls) are supported")
        sys.exit(1)
    
    try:
        print(f"Starting data ingestion for: {excel_file_path}")
        print("This may take a few minutes depending on file size...")
        
        # Ingest the data
        results = ingest_excel_data(excel_file_path)
        
        if "error" in results:
            print(f"Error during ingestion: {results['error']}")
            sys.exit(1)
        
        print("\n" + "="*50)
        print("DATA INGESTION COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Total Sheets Processed: {results.get('total_sheets', 0)}")
        print(f"Transactions Processed: {results.get('processed_transactions', 0)}")
        print(f"Customers Processed: {results.get('processed_customers', 0)}")
        print(f"Products Processed: {results.get('processed_products', 0)}")
        print(f"Stores Processed: {results.get('processed_stores', 0)}")
        
        if results.get('errors'):
            print(f"\nErrors encountered: {len(results['errors'])}")
            for error in results['errors'][:5]:  # Show first 5 errors
                print(f"  - {error}")
            if len(results['errors']) > 5:
                print(f"  ... and {len(results['errors']) - 5} more errors")
        
        print("\nNext steps:")
        print("1. Start the API server: python run.py api")
        print("2. Start the dashboard: python run.py dashboard")
        print("3. Access the dashboard at: http://localhost:8501")
        print("4. Access the API docs at: http://localhost:8000/docs")
        
    except Exception as e:
        logger.error(f"Error during data ingestion: {str(e)}")
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
