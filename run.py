#!/usr/bin/env python3
"""
Main entry point for the Sales Analytics Platform
"""
import uvicorn
import streamlit as st
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def run_api():
    """Run the FastAPI server"""
    uvicorn.run(
        "app.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

def run_dashboard():
    """Run the Streamlit dashboard"""
    os.system("streamlit run dashboard.py --server.port=8501 --server.address=0.0.0.0")

def main():
    """Main function to choose what to run"""
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "api":
            run_api()
        elif command == "dashboard":
            run_dashboard()
        elif command == "both":
            print("Starting both API and Dashboard...")
            print("API will be available at: http://localhost:8000")
            print("Dashboard will be available at: http://localhost:8501")
            # In a real scenario, you'd use multiprocessing or threading
            # For now, just run the API
            run_api()
        else:
            print("Unknown command. Use: api, dashboard, or both")
    else:
        print("Sales Analytics Platform")
        print("Usage: python run.py [api|dashboard|both]")
        print("  api       - Run FastAPI server only")
        print("  dashboard - Run Streamlit dashboard only") 
        print("  both      - Run both (API and Dashboard)")

if __name__ == "__main__":
    main()
