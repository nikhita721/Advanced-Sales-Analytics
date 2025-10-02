-- Initialize Sales Analytics Database
-- This script runs when the PostgreSQL container starts

-- Create database if it doesn't exist
SELECT 'CREATE DATABASE sales_analytics'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'sales_analytics')\gexec

-- Connect to the database
\c sales_analytics;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create indexes for better performance
-- These will be created by SQLAlchemy migrations, but we can add some custom ones here

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE sales_analytics TO user;
GRANT ALL PRIVILEGES ON SCHEMA public TO user;
