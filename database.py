import os
import psycopg2
from psycopg2 import pool
from contextlib import contextmanager

# 1. Get URL from Environment Variable
DATABASE_URL = os.environ.get('DATABASE_URL')

if not DATABASE_URL:
    # Fallback for local testing (Optional)
    print("WARNING: DATABASE_URL not found. Using localhost.")
    DATABASE_URL = "postgresql://postgres:password@localhost/fantasy_db"

# 2. Initialize Connection Pool
# Min 1 connection, Max 20 connections
try:
    db_pool = psycopg2.pool.SimpleConnectionPool(1, 20, DATABASE_URL)
    if db_pool:
        print("Connection pool created successfully")
except Exception as e:
    print(f"Error creating connection pool: {e}")
    db_pool = None

# 3. The Context Manager (Used by "with get_db_connection() as conn:")
@contextmanager
def get_db_connection():
    if not db_pool:
        raise Exception("Database connection pool is not initialized.")

    conn = db_pool.getconn()
    try:
        yield conn
    except Exception as e:
        conn.rollback() # Rollback transaction on error
        raise e
    finally:
        db_pool.putconn(conn) # Return connection to pool
