# database.py

import os
import psycopg2
from psycopg2 import pool, OperationalError, InterfaceError
from contextlib import contextmanager

# 1. Get URL
DATABASE_URL = os.environ.get('DATABASE_URL')

if not DATABASE_URL:
    print("WARNING: DATABASE_URL not found. Using localhost.")
    DATABASE_URL = "postgresql://postgres:password@localhost/fantasy_db"

# 2. Initialize Connection Pool
try:
    db_pool = psycopg2.pool.SimpleConnectionPool(1, 20, DATABASE_URL)
    if db_pool:
        print("Connection pool created successfully")
except Exception as e:
    print(f"Error creating connection pool: {e}")
    db_pool = None

# 3. Robust Context Manager
@contextmanager
def get_db_connection():
    if not db_pool:
        raise Exception("Database connection pool is not initialized.")

    conn = db_pool.getconn()
    try:
        # --- LIVENESS CHECK ---
        # If the connection is closed or the server dropped it, we need a new one.
        if conn.closed:
            db_pool.putconn(conn, close=True) # Discard dead connection
            conn = db_pool.getconn()          # Get a replacement
        else:
            # Execute a lightweight check to ensure the server is still listening
            try:
                with conn.cursor() as cur:
                    cur.execute('SELECT 1')
            except (OperationalError, InterfaceError):
                # Connection is dead; discard and retry once
                db_pool.putconn(conn, close=True)
                conn = db_pool.getconn()

        yield conn

    except Exception as e:
        # Only rollback if the connection is actually open
        if conn and not conn.closed:
            try:
                conn.rollback()
            except:
                pass
        raise e
    finally:
        # Return to pool (discard if broken)
        if conn:
            if conn.closed:
                db_pool.putconn(conn, close=True)
            else:
                db_pool.putconn(conn)
