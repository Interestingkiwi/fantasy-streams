"""
Schedules daily jobs for fantasystreams.app
Refactored for Multi-Tenancy (League ID) and Postgres Syntax

Author: Jason Druckenmiller
Created: 11/02/2025
Updated: 11/26/2025
"""

import os
import logging
import time
import json
import tempfile
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from yahoo_oauth import OAuth2
from yfpy.query import YahooFantasySportsQuery
import yahoo_fantasy_api as yfa
import db_builder
from database import get_db_connection

# Basic config
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def get_refreshed_token(user_row):
    # Unpack tuple from Postgres
    guid, access_token, refresh_token, token_type, expires_in, token_time, consumer_key, consumer_secret, _, _ = user_row

    creds = {
        "consumer_key": consumer_key,
        "consumer_secret": consumer_secret,
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": token_type,
        "expires_in": expires_in,
        "token_time": token_time,
        "xoauth_yahoo_guid": guid
    }

    fd, temp_path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, 'w') as f:
        json.dump(creds, f)

    try:
        sc = OAuth2(consumer_key, consumer_secret, from_file=temp_path)
        logger.info(f"Attempting refresh for {guid}...")
        sc.refresh_access_token()

        if not sc.token_is_valid():
            raise Exception("Token refresh failed")

        with open(temp_path, 'r') as f:
            new_creds = json.load(f)
        return new_creds
    except Exception as e:
        logger.error(f"Refresh failed: {e}")
        return None
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def process_subscription_expirations():
    today_str = datetime.utcnow().strftime('%Y-%m-%d')
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                UPDATE users SET is_premium = FALSE
                WHERE is_premium = TRUE
                AND premium_expiration_date < %s
            """, (today_str,))
            conn.commit()

def run_league_updates():
    logger.info("--- Starting Scheduled League Updates ---")
    process_subscription_expirations()

    with get_db_connection() as conn:
        with conn.cursor() as cursor:
#            cursor.execute("""
#                SELECT l.league_id, u.* FROM league_updaters l
#                JOIN users u ON l.user_guid = u.guid
#                WHERE u.is_premium = TRUE
#            """)
            cursor.execute("""
                SELECT l.league_id, u.* FROM league_updaters l
                JOIN users u ON l.user_guid = u.guid
            """)
            rows = cursor.fetchall()

            if not rows:
                logger.info("No premium leagues to update.")
                return

            for row in rows:
                league_id = row[0]
                user_data = row[1:] # users table columns
                user_guid = user_data[0]

                logger.info(f"Processing League {league_id}")
                creds = get_refreshed_token(user_data)
                if not creds: continue

                # Update Token in DB
                cursor.execute("""
                    UPDATE users SET access_token=%s, refresh_token=%s, token_time=%s, expires_in=%s
                    WHERE guid=%s
                """, (creds['access_token'], creds['refresh_token'], creds['token_time'], creds['expires_in'], user_guid))
                conn.commit()

                try:
                    auth_data = {
                        'consumer_key': creds['consumer_key'],
                        'consumer_secret': creds['consumer_secret'],
                        'access_token': creds['access_token'],
                        'refresh_token': creds['refresh_token'],
                        'token_type': creds['token_type'],
                        'token_time': creds['token_time'],
                        'guid': creds['xoauth_yahoo_guid']
                    }
                    yq = YahooFantasySportsQuery(league_id, game_code="nhl", yahoo_access_token_json=auth_data)

                    fd, temp_path = tempfile.mkstemp(suffix=".json")
                    with os.fdopen(fd, 'w') as f: json.dump(creds, f)
                    sc = OAuth2(creds['consumer_key'], creds['consumer_secret'], from_file=temp_path)
                    gm = yfa.Game(sc, 'nhl')
                    lg = gm.to_league(f"nhl.l.{league_id}")
                    os.remove(temp_path)

                    db_builder.update_league_db(yq, lg, league_id, logger)
                except Exception as e:
                    logger.error(f"Error updating {league_id}: {e}", exc_info=True)

# Import your global update function
from jobs.global_update import update_global_data

def start_scheduler():
    logger.info("Initializing scheduler...")
    scheduler = BackgroundScheduler(timezone="UTC")

    # Global data update
    scheduler.add_job(update_global_data, trigger='cron', hour=9, minute=0)

    # League updates
    scheduler.add_job(run_league_updates, trigger='cron', hour=9, minute=20)

    scheduler.start()

if __name__ == "__main__":
    start_scheduler()
    try:
        while True: time.sleep(60)
    except (KeyboardInterrupt, SystemExit): pass
