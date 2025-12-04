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

    # 1. Create temp file
    fd, temp_path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, 'w') as f:
        json.dump(creds, f)

    try:
        # 2. Refresh
        sc = OAuth2(None, None, from_file=temp_path)
        if not sc.token_is_valid():
            sc.refresh_access_token()

        # 3. Read back new token
        with open(temp_path, 'r') as f:
            new_creds = json.load(f)

        # 4. Update DB
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE users
                    SET access_token = %s, refresh_token = %s, token_time = %s, expires_in = %s
                    WHERE guid = %s
                """, (
                    new_creds['access_token'],
                    new_creds['refresh_token'],
                    new_creds['token_time'],
                    new_creds.get('expires_in', 3600),
                    guid
                ))
            conn.commit()

        # Return updated creds dict
        return new_creds

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def process_subscription_expirations():
    """Checks for expired premium subscriptions and downgrades them."""
    logger.info("Checking for expired subscriptions...")
    today = datetime.now().date()
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE users
                    SET is_premium = FALSE, premium_expiration_date = NULL
                    WHERE is_premium = TRUE AND premium_expiration_date < %s
                """, (today,))
                if cursor.rowcount > 0:
                    logger.info(f"Downgraded {cursor.rowcount} expired subscriptions.")
            conn.commit()
    except Exception as e:
        logger.error(f"Error processing expirations: {e}", exc_info=True)

def run_league_updates(target_league_id=None, force_full_history=False):
    logger.info(f"--- Starting Scheduled League Updates (Target: {target_league_id}, Force Full: {force_full_history}) ---")

    # Only run subscription expiration checks if we are doing a full run (no target)
    if not target_league_id:
        process_subscription_expirations()

    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            if target_league_id:
                # Targeted Query for one league
                cursor.execute("""
                    SELECT l.league_id, u.* FROM league_updaters l
                    JOIN users u ON l.user_guid = u.guid
                    WHERE l.league_id = %s
                """, (target_league_id,))
                logger.info(f"Running manual update for single league: {target_league_id}")
            else:
                # Standard Query for all leagues
                cursor.execute("""
                    SELECT l.league_id, u.* FROM league_updaters l
                    JOIN users u ON l.user_guid = u.guid
                """)

            rows = cursor.fetchall()

    for row in rows:
        # Row structure: (league_id, guid, access_token, ...)
        # Postgres returns tuples. Index 0 is league_id. Index 1 onwards is User columns.
        league_id = str(row[0])
        # User columns start at index 1
        user_row = row[1:]

        logger.info(f"Processing League {league_id}...")

        try:
            creds = get_refreshed_token(user_row)

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

            db_builder.update_league_db(
                yq,
                lg,
                league_id,
                logger,
                force_full_history=force_full_history
            )
        except Exception as e:
            logger.error(f"Error updating {league_id}: {e}", exc_info=True)

        # Pause for 2 minutes after each league to avoid rate limits
        logger.info("Pausing for 2 minutes to avoid rate limiting...")
        time.sleep(61)

# Import your global update function
from jobs.global_update import update_global_data

def start_scheduler():
    logger.info("Initializing scheduler...")
    scheduler = BackgroundScheduler(timezone="UTC")

    # Global data update
    scheduler.add_job(update_global_data, trigger='cron', hour=9, minute=30)

    # League updates
    scheduler.add_job(run_league_updates, trigger='cron', hour=10, minute=0)

    scheduler.start()
