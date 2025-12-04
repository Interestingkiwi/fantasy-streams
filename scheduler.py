"""
Schedules daily jobs for fantasystreams.app
Refactored for Priority Queues (Redis)

Author: Jason Druckenmiller
Created: 11/02/2025
Updated: 12/03/2025
"""

import os
import logging
import uuid
import redis
from rq import Queue
from apscheduler.schedulers.background import BackgroundScheduler
from database import get_db_connection
from datetime import datetime

# Basic config
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Redis Connection
redis_url = os.getenv('REDIS_URL', 'redis://red-d4ae0rur433s73eil750:6379')
conn = redis.from_url(redis_url)

# Define Queues
low_queue = Queue('low', connection=conn)

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
    logger.info(f"--- Scheduling League Updates (Target: {target_league_id}) ---")

    if not target_league_id:
        process_subscription_expirations()

    rows = []
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            if target_league_id:
                cursor.execute("""
                    SELECT l.league_id, u.guid FROM league_updaters l
                    JOIN users u ON l.user_guid = u.guid
                    WHERE l.league_id = %s
                """, (target_league_id,))
            else:
                cursor.execute("""
                    SELECT l.league_id, u.guid FROM league_updaters l
                    JOIN users u ON l.user_guid = u.guid
                """)
            rows = cursor.fetchall()

    for row in rows:
        league_id = str(row[0])
        guid = row[1]

        job_id = f"scheduled_{league_id}"

        task_data = {
            'league_id': league_id,
            'token': {'xoauth_yahoo_guid': guid},
            'dev_mode': False
        }

        # --- FIX: Add Rate Limit Instruction ---
        options = {
            'capture_lineups': True,
            'force_full_history': force_full_history,
            'freshness_minutes': 240,

            # TELL THE WORKER: "After you finish this job, sleep for 1.5 mins."
            'rate_limit_seconds': 90
        }

        try:
            low_queue.enqueue(
                'db_builder.run_task',
                args=(job_id, None, options, task_data),
                job_id=job_id,
                job_timeout=3600 # Increased timeout to account for the sleep
            )
            logger.info(f"Enqueued Low-Priority job for {league_id}")
        except Exception as e:
            logger.error(f"Failed to enqueue {league_id}: {e}")

# Import your global update function
from jobs.global_update import update_global_data

def start_scheduler():
    logger.info("Initializing scheduler...")
    scheduler = BackgroundScheduler(timezone="UTC")

    # Global data update
    scheduler.add_job(update_global_data, trigger='cron', hour=10, minute=0)

    # League updates (Runs every hour, but freshness check prevents over-updating)
    scheduler.add_job(run_league_updates, trigger='cron', hour='*/1', minute=0)

    scheduler.start()
