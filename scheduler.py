import os
import sys
import sqlite3
import logging
import time
import json
import tempfile
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from google.cloud import storage
import subprocess

# Import your db_builder and authentication libraries
import db_builder
import yahoo_fantasy_api as yfa
from yahoo_oauth import OAuth2
from yfpy.query import YahooFantasySportsQuery

# Basic config
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Path to the admin DB created in app.py
DATA_DIR = '/var/data/dbs'
ADMIN_DB_PATH = os.path.join(DATA_DIR, 'admin.db')
GCS_BUCKET_NAME = os.environ.get('GCS_BUCKET_NAME')

def get_refreshed_token(user_row):
    """
    Reconstructs OAuth session and refreshes token if needed.
    Returns the (possibly updated) token dictionary.
    """
    guid, access_token, refresh_token, token_type, expires_in, token_time, consumer_key, consumer_secret = user_row

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

    # Write to temp file for yahoo_oauth library
    fd, temp_path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, 'w') as f:
        json.dump(creds, f)

    try:
        # Initialize OAuth2. Pass consumer_key/secret explicitly for robustness during refresh.
        sc = OAuth2(consumer_key, consumer_secret, from_file=temp_path)

        # Force a token refresh attempt.
        logger.info(f"Attempting to refresh/validate token for user {guid}...")
        sc.refresh_access_token()

        # Check if the token is valid after the refresh attempt.
        if not sc.token_is_valid():
            raise Exception("Token refresh failed: Refresh token may be expired or revoked. User must re-authenticate.")

        # Read back the potentially updated token
        with open(temp_path, 'r') as f:
            new_creds = json.load(f)

        return new_creds

    except Exception as e:
        # This block catches hard failures, including the one we raise above
        logger.error(f"Failed to refresh token for user {guid}: {e}")
        return None
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def download_admin_db_from_gcs():
    """
    Fetches the authoritative admin.db from GCS before running jobs.
    """
    if not GCS_BUCKET_NAME:
        logger.warning("GCS_BUCKET_NAME not set. Cannot fetch admin.db.")
        return

    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob('system/admin.db')

        if blob.exists():
            logger.info("Downloading latest admin.db from GCS...")
            blob.download_to_filename(ADMIN_DB_PATH)
            logger.info("Download complete.")
        else:
            logger.warning("admin.db not found in GCS (system/admin.db). Assuming empty start.")

    except Exception as e:
        logger.error(f"Failed to download admin.db from GCS: {e}")

def upload_admin_db_to_gcs():
    """
    Uploads the local authoritative admin.db back to GCS.
    """
    if not GCS_BUCKET_NAME:
        logger.warning("GCS_BUCKET_NAME not set. Cannot upload admin.db.")
        return
    if not os.path.exists(ADMIN_DB_PATH):
        logger.error(f"Cannot upload: Admin DB not found locally at {ADMIN_DB_PATH}.")
        return

    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob('system/admin.db')

        logger.info("Uploading latest admin.db to GCS...")
        blob.upload_from_filename(ADMIN_DB_PATH)
        logger.info("Upload complete.")

    except Exception as e:
        logger.error(f"Failed to upload admin.db to GCS: {e}")

def run_script(script_path, *args):
    """Runs a script as a subprocess and logs the output."""
    # Use sys.executable to ensure we run with the current Python interpreter
    cmd = [sys.executable, script_path] + list(args)
    logger.info(f"Executing command: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False # Don't raise exception on non-zero exit code
        )
        if result.stdout:
            # Log stdout if present, stripping leading/trailing whitespace
            logger.info(f"--- {script_path} STDOUT ---\n{result.stdout.strip()}")
        if result.stderr:
            # Log stderr if present
            logger.error(f"--- {script_path} STDERR ---\n{result.stderr.strip()}")

        if result.returncode != 0:
            logger.error(f"Script {script_path} failed with return code {result.returncode}")
            return False

        logger.info(f"Script {script_path} completed successfully.")
        return True

    except Exception as e:
        logger.error(f"Exception during execution of {script_path}: {e}")
        return False


def run_league_updates():
    """
    Iterates through all assigned leagues in admin.db and runs the DB update.
    """
    logger.info("--- Starting Scheduled League Updates ---")
    download_admin_db_from_gcs()
    if not os.path.exists(ADMIN_DB_PATH):
        logger.warning("Admin DB not found locally or in GCS. Skipping updates.")
        return

    process_subscription_expirations()

    conn = sqlite3.connect(ADMIN_DB_PATH)
    cursor = conn.cursor()

    # Get list of leagues and their assigned updaters
    cursor.execute("""
        SELECT
            l.league_id,
            u.guid,
            u.access_token,
            u.refresh_token,
            u.token_type,
            u.expires_in,
            u.token_time,
            u.consumer_key,
            u.consumer_secret
        FROM league_updaters l
        JOIN users u ON l.user_guid = u.guid
    """)
#        SELECT l.league_id, u.* FROM league_updaters l
#        JOIN users u ON l.user_guid = u.guid
#        WHERE u.is_premium = 1
#    """)
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        logger.info("No leagues assigned for background updates.")
        return

    logger.info(f"Found {len(rows)} leagues to update.")

    for row in rows:
        league_id = row[0]
        user_data = row[1:] # The rest of the columns from 'users' table
        user_guid = user_data[0]

        logger.info(f"Processing League {league_id} (Updater: {user_guid})")

        # 1. Refresh Token
        creds = get_refreshed_token(user_data)

        if not creds:
            logger.error(f"Skipping league {league_id} due to token failure.")
            continue

        # 2. Update the DB with new token (so we don't use stale tokens next time)
        conn_update = sqlite3.connect(ADMIN_DB_PATH)
        cursor_update = conn_update.cursor()
        cursor_update.execute("""
            UPDATE users
            SET access_token = ?,
                refresh_token = ?,
                token_time = ?,
                expires_in = ?
            WHERE guid = ?
        """, (
            creds['access_token'],
            creds['refresh_token'],
            creds['token_time'],
            creds['expires_in'],
            user_guid
        ))
        conn_update.commit()
        conn_update.close()

        # ADDED: Upload the newly updated local admin.db back to GCS
        upload_admin_db_to_gcs()

        # 3. Initialize APIs
        try:
            # YFPY
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

            # YFA - Final robust fix for non-interactive mode.
            fd, temp_path = tempfile.mkstemp(suffix=".json")
            with os.fdopen(fd, 'w') as f:
                json.dump(creds, f)

            # Pass consumer keys AND from_file to ensure non-interactive token load.
            sc = OAuth2(creds['consumer_key'], creds['consumer_secret'], from_file=temp_path)

            gm = yfa.Game(sc, 'nhl')
            lg = gm.to_league(f"nhl.l.{league_id}")
            os.remove(temp_path) # Clean up temp file

            # 4. Run the Update
            logger.info(f"Starting DB build for {league_id}...")

            # You might need to adjust imports if db_builder isn't in the same dir
            result = db_builder.update_league_db(
                yq, lg, league_id, DATA_DIR, logger,
                capture_lineups=False # Scheduled jobs usually just grab latest, not full history
            )

            if result.get('success'):
                logger.info(f"Successfully updated league {league_id}.")
            else:
                logger.error(f"Failed to update league {league_id}: {result.get('error')}")

        except Exception as e:
            logger.error(f"Critical error updating league {league_id}: {e}", exc_info=True)

    logger.info("--- Scheduled League Updates Completed ---")


def process_subscription_expirations():
    """
    Checks all Premium users. If their expiration date has passed,
    downgrade them to Free (0).
    """
    conn = sqlite3.connect(ADMIN_DB_PATH)
    cursor = conn.cursor()

    # Get today's date
    today_str = datetime.utcnow().strftime('%Y-%m-%d')

    # Find expired users
    cursor.execute("""
        SELECT guid FROM users
        WHERE is_premium = 1
        AND premium_expiration_date < ?
        AND premium_expiration_date IS NOT NULL
    """, (today_str,))

    expired_users = cursor.fetchall()

    if expired_users:
        logger.info(f"Found {len(expired_users)} expired subscriptions. Downgrading...")

        # Bulk update them to 0
        cursor.execute("""
            UPDATE users
            SET is_premium = 0
            WHERE is_premium = 1
            AND premium_expiration_date < ?
        """, (today_str,))

        conn.commit()

        for user in expired_users:
            logger.info(f"Downgraded user {user[0]} to Free tier.")
    else:
        logger.info("No expired subscriptions found.")

    conn.close()


def run_daily_job_sequence():
    """
    Runs the full daily job sequence.
    """
    logger.info("Starting daily job sequence: run_daily_job_sequence")

    # Get required env vars for the subprocess
    league_id = os.environ.get('LEAGUE_ID')
    key = os.environ.get('YAHOO_CONSUMER_KEY')
    secret = os.environ.get('YAHOO_CONSUMER_SECRET')

    if not all([league_id, key, secret]):
        logger.error("Missing required environment variables (LEAGUE_ID, YAHOO_CONSUMER_KEY, YAHOO_CONSUMER_SECRET) for daily job.")
        return

    # Run the scripts in sequence. If one fails, stop.
    if run_script("jobs/fetch_player_ids.py", league_id, "-k", key, "-s", secret):
        if run_script("jobs/create_projection_db.py"):
            run_script("jobs/toi_script.py")


def start_scheduler():
    logger.info("Initializing background scheduler...")
    scheduler = BackgroundScheduler(timezone="UTC")

    # 1. Run League Updates (e.g., at 9:00 UTC / 4:00 AM EST)
    scheduler.add_job(
        run_daily_job_sequence,
        trigger='cron',
        hour=9,  # 6:00 AM UTC
        minute=0
    )

    scheduler.add_job(
        run_league_updates,
        trigger='cron',
        hour=9,
        minute=20
    )

    scheduler.start()
    logger.info("Scheduler started.")

if __name__ == "__main__":
    start_scheduler()
    try:
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        pass
