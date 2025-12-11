import sys
import uuid
import logging
import os

# Ensure we can import from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import get_db_connection
from db_builder import run_task

# Basic logging config to ensure we see output immediately
logging.basicConfig(level=logging.INFO)

def run_manual_update(league_id, force_full=False):
    print(f"\n--- Manual Update for League {league_id} ---\n")

    # 1. Find the assigned updater (User GUID)
    # We need to know WHO is updating this league to get the right token
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT user_guid FROM league_updaters WHERE league_id = %s", (league_id,))
                row = cursor.fetchone()

        if not row:
            print(f"❌ Error: League {league_id} has no assigned updater in the database.")
            return

        guid = row[0] # Tuple index 0 for postgres driver usually, or key access if RealDictCursor
        # Handle RealDictCursor just in case standard driver wasn't used
        if hasattr(row, 'get'):
            guid = row.get('user_guid') or row[0]

    except Exception as e:
        print(f"❌ Database Error: {e}")
        return

    # 2. Construct the Task Data
    # run_task expects a specific dictionary structure
    task_data = {
        'league_id': league_id,
        'token': {'xoauth_yahoo_guid': guid}, # Vital: Used to look up credentials in run_task
        'dev_mode': False,
        'consumer_key': os.environ.get("YAHOO_CONSUMER_KEY"),    # Fallbacks
        'consumer_secret': os.environ.get("YAHOO_CONSUMER_SECRET")
    }

    # 3. Configure Options
    options = {
        'capture_lineups': True,        # Capture daily stats
        'roster_updates_only': False,   # Do a standard update
        'force_full_history': force_full, # Force overwrite of history if requested
        'freshness_minutes': 0          # Bypass freshness check (Force run)
    }

    # Generate a dummy build ID
    build_id = f"manual_shell_{uuid.uuid4().hex[:6]}"

    print(f"✅ Found updater {guid}. Starting Job {build_id}...")

    # 4. Run the Task
    # We pass None for log_file_path as we want stdout
    try:
        run_task(build_id, None, options, task_data)
        print("\n✅ Update script finished successfully.")
    except Exception as e:
        print(f"\n❌ Script crashed: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        lid = sys.argv[1]
    else:
        lid = input("Enter League ID: ").strip()

    # Optional: Check for 'force' flag
    force = 'force' in sys.argv

    if lid:
        run_manual_update(lid, force_full=force)
    else:
        print("League ID required.")
