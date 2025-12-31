import os
import sys
import json
import logging
import psycopg2.extras
from database import get_db_connection
from yahoo_oauth import OAuth2
from yfpy.query import YahooFantasySportsQuery

# --- CONFIGURATION ---
LEAGUE_ID = "21022"
TARGET_GUID = "ULYBMB2VUJXZ62KPAUFZC6SCJA"

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("debug_scoring")

def main():
    logger.info(f"Starting debug for League {LEAGUE_ID} using GUID {TARGET_GUID}")

    # 1. Fetch Credentials from DB
    creds = None
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute("SELECT * FROM users WHERE guid = %s", (TARGET_GUID,))
                row = cursor.fetchone()

                if not row:
                    logger.error("User not found in database.")
                    return

                creds = {
                    "consumer_key": row['consumer_key'],
                    "consumer_secret": row['consumer_secret'],
                    "access_token": row['access_token'],
                    "refresh_token": row['refresh_token'],
                    "token_type": row['token_type'],
                    "expires_in": row['expires_in'],
                    "token_time": row['token_time'],
                    "xoauth_yahoo_guid": row['guid']
                }
    except Exception as e:
        logger.error(f"Database error: {e}")
        return

    # 2. Refresh Token
    logger.info("Refreshing token...")
    try:
        # Create a temp file for yahoo_oauth to read/write
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_cred_file:
            json.dump(creds, temp_cred_file)
            temp_path = temp_cred_file.name

        sc = OAuth2(None, None, from_file=temp_path)
        sc.refresh_access_token()

        # Read back the refreshed token
        with open(temp_path, 'r') as f:
            new_creds = json.load(f)

        os.remove(temp_path) # cleanup

        logger.info("Token refreshed successfully.")

    except Exception as e:
        logger.error(f"Auth failed: {e}")
        return

    # 3. Initialize Yahoo Query
    try:
        yq = YahooFantasySportsQuery(
            LEAGUE_ID,
            game_code="nhl",
            yahoo_access_token_json=new_creds,
            yahoo_consumer_key=new_creds['consumer_key'],
            yahoo_consumer_secret=new_creds['consumer_secret']
        )
    except Exception as e:
        logger.error(f"Failed to init yfpy: {e}")
        return

    # 4. Fetch and Inspect Settings
    logger.info("Fetching league settings...")
    try:
        settings = yq.get_league_settings()

        print("\n--- RAW SETTINGS OBJECT DUMP ---")
        print(settings)
        print("--------------------------------\n")

        print("--- INSPECTING STAT CATEGORIES ---")
        if hasattr(settings, 'stat_categories') and hasattr(settings.stat_categories, 'stats'):
            stats_list = settings.stat_categories.stats
            print(f"Found {len(stats_list)} stats.")

            for stat in stats_list:
                print(f"ID: {stat.stat_id} | Name: '{stat.name}' | Display: '{stat.display_name}' | Group: {stat.group}")

                # Test your specific logic
                cat = stat.display_name
                if cat == 'SV%': cat = 'SVpct'
                # print(f" -> DB Insert would be: ({LEAGUE_ID}, {stat.stat_id}, {cat}, {stat.group})")
        else:
            print("ERROR: settings object does not have stat_categories.stats")
            print(f"Available attributes: {dir(settings)}")

    except Exception as e:
        logger.error(f"Failed during query or parsing: {e}", exc_info=True)

if __name__ == "__main__":
    main()
