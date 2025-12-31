import os
import sys
import json
import logging
import psycopg2.extras
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from database import get_db_connection
from yahoo_oauth import OAuth2
from yfpy.query import YahooFantasySportsQuery

# --- CONFIGURATION ---
LEAGUE_ID = ""
TARGET_GUID = ""

# Setup logging
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
                # Ensure GUID is in top level for some versions of yahoo_oauth
                creds['guid'] = row['guid']

    except Exception as e:
        logger.error(f"Database error: {e}")
        return

    # 2. Refresh Token using LOCAL file
    logger.info("Refreshing token...")
    temp_path = "debug_creds.json"  # <--- USING LOCAL FILE

    try:
        # Write creds to local file
        with open(temp_path, 'w') as f:
            json.dump(creds, f, indent=4)

        # Verify file has content
        file_size = os.path.getsize(temp_path)
        logger.info(f"Created temp creds file at {temp_path} (Size: {file_size} bytes)")
        if file_size == 0:
            raise Exception("Credential file is empty before refresh!")

        # Initialize OAuth2 with local file
        sc = OAuth2(None, None, from_file=temp_path)

        # Refresh
        sc.refresh_access_token()

        # Read back the refreshed token
        with open(temp_path, 'r') as f:
            new_creds = json.load(f)

        logger.info("Token refreshed successfully.")

    except Exception as e:
        logger.error(f"Auth failed: {e}")
        # Print file content if it failed, for debugging
        if os.path.exists(temp_path):
            with open(temp_path, 'r') as f:
                content = f.read()
                logger.info(f"--- File Content at Failure ---\n{content}\n-----------------------------")
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

        print("\n--- INSPECTING STAT CATEGORIES ---")
        if hasattr(settings, 'stat_categories') and hasattr(settings.stat_categories, 'stats'):
            stats_list = settings.stat_categories.stats
            print(f"Found {len(stats_list)} stats.")

            data_to_insert = []
            for stat in stats_list:
                print(f"ID: {stat.stat_id} | Name: '{stat.name}' | Display: '{stat.display_name}' | Group: {stat.group}")

                cat = stat.display_name
                if cat == 'SV%': cat = 'SVpct'
                data_to_insert.append((LEAGUE_ID, stat.stat_id, cat, stat.group))

            print("\n--- SIMULATED DB INSERT ---")
            for row in data_to_insert:
                print(row)

        else:
            print("ERROR: settings object does not have stat_categories.stats")
            print(dir(settings))

    except Exception as e:
        logger.error(f"Failed during query or parsing: {e}", exc_info=True)

if __name__ == "__main__":
    main()
