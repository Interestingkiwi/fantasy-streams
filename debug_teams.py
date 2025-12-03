import os
import json
import logging
from database import get_db_connection
import psycopg2.extras
from yfpy.query import YahooFantasySportsQuery

# Configure basic logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_league_teams(league_id):
    """
    Fetches and prints raw team data for a specific league using stored credentials.
    """
    print(f"\n--- Debugging League {league_id} ---\n")

    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                # 1. Find the user responsible for updating this league
                print(f"Looking up updater for league {league_id}...")
                cursor.execute(
                    "SELECT user_guid FROM league_updaters WHERE league_id = %s",
                    (league_id,)
                )
                updater_row = cursor.fetchone()

                if not updater_row:
                    print("ERROR: No user assigned to update this league in 'league_updaters'.")
                    return

                guid = updater_row['user_guid']
                print(f"Found updater GUID: {guid}")

                # 2. Fetch credentials for this user
                cursor.execute(
                    "SELECT * FROM users WHERE guid = %s",
                    (guid,)
                )
                user = cursor.fetchone()

                if not user:
                    print("ERROR: User credentials not found in 'users' table.")
                    return

                # 3. Construct Auth Dictionary (matching app.py logic)
                # We prioritize DB keys, falling back to Env vars if missing in DB
                auth_data = {
                    'consumer_key': user['consumer_key'] or os.environ.get("YAHOO_CONSUMER_KEY"),
                    'consumer_secret': user['consumer_secret'] or os.environ.get("YAHOO_CONSUMER_SECRET"),
                    'access_token': user['access_token'],
                    'refresh_token': user['refresh_token'],
                    'token_type': user['token_type'],
                    'token_time': user['token_time'],
                    'expires_in': user['expires_in'],
                    'guid': user['guid']
                }

                print("Initializing Yahoo API connection...")

                # 4. Initialize YFPY
                # We use the game_code 'nhl' as seen in your app.py
                yq = YahooFantasySportsQuery(
                    league_id,
                    game_code="nhl",
                    yahoo_access_token_json=auth_data,
                    yahoo_consumer_key=auth_data['consumer_key'],
                    yahoo_consumer_secret=auth_data['consumer_secret']
                )

                # 5. Execute the Query
                print("Fetching League Teams from Yahoo API...\n")
                teams = yq.get_league_teams()

                # 6. Print Results
                print(f"{'Team ID':<10} | {'Team Name'}")
                print("-" * 40)

                raw_ids = []
                for team in teams:
                    # Decode bytes if necessary (handling your specific issue)
                    t_name = team.name.decode('utf-8') if isinstance(team.name, bytes) else team.name
                    t_id = team.team_id
                    raw_ids.append(int(t_id))
                    print(f"{t_id:<10} | {t_name}")

                print("-" * 40)
                print(f"Sorted IDs found: {sorted(raw_ids)}")

                # Check for gaps
                expected = list(range(1, len(raw_ids) + 1))
                if sorted(raw_ids) != expected:
                    print(f"\n[ALERT] Non-sequential IDs detected!")
                    print(f"Expected sequential: {expected}")
                    print(f"Actual IDs:          {sorted(raw_ids)}")
                else:
                    print("\nIDs appear sequential.")

    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    lid = input("Enter League ID to debug: ").strip()
    if lid:
        debug_league_teams(lid)
    else:
        print("League ID is required.")
