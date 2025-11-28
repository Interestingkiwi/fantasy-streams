"""
Fetches all players from the Yahoo Fantasy API and stores them in the Global Postgres DB.

Author: Jason Druckenmiller
Created: 11/02/2025
Updated: 11/28/2025
"""

import argparse
import logging
import os
import sys
import unicodedata
import re
import json
import tempfile
import time
from yfpy.query import YahooFantasySportsQuery
from yahoo_oauth import OAuth2
import psycopg2.extras

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import get_db_connection

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def initialize_yahoo_query(league_id, consumer_key=None, consumer_secret=None):
    """
    Initializes YQ by fetching AND REFRESHING the token from the Postgres 'users' table.
    """
    # 1. Find the User GUID assigned to update this league
    token_data = None

    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            # Join league_updaters -> users to get the token
            cursor.execute("""
                SELECT u.access_token, u.refresh_token, u.token_type,
                       u.expires_in, u.token_time, u.guid,
                       u.consumer_key, u.consumer_secret
                FROM league_updaters lu
                JOIN users u ON lu.user_guid = u.guid
                WHERE lu.league_id = %s
            """, (str(league_id),))

            row = cursor.fetchone()
            if row:
                token_data = row

    if not token_data:
        logger.error(f"No assigned updater found for league {league_id} in DB.")
        return None

    # 2. Construct the Token Dictionary
    ck = consumer_key or token_data['consumer_key']
    cs = consumer_secret or token_data['consumer_secret']

    if not ck or not cs:
         logger.error("Consumer Key/Secret missing.")
         return None

    creds = {
        "consumer_key": ck,
        "consumer_secret": cs,
        "access_token": token_data['access_token'],
        "refresh_token": token_data['refresh_token'],
        "token_type": token_data['token_type'],
        "expires_in": token_data['expires_in'],
        "token_time": token_data['token_time'],
        "xoauth_yahoo_guid": token_data['guid']
    }

    # 3. REFRESH TOKEN LOGIC
    temp_path = os.path.join(tempfile.gettempdir(), f"token_refresh_{league_id}.json")
    try:
        # --- CRITICAL FIX: Force Expiry ---
        # We trick the library into thinking the token expired 1 hour ago.
        # This forces a network call to Yahoo to get a fresh token.
        creds['token_time'] = time.time() - 3600
        # ----------------------------------

        # Write current creds to temp file
        with open(temp_path, 'w') as f:
            json.dump(creds, f)

        # Init OAuth2 to handle refresh
        sc = OAuth2(None, None, from_file=temp_path)
        sc.refresh_access_token()

        if not sc.token_is_valid():
            logger.error("Token refresh failed. User may need to re-login on the website.")
            return None

        # Read back fresh creds
        with open(temp_path, 'r') as f:
            new_creds = json.load(f)

        # --- FIX: Ensure 'guid' exists for yfpy ---
        if 'guid' not in new_creds:
            new_creds['guid'] = new_creds.get('xoauth_yahoo_guid', token_data['guid'])

        # 4. UPDATE DATABASE with fresh token
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE users SET
                        access_token = %s,
                        refresh_token = %s,
                        token_time = %s,
                        expires_in = %s
                    WHERE guid = %s
                """, (
                    new_creds['access_token'],
                    new_creds['refresh_token'],
                    new_creds['token_time'],
                    new_creds['expires_in'],
                    token_data['guid']
                ))
            conn.commit()
        logger.info("Token refreshed and saved to DB.")

        # Use the NEW credentials for yfpy
        access_token_json = new_creds

    except Exception as e:
        logger.error(f"Error during token refresh: {e}")
        if os.path.exists(temp_path): os.remove(temp_path)
        return None
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)

    try:
        # 5. Initialize YFPY with the FRESH dictionary
        yq = YahooFantasySportsQuery(
            league_id=league_id,
            game_code="nhl",
            yahoo_consumer_key=ck,
            yahoo_consumer_secret=cs,
            yahoo_access_token_json=access_token_json
        )
        return yq
    except Exception as e:
        logger.error(f"YQ Init failed: {e}")
        return None

def fetch_and_store_players(yq):
    logger.info("Fetching player info...")

    # Add a safety check before making the big call
    try:
        players = yq.get_league_players()
    except Exception as e:
        logger.error(f"CRITICAL: Failed to fetch league players from Yahoo. This usually means the token is invalid despite refresh. Error: {e}")
        return

    TEAM_TRICODE_MAP = {"TB": "TBL", "NJ": "NJD", "SJ": "SJS", "LA": "LAK", "MON": "MTL", "WAS": "WSH"}
    data = []

    for p in players:
        try:
            pid = str(p.player_id)
            norm = unicodedata.normalize('NFKD', p.name.full.lower())
            norm = re.sub(r'[^a-z0-9]', '', "".join([c for c in norm if not unicodedata.combining(c)]))

            if pid == "6777": norm += "f"
            elif pid == "7520": norm += "f"

            team = p.editorial_team_abbr.upper()
            team = TEAM_TRICODE_MAP.get(team, team)

            data.append((pid, p.name.full, team, p.display_position, p.status, norm))
        except: pass

    if not data:
        logger.warning("No players found/parsed.")
        return

    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS players (
                    player_id TEXT PRIMARY KEY,
                    player_name TEXT,
                    player_team TEXT,
                    positions TEXT,
                    status TEXT,
                    player_name_normalized TEXT
                )
            """)

            cursor.executemany("""
                INSERT INTO players (player_id, player_name, player_team, positions, status, player_name_normalized)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (player_id) DO UPDATE SET
                    player_name = EXCLUDED.player_name,
                    player_team = EXCLUDED.player_team,
                    status = EXCLUDED.status
            """, data)
            conn.commit()
            logger.info(f"Successfully updated {len(data)} players.")

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("league_id", type=int)
    parser.add_argument("-k", "--yahoo-consumer-key")
    parser.add_argument("-s", "--yahoo-consumer-secret")
    args = parser.parse_args()

    yq = initialize_yahoo_query(args.league_id, args.yahoo_consumer_key, args.yahoo_consumer_secret)
    if yq: fetch_and_store_players(yq)

if __name__ == "__main__":
    run()
