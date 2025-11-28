"""
Fetches all players from the Yahoo Fantasy API and stores them in the Global Postgres DB.

Author: Jason Druckenmiller
Created: 11/02/2025
Updated: 11/26/2025
"""

import argparse
import logging
import os
import sys
import sqlite3
import unicodedata
import re
import json
import shutil
from datetime import date
from yfpy.query import YahooFantasySportsQuery
import psycopg2.extras

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import get_db_connection

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# NOTE: We rely on env vars or arguments for keys, no longer rely on file mounting for SQL,
# but we DO rely on it for the token file bootstrapping.
MOUNT_PATH = "/var/data/dbs"

def initialize_yahoo_query(league_id, consumer_key=None, consumer_secret=None):
    """
    Initializes YQ by fetching the token from the Postgres 'users' table
    associated with the given league_id.
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
    # Use keys from DB if not provided in args (args take precedence)
    ck = consumer_key or token_data['consumer_key']
    cs = consumer_secret or token_data['consumer_secret']

    if not ck or not cs:
         logger.error("Consumer Key/Secret missing.")
         return None

    access_token_json = {
        "access_token": token_data['access_token'],
        "refresh_token": token_data['refresh_token'],
        "token_type": token_data['token_type'],
        "expires_in": token_data['expires_in'],
        "token_time": token_data['token_time'],
        "guid": token_data['guid'],
        "consumer_key": ck,
        "consumer_secret": cs
    }

    try:
        # 3. Initialize YFPY with the dictionary directly
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
    players = yq.get_league_players()

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
