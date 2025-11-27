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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import get_db_connection

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# NOTE: We rely on env vars or arguments for keys, no longer rely on file mounting for SQL,
# but we DO rely on it for the token file bootstrapping.
MOUNT_PATH = "/var/data/dbs"

def initialize_yahoo_query(league_id, consumer_key, consumer_secret):
    # --- Bootstrapping Logic (Preserved) ---
    SECRET_TOKEN_PATH = "/etc/secrets/token.json"
    PERSISTENT_TOKEN_PATH = os.path.join(MOUNT_PATH, "token.json")
    os.makedirs(MOUNT_PATH, exist_ok=True)

    if not os.path.exists(PERSISTENT_TOKEN_PATH) and os.path.exists(SECRET_TOKEN_PATH):
        shutil.copy2(SECRET_TOKEN_PATH, PERSISTENT_TOKEN_PATH)

    original_cwd = os.getcwd()
    try:
        os.chdir(MOUNT_PATH)
        token_file_path = "token.json"
        kwargs = {}
        if consumer_key and consumer_secret:
            kwargs["yahoo_consumer_key"] = consumer_key
            kwargs["yahoo_consumer_secret"] = consumer_secret

        if os.path.exists(token_file_path):
            with open(token_file_path, 'r') as f:
                kwargs["yahoo_access_token_json"] = json.load(f)

        yq = YahooFantasySportsQuery(league_id=league_id, game_code="nhl", **kwargs)

        # Trigger call to verify/refresh
        yq.get_current_game_info()

        # Save back refreshed token
        if yq._yahoo_access_token_dict:
            with open(token_file_path, 'w') as f:
                json.dump(yq._yahoo_access_token_dict, f)

        os.chdir(original_cwd)
        return yq
    except Exception as e:
        logger.error(f"Init failed: {e}")
        os.chdir(original_cwd)
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
