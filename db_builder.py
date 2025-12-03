"""
PostgreSQL Compatible Builder
Refactored for Multi-Tenancy (League ID) and Postgres Syntax

Author: Jason Druckenmiller
Created: 10/17/2025
Updated: 12/03/2025
"""

import os
import re
import logging
import time
import unicodedata
from datetime import date, timedelta, datetime
import ast
import threading
import json
import tempfile
import sys
from pathlib import Path
from database import get_db_connection
import psycopg2.extras
import pytz
from collections import defaultdict


# --- API Imports ---
from yfpy.query import YahooFantasySportsQuery
from yahoo_oauth import OAuth2
import yahoo_fantasy_api as yfa

db_build_status = {"running": False, "error": None, "current_build_id": None}
db_build_status_lock = threading.Lock()

# --- Custom Logging Handler ---
class PostgresHandler(logging.Handler):
    """
    Custom logging handler that writes logs to the PostgreSQL 'job_logs' table.
    """
    def __init__(self, job_id):
        super().__init__()
        self.job_id = job_id

    def emit(self, record):
        log_entry = self.format(record)
        try:
            # We open a fresh connection for each log to ensure thread safety
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "INSERT INTO job_logs (job_id, message) VALUES (%s, %s)",
                        (self.job_id, log_entry)
                    )
                conn.commit()
        except Exception as e:
            # Fallback to stderr if DB logging fails
            print(f"Failed to log to DB: {e}", file=sys.stderr)

def run_task(build_id, log_file_path, options, data):
    global db_build_status

    # 1. Setup Logger
    logger = logging.getLogger(f"db_build_{build_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # 2. Add Handlers
    if not logger.handlers:
        # Database Handler (For UI Streaming)
        db_handler = PostgresHandler(build_id)
        db_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        db_handler.setFormatter(formatter)
        logger.addHandler(db_handler)

        # Console Handler (For Render Dashboard visibility)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(stream_handler)

    logger.info(f"Build task {build_id} received. Preparing API connections...")

    yq = None
    lg = None

    try:
        if not data.get('dev_mode'):
            # --- STEP 1: FETCH CREDENTIALS FROM DB ---
            guid = data['token'].get('xoauth_yahoo_guid')

            db_creds = None
            with get_db_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                    cursor.execute("SELECT * FROM users WHERE guid = %s", (guid,))
                    db_creds = cursor.fetchone()

            if db_creds:
                logger.info(f"Fetched credentials for {guid} from DB.")
                creds = {
                    "consumer_key": db_creds['consumer_key'],
                    "consumer_secret": db_creds['consumer_secret'],
                    "access_token": db_creds['access_token'],
                    "refresh_token": db_creds['refresh_token'],
                    "token_type": db_creds['token_type'],
                    "expires_in": db_creds['expires_in'],
                    "token_time": db_creds['token_time'],
                    "xoauth_yahoo_guid": db_creds['guid']
                }
            else:
                logger.warning("User not found in DB, falling back to passed data (might be stale).")
                creds = {
                    "consumer_key": data['consumer_key'],
                    "consumer_secret": data['consumer_secret'],
                    "access_token": data['token'].get('access_token'),
                    "refresh_token": data['token'].get('refresh_token'),
                    "token_type": data['token'].get('token_type', 'bearer'),
                    "token_time": data['token'].get('expires_at', time.time() + 3600),
                    "xoauth_yahoo_guid": data['token'].get('xoauth_yahoo_guid')
                }

            # --- STEP 2: REFRESH TOKEN ---
            temp_dir = os.path.join(tempfile.gettempdir(), 'temp_creds')
            os.makedirs(temp_dir, exist_ok=True)
            temp_file_path = os.path.join(temp_dir, f"thread_{build_id}.json")

            # Force expiry so it ALWAYS refreshes
            creds['token_time'] = time.time() - 4000

            with open(temp_file_path, 'w') as f:
                json.dump(creds, f)

            logger.info("Validating/Refreshing token...")
            sc = OAuth2(None, None, from_file=temp_file_path)
            sc.refresh_access_token()

            if not sc.token_is_valid():
                 raise Exception("Token refresh failed. User may need to re-authenticate.")

            # --- STEP 3: READ & FIX TOKEN ---
            with open(temp_file_path, 'r') as f:
                new_creds = json.load(f)

            # Inject 'guid' key for yfpy
            if 'guid' not in new_creds:
                new_creds['guid'] = new_creds.get('xoauth_yahoo_guid') or guid

            # --- STEP 4: SAVE TO DB ---
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
                        guid
                    ))
                conn.commit()
            logger.info("Token refreshed and saved to DB.")

            # --- STEP 5: INIT LIBRARIES ---
            logger.info("Initializing Yahoo APIs with fresh token...")

            yq = YahooFantasySportsQuery(
                data['league_id'],
                game_code="nhl",
                yahoo_access_token_json=new_creds,
                yahoo_consumer_key=new_creds['consumer_key'],
                yahoo_consumer_secret=new_creds['consumer_secret']
            )

            gm = yfa.Game(sc, 'nhl')
            lg = gm.to_league(f"nhl.l.{data['league_id']}")

            logger.info("Yahoo API authentication successful.")

            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

        else:
            logger.info("Dev mode: Skipping real API object creation in thread.")
            yq = None
            lg = None

        logger.info("--- Starting Database Update ---")
        logger.info(f"League ID: {data['league_id']}")
        logger.info(f"Build ID: {build_id}")

        roster_updates_only = options.get('roster_updates_only', False)
        if roster_updates_only:
             logger.info("Mode: Roster Updates Only")
        else:
             logger.info(f"Mode: Standard Update (Capture Lineups: {options['capture_lineups']})")

        result = update_league_db(
            yq,
            lg,
            data['league_id'],
            logger,
            capture_lineups=options['capture_lineups'],
            roster_updates_only=options.get('roster_updates_only', False)
        )

        if result and result.get('success'):
            logger.info(f"--- SUCCESS: {result.get('league_name')} updated. ---")
        else:
            error_msg = result.get('error', 'Unknown error')
            logger.error(f"--- ERROR: {error_msg} ---")
            with db_build_status_lock:
                db_build_status["error"] = error_msg

    except Exception as e:
        error_str = f"--- FATAL ERROR: {str(e)} ---"
        logger.error(error_str, exc_info=True)
        with db_build_status_lock:
            db_build_status["error"] = str(e)
    finally:
        with db_build_status_lock:
            error_msg = db_build_status.get("error")
            db_build_status["running"] = False
            db_build_status["error"] = error_msg
            db_build_status["current_build_id"] = None

        logger.info(f"Build task {build_id} thread finished.")


# --- DB Finalizer Class ---
class DBFinalizer:
    """
    Processes statistics in the Postgres database.
    """
    def __init__(self, league_id, logger, conn):
        self.league_id = league_id
        self.logger = logger
        self.conn = conn

    def parse_and_store_player_stats(self):
        cursor = self.conn.cursor()

        cursor.execute("SELECT to_regclass('public.daily_lineups_dump');")
        if cursor.fetchone()[0] is None:
            self.logger.info("Table 'daily_lineups_dump' does not exist. Skipping stat parsing.")
            return

        # Create target table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_player_stats (
                league_id INTEGER NOT NULL,
                date_ TEXT NOT NULL,
                team_id INTEGER NOT NULL,
                player_id INTEGER NOT NULL,
                player_name_normalized TEXT,
                lineup_pos TEXT,
                stat_id INTEGER NOT NULL,
                category TEXT,
                stat_value REAL,
                PRIMARY KEY (league_id, date_, player_id, stat_id)
            );
        """)
        self.conn.commit()

        cursor.execute("SELECT MAX(date_) FROM daily_player_stats WHERE league_id = %s", (self.league_id,))
        max_processed_date_result = cursor.fetchone()
        last_processed_date = max_processed_date_result[0] if max_processed_date_result else None

        yesterday_iso = (date.today() - timedelta(days=1)).isoformat()

        if last_processed_date:
            self.logger.info(f"Parsing daily stats: Resuming after {last_processed_date} AND re-processing {yesterday_iso}.")
            dump_query = "SELECT * FROM daily_lineups_dump WHERE league_id = %s AND (date_ > %s OR date_ = %s)"
            query_params = (self.league_id, last_processed_date, yesterday_iso)
        else:
            self.logger.info("Parsing daily stats: Processing all dates for this league.")
            dump_query = "SELECT * FROM daily_lineups_dump WHERE league_id = %s"
            query_params = (self.league_id,)

        cursor.execute(dump_query, query_params)
        column_names = [desc[0] for desc in cursor.description]
        all_lineups = cursor.fetchall()

        if not all_lineups:
            self.logger.info("No new dates found in daily_lineups_dump to process.")
            return

        self.logger.info(f"Parsing raw player strings for {len(all_lineups)} new/updated rows...")

        stat_map = {
            1: 'G', 2: 'A', 3: 'P', 4: '+/-', 5: 'PIM', 6: 'PPG', 7: 'PPA', 8: 'PPP',
            9: 'SHG', 10: 'SHA', 11: 'SHP', 12: 'GWG', 13: 'GTG', 14: 'SOG', 15: 'SH%',
            16: 'FW', 17: 'FL', 31: 'HIT', 32: 'BLK', 18: 'GS', 19: 'W', 20: 'L',
            22: 'GA', 23: 'GAA', 24: 'SA', 25: 'SV', 26: 'SV%', 27: 'SHO', 28: 'TOI/G',
            29: 'GP/S', 30: 'GP/G', 33: 'TOI/S', 34: 'TOI/S/Gm'
        }

        # Get player name map (GLOBAL TABLE)
        cursor.execute("SELECT player_id, player_name_normalized FROM players")
        player_norm_name_map = dict(cursor.fetchall())

        stats_to_insert = []
        player_string_pattern = re.compile(r"ID: (\d+), Name: .*, Stats: (\[.*\])")
        pos_pattern = re.compile(r"([a-zA-Z]+)")
        active_roster_columns = ['c1', 'c2', 'l1', 'l2', 'r1', 'r2', 'd1', 'd2', 'd3', 'd4', 'g1', 'g2']

        for row in all_lineups:
            try:
                row_dict = dict(zip(column_names, row))
                date_ = row_dict['date_']
                team_id = row_dict['team_id']
            except Exception as e:
                self.logger.error(f"Error processing row: {e}")
                continue

            for col in active_roster_columns:
                if col in row_dict and row_dict[col]:
                    player_string = row_dict[col]
                    match = player_string_pattern.match(player_string)
                    if match:
                        player_id = int(match.group(1))
                        stats_list_str = match.group(2)
                        pos_match = pos_pattern.match(col)
                        lineup_pos = pos_match.group(1) if pos_match else None
                        player_name_normalized = player_norm_name_map.get(str(player_id))

                        try:
                            stats_list = ast.literal_eval(stats_list_str)
                            player_stats = dict(stats_list)

                            if (lineup_pos == 'g' and 22 in player_stats and 23 in player_stats):
                                val_22_ga = player_stats[22]
                                val_23_gaa = player_stats[23]
                                if val_23_gaa > 0:
                                    val_28_toi = (val_22_ga / val_23_gaa) * 60
                                    player_stats[28] = round(val_28_toi, 2)

                            for stat_id, stat_value in player_stats.items():
                                category = stat_map.get(stat_id, 'UNKNOWN')
                                stats_to_insert.append((
                                    self.league_id, date_, team_id, player_id, player_name_normalized,
                                    lineup_pos, stat_id, category, stat_value
                                ))
                        except (ValueError, SyntaxError) as e:
                            self.logger.warning(f"Could not parse stats for player {player_id} on {date_}: {e}")

        if stats_to_insert:
            self.logger.info(f"Found {len(stats_to_insert)} stats. Inserting...")
            cursor.executemany("""
                INSERT INTO daily_player_stats (
                    league_id, date_, team_id, player_id, player_name_normalized, lineup_pos,
                    stat_id, category, stat_value
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (league_id, date_, player_id, stat_id)
                DO UPDATE SET stat_value = EXCLUDED.stat_value
            """, stats_to_insert)
            self.conn.commit()
            self.logger.info("Stats inserted.")
        else:
            self.logger.info("No new stats found.")

    def parse_and_store_bench_stats(self):
        cursor = self.conn.cursor()

        cursor.execute("SELECT to_regclass('public.daily_lineups_dump');")
        if cursor.fetchone()[0] is None:
            return

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_bench_stats (
                league_id INTEGER NOT NULL,
                date_ TEXT NOT NULL,
                team_id INTEGER NOT NULL,
                player_id INTEGER NOT NULL,
                player_name_normalized TEXT,
                lineup_pos TEXT,
                stat_id INTEGER NOT NULL,
                category TEXT,
                stat_value REAL,
                PRIMARY KEY (league_id, date_, player_id, stat_id)
            );
        """)
        self.conn.commit()

        cursor.execute("SELECT MAX(date_) FROM daily_bench_stats WHERE league_id = %s", (self.league_id,))
        res = cursor.fetchone()
        last_processed = res[0] if res else None
        yesterday_iso = (date.today() - timedelta(days=1)).isoformat()

        if last_processed:
            dump_query = "SELECT * FROM daily_lineups_dump WHERE league_id = %s AND (date_ > %s OR date_ = %s)"
            params = (self.league_id, last_processed, yesterday_iso)
        else:
            dump_query = "SELECT * FROM daily_lineups_dump WHERE league_id = %s"
            params = (self.league_id,)

        cursor.execute(dump_query, params)
        column_names = [desc[0] for desc in cursor.description]
        all_lineups = cursor.fetchall()

        if not all_lineups:
            return

        stat_map = {
            1: 'G', 2: 'A', 3: 'P', 4: '+/-', 5: 'PIM', 6: 'PPG', 7: 'PPA', 8: 'PPP',
            9: 'SHG', 10: 'SHA', 11: 'SHP', 12: 'GWG', 13: 'GTG', 14: 'SOG', 15: 'SH%',
            16: 'FW', 17: 'FL', 31: 'HIT', 32: 'BLK', 18: 'GS', 19: 'W', 20: 'L',
            22: 'GA', 23: 'GAA', 24: 'SA', 25: 'SV', 26: 'SV%', 27: 'SHO', 28: 'TOI/G',
            29: 'GP/S', 30: 'GP/G', 33: 'TOI/S', 34: 'TOI/S/Gm'
        }

        cursor.execute("SELECT player_id, player_name_normalized FROM players")
        player_norm_name_map = dict(cursor.fetchall())

        stats_to_insert = []

        # --- FIX START: Added regex patterns ---
        player_string_pattern = re.compile(r"ID: (\d+), Name: .*, Stats: (\[.*\])")
        pos_pattern = re.compile(r"([a-zA-Z]+)")
        # --- FIX END ---

        # [FIX] This needs to be dynamic too if we want full coverage, but for now
        # the main roster fetch is the priority.
        # Since this finalizer parses columns that *exist*, we can just grab all columns
        # starting with 'b', 'i', or 'n' if we wanted to be truly dynamic.
        # For safety in this update, we will fetch all columns from the table schema.

        # Determine Bench Columns dynamically from the cursor description
        bench_roster_columns = [col for col in column_names if col.startswith('b') or col.startswith('i') or col.startswith('n') or col.startswith('u')]

        for row in all_lineups:
            try:
                row_dict = dict(zip(column_names, row))
                date_ = row_dict['date_']
                team_id = row_dict['team_id']
            except:
                continue

            for col in bench_roster_columns:
                if col in row_dict and row_dict[col]:
                    player_string = row_dict[col]
                    match = player_string_pattern.match(player_string)
                    if match:
                        player_id = int(match.group(1))
                        stats_list_str = match.group(2)
                        pos_match = pos_pattern.match(col)
                        lineup_pos = pos_match.group(1) if pos_match else None
                        player_name_normalized = player_norm_name_map.get(str(player_id))

                        try:
                            stats_list = ast.literal_eval(stats_list_str)
                            player_stats = dict(stats_list)

                            if (22 in player_stats and 23 in player_stats):
                                val_22_ga = player_stats[22]
                                val_23_gaa = player_stats[23]
                                if val_23_gaa > 0:
                                    val_28_toi = (val_22_ga / val_23_gaa) * 60
                                    player_stats[28] = round(val_28_toi, 2)

                            for stat_id, stat_value in player_stats.items():
                                category = stat_map.get(stat_id, 'UNKNOWN')
                                stats_to_insert.append((
                                    self.league_id, date_, team_id, player_id, player_name_normalized,
                                    lineup_pos, stat_id, category, stat_value
                                ))
                        except:
                            pass

        if stats_to_insert:
            self.logger.info(f"Found {len(stats_to_insert)} bench stats.")
            cursor.executemany("""
                INSERT INTO daily_bench_stats (
                    league_id, date_, team_id, player_id, player_name_normalized, lineup_pos,
                    stat_id, category, stat_value
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (league_id, date_, player_id, stat_id)
                DO UPDATE SET stat_value = EXCLUDED.stat_value
            """, stats_to_insert)
            self.conn.commit()


# --- Helper Functions ---

def _create_tables(cursor, logger):
    logger.info("Creating database tables if they don't exist...")

    # ------------------------------------------------------------------------------

    # Tables with league_id in PK
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS league_info (
            league_id INTEGER NOT NULL,
            key TEXT NOT NULL,
            value TEXT,
            PRIMARY KEY (league_id, key)
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS teams (
            league_id INTEGER NOT NULL,
            team_id TEXT NOT NULL,
            name TEXT,
            manager_nickname TEXT,
            PRIMARY KEY (league_id, team_id)
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_lineups_dump (
            league_id INTEGER NOT NULL,
            date_ TEXT NOT NULL,
            team_id INTEGER NOT NULL,
            c1 TEXT, c2 TEXT, l1 TEXT, l2 TEXT, r1 TEXT, r2 TEXT,
            d1 TEXT, d2 TEXT, d3 TEXT, d4 TEXT, g1 TEXT, g2 TEXT,
            b1 TEXT, b2 TEXT, b3 TEXT, b4 TEXT, b5 TEXT, b6 TEXT,
            b7 TEXT, b8 TEXT, b9 TEXT, b10 TEXT, b11 TEXT, b12 TEXT,
            b13 TEXT, b14 TEXT, b15 TEXT, b16 TEXT, b17 TEXT, b18 TEXT, b19 TEXT,
            i1 TEXT, i2 TEXT, i3 TEXT, i4 TEXT, i5 TEXT,
            PRIMARY KEY (league_id, date_, team_id)
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS scoring (
            league_id INTEGER NOT NULL,
            stat_id INTEGER NOT NULL,
            category TEXT NOT NULL,
            scoring_group TEXT NOT NULL,
            PRIMARY KEY (league_id, stat_id)
        )
    ''')
    # Constraint added here
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS lineup_settings (
            league_id INTEGER NOT NULL,
            position_id SERIAL PRIMARY KEY,
            position TEXT NOT NULL,
            position_count INTEGER NOT NULL,
            UNIQUE (league_id, position)
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS weeks (
            league_id INTEGER NOT NULL,
            week_num INTEGER NOT NULL,
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            PRIMARY KEY (league_id, week_num)
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS matchups (
            league_id INTEGER NOT NULL,
            week INTEGER NOT NULL,
            team1 TEXT NOT NULL,
            team2 TEXT NOT NULL,
            PRIMARY KEY (league_id, week, team1)
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS rosters (
            league_id INTEGER NOT NULL,
            team_id INTEGER NOT NULL,
            p1 INTEGER, p2 INTEGER, p3 INTEGER, p4 INTEGER, p5 INTEGER,
            p6 INTEGER, p7 INTEGER, p8 INTEGER, p9 INTEGER, p10 INTEGER,
            p11 INTEGER, p12 INTEGER, p13 INTEGER, p14 INTEGER, p15 INTEGER,
            p16 INTEGER, p17 INTEGER, p18 INTEGER, p19 INTEGER, p20 INTEGER,
            p21 INTEGER, p22 INTEGER, p23 INTEGER, p24 INTEGER, p25 INTEGER,
            p26 INTEGER, p27 INTEGER, p28 INTEGER, p29 INTEGER,
            PRIMARY KEY (league_id, team_id)
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS free_agents (
            league_id INTEGER NOT NULL,
            player_id INTEGER NOT NULL,
            status TEXT,
            PRIMARY KEY (league_id, player_id)
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS waiver_players (
            league_id INTEGER NOT NULL,
            player_id INTEGER NOT NULL,
            status TEXT,
            PRIMARY KEY (league_id, player_id)
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS rostered_players (
            league_id INTEGER NOT NULL,
            player_id INTEGER NOT NULL,
            status TEXT,
            eligible_positions TEXT,
            PRIMARY KEY (league_id, player_id)
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS db_metadata (
            league_id INTEGER NOT NULL,
            key TEXT NOT NULL,
            value TEXT,
            PRIMARY KEY (league_id, key)
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            league_id INTEGER NOT NULL,
            transaction_date TEXT NOT NULL,
            player_id INTEGER NOT NULL,
            player_name TEXT NOT NULL,
            fantasy_team TEXT,
            move_type TEXT,
            PRIMARY KEY (league_id, transaction_date, player_id, move_type)
        )
    ''')
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_player_stats (
            league_id INTEGER NOT NULL,
            date_ TEXT NOT NULL,
            team_id INTEGER NOT NULL,
            player_id INTEGER NOT NULL,
            player_name_normalized TEXT,
            lineup_pos TEXT,
            stat_id INTEGER NOT NULL,
            category TEXT,
            stat_value REAL,
            PRIMARY KEY (league_id, date_, player_id, stat_id)
        );
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_bench_stats (
            league_id INTEGER NOT NULL,
            date_ TEXT NOT NULL,
            team_id INTEGER NOT NULL,
            player_id INTEGER NOT NULL,
            player_name_normalized TEXT,
            lineup_pos TEXT,
            stat_id INTEGER NOT NULL,
            category TEXT,
            stat_value REAL,
            PRIMARY KEY (league_id, date_, player_id, stat_id)
        );
    """)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS rosters_tall (
            league_id INTEGER NOT NULL,
            team_id INTEGER NOT NULL,
            player_id INTEGER NOT NULL,
            PRIMARY KEY (league_id, team_id, player_id)
        );
    ''')

def _update_league_info(yq, cursor, league_id, league_name, league_metadata, logger):
    logger.info("Updating league_info table...")
    data_to_insert = [
        (league_id, 'league_id', str(league_id)),
        (league_id, 'league_name', league_name),
        (league_id, 'num_teams', str(league_metadata.num_teams)),
        (league_id, 'start_date', str(league_metadata.start_date)),
        (league_id, 'end_date', str(league_metadata.end_date))
    ]
    sql = """
        INSERT INTO league_info (league_id, key, value)
        VALUES (%s, %s, %s)
        ON CONFLICT (league_id, key)
        DO UPDATE SET value = EXCLUDED.value
    """
    cursor.executemany(sql, data_to_insert)

def _update_teams_info(yq, cursor, league_id, logger):
    logger.info("Updating teams table...")
    try:
        teams = yq.get_league_teams()
        teams_data_to_insert = []
        for team in teams:
            team_id = team.team_id

            # --- FIX: Decode team name if it is bytes ---
            team_name = team.name
            if isinstance(team_name, bytes):
                team_name = team_name.decode('utf-8')

            manager_nickname = None
            if team.managers and team.managers[0].nickname:
                manager_nickname = team.managers[0].nickname
                # --- FIX: Decode manager nickname if it is bytes ---
                if isinstance(manager_nickname, bytes):
                    manager_nickname = manager_nickname.decode('utf-8')

            teams_data_to_insert.append((league_id, team_id, team_name, manager_nickname))

        sql = """
            INSERT INTO teams (league_id, team_id, name, manager_nickname)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (league_id, team_id)
            DO UPDATE SET
                name = EXCLUDED.name,
                manager_nickname = EXCLUDED.manager_nickname
        """
        cursor.executemany(sql, teams_data_to_insert)
    except Exception as e:
        logger.error(f"Failed to update teams info: {e}", exc_info=True)

def _get_current_week_start_date(cursor, league_id, logger):
    try:
        today = date.today().isoformat()
        cursor.execute(
            "SELECT start_date FROM weeks WHERE league_id = %s AND start_date <= %s AND end_date >= %s",
            (league_id, today, today)
        )
        result = cursor.fetchone()
        if result:
            return result[0]
        else:
            return today
    except Exception as e:
        logger.error(f"Could not determine current week start date: {e}")
        return date.today().isoformat()

def _update_daily_lineups(yq, cursor, conn, league_id, num_teams, league_start_date, is_full_mode, logger):
    """
    Fetches daily roster snapshots.
    UPDATED:
    1. Fetches valid Team IDs first (handles non-sequential IDs).
    2. Dynamically generates required columns based on lineup_settings.
    3. Alters the table to add missing columns if necessary.
    """
    try:
        cursor.execute("SELECT MAX(date_) FROM daily_lineups_dump WHERE league_id = %s", (league_id,))
        res = cursor.fetchone()
        last_fetch_date_str = res[0] if res else None

        today_iso = date.today().isoformat()

        if is_full_mode or not last_fetch_date_str:
            start_date_for_fetch = league_start_date
            logger.info(f"Fetching full history from: {start_date_for_fetch}")
        else:
            last_fetch_date = date.fromisoformat(last_fetch_date_str)
            start_date_for_fetch = (last_fetch_date + timedelta(days=1)).isoformat()
            logger.info(f"Resuming from: {start_date_for_fetch}")

        if start_date_for_fetch >= today_iso:
            logger.info("Daily lineups up to date.")
            return

        # --- DYNAMIC COLUMN GENERATION ---
        logger.info("Generating dynamic roster columns based on settings...")
        cursor.execute("SELECT position, position_count FROM lineup_settings WHERE league_id = %s", (league_id,))
        settings = cursor.fetchall()

        if not settings:
            logger.warning("No lineup settings found! Falling back to standard defaults.")
            # Default fallback just in case
            roster_cols = ['c1', 'c2', 'l1', 'l2', 'r1', 'r2', 'd1', 'd2', 'd3', 'd4', 'g1', 'g2', 'b1', 'b2', 'b3', 'b4']
        else:
            # Map Yahoo positions to our DB prefixes
            prefix_map = {
                'C': 'c', 'LW': 'l', 'RW': 'r', 'F': 'f', 'D': 'd', 'G': 'g',
                'Util': 'u', 'NA': 'n', 'IR': 'i', 'IR+': 'i', 'BN': 'bn'
            }

            prefix_counts = defaultdict(int)
            total_roster_spots = 0

            # 1. Sum up counts per prefix (excluding BN first)
            for row in settings:
                pos = row[0] # position
                count = row[1] # position_count

                if pos == 'BN':
                    continue

                prefix = prefix_map.get(pos, pos[0].lower())
                prefix_counts[prefix] += count
                total_roster_spots += count

            # 2. Handle BN: "BN should have as many spots as total position_count"
            # We use the calculated total_roster_spots for the BN count
            # (plus some padding to be safe, e.g. +5, but user asked for total)
            prefix_counts['bn'] = total_roster_spots

            # 3. Build the final sorted list of columns
            # Order matters for readability but not for SQL logic.
            # We'll follow standard order: C, L, R, D, U, G, I, N, BN
            order = ['c', 'l', 'r', 'f', 'd', 'u', 'g', 'i', 'n', 'bn']
            roster_cols = []

            for p in order:
                count = prefix_counts.get(p, 0)
                for i in range(1, count + 1):
                    roster_cols.append(f"{p}{i}")

        # --- DYNAMIC SCHEMA UPDATE ---
        # Check which columns exist in the DB and add missing ones
        logger.info(f"Required roster columns: {roster_cols}")

        # Get existing columns in the table
        cursor.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'daily_lineups_dump'
        """)
        existing_cols = {row[0] for row in cursor.fetchall()}

        for col in roster_cols:
            if col not in existing_cols:
                logger.info(f"Adding missing column to daily_lineups_dump: {col}")
                cursor.execute(f"ALTER TABLE daily_lineups_dump ADD COLUMN IF NOT EXISTS {col} TEXT")

        # Commit schema changes immediately
        conn.commit()

        # --- FETCH TEAMS (Sequential ID Fix) ---
        logger.info("Fetching valid team IDs from Yahoo to handle non-sequential IDs...")
        league_teams = yq.get_league_teams()
        sorted_teams = sorted(league_teams, key=lambda t: int(t.team_id))

        lineup_data_to_insert = []

        for team in sorted_teams:
            team_id = int(team.team_id)
            current_date = start_date_for_fetch

            while current_date < today_iso:
                logger.info(f"Fetching daily lineups for team {team_id}, {current_date}...")

                try:
                    players = yq.get_team_roster_player_info_by_date(team_id, current_date)
                except Exception as e:
                    logger.error(f"Failed to fetch roster for team {team_id} on {current_date}: {e}")
                    # Skip day on error to prevent crashing entire loop
                    current_date = (date.fromisoformat(current_date) + timedelta(1)).isoformat()
                    continue

                # Counters for this specific roster
                counts = defaultdict(int)
                lineup_data_raw = {}

                for player in players:
                    pos_type = player.selected_position.position

                    # Map API position to our DB prefix
                    if pos_type == 'IR+':
                        bucket_key = 'i'
                    elif pos_type in prefix_map:
                        bucket_key = prefix_map[pos_type]
                    else:
                        # Fallback for unknown
                        bucket_key = pos_type.lower()[0]

                    counts[bucket_key] += 1
                    col_name = f"{bucket_key}{counts[bucket_key]}"

                    # Store data if this column is tracked in our roster_cols
                    if col_name in roster_cols:
                        p_stats = []
                        if player.player_stats and player.player_stats.stats:
                            p_stats = [(s.stat_id, s.value) for s in player.player_stats.stats]

                        # Safe name string
                        p_name = player.name.full.decode('utf-8') if isinstance(player.name.full, bytes) else player.name.full

                        lineup_data_raw[col_name] = f"ID: {player.player_id}, Name: {p_name}, Stats: {p_stats}"

                # Build row
                roster_values = [lineup_data_raw.get(col) for col in roster_cols]
                full_row = [league_id, current_date, team_id] + roster_values
                lineup_data_to_insert.append(tuple(full_row))

                current_date = (date.fromisoformat(current_date) + timedelta(1)).isoformat()

        if not lineup_data_to_insert:
            return

        # --- DYNAMIC INSERT QUERY ---
        update_clause = ", ".join([f"{col} = EXCLUDED.{col}" for col in roster_cols])
        placeholders = ", ".join(["%s"] * (3 + len(roster_cols)))
        col_list_str = ", ".join(roster_cols)

        sql = f"""
            INSERT INTO daily_lineups_dump (
                league_id, date_, team_id, {col_list_str}
            ) VALUES ({placeholders})
            ON CONFLICT (league_id, date_, team_id)
            DO UPDATE SET {update_clause}
        """
        cursor.executemany(sql, lineup_data_to_insert)
        logger.info(f"Inserted {len(lineup_data_to_insert)} lineup rows.")

    except Exception as e:
        logger.error(f"Failed to update lineup info: {e}", exc_info=True)

def _update_league_scoring_settings(yq, cursor, league_id, logger):
    logger.info("Fetching league scoring...")
    try:
        settings = yq.get_league_settings()
        data = []
        for stat in settings.stat_categories.stats:
            cat = stat.display_name
            if cat == 'SV%': cat = 'SVpct'
            data.append((league_id, stat.stat_id, cat, stat.group))

        sql = """
            INSERT INTO scoring (league_id, stat_id, category, scoring_group)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (league_id, stat_id) DO NOTHING
        """
        cursor.executemany(sql, data)
        return settings.playoff_start_week
    except Exception as e:
        logger.error(f"Failed to update scoring: {e}", exc_info=True)
        return None

def _update_lineup_settings(yq, cursor, league_id, logger):
    try:
        settings = yq.get_league_settings()
        data = []
        for pos in settings.roster_positions:
            data.append((league_id, pos.position, pos.count))

        sql = """
            INSERT INTO lineup_settings (league_id, position, position_count)
            VALUES (%s, %s, %s)
            ON CONFLICT (league_id, position) DO UPDATE SET position_count = EXCLUDED.position_count
        """
        cursor.executemany(sql, data)
    except Exception as e:
        logger.error(f"Failed to update lineup settings: {e}", exc_info=True)

def _update_fantasy_weeks(yq, cursor, league_id, league_key, logger):
    logger.info("Fetching fantasy weeks...")
    try:
        game_id = league_key.split('.')[0]
        weeks = yq.get_game_weeks_by_game_id(game_id)
        data = [(league_id, w.week, w.start, w.end) for w in weeks]

        sql = """
            INSERT INTO weeks (league_id, week_num, start_date, end_date)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (league_id, week_num) DO NOTHING
        """
        cursor.executemany(sql, data)
    except Exception as e:
        logger.error(f"Failed to update weeks: {e}", exc_info=True)

def _update_league_matchups(yq, cursor, league_id, playoff_start_week, logger):
    logger.info("Fetching league matchups...")
    try:
        # --- FIX: Clear old matchups to handle team name changes ---
        cursor.execute("DELETE FROM matchups WHERE league_id = %s", (league_id,))
        # -----------------------------------------------------------

        if not playoff_start_week: return
        last_reg_week = playoff_start_week - 1
        start_week = 1
        data = []

        while start_week <= last_reg_week:
            matchups = yq.get_league_matchups_by_week(start_week)
            for m in matchups:
                # Assumes 2 teams per matchup
                t1_name = m.teams[0].name
                if isinstance(t1_name, bytes): t1_name = t1_name.decode('utf-8')

                t2_name = m.teams[1].name
                if isinstance(t2_name, bytes): t2_name = t2_name.decode('utf-8')

                data.append((league_id, start_week, t1_name, t2_name))
            start_week += 1

        sql = """
            INSERT INTO matchups (league_id, week, team1, team2)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (league_id, week, team1) DO NOTHING
        """
        cursor.executemany(sql, data)
    except Exception as e:
        logger.error(f"Failed to update matchups: {e}", exc_info=True)

def _update_current_rosters(yq, cursor, conn, league_id, num_teams, logger):
    logger.info("Fetching current roster info...")
    try:
        cursor.execute("DELETE FROM rosters WHERE league_id = %s", (league_id,))
        conn.commit()
    except:
        conn.rollback()
        return

    try:
        data = []
        MAX_PLAYERS = 29
        for team_id in range(1, num_teams + 1):
            players = yq.get_team_roster_player_info_by_date(team_id, date.today().isoformat())
            p_ids = [p.player_id for p in players][:MAX_PLAYERS]
            padded = p_ids + [None] * (MAX_PLAYERS - len(p_ids))
            data.append([league_id, team_id] + padded)

        placeholders = ', '.join(['%s'] * (MAX_PLAYERS + 2))
        cols = ", ".join([f"p{i}" for i in range(1, MAX_PLAYERS + 1)])
        sql = f"INSERT INTO rosters (league_id, team_id, {cols}) VALUES ({placeholders})"
        cursor.executemany(sql, data)
        conn.commit()
    except Exception as e:
        logger.error(f"Failed to update rosters: {e}", exc_info=True)

def _create_rosters_tall(cursor, conn, league_id, logger):
    logger.info("Updating tall rosters table...")
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rosters_tall (
                league_id INTEGER NOT NULL,
                team_id INTEGER NOT NULL,
                player_id INTEGER NOT NULL,
                PRIMARY KEY (league_id, team_id, player_id)
            );
        """)
        cursor.execute("DELETE FROM rosters_tall WHERE league_id = %s", (league_id,))
        cols = ", ".join([f"(p{i})" for i in range(1, 30)])
        sql = f"""
            INSERT INTO rosters_tall (league_id, team_id, player_id)
            SELECT r.league_id, r.team_id, u.player_id
            FROM rosters r
            CROSS JOIN LATERAL (VALUES {cols}) AS u(player_id)
            WHERE r.league_id = %s AND u.player_id IS NOT NULL;
        """

        cursor.execute(sql, (league_id,))
        conn.commit()
    except Exception as e:
        logger.error(f"Failed to create rosters tall: {e}", exc_info=True)
        conn.rollback()

def _update_league_transactions(yq, cursor, league_id, logger):
    try:
        logger.info("Fetching transactions...")
        cursor.execute("DELETE FROM transactions WHERE league_id = %s", (league_id,))
        # -------------------------------------------------------------------

        transactions = yq.get_league_transactions()
        data = []
        processed = set()

        # Define Timezone
        pacific_tz = pytz.timezone('US/Pacific')

        for t in transactions:
            if t.status == 'successful':
                # Convert Unix timestamp to aware UTC, then to Pacific
                dt_utc = datetime.fromtimestamp(t.timestamp, pytz.utc)
                dt_pacific = dt_utc.astimezone(pacific_tz)
                t_date = dt_pacific.strftime('%Y-%m-%d')

                for p in t.players:
                    move = p.transaction_data.type

                    # Decode Player Name
                    p_name = p.name.full
                    if isinstance(p_name, bytes):
                        p_name = p_name.decode('utf-8')

                    # Decode Team Name
                    raw_team_name = p.transaction_data.destination_team_name if move == 'add' else p.transaction_data.source_team_name
                    if isinstance(raw_team_name, bytes):
                        raw_team_name = raw_team_name.decode('utf-8')

                    # Check uniqueness (timestamp + player + move)
                    key = (t.timestamp, p.player_id, move)
                    if key not in processed:
                        data.append((league_id, t_date, p.player_id, p_name, raw_team_name, move))
                        processed.add(key)

        sql = """
            INSERT INTO transactions (league_id, transaction_date, player_id, player_name, fantasy_team, move_type)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (league_id, transaction_date, player_id, move_type) DO NOTHING
        """
        cursor.executemany(sql, data)
    except Exception as e:
        logger.error(f"Failed to update transactions: {e}", exc_info=True)

def _update_free_agents(lg, cursor, league_id, logger):
    logger.info("Updating free agents...")
    cursor.execute("DELETE FROM free_agents WHERE league_id = %s", (league_id,))
    data = []
    for pos in ['C', 'LW', 'RW', 'D', 'G']:
        try:
            fas = lg.free_agents(pos)
            for p in fas:
                data.append((league_id, p['player_id'], 'FA'))
        except: pass

    cursor.executemany("""
        INSERT INTO free_agents (league_id, player_id, status)
        VALUES (%s, %s, %s) ON CONFLICT DO NOTHING
    """, data)

def _update_waivers(lg, cursor, league_id, logger):
    logger.info("Updating waivers...")
    cursor.execute("DELETE FROM waiver_players WHERE league_id = %s", (league_id,))
    data = []
    try:
        wvp = lg.waivers()
        for p in wvp:
            data.append((league_id, p['player_id'], 'W'))
    except: pass

    cursor.executemany("""
        INSERT INTO waiver_players (league_id, player_id, status)
        VALUES (%s, %s, %s) ON CONFLICT DO NOTHING
    """, data)

def _update_rostered_players(lg, cursor, league_id, logger):
    logger.info("Updating rostered players...")
    cursor.execute("DELETE FROM rostered_players WHERE league_id = %s", (league_id,))
    data = []
    try:
        tkp = lg.taken_players()
        for p in tkp:
            pos = ','.join(p['eligible_positions'])
            data.append((league_id, p['player_id'], 'R', pos))
    except: pass

    cursor.executemany("""
        INSERT INTO rostered_players (league_id, player_id, status, eligible_positions)
        VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING
    """, data)

def _update_db_metadata(cursor, league_id, logger, update_available_players_timestamp=False):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    ts_str = now.strftime("%Y-%m-%d %H:%M:%S")
    data = []
    if update_available_players_timestamp:
        data = [
            (league_id, 'available_players_last_updated_date', date_str),
            (league_id, 'available_players_last_updated_timestamp', ts_str)
        ]
    else:
        data = [
            (league_id, 'last_updated_date', date_str),
            (league_id, 'last_updated_timestamp', ts_str)
        ]

    sql = """
        INSERT INTO db_metadata (league_id, key, value) VALUES (%s, %s, %s)
        ON CONFLICT (league_id, key) DO UPDATE SET value = EXCLUDED.value
    """
    cursor.executemany(sql, data)

def update_league_db(yq, lg, league_id, logger, capture_lineups=False, roster_updates_only=False):
    logger.info(f"Starting DB update for league {league_id}...")

    if yq is None:
        return {'success': False, 'error': 'yfpy (yq) object not initialized.'}

    league_metadata = yq.get_league_metadata()
    league_name = league_metadata.name.decode('utf-8', 'ignore') if isinstance(league_metadata.name, bytes) else league_metadata.name

    # Use centralized DB connection
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            _create_tables(cursor, logger)
            _update_db_metadata(cursor, league_id, logger)
            _update_league_info(yq, cursor, league_id, league_name, league_metadata, logger)

            if not roster_updates_only:
                _update_teams_info(yq, cursor, league_id, logger)
                playoff_week = _update_league_scoring_settings(yq, cursor, league_id, logger)
                _update_lineup_settings(yq, cursor, league_id, logger)
                _update_fantasy_weeks(yq, cursor, league_id, league_metadata.league_key, logger)
                _update_league_matchups(yq, cursor, league_id, playoff_week, logger)
                _update_league_transactions(yq, cursor, league_id, logger)
                _update_daily_lineups(yq, cursor, conn, league_id, league_metadata.num_teams, league_metadata.start_date, capture_lineups, logger)

            _update_current_rosters(yq, cursor, conn, league_id, league_metadata.num_teams, logger)
            _create_rosters_tall(cursor, conn, league_id, logger)

            if lg:
                _update_free_agents(lg, cursor, league_id, logger)
                _update_waivers(lg, cursor, league_id, logger)
                _update_rostered_players(lg, cursor, league_id, logger)
                _update_db_metadata(cursor, league_id, logger, update_available_players_timestamp=True)

            conn.commit()

            if not roster_updates_only:
                fin = DBFinalizer(league_id, logger, conn)
                fin.parse_and_store_player_stats()
                fin.parse_and_store_bench_stats()

    return {
        'success': True,
        'league_name': league_name,
        'timestamp': int(time.time())
    }
