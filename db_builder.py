"""
Queries to create and update fantasystreams.app db

Author: Jason Druckenmiller
Date: 10/17/2025
Updated: 11/25/2025
"""

import os
import sqlite3
import re
import logging
import time
import unicodedata
from datetime import date, timedelta, datetime
try:
    from zoneinfo import ZoneInfo
except ImportError:
    import pytz
    print("WARNING: zoneinfo not found. Falling back to pytz. Please `pip install pytz` if needed.")
import ast
from google.cloud import storage
import threading
import json
import tempfile
from pathlib import Path
from yfpy.query import YahooFantasySportsQuery
import yahoo_fantasy_api as yfa
from yahoo_oauth import OAuth2


db_build_status = {"running": False, "error": None, "current_build_id": None}
db_build_status_lock = threading.Lock()

def run_task(build_id, log_file_path, options, data):
    global db_build_status

    logger = logging.getLogger(f"db_build_{build_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    file_handler = None
    try:
        file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
        if not logger.handlers:
            logger.addHandler(file_handler)

        logger.info(f"Build task {build_id} received. Preparing API connections...")

    except Exception as e:
        logging.error(f"Failed to create FileHandler for build {build_id}: {e}")
        with db_build_status_lock:
            db_build_status = {"running": False, "error": str(e), "current_build_id": None}
        return

    yq = None
    lg = None

    try:
        if not data.get('dev_mode'):

            logger.info("Authenticating with Yahoo API (yfpy)...")

            #Create yfpy (yq) instance
            auth_data = {
                'consumer_key': data['consumer_key'],
                'consumer_secret': data['consumer_secret'],
                'access_token': data['token'].get('access_token'),
                'refresh_token': data['token'].get('refresh_token'),
                'token_type': data['token'].get('token_type', 'bearer'),
                'token_time': data['token'].get('expires_at', time.time() + 3600),
                'guid': data['token'].get('xoauth_yahoo_guid')
            }
            yq = YahooFantasySportsQuery(
                data['league_id'],
                game_code="nhl",
                yahoo_access_token_json=auth_data
            )

            logger.info("yfpy authentication successful.")
            logger.info("Authenticating with Yahoo API (yfa)...")

            #Create yfa (lg) instance
            creds = {
                "consumer_key": data['consumer_key'],
                "consumer_secret": data['consumer_secret'],
                "access_token": data['token'].get('access_token'),
                "refresh_token": data['token'].get('refresh_token'),
                "token_type": data['token'].get('token_type', 'bearer'),
                "token_time": data['token'].get('expires_at', time.time() + 3600),
                "xoauth_yahoo_guid": data['token'].get('xoauth_yahoo_guid')
            }
            temp_dir = os.path.join(tempfile.gettempdir(), 'temp_creds')
            os.makedirs(temp_dir, exist_ok=True)
            temp_file_path = os.path.join(temp_dir, f"thread_{build_id}.json")

            with open(temp_file_path, 'w') as f:
                json.dump(creds, f)

            sc = OAuth2(None, None, from_file=temp_file_path)
            if not sc.token_is_valid():
                logger.info("Thread token expired, refreshing... (This may take a moment)")
                sc.refresh_access_token()

            gm = yfa.Game(sc, 'nhl')
            lg = gm.to_league(f"nhl.l.{data['league_id']}")

            logger.info("yfa authentication successful.")

            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

        else:
            logger.info("Dev mode: Skipping real API object creation in thread.")
            yq = None
            lg = None

        logger.info("--- Starting Database Update ---")
        logger.info(f"League ID: {data['league_id']}")
        logger.info(f"Build ID: {build_id}")

        DATA_DIR = '/var/data/dbs'
        roster_updates_only = options.get('roster_updates_only', False)
        if roster_updates_only:
             logger.info("Mode: Roster Updates Only")
        else:
             logger.info(f"Mode: Standard Update (Capture Lineups: {options['capture_lineups']})")
        result = update_league_db(
            yq,
            lg,
            data['league_id'],
            DATA_DIR, # Pass the data directory
            logger, # Pass the new logger
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

        try:
            #Create a .done file as a sentinel ---
            done_file_path = f"{log_file_path}.done"
            Path(done_file_path).touch()
        except Exception as e:
            logger.error(f"Build task {build_id} couldn't create .done file: {e}")

        if file_handler:
            file_handler.close()
            logger.removeHandler(file_handler)

        logger.info(f"Build task {build_id} thread finished.")



# --- DB Finalizer Class (from finalize_db.py) ---
class DBFinalizer:
    """
    Processes and joins data in the fantasy hockey database after initial creation.
    """
    def __init__(self, db_path, logger, league_id):
        self.db_path = db_path
        self.logger = logger
        self.league_id = league_id
        self.con = self.get_db_connection()

    def get_db_connection(self):
        """Gets a connection to the SQLite database."""
        if not os.path.exists(self.db_path):
            self.logger.error(f"Database not found at {self.db_path}. Please provide a valid database file.")
            return None
        return sqlite3.connect(self.db_path)

    def close_connection(self):
        """Closes the database connection if it's open."""
        if self.con:
            self.con.close()
            self.logger.info("Finalizer database connection closed.")

    def import_player_ids(self, player_ids_db_path):
        """
        Attaches the player IDs database, drops the existing players table,
        and imports the new one in a single transaction.
        """
        if not self.con:
            self.logger.error("No database connection for finalizer.")
            return

        if not os.path.exists(player_ids_db_path):
            self.logger.error(f"Player IDs database not found at: {player_ids_db_path}")
            return

        absolute_player_ids_path = os.path.abspath(player_ids_db_path)
        self.logger.info(f"Found player IDs DB. Absolute path: {absolute_player_ids_path}")

        attached_successfully = False
        try:
            self.logger.info("Attaching player IDs database...")
            self.con.execute(f"ATTACH DATABASE '{absolute_player_ids_path}' AS player_ids_db")
            attached_successfully = True
            cursor = self.con.cursor()

            self.logger.info("Importing 'players' table from player_ids_db...")
            cursor.execute("DROP TABLE IF EXISTS main.players")
            cursor.execute("CREATE TABLE main.players AS SELECT * FROM player_ids_db.players")

            self.con.commit()
            self.logger.info("Successfully imported the 'players' table.")

        except sqlite3.Error as e:
            self.logger.error(f"An error occurred while importing player IDs: {e}")
            self.logger.info("Rolling back any pending changes.")
            self.con.rollback()
        finally:
            if attached_successfully:
                self.logger.info("Detaching player IDs database.")
                self.con.execute("DETACH DATABASE player_ids_db")


    def process_with_projections(self, projections_db_path):
        """
        Attaches the projections DB and runs all related processing functions
        (imports and joins) within a single transaction.
        """
        if not self.con:
            self.logger.error("No database connection for finalizer.")
            return

        if not os.path.exists(projections_db_path):
            self.logger.error(f"Projections database not found at: {projections_db_path}")
            return

        absolute_proj_path = os.path.abspath(projections_db_path)
        self.logger.info(f"Found projections DB. Absolute path: {absolute_proj_path}")

        attached_successfully = False
        try:
            self.logger.info(f"Attaching projections database...")
            self.con.execute(f"ATTACH DATABASE '{absolute_proj_path}' AS projections")
            attached_successfully = True
            cursor = self.con.cursor()

            self.logger.info("Importing static tables (off_days, schedule, team_schedules, team_standings,team_stats_summary, team_stats_weekly)...")
            tables_to_import = ['off_days', 'schedule', 'team_schedules', 'team_standings','team_stats_summary', 'team_stats_weekly']
            for table in tables_to_import:
                self.logger.info(f"Importing table: {table}")
                cursor.execute(f"DROP TABLE IF EXISTS main.{table}")
                cursor.execute(f"CREATE TABLE main.{table} AS SELECT * FROM projections.{table}")

            #Create joined_player_stats ---
            self.logger.info("Joining player data with projections (projections)...")
            cursor.execute("DROP TABLE IF EXISTS main.joined_player_stats")
            cursor.execute("""
                CREATE TABLE main.joined_player_stats AS
                SELECT
                    p.player_id,
                    p.player_name,
                    p.player_team,
                    CASE
                        WHEN fa.player_id IS NOT NULL THEN 'F'
                        WHEN w.player_id IS NOT NULL THEN 'W'
                        WHEN r.player_id IS NOT NULL THEN 'R'
                        ELSE 'Unk'
                    END AS availability_status,
                    proj.*
                FROM main.players AS p
                LEFT JOIN projections.projections AS proj
                ON p.player_name_normalized = proj.player_name_normalized
                LEFT JOIN main.free_agents AS fa
                ON p.player_id = fa.player_id
                LEFT JOIN main.waiver_players AS w
                ON p.player_id = w.player_id
                LEFT JOIN main.rostered_players AS r
                ON p.player_id = r.player_id;
            """)

            #Create joined_player_stats_real ---
            self.logger.info("Joining player data with to-date stats (stats_to_date)...")
            cursor.execute("DROP TABLE IF EXISTS main.joined_player_stats_real")
            cursor.execute("""
                CREATE TABLE main.joined_player_stats_real AS
                SELECT
                    p.player_id,
                    p.player_name,
                    p.player_team,
                    CASE
                        WHEN fa.player_id IS NOT NULL THEN 'F'
                        WHEN w.player_id IS NOT NULL THEN 'W'
                        WHEN r.player_id IS NOT NULL THEN 'R'
                        ELSE 'Unk'
                    END AS availability_status,
                    proj_real.*
                FROM main.players AS p
                LEFT JOIN projections.stats_to_date AS proj_real
                ON p.player_name_normalized = proj_real.player_name_normalized
                LEFT JOIN main.free_agents AS fa
                ON p.player_id = fa.player_id
                LEFT JOIN main.waiver_players AS w
                ON p.player_id = w.player_id
                LEFT JOIN main.rostered_players AS r
                ON p.player_id = r.player_id;
            """)

            #Create joined_player_stats_combined ---
            self.logger.info("Joining player data with combined projections (combined_projections)...")
            cursor.execute("DROP TABLE IF EXISTS main.joined_player_stats_combined")
            cursor.execute("""
                CREATE TABLE main.joined_player_stats_combined AS
                SELECT
                    p.player_id,
                    p.player_name,
                    p.player_team,
                    CASE
                        WHEN fa.player_id IS NOT NULL THEN 'F'
                        WHEN w.player_id IS NOT NULL THEN 'W'
                        WHEN r.player_id IS NOT NULL THEN 'R'
                        ELSE 'Unk'
                    END AS availability_status,
                    proj_comb.*
                FROM main.players AS p
                LEFT JOIN projections.combined_projections AS proj_comb
                ON p.player_name_normalized = proj_comb.player_name_normalized
                LEFT JOIN main.free_agents AS fa
                ON p.player_id = fa.player_id
                LEFT JOIN main.waiver_players AS w
                ON p.player_id = w.player_id
                LEFT JOIN main.rostered_players AS r
                ON p.player_id = r.player_id;
            """)

            self.con.commit()
            self.logger.info("Successfully imported static tables and joined all player projection tables.")

        except sqlite3.Error as e:
            self.logger.error(f"An error occurred while processing with projections DB: {e}")
            self.logger.info("Rolling back any pending changes.")
            self.con.rollback()
        finally:
            if attached_successfully:
                self.logger.info("Detaching projections database.")
                self.con.execute("DETACH DATABASE projections")

    def parse_and_store_player_stats(self):
        """
        Parses raw player data from 'daily_lineups_dump' for dates not already
        processed, enriches it, calculates missing goalie stats (TOI/G), and
        stores the structured stats in 'daily_player_stats'.
        """
        if not self.con:
            self.logger.error("No database connection for finalizer.")
            return

        cursor = self.con.cursor()

        # Check if the source table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='daily_lineups_dump'")
        if cursor.fetchone() is None:
            self.logger.info("Table 'daily_lineups_dump' does not exist. Skipping stat parsing.")
            return

        # Create the target table if it doesn't exist yet
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
                stat_value DOUBLE PRECISION,
                PRIMARY KEY (league_id, date_, player_id, stat_id)
            );
        """)
        self.con.commit() # Commit table creation if it happened

        # Find the last date already processed in daily_player_stats
        cursor.execute("SELECT MAX(date_) FROM daily_player_stats")
        max_processed_date_result = cursor.fetchone()
        last_processed_date = max_processed_date_result[0] if max_processed_date_result else None

        yesterday_iso = (date.today() - timedelta(days=1)).isoformat() # Get yesterday's date

        # Determine the query and parameters for fetching unprocessed data
        if last_processed_date:
            self.logger.info(f"Parsing daily stats: Resuming from date after {last_processed_date} AND re-processing {yesterday_iso}.")
            dump_query = "SELECT * FROM daily_lineups_dump WHERE date_ > ? OR date_ = ?"
            query_params = (last_processed_date, yesterday_iso)
        else:
            self.logger.info("Parsing daily stats: Processing all dates from dump table.")
            dump_query = "SELECT * FROM daily_lineups_dump"
            query_params = ()

        # Fetch only the necessary data from the dump table
        cursor.execute(dump_query, query_params)

        column_names = [description[0] for description in cursor.description]
        all_lineups = cursor.fetchall()

        if not all_lineups:
            self.logger.info("No new dates found in daily_lineups_dump to process for daily_player_stats.")
            return

        self.logger.info(f"Parsing raw player strings for {len(all_lineups)} new/updated rows...")

        stat_map = {
            1: 'G', 2: 'A', 3: 'P', 4: '+/-', 5: 'PIM', 6: 'PPG', 7: 'PPA', 8: 'PPP',
            9: 'SHG', 10: 'SHA', 11: 'SHP', 12: 'GWG', 13: 'GTG', 14: 'SOG', 15: 'SH%',
            16: 'FW', 17: 'FL', 31: 'HIT', 32: 'BLK', 18: 'GS', 19: 'W', 20: 'L',
            22: 'GA', 23: 'GAA', 24: 'SA', 25: 'SV', 26: 'SV%', 27: 'SHO', 28: 'TOI/G',
            29: 'GP/S', 30: 'GP/G', 33: 'TOI/S', 34: 'TOI/S/Gm'
        }

        cursor.execute("SELECT player_id, player_name_normalized FROM players")
        player_norm_name_map = dict(cursor.fetchall())
        self.logger.info(f"Loaded {len(player_norm_name_map)} players for name normalization lookup.")


        stats_to_insert = []
        player_string_pattern = re.compile(r"ID: (\d+), Name: .*, Stats: (\[.*\])")
        pos_pattern = re.compile(r"([a-zA-Z]+)")
        active_roster_columns = ['c1', 'c2', 'l1', 'l2', 'r1', 'r2', 'd1', 'd2', 'd3', 'd4', 'g1', 'g2']

        for row in all_lineups:
            # --- Ensure row_dict is created correctly ---
            try:
                row_dict = dict(zip(column_names, row))
                # Now safely access keys
                date_ = row_dict['date_']
                team_id = row_dict['team_id']
            except KeyError as e:
                self.logger.error(f"Missing key {e} when creating row_dict. Column names: {column_names}. Row data: {row}")
                continue # Skip this row if it doesn't match columns
            except Exception as e:
                self.logger.error(f"Error processing row {row} with columns {column_names}: {e}")
                continue # Skip this row on other errors


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

                            if (lineup_pos == 'g' and
                                22 in player_stats and 23 in player_stats):
                                val_22_ga = player_stats[22]
                                val_23_gaa = player_stats[23]
                                if val_23_gaa > 0:
                                    val_28_toi = (val_22_ga / val_23_gaa) * 60
                                    player_stats[28] = round(val_28_toi, 2)


                            for stat_id, stat_value in player_stats.items():
                                category = stat_map.get(stat_id, 'UNKNOWN')
                                stats_to_insert.append((
                                    self.league_id, date_, team_id, player_id,
                                    player_name_normalized, lineup_pos, stat_id,
                                    category, stat_value
                                ))
                        except (ValueError, SyntaxError) as e:
                            self.logger.warning(f"Could not parse stats for player {player_id} on {date_} in daily_player_stats: {e}")

        if stats_to_insert:
            self.logger.info(f"Found {len(stats_to_insert)} individual stat entries to insert/replace into daily_player_stats.")
            cursor.executemany("""
                INSERT INTO daily_player_stats (
                    league_id, date_, team_id, player_id, player_name_normalized, lineup_pos,
                    stat_id, category, stat_value
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (league_id, date_, player_id, stat_id)
                DO UPDATE SET stat_value = EXCLUDED.stat_value
            """, stats_to_insert)
            self.con.commit()
            self.logger.info("Successfully stored/replaced parsed player stats in daily_player_stats.")
        else:
            self.logger.info("No new player stats to insert into daily_player_stats.")


    def parse_and_store_bench_stats(self):
        """
        Parses raw player data from 'daily_lineups_dump' for dates not already
        processed, enriches it, calculates missing goalie stats (TOI/G) for bench players,
        and stores the structured stats in 'daily_bench_stats'.
        """
        if not self.con:
            # --- MODIFIED ---
            self.logger.error("No database connection for finalizer.")
            return

        cursor = self.con.cursor()

        # Check if the source table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='daily_lineups_dump'")
        if cursor.fetchone() is None:
            # --- MODIFIED ---
            self.logger.info("Table 'daily_lineups_dump' does not exist. Skipping bench stat parsing.")
            return

        # Create the target table if it doesn't exist yet
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_bench_stats (
                date_ TEXT NOT NULL,
                team_id INTEGER NOT NULL,
                player_id INTEGER NOT NULL,
                player_name_normalized TEXT,
                lineup_pos TEXT,
                stat_id INTEGER NOT NULL,
                category TEXT,
                stat_value REAL,
                PRIMARY KEY (date_, player_id, stat_id)
            );
        """)
        self.con.commit() # Commit table creation if it happened

        # --- OPTIMIZATION START / MODIFICATION ---
        # Find the last date already processed in daily_bench_stats
        cursor.execute("SELECT MAX(date_) FROM daily_bench_stats")
        max_processed_date_result = cursor.fetchone()
        last_processed_date = max_processed_date_result[0] if max_processed_date_result else None

        # --- MODIFICATION ---
        yesterday_iso = (date.today() - timedelta(days=1)).isoformat() # Get yesterday's date
        # --- END MODIFICATION ---

        # Determine the query and parameters for fetching unprocessed data
        if last_processed_date:
            # --- MODIFICATION ---
            self.logger.info(f"Parsing bench stats: Resuming from date after {last_processed_date} AND re-processing {yesterday_iso}.")
            dump_query = "SELECT * FROM daily_lineups_dump WHERE date_ > ? OR date_ = ?"
            query_params = (last_processed_date, yesterday_iso)
            # --- END MODIFICATION ---
        else:
            # --- MODIFIED ---
            self.logger.info("Parsing bench stats: Processing all dates from dump table.")
            dump_query = "SELECT * FROM daily_lineups_dump"
            query_params = ()

        # Fetch only the necessary data from the dump table
        cursor.execute(dump_query, query_params)

        # --- FIX: Get column names IMMEDIATELY after execute ---
        column_names = [description[0] for description in cursor.description]
        # --- END FIX ---

        all_lineups = cursor.fetchall()
        # --- OPTIMIZATION END ---


        if not all_lineups:
            # --- MODIFIED ---
            self.logger.info("No new dates found in daily_lineups_dump to process for daily_bench_stats.")
            return

        # --- MODIFIED ---
        self.logger.info(f"Parsing raw bench player strings for {len(all_lineups)} new/updated rows...")

        # (Rest of the function remains mostly the same)
        stat_map = {
            1: 'G', 2: 'A', 3: 'P', 4: '+/-', 5: 'PIM', 6: 'PPG', 7: 'PPA', 8: 'PPP',
            9: 'SHG', 10: 'SHA', 11: 'SHP', 12: 'GWG', 13: 'GTG', 14: 'SOG', 15: 'SH%',
            16: 'FW', 17: 'FL', 31: 'HIT', 32: 'BLK', 18: 'GS', 19: 'W', 20: 'L',
            22: 'GA', 23: 'GAA', 24: 'SA', 25: 'SV', 26: 'SV%', 27: 'SHO', 28: 'TOI/G',
            29: 'GP/S', 30: 'GP/G', 33: 'TOI/S', 34: 'TOI/S/Gm'
        }

        cursor.execute("SELECT player_id, player_name_normalized FROM players")
        player_norm_name_map = dict(cursor.fetchall())
        # --- MODIFIED ---
        self.logger.info(f"Loaded {len(player_norm_name_map)} players for name normalization lookup.")


        stats_to_insert = []
        player_string_pattern = re.compile(r"ID: (\d+), Name: .*, Stats: (\[.*\])")
        pos_pattern = re.compile(r"([a-zA-Z]+)")
        bench_roster_columns = ['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9',
                                'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17', 'b18', 'b19',
                                'i1', 'i2', 'i3', 'i4', 'i5']

        for row in all_lineups:
            # --- Ensure row_dict is created correctly ---
            try:
                row_dict = dict(zip(column_names, row))
                # Now safely access keys
                date_ = row_dict['date_']
                team_id = row_dict['team_id']
            except KeyError as e:
                # --- MODIFIED ---
                self.logger.error(f"Missing key {e} when creating row_dict for bench stats. Column names: {column_names}. Row data: {row}")
                continue # Skip this row
            except Exception as e:
                # --- MODIFIED ---
                self.logger.error(f"Error processing row {row} for bench stats with columns {column_names}: {e}")
                continue # Skip this row
            # --- End safety check ---


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
                                    date_, team_id, player_id, player_name_normalized,
                                    lineup_pos, stat_id, category, stat_value
                                ))
                        except (ValueError, SyntaxError) as e:
                            # --- MODIFIED ---
                                self.logger.warning(f"Could not parse stats for player {player_id} on {date_} in daily_bench_stats: {e}")


        if stats_to_insert:
            # --- MODIFIED ---
            self.logger.info(f"Found {len(stats_to_insert)} individual bench stat entries to insert/replace into daily_bench_stats.")
            # --- MODIFICATION: Use INSERT OR REPLACE ---
            cursor.executemany("""
                INSERT OR REPLACE INTO daily_bench_stats (
                    date_, team_id, player_id, player_name_normalized, lineup_pos,
                    stat_id, category, stat_value
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, stats_to_insert)
            self.con.commit()
            # --- MODIFIED ---
            self.logger.info("Successfully stored/replaced parsed bench player stats in daily_bench_stats.")
        else:
            # --- MODIFIED ---
            self.logger.info("No new bench player stats to insert into daily_bench_stats.")

# --- MODIFIED: Accept logger ---
def _create_tables(cursor, logger):
    """
    Creates all necessary tables in the database if they don't already exist.
    """
    # --- MODIFIED ---
    logger.info("Creating database tables if they don't exist...")

    #league_info
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS league_info (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    ''')
    #teams
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS teams (
            team_id TEXT PRIMARY KEY,
            name TEXT,
            manager_nickname TEXT
        )
    ''')

    #daily_lineups_dump
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_lineups_dump (
            date_ TEXT NOT NULL,
            team_id INTEGER NOT NULL,
            c1 TEXT, c2 TEXT, l1 TEXT, l2 TEXT, r1 TEXT, r2 TEXT,
            d1 TEXT, d2 TEXT, d3 TEXT, d4 TEXT, g1 TEXT, g2 TEXT,
            b1 TEXT, b2 TEXT, b3 TEXT, b4 TEXT, b5 TEXT, b6 TEXT,
            b7 TEXT, b8 TEXT, b9 TEXT, b10 TEXT, b11 TEXT, b12 TEXT,
            b13 TEXT, b14 TEXT, b15 TEXT, b16 TEXT, b17 TEXT, b18 TEXT, b19 TEXT,
            i1 TEXT, i2 TEXT, i3 TEXT, i4 TEXT, i5 TEXT,
            PRIMARY KEY (date_, team_id)
        )
    ''')
    #scoring
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS scoring (
            stat_id INTEGER NOT NULL UNIQUE,
            category TEXT NOT NULL,
            scoring_group TEXT NOT NULL
        )
    ''')
    #lineup settings
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS lineup_settings (
            position_id INTEGER PRIMARY KEY AUTOINCREMENT,
            position TEXT NOT NULL,
            position_count INTEGER NOT NULL
        )
    ''')
    #weeks
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS weeks (
            week_num INTEGER NOT NULL UNIQUE,
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL
        )
    ''')
    #matchups
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS matchups (
            week INTEGER NOT NULL,
            team1 TEXT NOT NULL,
            team2 TEXT NOT NULL
        )
    ''')
    #rosters
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS rosters (
            team_id INTEGER NOT NULL UNIQUE,
            p1 INTEGER, p2 INTEGER, p3 INTEGER, p4 INTEGER, p5 INTEGER,
            p6 INTEGER, p7 INTEGER, p8 INTEGER, p9 INTEGER, p10 INTEGER,
            p11 INTEGER, p12 INTEGER, p13 INTEGER, p14 INTEGER, p15 INTEGER,
            p16 INTEGER, p17 INTEGER, p18 INTEGER, p19 INTEGER, p20 INTEGER,
            p21 INTEGER, p22 INTEGER, p23 INTEGER, p24 INTEGER, p25 INTEGER,
            p26 INTEGER, p27 INTEGER, p28 INTEGER, p29 INTEGER
        )
    ''')
    #free_agents
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS free_agents (
            player_id TEXT PRIMARY KEY,
            status TEXT
        )
    ''')
    #waiver_players
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS waiver_players (
            player_id TEXT PRIMARY KEY,
            status TEXT
        )
    ''')
    #rostered_players
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS rostered_players (
            player_id TEXT PRIMARY KEY,
            status TEXT,
            eligible_positions TEXT
        )
    ''')
    #db_metadata
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS db_metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    ''')
    #transactions
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            transaction_date TEXT NOT NULL,
            player_id TEXT NOT NULL,
            player_name TEXT NOT NULL,
            fantasy_team TEXT,
            move_type TEXT
        )
    ''')

# --- MODIFIED: Accept logger ---
def _update_league_info(yq, cursor, league_id, league_name, league_metadata, logger):
    """
    Fetches league metadata and updates the league_info table.
    """
    # --- MODIFIED ---
    logger.info("Updating league_info table...")
    num_teams = league_metadata.num_teams
    start_date = league_metadata.start_date
    end_date = league_metadata.end_date

    cursor.execute("INSERT OR REPLACE INTO league_info (key, value) VALUES (?, ?)",
                   ('league_id', league_id))
    cursor.execute("INSERT OR REPLACE INTO league_info (key, value) VALUES (?, ?)",
                   ('league_name', league_name))
    cursor.execute("INSERT OR REPLACE INTO league_info (key, value) VALUES (?, ?)",
                   ('num_teams', num_teams))
    cursor.execute("INSERT OR REPLACE INTO league_info (key, value) VALUES (?, ?)",
                   ('start_date', start_date))
    cursor.execute("INSERT OR REPLACE INTO league_info (key, value) VALUES (?, ?)",
                   ('end_date', end_date))


# --- MODIFIED: Accept logger ---
def _update_teams_info(yq, cursor, logger):
    """
    Fetches team data and updates the teams table.
    """
    # --- MODIFIED ---
    logger.info("Updating teams table...")
    try:
        teams = yq.get_league_teams()
        teams_data_to_insert = []
        for team in teams:
            team_id = team.team_id
            team_name = team.name
            manager_nickname = None
            if team.managers and team.managers[0].nickname:
                manager_nickname = team.managers[0].nickname
            teams_data_to_insert.append((team_id, team_name, manager_nickname))

        sql = "INSERT OR REPLACE INTO teams (team_id, name, manager_nickname) VALUES (?, ?, ?)"
        cursor.executemany(sql, teams_data_to_insert)
        # --- MODIFIED ---
        logger.info(f"Successfully inserted or ignored data for {len(teams_data_to_insert)} teams.")
    except Exception as e:
        # --- MODIFIED ---
        logger.error(f"Failed to update teams info: {e}", exc_info=True)


# --- MODIFIED: Accept logger ---
def _get_current_week_start_date(cursor, logger):
    """Gets the start date of the current fantasy week."""
    try:
        today = date.today().isoformat()
        cursor.execute(
            "SELECT start_date FROM weeks WHERE start_date <= ? AND end_date >= ?",
            (today, today)
        )
        result = cursor.fetchone()
        if result:
            # --- MODIFIED ---
            logger.info(f"Current week start date found: {result[0]}")
            return result[0]
        else:
            # --- MODIFIED ---
            logger.warning("Could not find a fantasy week for today's date. Defaulting to today.")
            return today
    except Exception as e:
        # --- MODIFIED ---
        logger.error(f"Could not determine current week start date from DB: {e}")
        return date.today().isoformat()


# --- MODIFIED: Accept logger ---
def _update_daily_lineups(yq, cursor, conn, num_teams, league_start_date, is_full_mode, logger):
    """
    Iterates through a teams lineup for every day of the season and writes
    player name, and stats to their assigned position in the daily lineup
    table. Repeats for each team in the league.
    """
    try:
        cursor.execute("SELECT MAX(date_) FROM daily_lineups_dump")
        last_fetch_date_str = cursor.fetchone()[0]

        # --- MODIFICATION ---
        today_obj = date.today()
        today_iso = today_obj.isoformat()
        yesterday_obj = today_obj - timedelta(days=1)
        yesterday_iso = yesterday_obj.isoformat()
        # --- END MODIFICATION ---

        last_fetch_date_plus_one = None
        start_date_override = None # New variable to force re-fetch

        if last_fetch_date_str:
            last_fetch_date = date.fromisoformat(last_fetch_date_str)

            # --- MODIFICATION: If last fetch was yesterday (or today), re-fetch YESTERDAY ---
            if last_fetch_date_str >= yesterday_iso:
                logger.info(f"Last fetch was {last_fetch_date_str}. Setting start date to re-fetch yesterday ({yesterday_iso}).")
                start_date_override = yesterday_iso
            # --- END MODIFICATION ---
            else:
                last_fetch_date_plus_one = (last_fetch_date + timedelta(days=1)).isoformat()

        if is_full_mode:
            start_date_for_fetch = league_start_date
            if start_date_override:
                start_date_for_fetch = start_date_override
            elif last_fetch_date_plus_one:
                    start_date_for_fetch = last_fetch_date_plus_one
            if last_fetch_date_str and not start_date_override: # Don't log resume if we're overriding
                # --- MODIFIED ---
                logger.info(f"Capture Daily Lineups is CHECKED. Resuming full history fetch from {start_date_for_fetch}.")
            else:
                # --- MODIFIED ---
                logger.info(f"Capture Daily Lineups is CHECKED. Starting full history fetch from league start date: {start_date_for_fetch}.")
        else:
            # --- MODIFIED: Pass logger ---
            current_week_start_date_str = _get_current_week_start_date(cursor, logger)
            current_week_start_obj = date.fromisoformat(current_week_start_date_str)
            start_of_week_minus_one_obj = current_week_start_obj - timedelta(days=1)
            start_of_week_minus_one_str = start_of_week_minus_one_obj.isoformat()

            if start_date_override:
                start_date_for_fetch = start_date_override
                # --- MODIFICATION ---
                logger.info(f"Capture Daily Lineups is UNCHECKED. Re-fetching yesterday's date: {start_date_for_fetch}.")
                # --- END MODIFICATION ---
            elif last_fetch_date_plus_one:
                start_date_for_fetch = max(start_of_week_minus_one_str, last_fetch_date_plus_one)
                # --- MODIFIED ---
                logger.info(f"Capture Daily Lineups is UNCHECKED. Resuming from more recent of week start-1 ({start_of_week_minus_one_str}) or last fetch+1 ({last_fetch_date_plus_one}): {start_date_for_fetch}.")
            else:
                start_date_for_fetch = start_of_week_minus_one_str
                # --- MODIFIED ---
                logger.info(f"No existing lineup data. Capture is UNCHECKED, starting from current week start date - 1 day: {start_date_for_fetch}.")


        team_id = 1
        # --- MODIFICATION: stop_date is TODAY, so loop runs up to (but not including) today ---
        stop_date = today_iso
        # --- END MODIFICATION ---
        lineup_data_to_insert = []

        if start_date_for_fetch >= stop_date:
            # --- MODIFIED ---
            logger.info(f"Daily lineups are already up to date (Start: {start_date_for_fetch}, Stop: {stop_date}). Nothing to fetch.")
            return

        while team_id <= num_teams:
            current_date = start_date_for_fetch
            # --- MODIFICATION: Loop condition will now run up to, but NOT including, today ---
            while current_date < stop_date:
                # --- MODIFIED ---
                logger.info(f"Fetching daily lineups for team {team_id}, for {current_date}...")
                players = yq.get_team_roster_player_info_by_date(team_id,current_date)
                c, lw, rw, d, g, bn, ir = 0, 0, 0, 0, 0, 0, 0
                lineup_data_raw = []
                for player in players:
                    player_id = player.player_id
                    player_name = player.name.full
                    pos = player.selected_position.position
                    if pos == "C": pos, c = 'c'+str(c+1), c+1
                    elif pos == "LW": pos, lw = 'l'+str(lw+1), lw+1
                    elif pos == "RW": pos, rw = 'r'+str(rw+1), rw+1
                    elif pos == "D": pos, d = 'd'+str(d+1), d+1
                    elif pos == "G": pos, g = 'g'+str(g+1), g+1
                    elif pos == "BN": pos, bn = 'b'+str(bn+1), bn+1
                    elif pos == "IR" or pos == "IR+": pos, ir = 'i'+str(ir+1), ir+1

                    player_stats = []
                    if player.player_stats and player.player_stats.stats:
                        stats_list = player.player_stats.stats
                        stats_dict = {stat_item.stat_id: stat_item.value for stat_item in stats_list}
                        for stat_id, stat_value in stats_dict.items():
                            player_stats.append((stat_id, stat_value))

                    player_data_string = f"ID: {player_id}, Name: {player_name}, Stats: {str(player_stats)}"
                    lineup_data_raw.append((player_data_string, pos))

                lineup_raw_dict = {position: data_string for data_string, position in lineup_data_raw}
                lineup_order = [
                    'c1', 'c2', 'l1', 'l2', 'r1', 'r2', 'd1', 'd2', 'd3', 'd4',
                    'g1', 'g2', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6',
                    'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14',
                    'b15', 'b16', 'b17', 'b18', 'b19', 'i1', 'i2', 'i3', 'i4', 'i5'
                ]
                lineup_data_values = [lineup_raw_dict.get(pos, None) for pos in lineup_order]
                full_row = [current_date, team_id] + lineup_data_values
                lineup_data_to_insert.append(tuple(full_row))
                current_date = (date.fromisoformat(current_date)+timedelta(1)).isoformat()
            team_id += 1

        if not lineup_data_to_insert:
            # --- MODIFIED ---
            logger.info("No new daily lineups to insert for the specified date range.")
            return

        placeholders = ', '.join(['?'] * 38)
        sql = f"""
            INSERT OR REPLACE INTO daily_lineups_dump (
                date_, team_id, c1, c2, l1, l2, r1, r2, d1, d2, d3, d4, g1, g2,
                b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15,
                b16, b17, b18, b19, i1, i2, i3, i4, i5
            ) VALUES ({placeholders})
        """
        cursor.executemany(sql, lineup_data_to_insert)
        # --- MODIFIED ---
        logger.info(f"Successfully inserted or replaced data for {len(lineup_data_to_insert)} dates.")
    except Exception as e:
        # --- MODIFIED ---
        logger.error(f"Failed to update lineup info: {e}", exc_info=True)

# --- MODIFIED: Accept logger ---
def _update_player_id(yq, cursor, logger):
    """
    ***CURRENTLY DISABLED AS STATIC FILE HANDLES THIS DATA***
    """
    # --- MODIFIED ---
    logger.info("Fetching all league players (this may take a while)...")
    try:
        TEAM_TRICODE_MAP = {"TB": "TBL", "NJ": "NJD", "SJ": "SJS", "LA": "LAK", "MON": "MTL", "WAS": "WSH"}
        player_data_to_insert = []
        batch_size = 100
        player_count = 0
        sql = "INSERT OR IGNORE INTO players (player_id, player_name, player_team, player_name_normalized) VALUES (?, ?, ?, ?)"

        for player in yq.get_league_players():
            player_count += 1
            player_name = player.name.full
            nfkd_form = unicodedata.normalize('NFKD', player_name.lower())
            ascii_name = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
            player_name_normalized = re.sub(r'[^a-z0-9]', '', ascii_name)
            player_team_abbr = player.editorial_team_abbr.upper()
            player_team = TEAM_TRICODE_MAP.get(player_team_abbr, player_team_abbr)
            player_data_to_insert.append((player.player_id, player_name, player_team, player_name_normalized))

            if len(player_data_to_insert) >= batch_size:
                # --- MODIFIED ---
                logger.info(f"Processed {player_count} players, inserting batch of {len(player_data_to_insert)}...")
                cursor.executemany(sql, player_data_to_insert)
                player_data_to_insert = []

        if player_data_to_insert:
            # --- MODIFIED ---
            logger.info(f"Inserting final batch of {len(player_data_to_insert)} players...")
            cursor.executemany(sql, player_data_to_insert)

        # --- MODIFIED ---
        logger.info(f"Successfully processed and inserted data for a total of {player_count} players.")

    except Exception as e:
        # --- MODIFIED ---
        logger.error(f"Failed to update player info: {e}", exc_info=True)


# --- MODIFIED: Accept logger ---
def _update_league_scoring_settings(yq, cursor, logger):
    """
    Writes the leagues scoring settings, and lineup settings
    """
    # --- MODIFIED ---
    logger.info("Fetching league scoring...")
    try:
        settings = yq.get_league_settings()
        playoff_start_week = settings.playoff_start_week
        scoring_settings_to_insert = []
        for stat_item in settings.stat_categories.stats:
            stat_details = stat_item
            category = stat_details.display_name
            if category == 'SV%':
                category = 'SVpct'
            scoring_group = stat_details.group
            stat_id = stat_details.stat_id
            scoring_settings_to_insert.append((stat_id, category, scoring_group))

        sql = "INSERT OR IGNORE INTO scoring (stat_id, category, scoring_group) VALUES (?, ?, ?)"
        cursor.executemany(sql, scoring_settings_to_insert)
        # --- MODIFIED ---
        logger.info(f"Successfully inserted or ignored data for {len(scoring_settings_to_insert)} categories.")
        return playoff_start_week
    except Exception as e:
        # --- MODIFIED ---
        logger.error(f"Failed to update scoring info: {e}", exc_info=True)
        return None


# --- MODIFIED: Accept logger ---
def _update_lineup_settings(yq, cursor, logger):
    try:
        settings = yq.get_league_settings()
        lineup_settings_data_to_insert = []
        for roster_position_item in settings.roster_positions:
            position_details = roster_position_item
            position = position_details.position
            position_count = position_details.count
            lineup_settings_data_to_insert.append((position, position_count))

        sql = "INSERT OR IGNORE INTO lineup_settings (position, position_count) VALUES (?, ?)"
        cursor.executemany(sql, lineup_settings_data_to_insert)
        # --- MODIFIED ---
        logger.info(f"Successfully inserted or ignored data for {len(lineup_settings_data_to_insert)} lineup positions.")
    except Exception as e:
        # --- MODIFIED ---
        logger.error(f"Failed to update lineup settings info: {e}", exc_info=True)

# --- MODIFIED: Accept logger ---
def _update_fantasy_weeks(yq, cursor, league_key, logger):
    """
    Fetches the weekly struture for the league
    """
    # --- MODIFIED ---
    logger.info("Fetching fantasy weeks...")
    try:
        game_id_end_pos = league_key.index('.')
        game_id = league_key[:game_id_end_pos]
        weeks = yq.get_game_weeks_by_game_id(game_id)

        weeks_to_insert = []
        for gameweek in weeks:
            week_num = gameweek.week
            start_date = gameweek.start
            end_date = gameweek.end
            weeks_to_insert.append((week_num, start_date, end_date))

        sql = "INSERT OR IGNORE INTO weeks (week_num, start_date, end_date) VALUES (?, ?, ?)"
        cursor.executemany(sql, weeks_to_insert)
        # --- MODIFIED ---
        logger.info(f"Successfully inserted or ignored data for {len(weeks_to_insert)} weeks.")
    except Exception as e:
        # --- MODIFIED ---
        logger.error(f"Failed to update week info: {e}", exc_info=True)


# --- MODIFIED: Accept logger ---
def _update_league_matchups(yq, cursor, playoff_start_week, logger):
    """
    Writes the leagues matchups
    """
    # --- MODIFIED ---
    logger.info("Fetching league matchups...")
    try:
        if not playoff_start_week:
            # --- MODIFIED ---
            logger.error("Cannot fetch matchups without playoff_start_week.")
            return

        last_reg_season_week = playoff_start_week-1
        start_week = 1
        matchup_data_to_insert = []

        while start_week <= last_reg_season_week:
            matchups = yq.get_league_matchups_by_week(start_week)
            for matchup in matchups:
                matchups_for_week = []
                for team_item in matchup.teams:
                    team_block = team_item
                    team_name = team_block.name
                    matchups_for_week.append(team_name)
                matchups_for_week.insert(0,start_week)
                matchup_data_to_insert.append(matchups_for_week)
            start_week += 1

        sql = "INSERT OR IGNORE INTO matchups (week, team1, team2) VALUES (?, ?, ?)"
        cursor.executemany(sql, matchup_data_to_insert)
        # --- MODIFIED ---
        logger.info(f"Successfully inserted or ignored data for {len(matchup_data_to_insert)} matchups.")
    except Exception as e:
        # --- MODIFIED ---
        logger.error(f"Failed to update matchup info: {e}", exc_info=True)


# --- MODIFIED: Accept logger ---
def _update_current_rosters(yq, cursor, conn, num_teams, logger):
    """
    Writes each team's current roster to the database.
    """
    # --- MODIFIED ---
    logger.info("Fetching current roster info...")
    try:
        # --- MODIFIED ---
        logger.info("Clearing existing data from rosters table.")
        cursor.execute("DELETE FROM rosters")
        conn.commit()
    except Exception as e:
        # --- MODIFIED ---
        logger.error("Failed to clear rosters table.", exc_info=True)
        conn.rollback()

    try:
        roster_data_to_insert = []
        MAX_PLAYERS = 29

        for team_id in range(1, num_teams + 1):
            players = yq.get_team_roster_player_info_by_date(team_id, date.today().isoformat())
            player_ids = [player.player_id for player in players][:MAX_PLAYERS]
            padded_player_ids = player_ids + [None] * (MAX_PLAYERS - len(player_ids))
            row_data = [team_id] + padded_player_ids
            roster_data_to_insert.append(row_data)

        placeholders = ', '.join(['?'] * (MAX_PLAYERS + 1))
        cols = ", ".join([f"p{i}" for i in range(1, MAX_PLAYERS + 1)])
        sql = f"""INSERT INTO rosters (
                    team_id, {cols})
                    VALUES ({placeholders})
        """
        cursor.executemany(sql, roster_data_to_insert)
        conn.commit()
        # --- MODIFIED ---
        logger.info(f"Successfully inserted data for {len(roster_data_to_insert)} teams.")
    except Exception as e:
        # --- MODIFIED ---
        logger.error(f"Failed to update roster info: {e}", exc_info=True)

# --- MODIFIED: Accept logger ---
def _create_rosters_tall_and_drop_rosters(cursor, conn, logger):
    """
    Creates a new 'rosters_tall' table from the wide 'rosters' table,
    and then drops the original 'rosters' table.
    """
    # --- MODIFIED ---
    logger.info("Creating tall rosters table and dropping the wide version...")
    try:
        cursor.execute("DROP TABLE IF EXISTS rosters_tall;")

        union_all_parts = []
        for i in range(1, 30):
            union_all_parts.append(
                f"SELECT team_id, p{i} AS player_id FROM rosters WHERE p{i} IS NOT NULL"
            )
        unpivot_query = "\nUNION ALL\n".join(union_all_parts)

        create_tall_table_query = f"CREATE TABLE rosters_tall AS\n{unpivot_query};"
        cursor.execute(create_tall_table_query)
        # --- MODIFIED ---
        logger.info("Successfully created 'rosters_tall' table.")

        cursor.execute("DROP TABLE IF EXISTS rosters;")
        # --- MODIFIED ---
        logger.info("Successfully dropped 'rosters' table.")

        conn.commit()
    except Exception as e:
        # --- MODIFIED ---
        logger.error(f"Failed during tall roster creation: {e}", exc_info=True)
        conn.rollback()


# --- MODIFIED: Accept logger ---
def _update_league_transactions(yq, cursor, logger):
    """
    Writes player name, fantasy team id, add or drop, and yahoo id to transactions
    table for all transactions in the league
    """
    try:
        # --- MODIFIED ---
        logger.info("Clearing existing transaction data...")
        cursor.execute("DELETE FROM transactions")
    except Exception as e:
        # --- MODIFIED ---
        logger.error(f"Failed to clear transactions table: {e}", exc_info=True)
        return

    # --- MODIFIED ---
    logger.info("Fetching player info...")

    # --- TIMEZONE CORRECTION START ---
    # Define the UTC timezone and your League's timezone.
    # 'America/New_York' (Eastern) fits your 11:28 PM example.
    # Use 'America/Los_Angeles' if you are sure it's Pacific.
    try:
        # Python 3.9+
        from zoneinfo import ZoneInfo
        UTC_TZ = ZoneInfo("UTC")
        LEAGUE_TZ = ZoneInfo("America/New_York")
    except ImportError:
        # Fallback for Python < 3.9 (requires `pip install pytz`)
        import pytz
        UTC_TZ = pytz.utc
        LEAGUE_TZ = pytz.timezone("America/New_York")
    # --- TIMEZONE CORRECTION END ---

    try:
        transactions = yq.get_league_transactions()
        transaction_data_to_insert = []
        processed_transactions = set()

        for transaction in transactions:
            if transaction.status == 'successful':
                timestamp_epoch = transaction.timestamp

                # --- TIMEZONE CORRECTION START ---
                # 1. Create a datetime object from epoch, aware it's in UTC
                utc_time = datetime.fromtimestamp(timestamp_epoch, tz=UTC_TZ)

                # 2. Convert the UTC time to the league's local timezone
                local_time = utc_time.astimezone(LEAGUE_TZ)

                # 3. Extract the date *from the local time*
                transaction_date = local_time.strftime('%Y-%m-%d')
                # --- TIMEZONE CORRECTION END ---

                for player_obj in transaction.players:
                    player_id = player_obj.player_id
                    player_name = player_obj.name.full
                    move_type = player_obj.transaction_data.type

                    if move_type == 'add':
                        fantasy_team = player_obj.transaction_data.destination_team_name
                    else:
                        fantasy_team = player_obj.transaction_data.source_team_name

                    unique_key = (timestamp_epoch, player_id, move_type)
                    if unique_key not in processed_transactions:
                        transaction_data_to_insert.append((transaction_date, player_id, player_name, fantasy_team, move_type))
                        processed_transactions.add(unique_key)

        sql = "INSERT INTO transactions (transaction_date, player_id, player_name, fantasy_team, move_type) VALUES (?, ?, ?, ?, ?)"
        cursor.executemany(sql, transaction_data_to_insert)
        # --- MODIFIED ---
        logger.info(f"Successfully inserted data for {len(transaction_data_to_insert)} transactions.")

    except Exception as e:
        # --- MODIFIED ---
        logger.error(f"Failed to update transaction info: {e}", exc_info=True)


# --- MODIFIED: Accept logger ---
def _update_free_agents(lg, conn, logger):
    """
    Writes all current free agents to free agent table
    """
    # --- MODIFIED ---
    logger.info("Fetching free agent info...")
    cursor = conn.cursor()
    try:
        # --- MODIFIED ---
        logger.info("Clearing existing data from free_agents table.")
        cursor.execute("DELETE FROM free_agents")
        conn.commit()
    except Exception as e:
        # --- MODIFIED ---
        logger.error("Failed to clear free_agents table.", exc_info=True)
        conn.rollback()
        return

    free_agents_to_insert = []
    for pos in ['C', 'LW', 'RW', 'D', 'G']:
        try:
            # --- MODIFIED ---
            logger.info(f"Fetching free agents for position: {pos}")
            fas = lg.free_agents(pos)
            for player in fas:
                player_id = player['player_id']
                free_agents_to_insert.append((player_id, 'FA'))
        except Exception as e:
            # --- MODIFIED ---
            logger.error(f"Could not fetch FAs for position {pos}: {e}")

    sql = "INSERT OR IGNORE INTO free_agents (player_id, status) VALUES (?, ?)"
    cursor.executemany(sql, free_agents_to_insert)
    conn.commit()
    # --- MODIFIED ---
    logger.info(f"Successfully inserted data for {len(free_agents_to_insert)} free agents.")


# --- MODIFIED: Accept logger ---
def _update_waivers(lg, conn, logger):
    """
    Writes all current waiver players to waiver_players table
    """
    # --- MODIFIED ---
    logger.info("Fetching waiver player info...")
    cursor = conn.cursor()
    try:
        # --- MODIFIED ---
        logger.info("Clearing existing data from waiver_players table.")
        cursor.execute("DELETE FROM waiver_players")
        conn.commit()
    except Exception as e:
        # --- MODIFIED ---
        logger.error("Failed to clear waiver_players table.", exc_info=True)
        conn.rollback()
        return

    waiver_players_to_insert = []
    try:
        # --- MODIFIED ---
        logger.info(f"Fetching all waiver players")
        wvp = lg.waivers()
        for player in wvp:
            player_id = player['player_id']
            waiver_players_to_insert.append((player_id, 'W'))
    except Exception as e:
        # --- MODIFIED ---
        logger.error(f"Could not fetch waiver players: {e}")

    sql = "INSERT OR IGNORE INTO waiver_players (player_id, status) VALUES (?, ?)"
    cursor.executemany(sql, waiver_players_to_insert)
    conn.commit()
    # --- MODIFIED ---
    logger.info(f"Successfully inserted data for {len(waiver_players_to_insert)} waiver players.")


# --- MODIFIED: Accept logger ---
def _update_rostered_players(lg, conn, logger):
    """
    Writes all currently rostered players to rostered_players table
    """
    # --- MODIFIED ---
    logger.info("Fetching rostered player info...")
    cursor = conn.cursor()
    try:
        # --- MODIFIED ---
        logger.info("Clearing existing data from rostered_players table.")
        cursor.execute("DELETE FROM rostered_players")
        conn.commit()
    except Exception as e:
        # --- MODIFIED ---
        logger.error("Failed to clear rostered_players table.", exc_info=True)
        conn.rollback()
        return

    rostered_players_to_insert = []
    try:
        # --- MODIFIED ---
        logger.info("Fetching all rostered players")
        tkp = lg.taken_players()
        for player in tkp:
            player_id = player['player_id']
            eligible_positions_list = player['eligible_positions']
            eligible_positions_str = ','.join(eligible_positions_list)
            rostered_players_to_insert.append((player_id, 'R', eligible_positions_str))
    except Exception as e:
        # --- MODIFIED ---
        logger.error(f"Could not fetch rostered players: {e}", exc_info=True)
        return

    if not rostered_players_to_insert:
        # --- MODIFIED ---
        logger.warning("No rostered players found to insert.")
        return

    try:
        sql = "INSERT OR IGNORE INTO rostered_players (player_id, status, eligible_positions) VALUES (?, ?, ?)"
        cursor.executemany(sql, rostered_players_to_insert)
        conn.commit()
        # --- MODIFIED ---
        logger.info(f"Successfully inserted data for {len(rostered_players_to_insert)} rostered players.")
    except Exception as e:
        # --- MODIFIED ---
        logger.error("Failed to insert rostered players into the database.", exc_info=True)
        conn.rollback()


# --- MODIFIED: Accept logger ---
def _update_db_metadata(cursor, logger, update_available_players_timestamp=False):
    """
    Updates the db_metadata table with the current timestamp.
    """
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")

    if update_available_players_timestamp:
        # --- MODIFIED ---
        logger.info("Updating available players timestamp in db_metadata...")
        cursor.execute("INSERT OR REPLACE INTO db_metadata (key, value) VALUES (?, ?)",
                       ('available_players_last_updated_date', date_str))
        cursor.execute("INSERT OR REPLACE INTO db_metadata (key, value) VALUES (?, ?)",
                       ('available_players_last_updated_timestamp', timestamp_str))
        # --- MODIFIED ---
        logger.info("Successfully updated available players timestamp.")
    else:
        # --- MODIFIED ---
        logger.info("Updating general db_metadata timestamp...")
        cursor.execute("INSERT OR REPLACE INTO db_metadata (key, value) VALUES (?, ?)",
                       ('last_updated_date', date_str))
        cursor.execute("INSERT OR REPLACE INTO db_metadata (key, value) VALUES (?, ?)",
                       ('last_updated_timestamp', timestamp_str))
        # --- MODIFIED ---
        logger.info("Successfully updated general db_metadata timestamp.")


# --- MODIFIED: Accept logger ---
def update_league_db(yq, lg, league_id, data_dir, logger, capture_lineups=False, roster_updates_only=False):
    """
    Creates or updates the league-specific SQLite database by calling
    individual query and update functions.
    """
    try:
        # --- MODIFIED ---
        logger.info(f"Starting DB update for league {league_id}...")
        logger.info("Fetching league metadata to determine filename...")

        if yq is None:
            logger.error("Yahoo Query object (yq) is None. Cannot proceed.")
            return {'success': False, 'error': 'yfpy (yq) object not initialized. Check dev mode.'}

        league_metadata = yq.get_league_metadata()
        league_name_str = league_metadata.name
        if isinstance(league_name_str, bytes):
            league_name_str = league_name_str.decode('utf-8', 'ignore')

        sanitized_name = re.sub(r'[\\/*?:"<>|]', "", league_name_str)
        db_filename = f"yahoo-{league_id}-{sanitized_name}.db"
        db_path = os.path.join(data_dir, db_filename)
        logger.info(f"Database will be built at: {db_path}")
        if not os.path.exists(db_path):
            logger.warning(f"No database found at {db_path}.")
            logger.warning(">>> FORCING FULL INITIALIZATION <<<")
            capture_lineups = True       # Force Full History
            roster_updates_only = False  # Cannot do partial on missing DB

        logger.info(f"Connecting to database: {db_path}")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        if roster_updates_only:
            logger.info(">>> ROSTER UPDATES ONLY MODE ENABLED <<<")
            logger.info("Skipping league settings, matchups, and historical stats.")

            # Create tables if they don't exist (safe to run)
            _create_tables(cursor, logger)
            _update_db_metadata(cursor, logger)

            # Core Roster Functions
            _update_current_rosters(yq, cursor, conn, league_metadata.num_teams, logger)
            _create_rosters_tall_and_drop_rosters(cursor, conn, logger)

            # Note: We skip _update_daily_lineups, teams info, weeks, etc.
            conn.commit()

        else:
            logger.info(f">>> STANDARD UPDATE MODE (Full History: {capture_lineups}) <<<")
            # --- FULL / PARTIAL MODE (Standard) ---
            _create_tables(cursor, logger)
            _update_db_metadata(cursor, logger)

            _update_league_info(yq, cursor, league_id, sanitized_name, league_metadata, logger)
            _update_teams_info(yq, cursor, logger)
            playoff_start_week = _update_league_scoring_settings(yq, cursor, logger)
            _update_lineup_settings(yq, cursor, logger)
            _update_fantasy_weeks(yq, cursor, league_metadata.league_key, logger)
            _update_league_matchups(yq, cursor, playoff_start_week, logger)
            _update_league_transactions(yq, cursor, logger)

            _update_daily_lineups(yq, cursor, conn, league_metadata.num_teams, league_metadata.start_date, capture_lineups, logger)
            _update_current_rosters(yq, cursor, conn, league_metadata.num_teams, logger)
            _create_rosters_tall_and_drop_rosters(cursor, conn, logger)


        # --- yfa API Call Functions ---
        if lg is None:
            logger.error("Yahoo Fantasy API (lg) object is None...")
        else:
            # Always run these in both modes to keep availability fresh
            _update_free_agents(lg, conn, logger)
            _update_waivers(lg, conn, logger)
            _update_rostered_players(lg, conn, logger)
            _update_db_metadata(cursor, logger, update_available_players_timestamp=True)

        conn.commit()
        conn.close()
        logger.info("Data import complete.")
        # --- DB Finalization ---
        # --- MODIFIED ---
        logger.info("--- Starting Database Finalization Process ---")

        PLAYER_IDS_DB_PATH = os.path.join(data_dir, 'yahoo_player_ids.db')
        PROJECTIONS_DB_PATH = os.path.join(data_dir, 'projections.db')

        logger.info(f"Using Player IDs DB from: {PLAYER_IDS_DB_PATH}")
        logger.info(f"Using Projections DB from: {PROJECTIONS_DB_PATH}")

        # --- MODIFIED: Pass logger ---
        finalizer = DBFinalizer(db_path, logger, league_id)
        if finalizer.con:
            finalizer.import_player_ids(PLAYER_IDS_DB_PATH)
            finalizer.process_with_projections(PROJECTIONS_DB_PATH)

            if not roster_updates_only:
                # Only parse stats if we actually fetched new ones
                finalizer.parse_and_store_player_stats()
                finalizer.parse_and_store_bench_stats()
            else:
                logger.info("Skipping stat parsing in Roster Only mode.")

            finalizer.close_connection()
        else:
            # --- MODIFIED ---
            logger.error("Failed to connect to the database for finalization.")
            return {'success': False, 'error': f"Could not open {db_path} for finalization."}

        # --- MODIFIED ---
        logger.info("--- Database Finalization Process Complete ---")

        logger.info("Uploading finalized database to Google Cloud Storage...")
        try:
            # This uses the GOOGLE_APPLICATION_CREDENTIALS secret file automatically
            storage_client = storage.Client()
            bucket_name = os.environ.get('GCS_BUCKET_NAME')
            if not bucket_name:
                raise Exception("GCS_BUCKET_NAME environment variable not set.")

            bucket = storage_client.bucket(bucket_name)

            # This is the "path" inside your GCS bucket
            remote_db_path = f'league-dbs/{db_filename}'
            blob = bucket.blob(remote_db_path)

            blob.upload_from_filename(db_path)
            logger.info(f"Successfully uploaded {db_filename} to GCS bucket {bucket_name} at {remote_db_path}")

        except Exception as e:
            logger.error(f"Failed to upload to GCS: {e}", exc_info=True)
            return {'success': False, 'error': f"DB built but failed to upload: {e}"}

        timestamp = os.path.getmtime(db_path)
        # --- MODIFIED ---
        logger.info(f"Database for '{sanitized_name}' updated successfully.")

        return {
            'success': True,
            'league_name': sanitized_name,
            'timestamp': int(timestamp)
        }

    except Exception as e:
        # --- MODIFIED ---
        logger.error(f"Database update process failed: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}
