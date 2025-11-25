from gevent import monkey
monkey.patch_all()

"""
Main run app for Fantasystreams.app

Author: Jason Druckenmiller
Date: 10/16/2025
Updated: 11/18/2025
"""

import os
import json
import logging
import sqlite3
from flask import Flask, Response, render_template, request, jsonify, session, redirect, url_for, send_from_directory
from yfpy.query import YahooFantasySportsQuery
import yahoo_fantasy_api as yfa
from yahoo_oauth import OAuth2
from requests_oauthlib import OAuth2Session
import time
import re
import db_builder
import uuid
from datetime import date, timedelta, datetime
import shutil
from collections import defaultdict, Counter
import itertools
import copy
from queue import Queue
import threading
import tempfile
from google.cloud import storage
import redis
from rq import Queue
from functools import wraps


# --- Flask App Configuration ---
# Assume a 'data' directory exists for storing database files
DATA_DIR = '/var/data/dbs'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

SERVER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'server')
TEST_DB_FILENAME = 'yahoo-22705-Albany Hockey Hooligans Test.db'
TEST_DB_PATH = os.path.join(SERVER_DIR, TEST_DB_FILENAME)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "a-strong-dev-secret-key-for-local-testing")
# Configure root logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


DB_BUILD_QUEUES = {}
DB_QUEUES_LOCK = threading.Lock() # To safely add/remove from the dict

db_build_status = {"running": False, "error": None, "current_build_id": None}
db_build_status_lock = threading.Lock()

# --- Yahoo OAuth2 Settings ---
authorization_base_url = 'https://api.login.yahoo.com/oauth2/request_auth'
token_url = 'https://api.login.yahoo.com/oauth2/get_token'

def model_to_dict(obj):
    """
    Recursively converts yfpy model objects, lists, and bytes into a structure
    that can be easily serialized to JSON.
    """
    if isinstance(obj, list):
        return [model_to_dict(i) for i in obj]

    if isinstance(obj, bytes):
        return obj.decode('utf-8', 'ignore')

    if not hasattr(obj, '__module__') or not obj.__module__.startswith('yfpy.'):
         return obj

    result = {}
    for key in dir(obj):
        if not key.startswith('_') and not callable(getattr(obj, key)):
            value = getattr(obj, key)
            result[key] = model_to_dict(value)
    return result

def get_yfpy_instance():
    """Helper function to get an authenticated yfpy instance."""
    # --- THIS FUNCTION IS NOT THREAD-SAFE (relies on session) ---
    if 'yahoo_token' not in session:
        return None

    if session.get('dev_mode'):
        logging.info("Dev mode: Skipping real yfpy init.")
        pass

    token = session['yahoo_token']
    auth_data = {
        'consumer_key': session.get('consumer_key', 'dev_key'), # Add defaults for dev_mode
        'consumer_secret': session.get('consumer_secret', 'dev_secret'), # Add defaults for dev_mode
        'access_token': token.get('access_token'),
        'refresh_token': token.get('refresh_token'),
        'token_type': token.get('token_type', 'bearer'),
        'token_time': token.get('expires_at', time.time() + token.get('expires_in', 3600)),
        'guid': token.get('xoauth_yahoo_guid')
    }
    try:
        yq = YahooFantasySportsQuery(
            session['league_id'],
            game_code="nhl",
            yahoo_access_token_json=auth_data
        )
        return yq
    except Exception as e:
        logging.error(f"Failed to init yfpy (expected in dev mode): {e}", exc_info=True)
        return None

def get_yfa_lg_instance():
    """Helper function to get an authenticated yfa league instance."""
    # --- THIS FUNCTION IS NOT THREAD-SAFE (relies on session) ---
    if 'yahoo_token' not in session:
        return None

    if session.get('dev_mode'):
        logging.info("Dev mode: Skipping real yfa init.")
        return None

    token = session['yahoo_token']
    consumer_key = session.get('consumer_key')
    consumer_secret = session.get('consumer_secret')
    league_id = session.get('league_id')

    if not all([token, consumer_key, consumer_secret, league_id]):
        logging.error("YFA instance requires token and credentials in session.")
        return None

    creds = {
        "consumer_key": consumer_key,
        "consumer_secret": consumer_secret,
        "access_token": token.get('access_token'),
        "refresh_token": token.get('refresh_token'),
        "token_type": token.get('token_type', 'bearer'),
        "token_time": token.get('expires_at', time.time() + token.get('expires_in', 3600)),
        "xoauth_yahoo_guid": token.get('xoauth_yahoo_guid')
    }

    temp_dir = os.path.join(tempfile.gettempdir(), 'temp_creds')
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}.json")

    try:
        with open(temp_file_path, 'w') as f:
            json.dump(creds, f)

        sc = OAuth2(None, None, from_file=temp_file_path)
        if not sc.token_is_valid():
            logging.info("YFA token expired, refreshing...")
            sc.refresh_access_token()
            # Read the *new* credentials back from the file
            with open(temp_file_path, 'r') as f:
                new_creds = json.load(f)

            # --- CRITICAL: Update the session ---
            session['yahoo_token']['access_token'] = new_creds.get('access_token')
            session['yahoo_token']['refresh_token'] = new_creds.get('refresh_token')
            session['yahoo_token']['expires_at'] = new_creds.get('token_time')
            session.modified = True
            logging.info("Session token updated after YFA refresh.")

        gm = yfa.Game(sc, 'nhl')
        lg = gm.to_league(f"nhl.l.{league_id}")
        return lg
    except Exception as e:
        logging.error(f"Failed to init yfa: {e}", exc_info=True)
        return None
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


storage_client = storage.Client()
gcs_bucket_name = os.environ.get('GCS_BUCKET_NAME')
if gcs_bucket_name:
    gcs_bucket = storage_client.bucket(gcs_bucket_name)
else:
    logging.warning("GCS_BUCKET_NAME not set. DB downloading will fail.")
    gcs_bucket = None


#MOBILE AUTH HELPER
def requires_auth(f):
    """
    A decorator to protect routes that require a logged-in user.
    It checks for 'leagues' in the session, which you set after
    a successful Yahoo login.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # 1. Check if user is logged in (do they have leagues in their session?)
        if 'leagues' not in session:
            # Not logged in. Return a 401 Unauthorized error.
            return jsonify({"error": "Unauthorized"}), 401

        # 2. (Optional but HIGHLY recommended)
        # Check if the league_id they are asking for is one of their leagues.
        if 'league_id' in kwargs:
            requested_league_id = kwargs['league_id']
            # session['leagues'] stores a list of dicts: [{'league_id': 'id', ...}]
            user_league_ids = [str(league['league_id']) for league in session['leagues']]

            if requested_league_id not in user_league_ids:
                # Logged in, but trying to access a league they don't own.
                return jsonify({"error": "Forbidden"}), 403

        # 3. If all checks pass, run the original route function
        return f(*args, **kwargs)
    return decorated_function


def requires_premium(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # 1. Standard Auth Check
        if 'yahoo_token' not in session:
            return jsonify({"error": "Unauthorized"}), 401

        # 2. Get the GUID
        guid = session['yahoo_token'].get('xoauth_yahoo_guid')

        # 3. Check Admin DB for premium status
        # We query the DB directly to catch upgrades that happened since login
        conn = sqlite3.connect(ADMIN_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT is_premium FROM users WHERE guid = ?", (guid,))
        row = cursor.fetchone()
        conn.close()

        if not row or row[0] != 1:
            return jsonify({
                "error": "Premium Required",
                "message": "This feature requires a premium subscription."
            }), 403

        return f(*args, **kwargs)
    return decorated_function


@app.route('/api/user_status')
def get_user_status():
    """Returns the current user's premium status and expiration."""
    guid = session['yahoo_token'].get('xoauth_yahoo_guid')

    conn = sqlite3.connect(ADMIN_DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT is_premium, premium_expiration_date FROM users WHERE guid = ?", (guid,))
    row = cursor.fetchone()
    conn.close()

    if row:
        return jsonify({
            "is_premium": bool(row['is_premium']),
            "expiration_date": row['premium_expiration_date']
        })
    return jsonify({"is_premium": False, "expiration_date": None})


@app.route('/api/gift_premium', methods=['POST'])
def gift_premium():
    """Sets the user to premium with a lifetime expiration."""
    guid = session['yahoo_token'].get('xoauth_yahoo_guid')
    lifetime_expiry = '9999-12-31'

    try:
        conn = sqlite3.connect(ADMIN_DB_PATH)
        cursor = conn.cursor()

        # Upsert logic mostly handled by login, so update is safe here
        cursor.execute("""
            UPDATE users
            SET is_premium = 1, premium_expiration_date = ?
            WHERE guid = ?
        """, (lifetime_expiry, guid))

        conn.commit()
        conn.close()

        # Sync changes to GCS immediately so Scheduler sees it
        upload_admin_db_to_gcs()

        return jsonify({"success": True, "expiration_date": lifetime_expiry})
    except Exception as e:
        logging.error(f"Error gifting premium: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


from api_v1 import api as api_v1_blueprint
app.register_blueprint(api_v1_blueprint)


def get_stat_source_table(sourcing_key):
    """
    Returns the correct, safe table name based on the sourcing key.
    Defaults to 'joined_player_stats' for safety.
    """
    if sourcing_key == 'todate':
        return 'joined_player_stats_real'
    elif sourcing_key == 'combined':
        return 'joined_player_stats_combined'
    else:
        # Default to 'projected' (the original table)
        return 'joined_player_stats'


def get_db_connection_for_league(league_id):
    """
    Finds the league's database, downloads it from GCS if it's newer,
    and connects to the local copy.
    """
    if session.get('use_test_db'):
        logging.info(f"Using test database: {TEST_DB_PATH}")
        if not os.path.exists(TEST_DB_PATH):
            return None, f"Test database '{TEST_DB_FILENAME}' not found in 'server' directory."
        try:
            writable_test_db_path = os.path.join(DATA_DIR, f"temp_{TEST_DB_FILENAME}")
            shutil.copy2(TEST_DB_PATH, writable_test_db_path)
            conn = sqlite3.connect(writable_test_db_path)
            conn.row_factory = sqlite3.Row
            logging.info(f"Successfully connected to temporary copy of test DB.")
            return conn, None
        except Exception as e:
            logging.error(f"Error connecting to test DB at {TEST_DB_PATH}: {e}")
            return None, "Could not connect to the test database."

    if not league_id:
        return None, "League ID not found in session."

    if not gcs_bucket:
        # --- MODIFIED: Use logging ---
        logging.error("GCS_BUCKET_NAME not set or GCS client failed to init.")
        # --- END MODIFIED ---
        return None, "GCS is not configured on the server."

    db_filename_prefix = f'yahoo-{league_id}-'
    remote_path_prefix = f'league-dbs/{db_filename_prefix}'

    # --- MODIFIED: Use logging ---
    logging.info(f"Searching GCS bucket for: {remote_path_prefix}")
    # --- END MODIFIED ---

    blob_to_download = None
    db_filename = None
    for blob in gcs_bucket.list_blobs(prefix=remote_path_prefix):
        blob_to_download = blob
        db_filename = blob.name.split('/')[-1]
        break

    if not blob_to_download or not db_filename:
        return None, "Database file not found in cloud storage. Please run a build."

    local_db_path = os.path.join(DATA_DIR, db_filename)

    try:
        should_download = False
        if not os.path.exists(local_db_path):
            should_download = True
            # --- MODIFIED: Use logging ---
            logging.info(f"No local copy of {db_filename}, downloading...")
            # --- END MODIFIED ---
        else:
            local_mtime = os.path.getmtime(local_db_path)
            remote_mtime = blob_to_download.updated.timestamp()

            if remote_mtime > local_mtime:
                should_download = True
                # --- MODIFIED: Use logging ---
                logging.info(f"Cloud copy of {db_filename} is newer, downloading...")
                # --- END MODIFIED ---

        if should_download:
            # --- MODIFIED: Use logging ---
            logging.info(f"Downloading {blob_to_download.name} to {local_db_path}...")
            # --- END MODIFIED ---
            os.makedirs(DATA_DIR, exist_ok=True)
            blob_to_download.download_to_filename(local_db_path)
            # --- MODIFIED: Use logging ---
            logging.info("Download complete.")
            # --- END MODIFIED ---

        conn = sqlite3.connect(local_db_path)
        conn.row_factory = sqlite3.Row
        return conn, None

    except Exception as e:
        # --- MODIFIED: Use logging ---
        logging.error(f"Error connecting to or downloading DB at {local_db_path}: {e}", exc_info=True)
        # --- END MODIFIED ---
        return None, "Could not connect to the database."


def decode_dict_values(data):
    """Recursively decodes byte strings in a dictionary or list of dictionaries."""
    if isinstance(data, list):
        return [decode_dict_values(item) for item in data]
    if isinstance(data, dict):
        return {k: v.decode('utf-8') if isinstance(v, bytes) else v for k, v in data.items()}
    return data


def _get_daily_simulated_roster(base_roster, simulated_moves, day_str):
    """
    Calculates the correct active roster for a given day, applying all
    simulated add/drops that have occurred up to and including that day.
    """
    # 1. Find all players dropped by this date
    dropped_player_ids_today = set()
    for m in simulated_moves:
        if m['date'] <= day_str and m.get('dropped_player'):
             dropped_player_ids_today.add(int(m['dropped_player']['player_id']))

    daily_active_roster = []

    # 2. Add players from the base roster who haven't been dropped
    for p in base_roster:
        if int(p.get('player_id', 0)) not in dropped_player_ids_today:
            daily_active_roster.append(p)

    # 3. Add simulated players who have been added AND have not been subsequently dropped
    for move in simulated_moves:
        added_player = move.get('added_player')

        # [FIX] Check if added_player exists (Drop-Only moves have this as None)
        if not added_player:
            continue

        add_date = move['date']
        added_player_id = int(added_player.get('player_id', 0))

        is_added = (add_date <= day_str)
        is_not_dropped = (added_player_id not in dropped_player_ids_today)

        if is_added and is_not_dropped:
            daily_active_roster.append(added_player)

    return daily_active_roster


def get_optimal_lineup(players, lineup_settings):
    """
    Calculates the optimal lineup using a three-pass greedy algorithm that prioritizes
    maximizing player starts and then optimizing for the best rank.
    """
    processed_players = []
    for p in players:
        player_copy = p.copy()
        if player_copy.get('total_rank') is None:
            player_copy['total_rank'] = 60
        processed_players.append(player_copy)

    ranked_players = sorted(
        processed_players,
        key=lambda p: p['total_rank']
    )

    lineup = {pos: [] for pos in lineup_settings}
    player_pool = list(ranked_players)

    # --- START MODIFICATION ---
    # Use player_id for tracking. It's guaranteed to exist and be unique.
    assigned_player_ids = set()

    def assign_player(player, pos, current_lineup, assigned_set):
        current_lineup[pos].append(player)
        # Use player_id, which is present on both base and simulated players
        assigned_set.add(player.get('player_id'))
        return True
    # --- END MODIFICATION ---

    # --- Helper to safely get position string ---
    def get_pos_str(p):
        return p.get('eligible_positions') or p.get('positions', '')

    # --- Pass 1: Place players with only one eligible position ---
    single_pos_players = sorted(
        [p for p in player_pool if len(get_pos_str(p).split(',')) == 1],
        key=lambda p: p['total_rank']
    )
    for player in single_pos_players:
        pos = get_pos_str(player).strip()
        if pos in lineup and len(lineup[pos]) < lineup_settings.get(pos, 0):
            # Use the new ID-based set
            assign_player(player, pos, lineup, assigned_player_ids)

    # Filter pool based on player_id
    player_pool = [p for p in player_pool if p.get('player_id') not in assigned_player_ids]

    # --- Pass 2: Place multi-position players using a scarcity-aware algorithm ---
    player_pool.sort(key=lambda p: p['total_rank'])
    for player in player_pool:
        eligible_positions = [pos.strip() for pos in get_pos_str(player).split(',')]
        available_slots_for_player = [
            pos for pos in eligible_positions if pos in lineup and len(lineup[pos]) < lineup_settings.get(pos, 0)
        ]

        if not available_slots_for_player: continue

        slot_scarcity = {}
        for slot in available_slots_for_player:
            scarcity_count = sum(1 for other in player_pool
                                     if other != player and
                                     other.get('player_id') not in assigned_player_ids and
                                     slot in [p.strip() for p in get_pos_str(other).split(',')])
            slot_scarcity[slot] = scarcity_count

        best_pos = min(slot_scarcity, key=slot_scarcity.get)
        # Use the new ID-based set
        assign_player(player, best_pos, lineup, assigned_player_ids)

    # Filter pool based on player_id
    player_pool = [p for p in player_pool if p.get('player_id') not in assigned_player_ids]

    # --- Pass 3: Upgrade Pass ---
    # (This pass is unaffected as it doesn't use the assigned_set)
    for benched_player in player_pool:
        for pos in [p.strip() for p in get_pos_str(benched_player).split(',')]:
            if pos not in lineup: continue

            if not lineup[pos]: continue

            worst_starter_in_pos = max(lineup[pos], key=lambda p: p['total_rank'])

            if benched_player['total_rank'] < worst_starter_in_pos['total_rank']:
                lineup[pos].remove(worst_starter_in_pos)
                lineup[pos].append(benched_player)

                is_re_slotted = False
                for other_pos in [p.strip() for p in get_pos_str(worst_starter_in_pos).split(',')]:
                    if other_pos in lineup and len(lineup[other_pos]) < lineup_settings.get(other_pos, 0):
                        lineup[other_pos].append(worst_starter_in_pos)
                        is_re_slotted = True
                        break
                break

    return lineup


def _get_ranked_roster_for_week(cursor, team_id, week_num, team_stats_map, sourcing='projected'):
    """
    Internal helper to fetch a team's full roster for a week and enrich it
    with game schedules and player performance ranks.
    """
    stat_table = get_stat_source_table(sourcing)
    cursor.execute("SELECT start_date, end_date FROM weeks WHERE week_num = ?", (week_num,))
    week_dates = cursor.fetchone()
    if not week_dates:
        return []
    start_date = week_dates['start_date'] # Keep as string for SQL
    end_date = week_dates['end_date']     # Keep as string for SQL


    cursor.execute("SELECT start_date, end_date FROM weeks WHERE week_num = ?", (week_num + 1,))
    week_dates_next = cursor.fetchone()
    start_date_next, end_date_next = None, None
    if week_dates_next:
        start_date_next = week_dates_next['start_date'] # Keep as string for SQL
        end_date_next = week_dates_next['end_date']     # Keep as string for SQL

    # --- [START] NEW: Fetch THIS WEEK'S schedule ---
    schedule_data_this_week = []
    cursor.execute("SELECT game_date, home_team, away_team FROM schedule WHERE game_date >= ? AND game_date <= ?", (start_date, end_date))
    schedule_data_this_week = decode_dict_values([dict(row) for row in cursor.fetchall()])

    # --- [START] NEW: Fetch NEXT WEEK'S schedule ---
    schedule_data_next_week = []
    if start_date_next and end_date_next:
        cursor.execute("SELECT game_date, home_team, away_team FROM schedule WHERE game_date >= ? AND game_date <= ?", (start_date_next, end_date_next))
        schedule_data_next_week = decode_dict_values([dict(row) for row in cursor.fetchall()])

    cursor.execute("""
        SELECT
            p.player_id, p.player_name, p.player_team as team,
            p.player_name_normalized, rp.eligible_positions
        FROM rosters_tall r
        JOIN rostered_players rp ON r.player_id = rp.player_id
        JOIN players p ON rp.player_id = p.player_id
        WHERE r.team_id = ?
    """, (team_id,))
    players_raw = cursor.fetchall()
    players = decode_dict_values([dict(row) for row in players_raw])

    cursor.execute("SELECT category FROM scoring")
    scoring_categories = [row['category'] for row in cursor.fetchall()]
    cat_rank_columns = [f"{cat}_cat_rank" for cat in scoring_categories]

    # --- [START] MODIFICATION: Use schedule_data list ---
    for player in players:
        player['game_dates_this_week'] = []
        player['games_this_week'] = []
        player['games_next_week'] = []
        player['opponents_list'] = []
        player['opponent_stats_this_week'] = []

        player_team = player.get('team')
        if not player_team:
            continue

        # 1. Find This Week's Games & Opponents
        games_this_week = [
            g for g in schedule_data_this_week
            if g['home_team'] == player_team or g['away_team'] == player_team
        ]
        games_this_week.sort(key=lambda g: g['game_date'])

        for game in games_this_week:
            game_date = datetime.strptime(game['game_date'], '%Y-%m-%d').date()
            player['game_dates_this_week'].append(game['game_date'])
            player['games_this_week'].append(game_date.strftime('%a'))

            opponent_tricode = game['away_team'] if game['home_team'] == player_team else game['home_team']

            if opponent_tricode:
                player['opponents_list'].append(opponent_tricode)
                opponent_stats = team_stats_map.get(opponent_tricode, {})

                player['opponent_stats_this_week'].append({
                    'game_date': game_date.strftime('%a, %b %d'),
                    'opponent_tricode': opponent_tricode,
                    'ga_gm': opponent_stats.get('ga_gm'),
                    'soga_gm': opponent_stats.get('soga_gm'),
                    'ga_gm_weekly': opponent_stats.get('ga_gm_weekly'),
                    'soga_gm_weekly': opponent_stats.get('soga_gm_weekly'),
                    'gf_gm': opponent_stats.get('gf_gm'),
                    'sogf_gm': opponent_stats.get('sogf_gm'),
                    'gf_gm_weekly': opponent_stats.get('gf_gm_weekly'),
                    'sogf_gm_weekly': opponent_stats.get('sogf_gm_weekly'),
                    'pk_pct': opponent_stats.get('pk_pct'),
                    'pk_pct_weekly': opponent_stats.get('pk_pct_weekly')
                })

        # 2. Find Next Week's Games
        games_next_week = [
            g for g in schedule_data_next_week
            if g['home_team'] == player_team or g['away_team'] == player_team
        ]
        games_next_week.sort(key=lambda g: g['game_date'])
        for game in games_next_week:
            game_date = datetime.strptime(game['game_date'], '%Y-%m-%d').date()
            player['games_next_week'].append(game_date.strftime('%a'))
    # --- [END] MODIFICATION ---

    active_players = [p for p in players if not any(pos.strip().startswith('IR') for pos in p['eligible_positions'].split(','))]
    normalized_names = [p['player_name_normalized'] for p in active_players]

    if normalized_names:
        placeholders = ','.join('?' for _ in normalized_names)
        query = f"""
            SELECT player_name_normalized, {', '.join(cat_rank_columns)}
            FROM {stat_table} j
            WHERE player_name_normalized IN ({placeholders})
        """
        cursor.execute(query, normalized_names)
        player_stats = {row['player_name_normalized']: dict(row) for row in cursor.fetchall()}

        for player in active_players:
            stats = player_stats.get(player['player_name_normalized'])
            if stats:
                total_rank = sum(stats.get(col, 0) or 0 for col in cat_rank_columns)
                player['total_rank'] = round(total_rank, 2)
            else:
                player['total_rank'] = None
            if stats:
                for col in cat_rank_columns:
                    player[col] = stats.get(col) if stats.get(col) is not None else None

    return active_players

def _calculate_unused_spots(days_in_week, active_players, lineup_settings, simulated_moves=None):
    """
    Calculates the unused roster spots for each day of the week and identifies
    potential player movements, applying simulated add/drops if provided.
    """
    if simulated_moves is None:
        simulated_moves = []

    unused_spots_data = {}
    position_order = ['C', 'LW', 'RW', 'D', 'G']

    today = date.today()
    for day_date in days_in_week:
        day_str = day_date.strftime('%Y-%m-%d')
        day_name = day_date.strftime('%a')

        daily_active_roster = _get_daily_simulated_roster(active_players, simulated_moves, day_str)

        players_playing_today = []
        for p in daily_active_roster:
            # Check both 'game_dates_this_week' (for base roster) and 'game_dates_this_week_full' (for added players)
            game_dates = p.get('game_dates_this_week') or p.get('game_dates_this_week_full', [])
            if day_str in game_dates:
                # --- FIX START: Filter out IR/IR+ players ---
                eligible_ops = (p.get('eligible_positions') or p.get('positions', '')).split(',')
                if any(pos.strip().startswith('IR') for pos in eligible_ops):
                    continue
                # --- FIX END ---
                players_playing_today.append(p)

        daily_lineup = get_optimal_lineup(players_playing_today, lineup_settings)

        if day_date < today:
            open_slots = {pos: '-' for pos in position_order}
        else:
            open_slots = {pos: lineup_settings.get(pos, 0) - len(daily_lineup.get(pos, [])) for pos in position_order}

        # Asterisk logic: check if a starter could move to an open slot
        for pos, players in daily_lineup.items():
            if pos not in position_order: continue

            # If this position is full, check if any of its players could move
            if open_slots[pos] == 0:
                for player in players:
                    eligible_positions_str = player.get('eligible_positions') or player.get('positions', '')
                    eligible = [p.strip() for p in eligible_positions_str.split(',')]
                    for other_pos in eligible:
                        current_val = open_slots.get(other_pos)
                        if current_val is not None:
                            # Safely check the value before comparing
                            numeric_val = int(str(current_val).replace('*',''))
                            if numeric_val > 0:
                                open_slots[pos] = f"{open_slots[pos]}*"
                                break
                    if isinstance(open_slots[pos], str):
                        break

        unused_spots_data[day_name] = open_slots

    return unused_spots_data

def _get_ranked_players(cursor, player_ids, cat_rank_columns, raw_stat_columns, week_num, team_stats_map, sourcing='projected'):
    """
    Internal helper to fetch player details, ranks, and schedules for a list of player IDs.
    """
    stat_table = get_stat_source_table(sourcing)
    if not player_ids:
        return []

    # Get dates for current and next week
    cursor.execute("SELECT start_date, end_date FROM weeks WHERE week_num = ?", (week_num,))
    week_dates = cursor.fetchone()
    start_date, end_date = None, None
    if week_dates:
        start_date = week_dates['start_date'] # Keep as string for SQL
        end_date = week_dates['end_date']     # Keep as string for SQL

    cursor.execute("SELECT start_date, end_date FROM weeks WHERE week_num = ?", (week_num + 1,))
    week_dates_next = cursor.fetchone()
    start_date_next, end_date_next = None, None
    if week_dates_next:
        start_date_next = week_dates_next['start_date'] # Keep as string for SQL
        end_date_next = week_dates_next['end_date']     # Keep as string for SQL

    # --- [START] NEW: Fetch THIS WEEK'S schedule ---
    schedule_data_this_week = []
    if start_date and end_date:
        cursor.execute("SELECT game_date, home_team, away_team FROM schedule WHERE game_date >= ? AND game_date <= ?", (start_date, end_date))
        schedule_data_this_week = decode_dict_values([dict(row) for row in cursor.fetchall()])

    # --- [START] NEW: Fetch NEXT WEEK'S schedule ---
    schedule_data_next_week = []
    if start_date_next and end_date_next:
        cursor.execute("SELECT game_date, home_team, away_team FROM schedule WHERE game_date >= ? AND game_date <= ?", (start_date_next, end_date_next))
        schedule_data_next_week = decode_dict_values([dict(row) for row in cursor.fetchall()])

    placeholders = ','.join('?' for _ in player_ids)
    base_columns = ['player_id', 'player_name', 'player_team', 'positions', 'status', 'player_name_normalized']
    pp_stat_columns = [
        'avg_ppTimeOnIcePctPerGame', 'lg_ppTimeOnIce', 'lg_ppTimeOnIcePctPerGame',
        'lg_ppAssists', 'lg_ppGoals', 'avg_ppTimeOnIce', 'total_ppAssists',
        'total_ppGoals', 'team_games_played'
    ]
    columns_to_select = base_columns + cat_rank_columns + pp_stat_columns + raw_stat_columns

    query = f"""
        SELECT {', '.join(columns_to_select)}
        FROM {stat_table} j
        WHERE player_id IN ({placeholders})
    """
    cursor.execute(query, player_ids)
    players_raw = cursor.fetchall()
    players = decode_dict_values([dict(row) for row in players_raw])

    # Calculate total rank and add schedules
    for player in players:
        total_rank = sum(player.get(col, 0) or 0 for col in cat_rank_columns)
        player['total_cat_rank'] = round(total_rank, 2)

        player['games_this_week'] = []
        player['games_next_week'] = []
        player['game_dates_this_week_full'] = []
        player['opponents_list'] = []
        player['opponent_stats_this_week'] = []

        player_team = player.get('player_team')
        if not player_team:
            continue

        # --- [START] REPLACEMENT LOGIC (The Performance & Accuracy Fix) ---

        # 1. Find This Week's Games & Opponents
        games_this_week = [
            g for g in schedule_data_this_week
            if g['home_team'] == player_team or g['away_team'] == player_team
        ]
        games_this_week.sort(key=lambda g: g['game_date']) # Sort just in case

        for game in games_this_week:
            game_date = datetime.strptime(game['game_date'], '%Y-%m-%d').date()
            player['games_this_week'].append(game_date.strftime('%a'))
            player['game_dates_this_week_full'].append(game['game_date'])

            opponent_tricode = game['away_team'] if game['home_team'] == player_team else game['home_team']

            if opponent_tricode:
                player['opponents_list'].append(opponent_tricode)
                opponent_stats = team_stats_map.get(opponent_tricode, {})

                player['opponent_stats_this_week'].append({
                    'game_date': game_date.strftime('%a, %b %d'),
                    'opponent_tricode': opponent_tricode,
                    'ga_gm': opponent_stats.get('ga_gm'),
                    'soga_gm': opponent_stats.get('soga_gm'),
                    'ga_gm_weekly': opponent_stats.get('ga_gm_weekly'),
                    'soga_gm_weekly': opponent_stats.get('soga_gm_weekly'),
                    'gf_gm': opponent_stats.get('gf_gm'),
                    'sogf_gm': opponent_stats.get('sogf_gm'),
                    'gf_gm_weekly': opponent_stats.get('gf_gm_weekly'),
                    'sogf_gm_weekly': opponent_stats.get('sogf_gm_weekly'),
                    'pk_pct': opponent_stats.get('pk_pct'),
                    'pk_pct_weekly': opponent_stats.get('pk_pct_weekly')
                })

        # 2. Find Next Week's Games
        games_next_week = [
            g for g in schedule_data_next_week
            if g['home_team'] == player_team or g['away_team'] == player_team
        ]
        games_next_week.sort(key=lambda g: g['game_date'])
        for game in games_next_week:
            game_date = datetime.strptime(game['game_date'], '%Y-%m-%d').date()
            player['games_next_week'].append(game_date.strftime('%a'))
        # --- [END] REPLACEMENT LOGIC ---

    return players


# --- Admin DB Configuration ---
ADMIN_DB_PATH = os.path.join(DATA_DIR, 'admin.db')
gcs_bucket = storage_client.bucket(gcs_bucket_name)

def upload_admin_db_to_gcs():
    """Syncs the local admin.db to GCS."""
    if not gcs_bucket:
        logging.warning("GCS not configured. Cannot sync admin.db.")
        return

    try:
        # Upload to the root of the bucket or a specific folder
        blob = gcs_bucket.blob('system/admin.db')
        blob.upload_from_filename(ADMIN_DB_PATH)
        logging.info("Synced admin.db to GCS.")
    except Exception as e:
        logging.error(f"Failed to upload admin.db to GCS: {e}")

def init_admin_db():
    """Creates the admin database tables and ensures schema is up to date."""
    conn = sqlite3.connect(ADMIN_DB_PATH)
    cursor = conn.cursor()

    # Create table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            guid TEXT PRIMARY KEY,
            access_token TEXT,
            refresh_token TEXT,
            token_type TEXT,
            expires_in INTEGER,
            token_time REAL,
            consumer_key TEXT,
            consumer_secret TEXT,
            is_premium INTEGER DEFAULT 0,
            premium_expiration_date TEXT
        )
    ''')

    # Create league_updaters table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS league_updaters (
            league_id TEXT PRIMARY KEY,
            user_guid TEXT,
            last_updated_ts REAL,
            FOREIGN KEY(user_guid) REFERENCES users(guid)
        )
    ''')

    # MIGRATION: Attempt to add columns to existing DBs if they are missing
    try:
        cursor.execute("ALTER TABLE users ADD COLUMN is_premium INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE users ADD COLUMN premium_expiration_date TEXT")
    except sqlite3.OperationalError:
        pass

    conn.commit()
    conn.close()

# Run this once on startup
init_admin_db()

def save_user_credentials(token, consumer_key, consumer_secret):
    """Saves or updates user credentials in the admin DB."""
    conn = sqlite3.connect(ADMIN_DB_PATH)
    cursor = conn.cursor()

    guid = token.get('xoauth_yahoo_guid')
    if not guid:
        logging.error("Cannot save credentials: No GUID in token.")
        return

    cursor.execute('''
        INSERT OR REPLACE INTO users
        (guid, access_token, refresh_token, token_type, expires_in, token_time, consumer_key, consumer_secret)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        guid,
        token.get('access_token'),
        token.get('refresh_token'),
        token.get('token_type'),
        token.get('expires_in'),
        token.get('expires_at', time.time()), # Use provided time or current
        consumer_key,
        consumer_secret
    ))
    conn.commit()
    conn.close()
    logging.info(f"Saved credentials for user {guid}")
    upload_admin_db_to_gcs()

def assign_league_updater(league_id, user_guid):
    """
    Assigns a league to a user.
    Logic: Premium users always 'steal' the updater role from Free users.
    """
    conn = sqlite3.connect(ADMIN_DB_PATH)
    cursor = conn.cursor()

    # 1. Get the Premium Status of the user currently logging in
    cursor.execute("SELECT is_premium FROM users WHERE guid = ?", (user_guid,))
    user_row = cursor.fetchone()
    is_premium_user = (user_row and user_row[0] == 1)

    # 2. Check who currently owns this league
    cursor.execute("""
        SELECT lu.user_guid, u.is_premium
        FROM league_updaters lu
        JOIN users u ON lu.user_guid = u.guid
        WHERE lu.league_id = ?
    """, (league_id,))
    row = cursor.fetchone()

    if row is None:
        # No one updates this league yet. Claim it.
        cursor.execute("INSERT INTO league_updaters (league_id, user_guid) VALUES (?, ?)", (league_id, user_guid))
        logging.info(f"League {league_id} assigned to {user_guid} (New Assignment).")
        conn.commit()
        upload_admin_db_to_gcs()
    else:
        current_updater_guid = row[0]
        current_updater_is_premium = (row[1] == 1)

        # 3. Takeover Logic: If current is Free and I am Premium -> I take over.
        if not current_updater_is_premium and is_premium_user:
            cursor.execute("UPDATE league_updaters SET user_guid = ? WHERE league_id = ?", (user_guid, league_id))
            logging.info(f"League {league_id} TAKEOVER: Premium user {user_guid} replaced Free user {current_updater_guid}.")
            conn.commit()
            upload_admin_db_to_gcs()
        else:
            logging.info(f"League {league_id} already has a valid updater. No change.")

    conn.close()


@app.route('/healthz')
def health_check():
    return "OK", 200

@app.route('/')
def index():
    if 'yahoo_token' in session:
        return redirect(url_for('home'))
    return render_template('index.html')

@app.route('/home')
def home():
    if 'yahoo_token' not in session:
        return redirect(url_for('index'))
    return render_template('home.html')

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    league_id = data.get('league_id')

    # --- [START] DEV CODE BYPASS ---
    if league_id == '99999':
        session['league_id'] = '22705' # Use the test DB's league ID
        session['use_test_db'] = True
        session['dev_mode'] = True
        session['yahoo_token'] = {
            'access_token': 'dev_token',
            'refresh_token': 'dev_refresh',
            'expires_at': time.time() + 3600
        }
        logging.info("Developer login successful using code 99999. Using test DB.")
        return jsonify({'dev_login': True, 'redirect_url': url_for('home')})
    # --- [END] DEV CODE BYPASS ---

    session['league_id'] = league_id
    session['consumer_key'] = os.environ.get("YAHOO_CONSUMER_KEY")
    session['consumer_secret'] = os.environ.get("YAHOO_CONSUMER_SECRET")

    if not all([session['league_id'], session['consumer_key'], session['consumer_secret']]):
        if not session['consumer_key'] or not session['consumer_secret']:
            logging.error("YAHOO_CONSUMER_KEY or YAHOO_CONSUMER_SECRET not set.")
            return jsonify({"error": "Server is not configured correctly."}), 500
        return jsonify({"error": "League ID is required."}), 400

    redirect_uri = url_for('callback', _external=True, _scheme='https')
    yahoo = OAuth2Session(session['consumer_key'], redirect_uri=redirect_uri)
    authorization_url, state = yahoo.authorization_url(authorization_base_url)
    session['oauth_state'] = state
    return jsonify({'auth_url': authorization_url})

@app.route('/callback')
def callback():
    if 'error' in request.args:
        error_msg = request.args.get('error_description', 'An unknown error occurred.')
        return f'<h1>Error: {error_msg}</h1>', 400

    if request.args.get('state') != session.get('oauth_state'):
        return '<h1>Error: State mismatch.</h1>', 400

    redirect_uri = url_for('callback', _external=True, _scheme='https')
    yahoo = OAuth2Session(session['consumer_key'], state=session.get('oauth_state'), redirect_uri=redirect_uri)

    try:
        token = yahoo.fetch_token(
            token_url,
            client_secret=session['consumer_secret'],
            code=request.args.get('code')
        )

        # 1. Basic Token Save
        session['yahoo_token'] = token

        # 2. Ensure we have the GUID (The unique user ID)
        # We check the token first (Fast Path). If missing, we ask the Fantasy API (Robust Path).
        if 'xoauth_yahoo_guid' not in token:
            logging.info("GUID missing from initial token. Fetching via Fantasy API...")
            try:
                # This endpoint confirms the user identity associated with the token
                resp = yahoo.get('https://fantasysports.yahooapis.com/fantasy/v2/users;use_login=1?format=json')

                if resp.status_code == 200:
                    data = resp.json()
                    # Parse: fantasy_content -> users -> 0 -> user -> 0 -> guid
                    user_obj = data['fantasy_content']['users']['0']['user'][0]
                    guid = user_obj['guid']

                    # Retrofit the GUID into the token object so the rest of the app works normally
                    token['xoauth_yahoo_guid'] = guid
                    session['yahoo_token'] = token # Re-save the complete token to session
                    logging.info(f"GUID retrieved successfully: {guid}")
                else:
                    logging.error(f"Failed to fetch user GUID. Status: {resp.status_code} Body: {resp.text}")
                    return '<h1>Error: Could not identify user.</h1>', 500
            except Exception as e:
                logging.error(f"Exception fetching fallback GUID: {e}")
                return '<h1>Error: Authentication failed during user identification.</h1>', 500

        # 3. Persist Credentials to Admin DB (and sync to GCS)
        save_user_credentials(
            session['yahoo_token'],
            session.get('consumer_key'),
            session.get('consumer_secret')
        )

        # 4. Assign as Updater if needed
        current_league_id = session.get('league_id')
        user_guid = session['yahoo_token'].get('xoauth_yahoo_guid')

        if current_league_id and user_guid:
            assign_league_updater(current_league_id, user_guid)

    except Exception as e:
        logging.error(f"Error in callback sequence: {e}", exc_info=True)
        return '<h1>Error: Login sequence failed.</h1>', 500

    return redirect(url_for('home'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

@app.route('/query', methods=['POST'])
def handle_query():
    yq = get_yfpy_instance()
    if not yq:
        return jsonify({"error": "Could not connect to Yahoo API. Your session may have expired."}), 401

    query_str = request.get_json().get('query')
    if not query_str:
        return jsonify({"error": "No query provided."}), 400

    logging.info(f"Executing query: {query_str}")
    try:
        result = eval(query_str, {"yq": yq})
        dict_result = model_to_dict(result)
        json_result = json.dumps(dict_result, indent=2)
        return jsonify({"result": json_result})
    except Exception as e:
        logging.error(f"Query error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/yfa_query', methods=['POST'])
def handle_yfa_query():
    lg = get_yfa_lg_instance()
    if not lg:
        return jsonify({"error": "Could not connect to Yahoo API. Your session may have expired."}), 401

    query_str = request.get_json().get('query')
    if not query_str:
        return jsonify({"error": "No query provided."}), 400

    logging.info(f"Executing YFA query: {query_str}")
    try:
        result = eval(query_str, {"lg": lg})
        pretty_result = json.dumps(result, indent=2)
        return jsonify({"result": pretty_result})
    except Exception as e:
        logging.error(f"YFA Query error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/matchup_page_data')
def matchup_page_data():
    league_id = session.get('league_id')
    conn, error_msg = get_db_connection_for_league(league_id)

    if not conn:
        return jsonify({'db_exists': False, 'error': error_msg})

    try:
        cursor = conn.cursor()

        # Fetch weeks
        cursor.execute("SELECT week_num, start_date, end_date FROM weeks ORDER BY week_num")
        weeks = decode_dict_values([dict(row) for row in cursor.fetchall()])

        # Fetch teams
        cursor.execute("SELECT team_id, name FROM teams ORDER BY name")
        teams = decode_dict_values([dict(row) for row in cursor.fetchall()])

        # Fetch matchups
        cursor.execute("SELECT week, team1, team2 FROM matchups")
        matchups = decode_dict_values([dict(row) for row in cursor.fetchall()])

        # Fetch scoring categories, ordered by group (offense, then goalie) then ID
        cursor.execute("SELECT category, stat_id, scoring_group FROM scoring ORDER BY scoring_group DESC, stat_id")
        scoring_categories = decode_dict_values([dict(row) for row in cursor.fetchall()])

        # Determine current week
        today = date.today().isoformat()
        cursor.execute("SELECT week_num FROM weeks WHERE start_date <= ? AND end_date >= ?", (today, today))
        current_week_row = cursor.fetchone()
        current_week = current_week_row['week_num'] if current_week_row else (weeks[0]['week_num'] if weeks else 1)

        return jsonify({
            'db_exists': True,
            'weeks': weeks,
            'teams': teams,
            'matchups': matchups,
            'scoring_categories': scoring_categories,
            'current_week': current_week
        })

    except Exception as e:
        logging.error(f"Error fetching matchup page data: {e}", exc_info=True)
        return jsonify({'db_exists': False, 'error': f"An error occurred: {e}"}), 500
    finally:
        if conn:
            conn.close()


@app.route('/api/matchup_team_stats', methods=['POST'])
def get_matchup_stats():
    league_id = session.get('league_id')
    data = request.get_json()
    sourcing = data.get('sourcing', 'projected')
    stat_table = get_stat_source_table(sourcing)

    week_num_str = data.get('week')
    team1_name = data.get('team1_name')
    team2_name = data.get('team2_name')
    simulated_moves = data.get('simulated_moves', [])

    if not week_num_str:
        return jsonify({'error': 'Week number is required.'}), 400
    try:
        week_num = int(week_num_str)
    except ValueError:
        return jsonify({'error': 'Invalid week number format.'}), 400

    conn, error_msg = get_db_connection_for_league(league_id)
    if not conn:
        return jsonify({'error': error_msg}), 404

    cursor = conn.cursor()
    cursor.execute("SELECT category FROM scoring")
    all_scoring_categories = [row['category'] for row in cursor.fetchall()]

    checked_categories = data.get('categories')
    if checked_categories is None:
        checked_categories = all_scoring_categories
    unchecked_categories = [cat for cat in all_scoring_categories if cat not in checked_categories]

    try:
        cursor = conn.cursor()

        cursor.execute("SELECT team_id FROM teams WHERE CAST(name AS TEXT) = ?", (team1_name,))
        team1_id_row = cursor.fetchone()
        if not team1_id_row: return jsonify({'error': f'Team not found: {team1_name}'}), 404
        team1_id = team1_id_row['team_id']

        cursor.execute("SELECT team_id FROM teams WHERE CAST(name AS TEXT) = ?", (team2_name,))
        team2_id_row = cursor.fetchone()
        if not team2_id_row: return jsonify({'error': f'Team not found: {team2_name}'}), 404
        team2_id = team2_id_row['team_id']

        cursor.execute("SELECT start_date, end_date FROM weeks WHERE week_num = ?", (week_num,))
        week_dates = cursor.fetchone()
        if not week_dates: return jsonify({'error': f'Week not found: {week_num}'}), 404
        start_date_str = week_dates['start_date']
        end_date_str = week_dates['end_date']
        start_date_obj = datetime.strptime(start_date_str, '%Y-%m-%d').date()
        end_date_obj = datetime.strptime(end_date_str, '%Y-%m-%d').date()
        days_in_week = [(start_date_obj + timedelta(days=i)) for i in range((end_date_obj - start_date_obj).days + 1)]

        cursor.execute("SELECT category FROM scoring")
        scoring_categories = [row['category'] for row in cursor.fetchall()]

        required_cats = {'SV', 'SA', 'GA', 'TOI/G'}
        all_categories_to_fetch = list(set(scoring_categories) | required_cats)
        projection_cats = list(set(all_categories_to_fetch) - {'TOI/G', 'SVpct'})

        cursor.execute("SELECT position, position_count FROM lineup_settings WHERE position NOT IN ('BN', 'IR', 'IR+')")
        lineup_settings = {row['position']: row['position_count'] for row in cursor.fetchall()}

        cursor.execute("""
            SELECT team_id, category, SUM(stat_value) as total
            FROM daily_player_stats
            WHERE date_ >= ? AND date_ <= ? AND (team_id = ? OR team_id = ?)
            GROUP BY team_id, category
        """, (start_date_str, end_date_str, team1_id, team2_id))

        live_stats_raw = cursor.fetchall()
        live_stats_decoded = decode_dict_values([dict(row) for row in live_stats_raw])

        stats = {
            'team1': {'live': {cat: 0 for cat in all_categories_to_fetch}, 'row': {}},
            'team2': {'live': {cat: 0 for cat in all_categories_to_fetch}, 'row': {}},
            'game_counts': { 'team1_total': 0, 'team2_total': 0, 'team1_remaining': 0, 'team2_remaining': 0 }
        }

        for row in live_stats_decoded:
          team_key = 'team1' if str(row['team_id']) == str(team1_id) else 'team2'
          if row['category'] in all_categories_to_fetch:
              stats[team_key]['live'][row['category']] = row.get('total', 0)

        for team_key in ['team1', 'team2']:
          live_stats = stats[team_key]['live']
          if 'SHO' in live_stats and live_stats['SHO'] > 0:
              live_stats['TOI/G'] += (live_stats['SHO'] * 60)
          if 'GAA' in live_stats:
              live_stats['GAA'] = (live_stats.get('GA', 0) * 60) / live_stats['TOI/G'] if live_stats.get('TOI/G', 0) > 0 else 0
          if 'SVpct' in live_stats:
              live_stats['SVpct'] = live_stats.get('SV', 0) / live_stats['SA'] if live_stats.get('SA', 0) > 0 else 0

        stats['team1']['row'] = copy.deepcopy(stats['team1']['live'])
        stats['team2']['row'] = copy.deepcopy(stats['team2']['live'])

        # --- [START] MODIFICATION: Only fetch team_stats_map. Schedule is removed. ---
        team_stats_map = {}
        cursor.execute("SELECT * FROM team_stats_summary")
        for row in cursor.fetchall():
            team_stats_map[row['team_tricode']] = dict(row)

        cursor.execute("SELECT * FROM team_stats_weekly")
        for row in cursor.fetchall():
            team_tricode = row['team_tricode']
            if team_tricode in team_stats_map:
                team_stats_map[team_tricode].update(dict(row))
        # --- [END] MODIFICATION ---

        # --- MODIFIED: Pass only team_stats_map ---
        team1_ranked_roster = _get_ranked_roster_for_week(cursor, team1_id, week_num, team_stats_map, sourcing)
        team2_ranked_roster = _get_ranked_roster_for_week(cursor, team2_id, week_num, team_stats_map, sourcing)

        # ... (rest of function is unchanged) ...
        rosters_to_update = [team1_ranked_roster, team2_ranked_roster]
        today = date.today()
        projection_start_date = max(today, start_date_obj)
        current_date = projection_start_date
        while current_date <= end_date_obj:
            current_date_str = current_date.strftime('%Y-%m-%d')
            t1_daily_roster = _get_daily_simulated_roster(team1_ranked_roster, simulated_moves, current_date_str)
            t1_players_today = []
            for p in t1_daily_roster:
                game_dates = p.get('game_dates_this_week') or p.get('game_dates_this_week_full', [])
                if current_date_str in game_dates:
                    t1_players_today.append(p)
            team2_players_today = [p for p in team2_ranked_roster if current_date_str in p.get('game_dates_this_week', [])]
            team1_lineup = get_optimal_lineup(t1_players_today, lineup_settings)
            team2_lineup = get_optimal_lineup(team2_players_today, lineup_settings)
            team1_starters = [player for pos_players in team1_lineup.values() for player in pos_players]
            team2_starters = [player for pos_players in team2_lineup.values() for player in pos_players]
            stats['game_counts']['team1_remaining'] += len(team1_starters)
            stats['game_counts']['team2_remaining'] += len(team2_starters)
            all_starter_ids_today = [p['player_id'] for p in team1_starters + team2_starters]
            if all_starter_ids_today:
                placeholders = ','.join('?' for _ in all_starter_ids_today)
                query = f"SELECT player_id, {', '.join(projection_cats)} FROM {stat_table} j WHERE player_id IN ({placeholders})"
                cursor.execute(query, tuple(all_starter_ids_today))
                player_avg_stats = {row['player_id']: dict(row) for row in cursor.fetchall()}
                for starter in team1_starters:
                    if starter['player_id'] in player_avg_stats:
                        player_proj = player_avg_stats[starter['player_id']]
                        for category in projection_cats:
                            stat_val = player_proj.get(category) or 0
                            stats['team1']['row'][category] += stat_val
                        pos_str = starter.get('eligible_positions') or starter.get('positions', '')
                        if 'G' in pos_str.split(','):
                            stats['team1']['row']['TOI/G'] += 60
                for starter in team2_starters:
                    if starter['player_id'] in player_avg_stats:
                        player_proj = player_avg_stats[starter['player_id']]
                        for category in projection_cats:
                            stat_val = player_proj.get(category) or 0
                            stats['team2']['row'][category] += stat_val
                        pos_str = starter.get('eligible_positions') or starter.get('positions', '')
                        if 'G' in pos_str.split(','):
                            stats['team2']['row']['TOI/G'] += 60
            current_date += timedelta(days=1)
        for team_key in ['team1', 'team2']:
            row_stats = stats[team_key]['row']
            gaa = (row_stats.get('GA', 0) * 60) / row_stats['TOI/G'] if row_stats.get('TOI/G', 0) > 0 else 0
            sv_pct = row_stats.get('SV', 0) / row_stats['SA'] if row_stats.get('SA', 0) > 0 else 0
            for cat, value in row_stats.items():
                if cat == 'GAA':
                    row_stats[cat] = round(gaa, 2)
                elif cat == 'SVpct':
                    row_stats[cat] = round(sv_pct, 3)
                elif isinstance(value, (int, float)) and cat not in ['GAA', 'SVpct']:
                    row_stats[cat] = round(value, 1)
        for day_date in days_in_week:
            day_str = day_date.strftime('%Y-%m-%d')
            t1_daily_roster = _get_daily_simulated_roster(team1_ranked_roster, simulated_moves, day_str)
            t1_players_today = []
            for p in t1_daily_roster:
                game_dates = p.get('game_dates_this_week') or p.get('game_dates_this_week_full', [])
                if day_str in game_dates:
                    t1_players_today.append(p)
            team2_players_today = [p for p in team2_ranked_roster if day_str in p.get('game_dates_this_week', [])]
            team1_lineup = get_optimal_lineup(t1_players_today, lineup_settings)
            team2_lineup = get_optimal_lineup(team2_players_today, lineup_settings)
            team1_starters = [player for pos_players in team1_lineup.values() for player in pos_players]
            team2_starters = [player for pos_players in team2_lineup.values() for player in pos_players]
            stats['game_counts']['team1_total'] += len(team1_starters)
            stats['game_counts']['team2_total'] += len(team2_starters)
        stats['team1_unused_spots'] = _calculate_unused_spots(days_in_week, team1_ranked_roster, lineup_settings, simulated_moves)
        return jsonify(stats)

    except Exception as e:
        logging.error(f"Error fetching matchup stats: {e}", exc_info=True)
        return jsonify({'error': f"An error occurred: {e}"}), 500
    finally:
        if conn:
            conn.close()

@app.route('/api/lineup_page_data')
def lineup_page_data():
    league_id = session.get('league_id')
    conn, error_msg = get_db_connection_for_league(league_id)

    if not conn:
        return jsonify({'db_exists': False, 'error': error_msg})

    try:
        cursor = conn.cursor()

        # Fetch weeks
        cursor.execute("SELECT week_num, start_date, end_date FROM weeks ORDER BY week_num")
        weeks = decode_dict_values([dict(row) for row in cursor.fetchall()])

        # Fetch teams
        cursor.execute("SELECT team_id, name FROM teams ORDER BY name")
        teams = decode_dict_values([dict(row) for row in cursor.fetchall()])

        # Determine current week
        today = date.today().isoformat()
        cursor.execute("SELECT week_num FROM weeks WHERE start_date <= ? AND end_date >= ?", (today, today))
        current_week_row = cursor.fetchone()
        current_week = current_week_row['week_num'] if current_week_row else (weeks[0]['week_num'] if weeks else 1)

        return jsonify({
            'db_exists': True,
            'weeks': weeks,
            'teams': teams,
            'current_week': current_week
        })

    except Exception as e:
        logging.error(f"Error fetching lineup page data: {e}", exc_info=True)
        return jsonify({'db_exists': False, 'error': f"An error occurred: {e}"}), 500
    finally:
        if conn:
            conn.close()


@app.route('/api/season_history_page_data')
def season_history_page_data():
    league_id = session.get('league_id')
    conn, error_msg = get_db_connection_for_league(league_id)

    if not conn:
        return jsonify({'db_exists': False, 'error': error_msg})

    try:
        cursor = conn.cursor()

        # Fetch weeks
        cursor.execute("SELECT week_num, start_date, end_date FROM weeks ORDER BY week_num")
        weeks = decode_dict_values([dict(row) for row in cursor.fetchall()])

        # Fetch teams
        cursor.execute("SELECT team_id, name FROM teams ORDER BY name")
        teams = decode_dict_values([dict(row) for row in cursor.fetchall()])

        # Determine current week
        today = date.today().isoformat()
        cursor.execute("SELECT week_num FROM weeks WHERE start_date <= ? AND end_date >= ?", (today, today))
        current_week_row = cursor.fetchone()
        current_week = current_week_row['week_num'] if current_week_row else (weeks[0]['week_num'] if weeks else 1)

        return jsonify({
            'db_exists': True,
            'weeks': weeks,
            'teams': teams,
            'current_week': current_week
        })

    except Exception as e:
        logging.error(f"Error fetching season history page data: {e}", exc_info=True)
        return jsonify({'db_exists': False, 'error': f"An error occurred: {e}"}), 500
    finally:
        if conn:
            conn.close()


def _get_live_matchup_stats(cursor, team1_id, team2_id, start_date_str, end_date_str):
    """
    Fetches only the 'live' stats for two teams for a given date range.
    """

    # Get official scoring categories in display order
    cursor.execute("SELECT category FROM scoring ORDER BY scoring_group DESC, stat_id")
    scoring_categories = [row['category'] for row in cursor.fetchall()]

    # Ensure all necessary sub-categories for calculations are included
    required_cats = {'SV', 'SA', 'GA', 'TOI/G'}
    all_categories_to_fetch = list(set(scoring_categories) | required_cats)

    # --- Calculate Live Stats ---
    cursor.execute("""
        SELECT team_id, category, SUM(stat_value) as total
        FROM daily_player_stats
        WHERE date_ >= ? AND date_ <= ? AND (team_id = ? OR team_id = ?)
        GROUP BY team_id, category
    """, (start_date_str, end_date_str, team1_id, team2_id))

    live_stats_raw = cursor.fetchall()
    live_stats_decoded = decode_dict_values([dict(row) for row in live_stats_raw])

    stats = {
        'team1': {'live': {cat: 0 for cat in all_categories_to_fetch}},
        'team2': {'live': {cat: 0 for cat in all_categories_to_fetch}}
    }

    for row in live_stats_decoded:
        team_key = 'team1' if str(row['team_id']) == str(team1_id) else 'team2'
        if row['category'] in all_categories_to_fetch:
            stats[team_key]['live'][row['category']] = row.get('total', 0)

    # --- Calculate Live Derived Stats & Apply SHO Fix ---
    for team_key in ['team1', 'team2']:
        live_stats = stats[team_key]['live']

        if 'SHO' in live_stats and live_stats['SHO'] > 0:
            live_stats['TOI/G'] += (live_stats['SHO'] * 60)

        if 'GAA' in live_stats:
            live_stats['GAA'] = (live_stats.get('GA', 0) * 60) / live_stats['TOI/G'] if live_stats.get('TOI/G', 0) > 0 else 0

        if 'SVpct' in live_stats:
            live_stats['SVpct'] = live_stats.get('SV', 0) / live_stats['SA'] if live_stats.get('SA', 0) > 0 else 0

    # Rounding for display
    for team_key in ['team1', 'team2']:
        live_stats = stats[team_key]['live']
        for cat, value in live_stats.items():
            if cat == 'GAA':
                live_stats[cat] = round(value, 2)
            elif cat == 'SVpct':
                live_stats[cat] = round(value, 3)
            elif isinstance(value, (int, float)):
                live_stats[cat] = round(value, 1)

    return {
        'your_team_stats': stats['team1']['live'],
        'opponent_team_stats': stats['team2']['live'],
        'scoring_categories': scoring_categories # Return the ordered list
    }


def _calculate_bench_optimization(cursor, team_id, week_num, start_date, end_date, matchup_data):
    """
    Performs a daily greedy simulation to find the "optimal" lineup
    by swapping bench players for the weakest starters.
    """
    try:
        logging.info("--- Starting Bench Optimization ---")

        # 1. Get data needed for simulation

        # Get lineup settings
        cursor.execute("SELECT position FROM lineup_settings WHERE position NOT IN ('BN', 'IR', 'IR+')")
        starter_positions = {row['position'] for row in cursor.fetchall()} # e.g., {'C', 'LW', 'RW', 'D', 'G'}
        logging.info(f"Starter positions: {starter_positions}")

        # Create position mapping
        pos_map = {
            'c': 'C', 'l': 'LW', 'r': 'RW', 'd': 'D', 'g': 'G',
            'b': 'BN', 'i': 'IR'
        }

        # Get scoring categories
        scoring_categories = matchup_data['scoring_categories']
        reverse_cats = {'GA', 'GAA'}

        # Get opponent stats
        opponent_stats = matchup_data['opponent_team_stats']

        # Create a deep copy of our stats to modify
        optimized_stats = copy.deepcopy(matchup_data['your_team_stats'])

        # Query BOTH tables
        logging.info("Querying for ALL player stats (starters and bench)...")

        cursor.execute("""
            SELECT
                d.date_, d.player_id, d.lineup_pos, d.category, d.stat_value,
                p.player_name, p.positions
            FROM daily_player_stats d
            JOIN players p ON d.player_id = p.player_id
            WHERE d.team_id = ? AND d.date_ >= ? AND d.date_ <= ?

            UNION ALL

            SELECT
                b.date_, b.player_id, b.lineup_pos, b.category, b.stat_value,
                p.player_name, p.positions
            FROM daily_bench_stats b
            JOIN players p ON b.player_id = p.player_id
            WHERE b.team_id = ? AND b.date_ >= ? AND b.date_ <= ?

            ORDER BY 1, 2
        """, (team_id, start_date, end_date, team_id, start_date, end_date))

        all_stats_raw = decode_dict_values([dict(row) for row in cursor.fetchall()])

        if not all_stats_raw:
            logging.warning("No daily_player_stats or daily_bench_stats found for optimization.")
            return matchup_data, []

        logging.info(f"Found {len(all_stats_raw)} total stat rows for simulation.")

        # 2. Pivot data by day and player
        daily_player_performances = defaultdict(lambda: defaultdict(lambda: {
            'stats': defaultdict(float), 'player_id': None, 'player_name': None,
            'lineup_pos': None, 'eligible_positions': []
        }))

        for row in all_stats_raw:
            day = row['date_']
            pid = row['player_id']
            player = daily_player_performances[day][pid]

            player['stats'][row['category']] = row['stat_value']
            player['player_id'] = pid
            player['player_name'] = row['player_name']
            player['lineup_pos'] = pos_map.get(row['lineup_pos'], row['lineup_pos']) # Normalize pos
            player['eligible_positions'] = row['positions'].split(',')

        # 3. Start the simulation
        swaps_log = []

        week_dates = sorted(daily_player_performances.keys())
        logging.info(f"Simulating days: {week_dates}")

        for day in week_dates:
            logging.info(f"--- Simulating Day: {day} ---")
            performances = daily_player_performances[day]

            starters = [p for p in performances.values() if p['lineup_pos'] in starter_positions]
            bench = [p for p in performances.values() if p['lineup_pos'] == 'BN' and sum(p['stats'].values()) > 0]

            logging.info(f"Found {len(starters)} starters and {len(bench)} scoring bench players.")

            if not bench or not starters:
                logging.info("No bench players or starters, skipping day.")
                continue

            replaced_starters_today = set() # Prevent double-swapping

            # Iterate through each bench player
            for bench_player in bench:
                logging.info(f"Evaluating bench player: {bench_player['player_name']} (Eligible: {bench_player['eligible_positions']})")
                best_swap = {
                    'starter_to_replace': None,
                    'net_gain_score': 0
                }

                bench_stats = bench_player['stats']

                # Find all starters this player is eligible to replace
                for starter in starters:
                    if starter['player_id'] in replaced_starters_today:
                        continue

                    if starter['lineup_pos'] not in bench_player['eligible_positions']:
                        logging.debug(f"  -> Skipping {starter['player_name']}: Bench player not eligible for {starter['lineup_pos']}")
                        continue

                    # This is a valid swap. Let's score it.
                    starter_stats = starter['stats']
                    current_swap_score = 0

                    for cat in scoring_categories:
                        stat_diff = bench_stats.get(cat, 0) - starter_stats.get(cat, 0)
                        if stat_diff == 0:
                            continue

                        my_current_total = optimized_stats[cat]
                        opp_total = opponent_stats.get(cat, 0)
                        my_new_total = my_current_total + stat_diff

                        is_reverse = cat in reverse_cats

                        current_points = 0
                        if (my_current_total > opp_total and not is_reverse) or (my_current_total < opp_total and is_reverse):
                            current_points = 2 # Current Win
                        elif my_current_total == opp_total:
                            current_points = 1 # Current Tie

                        new_points = 0
                        if (my_new_total > opp_total and not is_reverse) or (my_new_total < opp_total and is_reverse):
                            new_points = 2 # New Win
                        elif my_new_total == opp_total:
                            new_points = 1 # New Tie

                        current_swap_score += (new_points - current_points)

                    logging.info(f"  -> vs {starter['player_name']} ({starter['lineup_pos']}): net score = {current_swap_score}")

                    if current_swap_score > best_swap['net_gain_score']:
                        best_swap['net_gain_score'] = current_swap_score
                        best_swap['starter_to_replace'] = starter

                # After checking all starters, make the best swap if it's beneficial
                if best_swap['net_gain_score'] > 0 and best_swap['starter_to_replace']:
                    starter_to_replace = best_swap['starter_to_replace']

                    logging.info(f"  ==> SWAP FOUND: {bench_player['player_name']} for {starter_to_replace['player_name']} (Score: {best_swap['net_gain_score']})")

                    # --- START MODIFICATION ---
                    # Calculate and store the stat diffs for this specific swap
                    stat_diffs = {}
                    starter_stats = starter_to_replace['stats']
                    for cat in scoring_categories:
                        stat_diff = bench_stats.get(cat, 0) - starter_stats.get(cat, 0)
                        if stat_diff != 0:
                            stat_diffs[cat] = stat_diff
                    # --- END MODIFICATION ---

                    # 1. Log the swap
                    swaps_log.append({
                        'date': day,
                        'position': starter_to_replace['lineup_pos'],
                        'bench_player': bench_player['player_name'],
                        'replaced_player': starter_to_replace['player_name'],
                        'stat_diffs': stat_diffs  # <-- ADDED
                    })

                    # 2. Mark starter as "used" for this day
                    replaced_starters_today.add(starter_to_replace['player_id'])

                    # 3. Apply the stat changes to our optimized totals
                    for cat, diff in stat_diffs.items():
                        logging.info(f"    Applying {cat}: {optimized_stats[cat]:.1f} + ({diff:.1f}) = {optimized_stats[cat] + diff:.1f}")
                        optimized_stats[cat] += diff
                else:
                    logging.info(f"  -> No beneficial swap found for {bench_player['player_name']}.")


        # 4. Create the final optimized matchup data object
        # Recalculate derived stats (GAA, SVpct)
        if 'GAA' in optimized_stats:
            optimized_stats['GAA'] = (optimized_stats.get('GA', 0) * 60) / optimized_stats['TOI/G'] if optimized_stats.get('TOI/G', 0) > 0 else 0
        if 'SVpct' in optimized_stats:
            optimized_stats['SVpct'] = optimized_stats.get('SV', 0) / optimized_stats['SA'] if optimized_stats.get('SA', 0) > 0 else 0

        # Rounding
        for cat, value in optimized_stats.items():
            if cat == 'GAA':
                optimized_stats[cat] = round(value, 2)
            elif cat == 'SVpct':
                optimized_stats[cat] = round(value, 3)
            elif isinstance(value, (int, float)):
                optimized_stats[cat] = round(value, 1)

        optimized_matchup_data = copy.deepcopy(matchup_data)
        optimized_matchup_data['your_team_stats'] = optimized_stats

        logging.info(f"--- Bench Optimization Complete. Found {len(swaps_log)} swaps. ---")
        return optimized_matchup_data, swaps_log

    except Exception as e:
        logging.error(f"Error in _calculate_bench_optimization: {e}", exc_info=True)
        # Return the original data if simulation fails
        return matchup_data, []

@app.route('/api/history/bench_points', methods=['POST'])
def get_bench_points_data():
    league_id = session.get('league_id')
    conn, error_msg = get_db_connection_for_league(league_id)
    if not conn:
        return jsonify({'error': error_msg}), 404

    try:
        cursor = conn.cursor()
        data = request.get_json()
        team_name = data.get('team_name')
        week = data.get('week')

        logging.info("--- History Report ---")
        logging.info(f"Selected team: {team_name}, week: '{week}'")

        # 1. Get team_id
        cursor.execute("SELECT team_id FROM teams WHERE CAST(name AS TEXT) = ?", (team_name,))
        team_id_row = cursor.fetchone()
        if not team_id_row:
            return jsonify({'error': f'Team not found: {team_name}'}), 404
        team_id = team_id_row['team_id']
        logging.info(f"Found team_id: {team_id}")

        # 2. Get Dates & Matchup Data
        start_date, end_date = None, None
        matchup_data = None
        optimized_matchup_data = None
        swaps_log = []
        week_num_int = None # Initialize

        if week != 'all':
            try:
                week_num_int = int(week)
            except (ValueError, TypeError):
                return jsonify({'error': 'Invalid week format.'}), 400

            logging.info(f"Querying 'weeks' table for week_num = {week_num_int}")
            cursor.execute("SELECT start_date, end_date FROM weeks WHERE week_num = ?", (week_num_int,))
            week_dates = cursor.fetchone()

            if week_dates:
                start_date = week_dates['start_date']
                end_date = week_dates['end_date']
                logging.info(f"Found week dates: {start_date} to {end_date}")

            if start_date and end_date:
                logging.info(f"Querying 'matchups' for week = {week_num_int}, team = '{team_name}'")
                cursor.execute(
                    "SELECT team1, team2 FROM matchups WHERE week = ? AND (CAST(team1 AS TEXT) = ? OR CAST(team2 AS TEXT) = ?)",
                    (week_num_int, team_name, team_name)
                )
                matchup_row = cursor.fetchone()

                if matchup_row:
                    matchup_row_decoded = decode_dict_values(dict(matchup_row))
                    opponent_name = matchup_row_decoded['team2'] if matchup_row_decoded['team1'] == team_name else matchup_row_decoded['team1']
                    logging.info(f"Found opponent_name: {opponent_name}")

                    cursor.execute("SELECT team_id FROM teams WHERE CAST(name AS TEXT) = ?", (opponent_name,))
                    opponent_id_row = cursor.fetchone()

                    if opponent_id_row:
                        opponent_id = opponent_id_row['team_id']
                        logging.info(f"Found opponent_id: {opponent_id}")

                        # Get original matchup results
                        matchup_data = _get_live_matchup_stats(cursor, team_id, opponent_id, start_date, end_date)
                        matchup_data['opponent_name'] = opponent_name
                        logging.info("Successfully generated base matchup_data.")

                        # --- RUN OPTIMIZATION ---
                        optimized_matchup_data, swaps_log = _calculate_bench_optimization(
                            cursor, team_id, week_num_int, start_date, end_date, matchup_data
                        )
                        # --- END OPTIMIZATION ---
                    else:
                        logging.warning(f"Could not find team_id for opponent_name = {opponent_name}")
                else:
                    logging.warning(f"Query 2 FAILED: Could not find matchup_row for week = {week_num_int}")
            else:
                logging.warning("Query 1 FAILED: Could not find week_dates.")
        else:
            logging.info("Week is 'all', skipping matchup data fetch.")

        # --- 3. GET BENCH STATS (for the table) ---
        # This logic is now separate from the optimization

        logging.info("Proceeding to fetch bench stats for table display...")
        cursor.execute("SELECT category FROM scoring ORDER BY stat_id")
        all_cats_raw = cursor.fetchall()
        known_goalie_stats = {'W', 'L', 'GA', 'SV', 'SA', 'SHO', 'TOI/G', 'GAA', 'SVpct'}
        all_categories = [row['category'] for row in all_cats_raw]
        goalie_categories = [cat for cat in all_categories if cat in known_goalie_stats]
        skater_categories = [cat for cat in all_categories if cat not in known_goalie_stats]

        sql_params = [team_id]
        sql_query = """
            SELECT d.date_, d.player_id, p.player_name, p.positions, d.category, d.stat_value
            FROM daily_bench_stats d
            JOIN players p ON d.player_id = p.player_id
            WHERE d.team_id = ?
        """

        if start_date and end_date: # Only show week-specific bench stats
            sql_query += " AND d.date_ >= ? AND d.date_ <= ?"
            sql_params.extend([start_date, end_date])
        elif week != 'all': # Week was selected but no dates found (error)
             sql_query += " AND 1=0" # Force no results
        else: # 'all' weeks selected
             pass # Get all bench stats for the season

        sql_query += " ORDER BY d.date_, p.player_name"
        cursor.execute(sql_query, tuple(sql_params))
        raw_stats = decode_dict_values([dict(row) for row in cursor.fetchall()])
        logging.info(f"Found {len(raw_stats)} raw bench stat rows for table.")

        # Pivot the data
        daily_player_stats = defaultdict(lambda: defaultdict(float))
        player_positions = {}
        for row in raw_stats:
            key = (row['date_'], row['player_id'], row['player_name'])
            daily_player_stats[key][row['category']] = row['stat_value']
            player_positions[key] = row['positions']

        skater_rows, goalie_rows = [], []
        for (date, player_id, player_name), stats in daily_player_stats.items():
            if sum(stats.values()) == 0:
                continue
            key = (date, player_id, player_name)
            positions_str = player_positions.get(key, '')
            base_row = {'Date': date, 'Player': player_name, 'Positions': positions_str}
            is_goalie = 'G' in positions_str.split(',')
            if is_goalie:
                for cat in goalie_categories:
                    base_row[cat] = stats.get(cat, 0)
                goalie_rows.append(base_row)
            else:
                for cat in skater_categories:
                    base_row[cat] = stats.get(cat, 0)
                skater_rows.append(base_row)

        logging.info(f"Processed into {len(skater_rows)} skater and {len(goalie_rows)} goalie rows.")

        # 4. Return all data
        logging.info("--- History Report End ---")
        return jsonify({
            'skater_data': skater_rows,
            'skater_headers': skater_categories,
            'goalie_data': goalie_rows,
            'goalie_headers': goalie_categories,
            'matchup_data': matchup_data, # Original
            'optimized_matchup_data': optimized_matchup_data, # New
            'swaps_log': swaps_log # New
        })

    except Exception as e:
        logging.error(f"Error fetching bench points data: {e}", exc_info=True)
        return jsonify({'error': f"An error occurred: {e}"}), 500
    finally:
        if conn:
            conn.close()


@app.route('/api/history/transaction_history', methods=['POST'])
def get_transaction_history_data():
    league_id = session.get('league_id')
    conn, error_msg = get_db_connection_for_league(league_id)
    if not conn:
        return jsonify({'error': error_msg}), 404

    try:
        cursor = conn.cursor()
        data = request.get_json()
        team_name = data.get('team_name')
        week = data.get('week')
        # NEW: Get the view mode, default to 'team'
        view_mode = data.get('view_mode', 'team')

        logging.info(f"--- Transaction Success Report ---")
        logging.info(f"Selected team: {team_name}, week: '{week}', view_mode: '{view_mode}'")

        # --- MODIFIED: Get scoring group to differentiate skaters/goalies ---
        cursor.execute("SELECT category, scoring_group FROM scoring ORDER BY scoring_group DESC, stat_id")
        all_categories_raw = cursor.fetchall()
        skater_categories = [row['category'] for row in all_categories_raw if row['scoring_group'] == 'offense']
        goalie_categories = [row['category'] for row in all_categories_raw if row['scoring_group'] == 'goaltending']
        # --- END MODIFIED ---

        start_date, end_date = None, None
        transaction_start_date = None # NEW: Separate date for transaction lookup
        is_weekly_view = False

        if week != 'all':
            is_weekly_view = True
            try:
                week_num_int = int(week)
                cursor.execute("SELECT start_date, end_date FROM weeks WHERE week_num = ?", (week_num_int,))
                week_dates = cursor.fetchone()
                if week_dates:
                    start_date = week_dates['start_date']
                    end_date = week_dates['end_date']

                    # --- MODIFIED: Calculate transaction window start (2 days prior) ---
                    s_date_obj = datetime.strptime(start_date, '%Y-%m-%d').date()
                    transaction_start_date = (s_date_obj - timedelta(days=2)).strftime('%Y-%m-%d')

                    logging.info(f"Found week dates: {start_date} to {end_date}")
                    logging.info(f"Transaction window: {transaction_start_date} to {end_date}")
                else:
                    logging.warning(f"Could not find week_num = {week_num_int}")
                    is_weekly_view = False
            except (ValueError, TypeError):
                return jsonify({'error': 'Invalid week format.'}), 400

        # --- NEW: Logic split for Team vs League View ---

        if view_mode == 'team':
            # --- This is the existing logic for Team View ---
            # --- MODIFIED: Create separate lists for skaters and goalies ---
            added_skater_stats = []
            added_goalie_stats = []
            # --- END MODIFIED ---

            def fetch_transactions(move_type):
                sql_params = [team_name, move_type]
                sql_query = """
                    SELECT transaction_date, player_name, player_id
                    FROM transactions
                    WHERE CAST(fantasy_team AS TEXT) = ? AND move_type = ?
                """
                if start_date and end_date:
                    # --- MODIFIED: Use transaction_start_date ---
                    sql_query += " AND transaction_date >= ? AND transaction_date <= ?"
                    # Use the 2-day buffer start date for transactions
                    sql_params.extend([transaction_start_date, end_date])

                sql_query += " ORDER BY transaction_date, player_name"
                cursor.execute(sql_query, tuple(sql_params))
                return decode_dict_values([dict(row) for row in cursor.fetchall()])

            add_rows = fetch_transactions('add')
            drop_rows = fetch_transactions('drop')
            logging.info(f"Found {len(add_rows)} adds and {len(drop_rows)} drops for team '{team_name}'.")

            if is_weekly_view and add_rows:
                logging.info("Fetching weekly stats for added players (Team View)...")
                for player in add_rows:
                    player_stats = {'Player': player['player_name'], 'GP': 0}
                    player_id_str = str(player['player_id'])

                    # --- REVERTED: Query 1: Check for 'g' in daily_player_stats ---
                    cursor.execute("""
                        SELECT 1
                        FROM daily_player_stats
                        WHERE CAST(player_id AS TEXT) = ?
                          AND date_ >= ? AND date_ <= ?
                          AND lineup_pos = 'g'
                        LIMIT 1
                    """, (player_id_str, start_date, end_date)) # Keep original start_date for stats
                    is_goalie = cursor.fetchone() is not None
                    # --- END REVERTED ---

                    # --- Query 2: Get aggregated stats (simplified) ---
                    cursor.execute("""
                        SELECT category, SUM(stat_value) as total
                        FROM daily_player_stats
                        WHERE CAST(player_id AS TEXT) = ?
                          AND date_ >= ? AND date_ <= ?
                        GROUP BY category
                    """, (player_id_str, start_date, end_date)) # Keep original start_date for stats

                    stats_raw = cursor.fetchall()
                    player_stat_map = {row['category']: row['total'] for row in stats_raw}

                    # --- MODIFIED: Query 3: Get Games Played with non-zero stats ---
                    cursor.execute("""
                        SELECT COUNT(T.date_) as games_played
                        FROM (
                            SELECT date_, SUM(stat_value) as total_stats
                            FROM daily_player_stats
                            WHERE CAST(player_id AS TEXT) = ?
                              AND date_ >= ? AND date_ <= ?
                            GROUP BY date_
                            HAVING total_stats > 0
                        ) T
                    """, (player_id_str, start_date, end_date)) # Keep original start_date for stats

                    gp_row = cursor.fetchone()
                    if gp_row:
                        player_stats['GP'] = gp_row['games_played']
                    # --- END MODIFIED ---

                    # --- MODIFIED: Populate based on position and fill 0s ---
                    if is_goalie:
                        # --- NEW: Calculate SVpct and GAA ---
                        sv = player_stat_map.get('SV', 0)
                        sa = player_stat_map.get('SA', 0)
                        ga = player_stat_map.get('GA', 0)
                        toi = player_stat_map.get('TOI/G', 0) # Assumes 'TOI/G' is the time-on-ice stat

                        if 'SVpct' in goalie_categories:
                            player_stat_map['SVpct'] = (sv / sa) if sa > 0 else 0.0

                        if 'GAA' in goalie_categories:
                            player_stat_map['GAA'] = ((float(ga) * 60) / toi) if toi > 0 else 0.0
                        # --- END NEW ---

                        for cat in goalie_categories:
                            player_stats[cat] = player_stat_map.get(cat, 0)

                        # --- NEW: Add sub-stats for JS to use ---
                        player_stats['SV'] = sv
                        player_stats['SA'] = sa
                        player_stats['GA'] = ga
                        player_stats['TOI/G'] = toi
                        # --- END NEW ---

                        added_goalie_stats.append(player_stats)
                    else:
                        for cat in skater_categories:
                            player_stats[cat] = player_stat_map.get(cat, 0)
                        added_skater_stats.append(player_stats)
                    # --- END MODIFIED ---

            return jsonify({
                'view_mode': 'team',
                'adds': add_rows,
                'drops': drop_rows,
                # --- MODIFIED: Send separate lists and headers ---
                'added_skater_stats': added_skater_stats,
                'added_goalie_stats': added_goalie_stats,
                'skater_stat_headers': skater_categories,
                'goalie_stat_headers': goalie_categories,
                # --- END MODIFIED ---
                'is_weekly_view': is_weekly_view
            })

        elif view_mode == 'league':
            # --- This is the new logic for League View ---
            if not is_weekly_view:
                return jsonify({'error': 'League View requires a specific week to be selected.'}), 400

            # Get team name -> team_id map
            cursor.execute("SELECT team_id, name FROM teams")

            # --- MODIFIED: Decode bytes to string, then strip whitespace ---
            teams_map = {row['name'].decode('utf-8').strip(): row['team_id'] for row in cursor.fetchall()}

            # --- NEW DEBUGGING ---
            logging.info(f"Team map keys: {list(teams_map.keys())}")
            # --- END DEBUGGING ---

            # Get all 'add' transactions for the week
            # --- MODIFIED: Use transaction_start_date (2 days prior) ---
            cursor.execute("""
                SELECT transaction_date, player_name, player_id, fantasy_team
                FROM transactions
                WHERE move_type = 'add'
                  AND transaction_date >= ? AND transaction_date <= ?
                ORDER BY fantasy_team, transaction_date, player_name
            """, (transaction_start_date, end_date))

            all_adds = decode_dict_values([dict(row) for row in cursor.fetchall()])
            logging.info(f"Found {len(all_adds)} total adds for the league in week {week}.")

            # --- MODIFIED: league_data to hold skater/goalie lists ---
            league_data = defaultdict(lambda: {'skaters': [], 'goalies': []})
            # --- END MODIFIED ---

            for player in all_adds:
                # --- MODIFIED: Strip whitespace from transaction team name before lookup ---
                team_name = player['fantasy_team'].strip()
                team_id = teams_map.get(team_name)

                # --- NEW DEBUGGING ---
                if team_id is None:
                    logging.warning(f"Lookup failed for team: '{team_name}' (Original from transactions: '{player['fantasy_team']}')")
                # --- END DEBUGGING ---

                if not team_id:
                    logging.warning(f"Skipping player {player['player_name']}: could not find team_id for team '{team_name}'.")
                    continue

                player_id_str = str(player['player_id'])
                player_stats = {'Player': player['player_name'], 'GP': 0}

                # --- REVERTED: Query 1: Check for 'g' in daily_player_stats ---
                # Keep using original start_date for stats
                cursor.execute("""
                    SELECT 1
                    FROM daily_player_stats
                    WHERE CAST(player_id AS TEXT) = ? AND team_id = ?
                      AND date_ >= ? AND date_ <= ?
                      AND lineup_pos = 'g'
                    LIMIT 1
                """, (player_id_str, team_id, start_date, end_date))
                is_goalie = cursor.fetchone() is not None
                # --- END REVERTED ---

                # --- Query 2: Get aggregated stats (simplified) ---
                # Keep using original start_date for stats
                cursor.execute("""
                    SELECT category, SUM(stat_value) as total
                    FROM daily_player_stats
                    WHERE CAST(player_id AS TEXT) = ? AND team_id = ?
                      AND date_ >= ? AND date_ <= ?
                    GROUP BY category
                """, (player_id_str, team_id, start_date, end_date))

                stats_raw = cursor.fetchall()
                player_stat_map = {row['category']: row['total'] for row in stats_raw}

                # --- MODIFIED: Query 3: Get Games Played with non-zero stats *for that team* ---
                # Keep using original start_date for stats
                cursor.execute("""
                    SELECT COUNT(T.date_) as games_played
                    FROM (
                        SELECT date_, SUM(stat_value) as total_stats
                        FROM daily_player_stats
                        WHERE CAST(player_id AS TEXT) = ? AND team_id = ?
                          AND date_ >= ? AND date_ <= ?
                        GROUP BY date_
                        HAVING total_stats > 0
                    ) T
                """, (player_id_str, team_id, start_date, end_date))

                gp_row = cursor.fetchone()
                if gp_row:
                    player_stats['GP'] = gp_row['games_played']
                # --- END MODIFIED ---

                # --- MODIFIED: Populate based on position and fill 0s ---
                if is_goalie:
                    # --- NEW: Calculate SVpct and GAA ---
                    sv = player_stat_map.get('SV', 0)
                    sa = player_stat_map.get('SA', 0)
                    ga = player_stat_map.get('GA', 0)
                    toi = player_stat_map.get('TOI/G', 0) # Assumes 'TOI/G' is the time-on-ice stat

                    if 'SVpct' in goalie_categories:
                        player_stat_map['SVpct'] = (sv / sa) if sa > 0 else 0.0

                    if 'GAA' in goalie_categories:
                        player_stat_map['GAA'] = ((float(ga) * 60) / toi) if toi > 0 else 0.0
                    # --- END NEW ---

                    for cat in goalie_categories:
                        player_stats[cat] = player_stat_map.get(cat, 0)

                    # --- NEW: Add sub-stats for JS to use ---
                    player_stats['SV'] = sv
                    player_stats['SA'] = sa
                    player_stats['GA'] = ga
                    player_stats['TOI/G'] = toi
                    # --- END NEW ---

                    league_data[team_name]['goalies'].append(player_stats)
                else:
                    for cat in skater_categories:
                        player_stats[cat] = player_stat_map.get(cat, 0)
                    league_data[team_name]['skaters'].append(player_stats)
                # --- END MODIFIED ---

            return jsonify({
                'view_mode': 'league',
                'league_data': league_data,
                # --- MODIFIED: Send separate headers ---
                'skater_stat_headers': skater_categories,
                'goalie_stat_headers': goalie_categories,
                # --- END MODIFIED ---
                'is_weekly_view': True # League view is always weekly for now
            })

    except Exception as e:
        logging.error(f"Error fetching transaction data: {e}", exc_info=True)
        return jsonify({'error': f"An error occurred: {e}"}), 500
    finally:
        if conn:
            conn.close()


def _get_ranks_for_one_week(cursor, all_team_ids, selected_team_id, categories_to_process, categories_to_fetch, reverse_scoring_cats, week_num):
    """
    Helper function to calculate the ranks for a selected team for a single week.
    Returns a dict: {category: rank}
    """
    ranks_map = {}

    # 1. Get week dates
    cursor.execute("SELECT start_date, end_date FROM weeks WHERE week_num = ?", (week_num,))
    week_dates = cursor.fetchone()
    if not week_dates:
        return {} # No data for this week

    start_date = week_dates['start_date']
    end_date = week_dates['end_date']

    # 2. Initialize stats dict for all teams
    all_team_stats = {
        team_id: {cat: 0 for cat in categories_to_fetch}
        for team_id in all_team_ids
    }

    # 3. Run aggregation query for this specific week
    sql_params = [start_date, end_date]
    sql_query = """
        SELECT team_id, category, SUM(stat_value) as total
        FROM daily_player_stats
        WHERE date_ >= ? AND date_ <= ?
        GROUP BY team_id, category
    """
    cursor.execute(sql_query, tuple(sql_params))
    raw_stats = decode_dict_values([dict(row) for row in cursor.fetchall()])

    # 4. Pivot data
    for row in raw_stats:
        team_id = str(row['team_id'])
        if team_id in all_team_stats and row['category'] in all_team_stats[team_id]:
            all_team_stats[team_id][row['category']] = row.get('total', 0)

    # 5. Recalculate goalie stats
    for team_id in all_team_stats:
        stats = all_team_stats[team_id]
        sv = stats.get('SV', 0)
        sa = stats.get('SA', 0)
        ga = stats.get('GA', 0)
        toi = stats.get('TOI/G', 0)
        sho = stats.get('SHO', 0)
        if sho > 0:
            toi += (sho * 60)
            stats['TOI/G'] = toi
        if 'GAA' in stats:
            stats['GAA'] = (ga * 60) / toi if toi > 0 else 0
        if 'SVpct' in stats:
            stats['SVpct'] = sv / sa if sa > 0 else 0

    # 6. Calculate ranks for selected team
    for cat in categories_to_process:
        # --- [START] FIX ---
        # Removed the faulty 'if cat not in ...' check.
        # We will get the value (defaulting to 0) and rank it.
        # --- [END] FIX ---

        my_value = all_team_stats[selected_team_id].get(cat, 0)
        is_reverse = cat in reverse_scoring_cats

        all_values = [all_team_stats[team_id].get(cat, 0) for team_id in all_team_ids]
        sorted_values = sorted(list(set(all_values)), reverse=(not is_reverse))

        rank = sorted_values.index(my_value) + 1
        ranks_map[cat] = rank

    return ranks_map



@app.route('/api/history/category_strengths', methods=['POST'])
def get_category_strengths_data():
    league_id = session.get('league_id')
    conn, error_msg = get_db_connection_for_league(league_id)
    if not conn:
        return jsonify({'error': error_msg}), 404

    try:
        cursor = conn.cursor()
        data = request.get_json()
        selected_team_name = data.get('team_name')
        week = data.get('week')
        week_num_int = None # Initialize

        logging.info("--- Category Strengths Report (League) ---")
        logging.info(f"Selected team: {selected_team_name}, week: '{week}'")

        # 1. Get *all* teams from the DB
        cursor.execute("SELECT team_id, name FROM teams")
        all_teams_raw = cursor.fetchall()

        teams_map_id_to_name = {str(row['team_id']): row['name'].decode('utf-8').strip() for row in all_teams_raw}
        teams_map_name_to_id = {v: k for k, v in teams_map_id_to_name.items()}
        all_team_ids = list(teams_map_id_to_name.keys()) # List of string IDs

        if selected_team_name not in teams_map_name_to_id:
             return jsonify({'error': f'Team not found: {selected_team_name}'}), 404

        selected_team_id = teams_map_name_to_id[selected_team_name]

        # 2. Get Dates and Opponent
        start_date, end_date = None, None
        opponent_name = None

        if week != 'all':
            try:
                week_num_int = int(week)
                cursor.execute("SELECT start_date, end_date FROM weeks WHERE week_num = ?", (week_num_int,))
                week_dates = cursor.fetchone()

                if week_dates:
                    start_date = week_dates['start_date']
                    end_date = week_dates['end_date']
                    logging.info(f"Found week dates: {start_date} to {end_date}")

                    cursor.execute(
                        "SELECT team1, team2 FROM matchups WHERE week = ? AND (CAST(team1 AS TEXT) = ? OR CAST(team2 AS TEXT) = ?)",
                        (week_num_int, selected_team_name, selected_team_name)
                    )
                    matchup_row = cursor.fetchone()
                    if matchup_row:
                        matchup_row_decoded = decode_dict_values(dict(matchup_row))
                        opponent_name = matchup_row_decoded['team2'] if matchup_row_decoded['team1'] == selected_team_name else matchup_row_decoded['team1']
                        logging.info(f"Found opponent_name: {opponent_name}")
                else:
                    logging.warning(f"Could not find week_num = {week_num_int}")
            except (ValueError, TypeError):
                return jsonify({'error': 'Invalid week format.'}), 400
        else:
            logging.info("Week is 'all', aggregating all season stats.")

        # 3. Get Scoring Categories
        cursor.execute("SELECT category, scoring_group FROM scoring ORDER BY scoring_group DESC, stat_id")
        all_categories_raw = cursor.fetchall()
        skater_categories = [row['category'] for row in all_categories_raw if row['scoring_group'] == 'offense']
        goalie_categories = [row['category'] for row in all_categories_raw if row['scoring_group'] == 'goaltending']

        # --- MODIFIED: Create a combined list in order ---
        categories_to_process = skater_categories + goalie_categories

        all_scoring_categories = set(categories_to_process)
        categories_to_fetch = all_scoring_categories | {'SV', 'SA', 'GA', 'TOI/G'}
        reverse_scoring_cats = {'GA', 'GAA'}

        # 4. Build and execute the *league-wide* aggregation query
        sql_params = []
        sql_query = """
            SELECT team_id, category, SUM(stat_value) as total
            FROM daily_player_stats
        """

        if start_date and end_date: # Week-specific
            sql_query += " WHERE date_ >= ? AND date_ <= ?"
            sql_params.extend([start_date, end_date])
        elif week != 'all': # Week was selected but no dates found (error)
             sql_query += " WHERE 1=0" # Force no results

        sql_query += " GROUP BY team_id, category"

        cursor.execute(sql_query, tuple(sql_params))
        raw_stats = decode_dict_values([dict(row) for row in cursor.fetchall()])

        # 5. Pivot data for *all* teams and recalculate derived stats
        all_team_stats = {
            team_id: {cat: 0 for cat in categories_to_fetch}
            for team_id in all_team_ids
        }

        for row in raw_stats:
            team_id = str(row['team_id'])
            if team_id in all_team_stats and row['category'] in all_team_stats[team_id]:
                all_team_stats[team_id][row['category']] = row.get('total', 0)

        for team_id in all_team_stats:
            stats = all_team_stats[team_id]
            sv = stats.get('SV', 0)
            sa = stats.get('SA', 0)
            ga = stats.get('GA', 0)
            toi = stats.get('TOI/G', 0)
            sho = stats.get('SHO', 0)
            if sho > 0:
                toi += (sho * 60)
                stats['TOI/G'] = toi
            if 'GAA' in stats:
                stats['GAA'] = (ga * 60) / toi if toi > 0 else 0
            if 'SVpct' in stats:
                stats['SVpct'] = sv / sa if sa > 0 else 0

        opponent_team_ids = [team_id for team_id in all_team_ids if team_id != selected_team_id]
        num_opponents = len(opponent_team_ids)

        # 6. Determine Column Order
        team_headers = [selected_team_name]
        other_team_names = [name for name in teams_map_name_to_id if name != selected_team_name]
        if opponent_name:
            team_headers.append(opponent_name)
            if opponent_name in other_team_names:
                other_team_names.remove(opponent_name)
        team_headers.extend(sorted(other_team_names))

        # 7. Format data for main tables (and get current ranks)
        skater_data_rows = []
        goalie_data_rows = []
        current_period_ranks = {} # Store ranks for Step 8

        for cat in skater_categories:
            row = {'category': cat}
            my_value = all_team_stats[selected_team_id].get(cat, 0)
            is_reverse = cat in reverse_scoring_cats
            all_values = [all_team_stats[team_id].get(cat, 0) for team_id in all_team_ids]
            sorted_values = sorted(list(set(all_values)), reverse=(not is_reverse))
            rank = sorted_values.index(my_value) + 1
            row['Rank'] = rank
            current_period_ranks[cat] = rank # Save rank

            if num_opponents > 0:
                opponent_values = [all_team_stats[team_id].get(cat, 0) for team_id in opponent_team_ids]
                deltas = [my_value - opp_value for opp_value in opponent_values]
                avg_delta = sum(deltas) / num_opponents
                if is_reverse:
                    avg_delta = -avg_delta
                row['Average Delta'] = round(avg_delta, 2)
            else:
                row['Average Delta'] = 0

            for team_name in team_headers:
                row[team_name] = round(all_team_stats[teams_map_name_to_id[team_name]].get(cat, 0), 1)
            skater_data_rows.append(row)

        for cat in goalie_categories:
            row = {'category': cat}
            my_value = all_team_stats[selected_team_id].get(cat, 0)
            is_reverse = cat in reverse_scoring_cats
            all_values = [all_team_stats[team_id].get(cat, 0) for team_id in all_team_ids]
            sorted_values = sorted(list(set(all_values)), reverse=(not is_reverse))
            rank = sorted_values.index(my_value) + 1
            row['Rank'] = rank
            current_period_ranks[cat] = rank # Save rank

            if num_opponents > 0:
                opponent_values = [all_team_stats[team_id].get(cat, 0) for team_id in opponent_team_ids]
                deltas = [my_value - opp_value for opp_value in opponent_values]
                avg_delta = sum(deltas) / num_opponents
                if is_reverse:
                    avg_delta = -avg_delta
                row['Average Delta'] = round(avg_delta, 2)
            else:
                row['Average Delta'] = 0

            for team_name in team_headers:
                value = all_team_stats[teams_map_name_to_id[team_name]].get(cat, 0)
                if cat == 'GAA': value = round(value, 2)
                elif cat == 'SVpct': value = round(value, 3)
                else: value = round(value, 1)
                row[team_name] = value
            goalie_data_rows.append(row)

        # --- [START] NEW STEP 8: Calculate Rank Trends ---
        trend_data = {}

        # Get max week (current week - 1)
        today = date.today().isoformat()
        cursor.execute("SELECT week_num FROM weeks WHERE start_date <= ? AND end_date >= ?", (today, today))
        current_week_row = cursor.fetchone()
        # Use 1 as a fallback, but default to current week
        current_week_num = current_week_row['week_num'] if current_week_row else 1

        # Max week for trend is the *last completed week*
        max_week = current_week_num - 1

        if week == 'all':
            logging.info(f"Calculating 'All Season' trend data up to week {max_week}")
            trend_data['type'] = 'matrix'
            matrix_data = {cat: {} for cat in categories_to_process}
            prior_ranks = {cat: None for cat in categories_to_process}

            weeks_to_process = list(range(1, max_week + 1))
            if not weeks_to_process:
                logging.warning("No completed weeks found for trend data.")

            for w in weeks_to_process:
                # This is our new helper function
                weekly_ranks = _get_ranks_for_one_week(
                    cursor, all_team_ids, selected_team_id,
                    categories_to_process, categories_to_fetch,
                    reverse_scoring_cats, w
                )

                for cat in categories_to_process:
                    rank = weekly_ranks.get(cat)
                    prior_rank = prior_ranks.get(cat)
                    delta = None
                    if rank is not None and prior_rank is not None:
                        delta = prior_rank - rank # User's logic: +3 for 4 -> 1

                    matrix_data[cat][w] = (rank, delta)
                    prior_ranks[cat] = rank # Set for next loop

            trend_data['data'] = matrix_data
            trend_data['weeks'] = weeks_to_process

        elif week_num_int is not None: # Individual week
            logging.info(f"Calculating 'Individual Week' trend data for week {week_num_int}")
            trend_data['type'] = 'list'
            list_data = []
            prior_week_num = week_num_int - 1
            prior_ranks = {}

            if prior_week_num > 0:
                prior_ranks = _get_ranks_for_one_week(
                    cursor, all_team_ids, selected_team_id,
                    categories_to_process, categories_to_fetch,
                    reverse_scoring_cats, prior_week_num
                )

            for cat in categories_to_process:
                current_rank = current_period_ranks.get(cat)
                prior_rank = prior_ranks.get(cat)
                delta = None
                if current_rank is not None and prior_rank is not None:
                    delta = prior_rank - current_rank

                list_data.append({
                    'category': cat,
                    'rank': current_rank,
                    'delta': delta
                })
            trend_data['data'] = list_data
        # --- [END] NEW STEP 8 ---

        logging.info("--- Category Strengths Report End ---")
        return jsonify({
            'team_headers': team_headers,
            'skater_stats': skater_data_rows,
            'goalie_stats': goalie_data_rows,
            'trend_data': trend_data # Add new data to response
        })

    except Exception as e:
        logging.error(f"Error fetching category strengths data: {e}", exc_info=True)
        return jsonify({'error': f"An error occurred: {e}"}), 500
    finally:
        if conn:
            conn.close()


@app.route('/api/trade_helper_data', methods=['POST'])
def get_trade_helper_data():
    """
    Fetches category strength data and generates a league rank matrix.
    Now includes 'league_raw_stats' so the frontend can simulate trades accurately.
    """
    league_id = session.get('league_id')
    conn, error_msg = get_db_connection_for_league(league_id)
    if not conn:
        return jsonify({'error': error_msg}), 404

    try:
        cursor = conn.cursor()
        data = request.get_json()
        selected_team_name = data.get('team_name')
        week = data.get('week')

        # 1. Get Teams
        cursor.execute("SELECT team_id, name FROM teams")
        all_teams_raw = cursor.fetchall()
        teams_map_id_to_name = {str(row['team_id']): row['name'].decode('utf-8').strip() for row in all_teams_raw}
        teams_map_name_to_id = {v: k for k, v in teams_map_id_to_name.items()}
        all_team_ids = list(teams_map_id_to_name.keys())

        if not selected_team_name or selected_team_name not in teams_map_name_to_id:
             return jsonify({'error': f'Valid team_name is required.'}), 404

        selected_team_id = teams_map_name_to_id[selected_team_name]
        opponent_team_ids = [tid for tid in all_team_ids if tid != selected_team_id]
        num_opponents = len(opponent_team_ids)

        # 2. Get Dates
        start_date, end_date = None, None
        if week != 'all':
            try:
                week_num_int = int(week)
                cursor.execute("SELECT start_date, end_date FROM weeks WHERE week_num = ?", (week_num_int,))
                week_dates = cursor.fetchone()
                if week_dates:
                    start_date = week_dates['start_date']
                    end_date = week_dates['end_date']
            except (ValueError, TypeError):
                pass

        # 3. Get Categories
        cursor.execute("SELECT category, scoring_group FROM scoring ORDER BY scoring_group DESC, stat_id")
        all_categories_raw = cursor.fetchall()
        skater_categories = [row['category'] for row in all_categories_raw if row['scoring_group'] == 'offense']
        goalie_categories = [row['category'] for row in all_categories_raw if row['scoring_group'] == 'goaltending']
        all_scoring_categories_list = skater_categories + goalie_categories

        goalie_sub_cats_map = {'GAA': ['GA', 'TOI/G'], 'SVpct': ['SV', 'SA']}
        extra_stats = {'SV', 'SA', 'GA', 'TOI/G'}
        categories_to_fetch = set(all_scoring_categories_list) | extra_stats
        reverse_scoring_cats = {'GA', 'GAA', 'L'}

        # 4. Aggregate Stats
        sql_params = []
        sql_query = "SELECT team_id, category, SUM(stat_value) as total FROM daily_player_stats"
        if start_date and end_date:
            sql_query += " WHERE date_ >= ? AND date_ <= ?"
            sql_params.extend([start_date, end_date])
        sql_query += " GROUP BY team_id, category"

        cursor.execute(sql_query, tuple(sql_params))
        raw_stats = decode_dict_values([dict(row) for row in cursor.fetchall()])

        # 5. Pivot Data
        all_team_stats = {tid: {cat: 0 for cat in categories_to_fetch} for tid in all_team_ids}

        for row in raw_stats:
            tid = str(row['team_id'])
            if tid in all_team_stats and row['category'] in all_team_stats[tid]:
                all_team_stats[tid][row['category']] = row.get('total', 0)

        # Calculate Derived Stats
        for tid in all_team_stats:
            stats = all_team_stats[tid]
            sv, sa, ga = stats.get('SV', 0), stats.get('SA', 0), stats.get('GA', 0)
            toi, sho = stats.get('TOI/G', 0), stats.get('SHO', 0)
            if sho > 0: toi += (sho * 60)
            stats['TOI/G'] = toi

            if 'GAA' in categories_to_fetch:
                stats['GAA'] = (ga * 60) / toi if toi > 0 else 0
            if 'SVpct' in categories_to_fetch:
                stats['SVpct'] = sv / sa if sa > 0 else 0

        # --- NEW: Build Readable League Stats Dict (Name -> Stats) ---
        league_raw_stats = {}
        for tid, stats in all_team_stats.items():
            tname = teams_map_id_to_name.get(tid, f"Unknown ({tid})")
            league_raw_stats[tname] = stats
        # ------------------------------------------------------------

        # 6. Generate League Rank Matrix
        league_rank_matrix = {}
        for team_name in teams_map_name_to_id.keys():
            league_rank_matrix[team_name] = {}

        for cat in categories_to_fetch:
            is_reverse = cat in reverse_scoring_cats
            values_map = {tid: all_team_stats[tid].get(cat, 0) for tid in all_team_ids}
            sorted_values = sorted(list(set(values_map.values())), reverse=(not is_reverse))

            for tid in all_team_ids:
                val = values_map[tid]
                rank = sorted_values.index(val) + 1
                t_name = teams_map_id_to_name[tid]
                league_rank_matrix[t_name][cat] = rank

        # 7. Format Response
        skater_data_rows = []
        goalie_data_rows = []

        def build_row_dict(cat):
            my_rank = league_rank_matrix[selected_team_name].get(cat, '-')
            my_val = all_team_stats[selected_team_id].get(cat, 0)

            is_rev = cat in reverse_scoring_cats
            avg_delta = 0
            if num_opponents > 0:
                opp_vals = [all_team_stats[tid].get(cat, 0) for tid in opponent_team_ids]
                deltas = [my_val - ov for ov in opp_vals]
                avg_delta = sum(deltas) / num_opponents
                if is_rev: avg_delta = -avg_delta

            val_display = round(my_val, 3) if cat == 'SVpct' else (round(my_val, 2) if cat == 'GAA' else round(my_val, 1))

            return {
                'category': cat,
                'Rank': my_rank,
                'Total': val_display,
                'Average Delta': round(avg_delta, 2)
            }

        def build_rows(cats, target_list):
            for cat in cats:
                row = build_row_dict(cat)
                row['sub_rows'] = []
                if cat in goalie_sub_cats_map:
                    for sub_cat in goalie_sub_cats_map[cat]:
                        sub_row = build_row_dict(sub_cat)
                        row['sub_rows'].append(sub_row)
                target_list.append(row)

        build_rows(skater_categories, skater_data_rows)
        build_rows(goalie_categories, goalie_data_rows)

        return jsonify({
            'skater_stats': skater_data_rows,
            'goalie_stats': goalie_data_rows,
            'all_scoring_categories': all_scoring_categories_list,
            'league_rank_matrix': league_rank_matrix,
            'league_raw_stats': league_raw_stats,  # <--- SENDING THIS NOW
            'total_teams': len(all_team_ids)
        })

    except Exception as e:
        logging.error(f"Error fetching trade helper data: {e}", exc_info=True)
        return jsonify({'error': f"An error occurred: {e}"}), 500
    finally:
        if conn: conn.close()




@app.route('/api/trade_helper_league_roster_data', methods=['POST'])
def get_trade_helper_league_roster_data():
    """
    Fetches ALL rostered players, including Raw Stats, Ranks, and PP Stats.
    Fixed: Quotes column names ONCE to handle special characters like 'TOI/G'.
    """
    league_id = session.get('league_id')
    data = request.get_json()
    sourcing = data.get('sourcing', 'projected')
    stat_table = get_stat_source_table(sourcing)

    conn, error_msg = get_db_connection_for_league(league_id)
    if not conn:
        return jsonify({'error': error_msg}), 404

    try:
        cursor = conn.cursor()

        # 0. Get Current Week
        today = date.today().isoformat()
        cursor.execute("SELECT week_num FROM weeks WHERE start_date <= ? AND end_date >= ?", (today, today))
        current_week_row = cursor.fetchone()
        current_week = current_week_row['week_num'] if current_week_row else 1

        # 1. Get Team Names/IDs
        cursor.execute("SELECT team_id, name FROM teams")
        teams_map = {str(row['team_id']): row['name'].decode('utf-8').strip() for row in cursor.fetchall()}

        # 2. Get Scoring Categories
        cursor.execute("SELECT category, scoring_group FROM scoring ORDER BY scoring_group DESC, stat_id")
        all_categories_raw = cursor.fetchall()
        skater_categories = [row['category'] for row in all_categories_raw if row['scoring_group'] == 'offense']
        goalie_categories = [row['category'] for row in all_categories_raw if row['scoring_group'] == 'goaltending']
        all_scoring_categories = skater_categories + goalie_categories

        # 3. Define Columns to Fetch
        pp_cols = [
            'avg_ppTimeOnIcePctPerGame', 'lg_ppTimeOnIce', 'lg_ppTimeOnIcePctPerGame',
            'lg_ppAssists', 'lg_ppGoals', 'avg_ppTimeOnIce', 'total_ppAssists',
            'total_ppGoals', 'team_games_played'
        ]

        # 4. Get all players
        cursor.execute("""
            SELECT
                p.player_id, p.player_name, p.player_team as team,
                rp.eligible_positions, p.player_name_normalized,
                r.team_id as fantasy_team_id
            FROM rosters_tall r
            JOIN rostered_players rp ON r.player_id = rp.player_id
            JOIN players p ON rp.player_id = p.player_id
        """)
        all_players_raw = cursor.fetchall()
        all_players = decode_dict_values([dict(row) for row in all_players_raw])

        # 5. Add Fantasy Team Name
        for player in all_players:
            player['fantasy_team_name'] = teams_map.get(str(player['fantasy_team_id']), 'Unknown Team')

        # 6. Get Ranks AND Stats
        cat_rank_columns = [f"{cat}_cat_rank" for cat in all_scoring_categories]

        # --- FIX: Define raw stats WITHOUT quotes here ---
        # This prevents double-quoting later (e.g. ""TOI/G"") which causes SQL errors
        raw_stats_to_fetch = list(set(all_scoring_categories) | {'GA', 'SV', 'SA', 'TOI/G'})

        valid_normalized_names = [p.get('player_name_normalized') for p in all_players if p.get('player_name_normalized')]

        if valid_normalized_names:
            # Combine all column names (plain strings)
            cols_to_select = list(set(cat_rank_columns + pp_cols + raw_stats_to_fetch))

            # --- FIX: Quote columns ONCE here ---
            # This turns G into "G" and TOI/G into "TOI/G" for the SQL query
            quoted_cols = [f'"{col}"' for col in cols_to_select]

            placeholders = ','.join('?' for _ in valid_normalized_names)
            query = f"""
                SELECT player_name_normalized, {', '.join(quoted_cols)}
                FROM {stat_table} j WHERE player_name_normalized IN ({placeholders})
            """
            cursor.execute(query, valid_normalized_names)
            player_stats = {row['player_name_normalized']: dict(row) for row in cursor.fetchall()}

            # 7. Enrich players
            for player in all_players:
                p_stats = player_stats.get(player.get('player_name_normalized'))
                if p_stats:
                    # Map all columns (Ranks + Raw Stats + PP) to the player object
                    for key, val in p_stats.items():
                        player[key] = val
                else:
                    for cat in all_scoring_categories:
                        player[f"{cat}_cat_rank"] = None

        return jsonify({
            'players': all_players,
            'skater_categories': skater_categories,
            'goalie_categories': goalie_categories,
            'current_week': current_week
        })

    except Exception as e:
        logging.error(f"Error fetching league roster data: {e}", exc_info=True)
        return jsonify({'error': f"An error occurred: {e}"}), 500
    finally:
        if conn: conn.close()



@app.route('/api/schedules_page_data')
def schedules_page_data():
    """
    Provides the necessary data to populate the Schedules page.
    This includes all weeks in the season.
    """
    league_id = session.get('league_id')
    conn, error_msg = get_db_connection_for_league(league_id)

    if not conn:
        return jsonify({'db_exists': False, 'error': error_msg})

    try:
        cursor = conn.cursor()

        # Fetch all weeks (as requested for the schedules page)
        cursor.execute("SELECT week_num, start_date, end_date FROM weeks ORDER BY week_num")
        weeks = decode_dict_values([dict(row) for row in cursor.fetchall()])

        return jsonify({
            'db_exists': True,
            'weeks': weeks
        })

    except Exception as e:
        logging.error(f"Error fetching schedules page data: {e}", exc_info=True)
        return jsonify({'db_exists': False, 'error': f"An error occurred: {e}"}), 500
    finally:
        if conn:
            conn.close()


TEAM_TRICODES = [
    "FLA", "CHI", "NYR", "PIT", "LAK", "COL", "TOR", "MTL", "WSH", "BOS",
    "EDM", "CGY", "VGK", "BUF", "DET", "TBL", "OTT", "PHI", "NYI", "CAR",
    "NJD", "STL", "MIN", "NSH", "CBJ", "WPG", "DAL", "UTA", "VAN", "SJS",
    "SEA", "ANA"
]

@app.route('/api/schedules/off_days', methods=['POST'])
def schedules_off_days():
    """
    Fetches and processes "Off Days" data based on the selected week.
    """
    league_id = session.get('league_id')
    conn, error_msg = get_db_connection_for_league(league_id)
    if not conn:
        return jsonify({'error': error_msg}), 500

    try:
        data = request.get_json()
        selected_week = data.get('week')
        cursor = conn.cursor()

        # 1a. Fetch team standings data
        cursor.execute("SELECT team_tricode, point_pct, goals_against_per_game FROM team_standings")
        standings_rows = cursor.fetchall()
        # Create a map for easy lookup: {'STL': {'point_pct': '0.550', 'goals_against_per_game': 2.80}, ...}
        standings_map = {row['team_tricode']: {'point_pct': row['point_pct'], 'goals_against_per_game': row['goals_against_per_game']} for row in standings_rows}


        # 1b. Fetch data from all three tables (Unchanged)
        cursor.execute("SELECT off_day_date FROM off_days")
        off_days_set = set(row['off_day_date'] for row in cursor.fetchall())

        cursor.execute("SELECT week_num, start_date, end_date FROM weeks ORDER BY week_num")
        weeks = decode_dict_values([dict(row) for row in cursor.fetchall()])

        cursor.execute("SELECT game_date, home_team, away_team FROM schedule")
        schedule = decode_dict_values([dict(row) for row in cursor.fetchall()])

        # 2. Determine current week (Unchanged)
        today = date.today().isoformat()
        cursor.execute("SELECT week_num FROM weeks WHERE start_date <= ? AND end_date >= ?", (today, today))
        current_week_row = cursor.fetchone()
        current_week = current_week_row['week_num'] if current_week_row else (weeks[0]['week_num'] if weeks else 1)
        if not weeks:
            return jsonify({'error': 'No week data found in database.'}), 500

        # 3. Process the data into a master structure (Unchanged)
        all_weeks_data = {}
        for week in weeks:
            week_num = week['week_num']
            start_date = week['start_date']
            end_date = week['end_date']
            all_weeks_data[week_num] = {team: {'off_days': 0, 'total_games': 0} for team in TEAM_TRICODES}

            week_schedule = [g for g in schedule if start_date <= g['game_date'] <= end_date]

            for game in week_schedule:
                is_off_day = game['game_date'] in off_days_set
                teams_in_game = [game['home_team'], game['away_team']]

                for team in teams_in_game:
                    if team in all_weeks_data[week_num]:
                        all_weeks_data[week_num][team]['total_games'] += 1
                        if is_off_day:
                            all_weeks_data[week_num][team]['off_days'] += 1

        # 4. Format the response based on selected_week
        if selected_week == 'all':
            # --- "All Season" logic (Unchanged) ---
            ros_data = {'headers': [], 'rows': []}
            past_data = {'headers': [], 'rows': []}

            ros_headers = [f"Week {w['week_num']}" for w in weeks if w['week_num'] >= current_week]
            past_headers = [f"Week {w['week_num']}" for w in weeks if w['week_num'] < current_week]
            ros_data['headers'] = ros_headers
            past_data['headers'] = past_headers

            for team in TEAM_TRICODES:
                ros_row = {'team': team}
                past_row = {'team': team}
                ros_total = 0
                for week_header in ros_headers:
                    week_num = int(week_header.split(' ')[1])
                    off_days_count = all_weeks_data[week_num][team]['off_days']
                    ros_row[week_header] = off_days_count
                    ros_total += off_days_count
                ros_row['Total'] = ros_total
                for week_header in past_headers:
                    week_num = int(week_header.split(' ')[1])
                    past_row[week_header] = all_weeks_data[week_num][team]['off_days']
                ros_data['rows'].append(ros_row)
                past_data['rows'].append(past_row)

            return jsonify({
                'report_type': 'all_season',
                'ros_data': ros_data,
                'past_data': past_data
            })

        else:
            # --- Single week logic ---
            week_num_int = int(selected_week)
            table_data = []
            if week_num_int not in all_weeks_data:
                 return jsonify({'error': f'Data for week {week_num_int} not found.'}), 404

            # Find the correct week's start/end dates
            selected_week_details = next((w for w in weeks if w['week_num'] == week_num_int), None)
            if not selected_week_details:
                 return jsonify({'error': f'Week details for {week_num_int} not found.'}), 404

            start_date = selected_week_details['start_date']
            end_date = selected_week_details['end_date']
            week_data = all_weeks_data[week_num_int]

            for team in TEAM_TRICODES:
                # Find opponents for this team in this week
                games_this_week = [g for g in schedule if start_date <= g['game_date'] <= end_date and (g['home_team'] == team or g['away_team'] == team)]
                opponents = []
                for game in games_this_week:
                    opponent = game['away_team'] if game['home_team'] == team else game['home_team']
                    opponents.append(opponent)

                # --- [START] MODIFIED Calculate opponent averages ---
                if not opponents:
                    avg_ga_str = 'N/A'
                    avg_pt_pct_str = 'N/A'
                else:
                    total_ga = 0.0      # MODIFIED: Use float
                    total_pt_pct = 0.0  # MODIFIED: Use float
                    game_count = len(opponents)

                    for opp in opponents:
                        team_stats = standings_map.get(opp) # Get stats for the opponent
                        if team_stats:
                            # Coalesce None (from DB) to 0.0
                            total_ga += team_stats.get('goals_against_per_game') or 0.0
                            # MODIFIED: Cast TEXT 'point_pct' to float
                            total_pt_pct += float(team_stats.get('point_pct') or 0.0)

                    avg_ga = total_ga / game_count
                    avg_pt_pct = total_pt_pct / game_count
                    avg_ga_str = f"{avg_ga:.2f}"
                    avg_pt_pct_str = f"{avg_pt_pct:.3f}"
                # --- [END] MODIFIED ---

                table_data.append({
                    'team': team,
                    'off_days': week_data[team]['off_days'],
                    'total_games': week_data[team]['total_games'],
                    'opponents': ", ".join(opponents),
                    'opponent_avg_ga': avg_ga_str,
                    'opponent_avg_pt_pct': avg_pt_pct_str
                })

            return jsonify({
                'report_type': 'single_week',
                'table_data': table_data
            })

    except Exception as e:
        logging.error(f"Error fetching schedules/off_days data: {e}", exc_info=True)
        return jsonify({'error': f"An error occurred: {e}"}), 500
    finally:
        if conn:
            conn.close()



@app.route('/api/schedules/playoff_schedules', methods=['GET'])
def schedules_playoff_schedules():
    """
    Determines the league's playoff weeks and fetches the schedule,
    off-day games, and opponents for each team during those weeks.
    """
    league_id = session.get('league_id')
    conn, error_msg = get_db_connection_for_league(league_id)
    if not conn:
        return jsonify({'error': error_msg}), 500

    try:
        cursor = conn.cursor()

        # ... (Steps 1-4 are unchanged) ...

        # 1. Get league playoff end date
        cursor.execute("SELECT value FROM league_info WHERE key = 'end_date'")
        league_end_date_row = cursor.fetchone()
        if not league_end_date_row:
            return jsonify({'error': 'League end_date not found in league_info table.'}), 404
        league_end_date = league_end_date_row['value']

        # 2. Get max regular season matchup week
        cursor.execute("SELECT MAX(week) as max_week FROM matchups")
        max_week_row = cursor.fetchone()
        if not max_week_row or max_week_row['max_week'] is None:
            return jsonify({'error': 'No matchup data found to determine playoff start.'}), 404
        start_playoff_week_num = max_week_row['max_week'] + 1

        # 3. Get all weeks from the database
        cursor.execute("SELECT week_num, start_date, end_date FROM weeks ORDER BY week_num")
        all_weeks = decode_dict_values([dict(row) for row in cursor.fetchall()])

        # 4. Filter to find the exact playoff weeks
        playoff_weeks = []
        found_start = False
        for week in all_weeks:
            if week['week_num'] == start_playoff_week_num:
                found_start = True

            if found_start:
                playoff_weeks.append(week)
                if week['end_date'] == league_end_date:
                    break

        if not playoff_weeks:
             return jsonify({
                  'title': 'Playoff Weeks',
                  'headers': [],
                  'rows': []
             }), 200

        # 5. Get data for schedule and off-days
        cursor.execute("SELECT off_day_date FROM off_days")
        off_days_set = set(row['off_day_date'] for row in cursor.fetchall())

        cursor.execute("SELECT game_date, home_team, away_team FROM schedule")
        schedule = decode_dict_values([dict(row) for row in cursor.fetchall()])

        # 5a. Fetch team standings data
        cursor.execute("SELECT team_tricode, point_pct, goals_against_per_game FROM team_standings")
        standings_rows = cursor.fetchall()
        standings_map = {row['team_tricode']: {'point_pct': row['point_pct'], 'goals_against_per_game': row['goals_against_per_game']} for row in standings_rows}

        # 6. Process data for each team for each playoff week
        team_data = {team: {} for team in TEAM_TRICODES}
        for team in TEAM_TRICODES:
            for week in playoff_weeks:
                week_num = week['week_num']
                start_date = week['start_date']
                end_date = week['end_date']

                games_this_week = [g for g in schedule if start_date <= g['game_date'] <= end_date and (g['home_team'] == team or g['away_team'] == team)]

                total_games = len(games_this_week)
                off_day_games = 0
                opponents = [] # This is a list

                for game in games_this_week:
                    if game['game_date'] in off_days_set:
                        off_day_games += 1

                    opponent = game['away_team'] if game['home_team'] == team else game['home_team']
                    opponents.append(opponent)

                # --- [START] MODIFIED Calculate opponent averages ---
                if not opponents:
                    avg_ga_str = 'N/A'
                    avg_pt_pct_str = 'N/A'
                else:
                    total_ga = 0.0      # MODIFIED: Use float
                    total_pt_pct = 0.0  # MODIFIED: Use float
                    game_count = len(opponents)

                    for opp in opponents:
                        team_stats = standings_map.get(opp)
                        if team_stats:
                            total_ga += team_stats.get('goals_against_per_game') or 0.0
                            # MODIFIED: Cast TEXT 'point_pct' to float
                            total_pt_pct += float(team_stats.get('point_pct') or 0.0)

                    avg_ga = total_ga / game_count
                    avg_pt_pct = total_pt_pct / game_count
                    avg_ga_str = f"{avg_ga:.2f}"
                    avg_pt_pct_str = f"{avg_pt_pct:.3f}"
                # --- [END] MODIFIED ---

                team_data[team][week_num] = {
                    'games': total_games,
                    'off_days': off_day_games,
                    'opponents': ", ".join(opponents),
                    'opponent_avg_ga': avg_ga_str,
                    'opponent_avg_pt_pct': avg_pt_pct_str
                }

        # 7. Format for the frontend table (Unchanged)
        headers = ['Team']
        for week in playoff_weeks:
            week_num = week['week_num']
            headers.append(f'Week {week_num} Games')
            headers.append(f'Week {week_num} Opponents')
            headers.append(f'Week {week_num} Opponent Avg GA')
            headers.append(f'Week {week_num} Opponent Avg Pt %')

        rows = []
        for team in TEAM_TRICODES:
            row = {'Team': team}
            for week in playoff_weeks:
                week_num = week['week_num']
                data = team_data[team][week_num]

                row[f'Week {week_num} Games'] = f"{data['games']} ({data['off_days']})"
                row[f'Week {week_num} Opponents'] = data['opponents']
                row[f'Week {week_num} Opponent Avg GA'] = data['opponent_avg_ga']
                row[f'Week {week_num} Opponent Avg Pt %'] = data['opponent_avg_pt_pct']
            rows.append(row)

        return jsonify({
            'title': 'Playoff Weeks',
            'headers': headers,
            'rows': rows
        })

    except Exception as e:
        logging.error(f"Error fetching schedules/playoff_schedules data: {e}", exc_info=True)
        return jsonify({'error': f"An error occurred: {e}"}), 500
    finally:
        if conn:
            conn.close()


@app.route('/api/roster_data', methods=['POST'])
def get_roster_data():
    league_id = session.get('league_id')
    data = request.get_json()
    sourcing = data.get('sourcing', 'projected')
    stat_table = get_stat_source_table(sourcing)
    week_num_str = data.get('week')
    team_name = data.get('team_name')
    simulated_moves = data.get('simulated_moves', [])

    if not week_num_str: return jsonify({'error': 'Week number is required.'}), 400
    try:
        week_num = int(week_num_str)
    except ValueError: return jsonify({'error': 'Invalid week number format.'}), 400

    conn, error_msg = get_db_connection_for_league(league_id)
    if not conn: return jsonify({'error': error_msg}), 404

    try:
        cursor = conn.cursor()

        # 1. Setup Categories
        cursor.execute("SELECT category, scoring_group FROM scoring ORDER BY scoring_group DESC, stat_id")
        all_categories_raw = cursor.fetchall()
        skater_categories = [row['category'] for row in all_categories_raw if row['scoring_group'] == 'offense']
        goalie_categories = [row['category'] for row in all_categories_raw if row['scoring_group'] == 'goaltending']
        all_scoring_categories = skater_categories + goalie_categories

        checked_categories = data.get('categories') or all_scoring_categories
        unchecked_categories = [cat for cat in all_scoring_categories if cat not in checked_categories]

        # 2. Get Team ID
        cursor.execute("SELECT team_id FROM teams WHERE CAST(name AS TEXT) = ?", (team_name,))
        team_id_row = cursor.fetchone()
        if not team_id_row: return jsonify({'error': f'Team not found: {team_name}'}), 404
        team_id = team_id_row['team_id']

        # 3. Get Week Dates
        cursor.execute("SELECT start_date, end_date FROM weeks WHERE week_num = ?", (week_num,))
        week_dates = cursor.fetchone()
        if not week_dates: return jsonify({'error': f'Week not found: {week_num}'}), 404
        start_date_str = week_dates['start_date']
        end_date_str = week_dates['end_date']
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
        days_in_week = [(start_date + timedelta(days=i)) for i in range((end_date - start_date).days + 1)]

        cursor.execute("SELECT start_date, end_date FROM weeks WHERE week_num = ?", (week_num + 1,))
        week_next = cursor.fetchone()

        # 4. Fetch Schedule Data (Needed for Simulated Players)
        schedule_this_week = []
        cursor.execute("SELECT game_date, home_team, away_team FROM schedule WHERE game_date >= ? AND game_date <= ?", (start_date_str, end_date_str))
        schedule_this_week = decode_dict_values([dict(row) for row in cursor.fetchall()])

        schedule_next_week = []
        if week_next:
            cursor.execute("SELECT game_date, home_team, away_team FROM schedule WHERE game_date >= ? AND game_date <= ?", (week_next['start_date'], week_next['end_date']))
            schedule_next_week = decode_dict_values([dict(row) for row in cursor.fetchall()])

        # 5. Get Team Stats Map
        team_stats_map = {}
        cursor.execute("SELECT * FROM team_stats_summary")
        for row in cursor.fetchall(): team_stats_map[row['team_tricode']] = dict(row)
        cursor.execute("SELECT * FROM team_stats_weekly")
        for row in cursor.fetchall():
            if row['team_tricode'] in team_stats_map: team_stats_map[row['team_tricode']].update(dict(row))

        # 6. Get Base Roster (Active Players)
        active_players = _get_ranked_roster_for_week(cursor, team_id, week_num, team_stats_map, sourcing)

        # 7. Get All Players (Base + Simulated)
        cursor.execute("""
            SELECT p.player_id, p.player_name, p.player_team as team, rp.eligible_positions, p.player_name_normalized, p.status
            FROM rosters_tall r
            JOIN rostered_players rp ON r.player_id = rp.player_id
            JOIN players p ON rp.player_id = p.player_id
            WHERE r.team_id = ?
        """, (team_id,))
        all_players_raw = cursor.fetchall()
        base_roster_players = decode_dict_values([dict(row) for row in all_players_raw])

        # Create a master list for display that includes base roster + added players
        all_players = list(base_roster_players)

        if simulated_moves:
            # Logic change: We do NOT filter drops from all_players (display list).
            # We only append adds. Daily logic handles who is active when.
            existing_ids = {int(p.get('player_id', 0)) for p in all_players}

            for move in simulated_moves:
                added = move.get('added_player')

                # --- [FIX] Check if added is None (Drop-Only moves) ---
                if not added:
                    continue
                # ------------------------------------------------------

                # Map fields if they differ (free agent object vs roster object)
                if 'positions' in added and 'eligible_positions' not in added:
                    added['eligible_positions'] = added['positions']
                if 'player_team' in added and 'team' not in added:
                    added['team'] = added['player_team']

                # Only append if not already in the list (e.g. drop/add same player)
                if int(added.get('player_id', 0)) not in existing_ids:
                    all_players.append(added)
                    existing_ids.add(int(added.get('player_id', 0)))

        # 8. Fetch Stats & Ranks for EVERYONE (Base + Sim)
        cat_rank_columns = [f"{cat}_cat_rank" for cat in all_scoring_categories]
        raw_stat_columns = [f'"{cat}"' for cat in all_scoring_categories]
        all_cols = list(set(cat_rank_columns + raw_stat_columns))
        pp_stat_columns = [
            'avg_ppTimeOnIcePctPerGame', 'lg_ppTimeOnIce', 'lg_ppTimeOnIcePctPerGame',
            'lg_ppAssists', 'lg_ppGoals', 'avg_ppTimeOnIce', 'total_ppAssists',
            'total_ppGoals', 'team_games_played'
        ]

        valid_names = [p.get('player_name_normalized') for p in all_players if p.get('player_name_normalized')]
        player_stats = {}
        if valid_names:
            placeholders = ','.join('?' for _ in valid_names)
            query = f"SELECT player_name_normalized, {', '.join(all_cols + pp_stat_columns)} FROM {stat_table} j WHERE player_name_normalized IN ({placeholders})"
            cursor.execute(query, valid_names)
            player_stats = {row['player_name_normalized']: dict(row) for row in cursor.fetchall()}

        # 9. Enrich All Players (Schedule + Stats)
        player_custom_rank_map = {}
        active_player_map = {p['player_name']: p for p in active_players} # Base roster only

        for player in all_players:
            # A. Schedule & Opponents
            if player['player_name'] in active_player_map:
                # Copy pre-calculated data for existing players
                source = active_player_map[player['player_name']]
                for k in ['total_rank', 'game_dates_this_week', 'games_this_week', 'games_next_week', 'opponents_list', 'opponent_stats_this_week']:
                    player[k] = source.get(k, [])
            else:
                # --- NEW: Calculate for Simulated Players ---
                player['games_this_week'] = []
                player['game_dates_this_week'] = []
                player['games_next_week'] = []
                player['opponents_list'] = []
                player['opponent_stats_this_week'] = []

                p_team = player.get('team')
                if p_team:
                    # This Week
                    p_games = [g for g in schedule_this_week if g['home_team'] == p_team or g['away_team'] == p_team]
                    p_games.sort(key=lambda x: x['game_date'])
                    for g in p_games:
                        g_date = datetime.strptime(g['game_date'], '%Y-%m-%d').date()
                        player['games_this_week'].append(g_date.strftime('%a'))
                        player['game_dates_this_week'].append(g['game_date'])

                        opp = g['away_team'] if g['home_team'] == p_team else g['home_team']
                        player['opponents_list'].append(opp)

                        opp_stats = team_stats_map.get(opp, {})
                        player['opponent_stats_this_week'].append({
                            'game_date': g_date.strftime('%a, %b %d'),
                            'opponent_tricode': opp,
                            **{k: opp_stats.get(k) for k in ['ga_gm', 'soga_gm', 'gf_gm', 'sogf_gm', 'pk_pct']}
                        })

                    # Next Week
                    p_next = [g for g in schedule_next_week if g['home_team'] == p_team or g['away_team'] == p_team]
                    p_next.sort(key=lambda x: x['game_date'])
                    for g in p_next:
                        g_date = datetime.strptime(g['game_date'], '%Y-%m-%d').date()
                        player['games_next_week'].append(g_date.strftime('%a'))

            # B. Stats & Ranks
            p_data = player_stats.get(player.get('player_name_normalized'))
            new_total_rank = 0
            if p_data:
                for cat in all_scoring_categories:
                    # Rank
                    r_val = p_data.get(f"{cat}_cat_rank")
                    player[f"{cat}_cat_rank"] = r_val
                    if r_val is not None:
                        if cat in unchecked_categories: new_total_rank += r_val / 10.0
                        else: new_total_rank += r_val

                    # Raw Stat
                    player[cat] = p_data.get(cat)

                # PP Stats
                for col in pp_stat_columns:
                    player[col] = p_data.get(col)

            player['total_rank'] = round(new_total_rank, 2) if p_data else None
            if player.get('player_id'):
                player_custom_rank_map[int(player['player_id'])] = player['total_rank']

        # 10. Final Lineup/Usage Calc
        lineup_settings = {row['position']: row['position_count'] for row in cursor.execute("SELECT position, position_count FROM lineup_settings WHERE position NOT IN ('BN', 'IR', 'IR+')")}

        daily_optimal_lineups = {}
        player_starts_counter = Counter()

        for day_date in days_in_week:
            day_str = day_date.strftime('%Y-%m-%d')
            # Pass base_roster_players (the original DB roster) to the calculation.
            # The helper function will:
            # 1. Check base roster players against drops for this date.
            # 2. Check simulated moves for adds active on this date.
            daily_active_roster = _get_daily_simulated_roster(base_roster_players, simulated_moves, day_str)

            players_playing_today = []
            for p in daily_active_roster:
                # Check if they have a game today
                if day_str in p.get('game_dates_this_week', []):
                    # --- FIX START: Filter out IR/IR+ players ---
                    eligible_ops = (p.get('eligible_positions') or p.get('positions', '')).split(',')
                    if any(pos.strip().startswith('IR') for pos in eligible_ops):
                        continue
                    # --- FIX END ---
                    players_playing_today.append(p)

            if players_playing_today:
                optimal = get_optimal_lineup(players_playing_today, lineup_settings)
                daily_optimal_lineups[day_date.strftime('%A, %b %d')] = optimal
                for pos_players in optimal.values():
                    for player in pos_players:
                        player_starts_counter[player['player_id']] += 1

        for player in all_players:
            player['starts_this_week'] = player_starts_counter.get(player.get('player_id'), 0)

        # Pass base_roster_players here as well
        unused_roster_spots = _calculate_unused_spots(days_in_week, base_roster_players, lineup_settings, simulated_moves)

        return jsonify({
            'players': all_players,
            'daily_optimal_lineups': daily_optimal_lineups,
            'scoring_categories': all_scoring_categories,
            'skater_categories': skater_categories,
            'goalie_categories': goalie_categories,
            'lineup_settings': lineup_settings,
            'checked_categories': checked_categories,
            'unused_roster_spots': unused_roster_spots
        })

    except Exception as e:
        logging.error(f"Error fetching roster data: {e}", exc_info=True)
        return jsonify({'error': f"An error occurred: {e}"}), 500
    finally:
        if conn: conn.close()


@app.route('/api/free_agent_data', methods=['GET', 'POST'])
def get_free_agent_data():
    league_id = session.get('league_id')
    conn, error_msg = get_db_connection_for_league(league_id)

    if not conn:
        return jsonify({'error': error_msg}), 404

    try:
        cursor = conn.cursor()
        request_data = request.get_json(silent=True) or {}
        sourcing = request_data.get('sourcing', 'projected')
        cursor.execute("SELECT category, scoring_group FROM scoring ORDER BY scoring_group DESC, stat_id")
        all_categories_raw = cursor.fetchall()
        skater_categories = [row['category'] for row in all_categories_raw if row['scoring_group'] == 'offense']
        goalie_categories = [row['category'] for row in all_categories_raw if row['scoring_group'] == 'goaltending']
        all_scoring_categories = skater_categories + goalie_categories

        checked_categories = request_data.get('categories')
        if checked_categories is None:
            checked_categories = all_scoring_categories

        unchecked_categories = [cat for cat in all_scoring_categories if cat not in checked_categories]
        all_cat_rank_columns = [f"{cat}_cat_rank" for cat in all_scoring_categories]
        raw_stat_columns = [f'"{cat}"' for cat in all_scoring_categories]
        all_cols = list(set(all_cat_rank_columns + raw_stat_columns))
        selected_week_str = request_data.get('selected_week')
        target_week = None

        if selected_week_str:
            try:
                target_week = int(selected_week_str)
                cursor.execute("SELECT 1 FROM weeks WHERE week_num = ?", (target_week,))
                if not cursor.fetchone():
                    target_week = None
                    logging.warn(f"Selected week '{selected_week_str}' not found in database. Falling back to current week.")
            except ValueError:
                logging.warn(f"Invalid selected_week value: '{selected_week_str}'. Falling back to current week.")

        if target_week is None:
            today = date.today().isoformat()
            cursor.execute("SELECT week_num FROM weeks WHERE start_date <= ? AND end_date >= ?", (today, today))
            current_week_row = cursor.fetchone()
            target_week = current_week_row['week_num'] if current_week_row else 1
            logging.info(f"No valid selected week provided. Using current week: {target_week}")
        else:
            logging.info(f"Using selected week: {target_week}")

        # --- [START] MODIFICATION: Only fetch team_stats_map. Schedule is removed. ---
        team_stats_map = {}
        cursor.execute("SELECT * FROM team_stats_summary")
        for row in cursor.fetchall():
            team_stats_map[row['team_tricode']] = dict(row)

        cursor.execute("SELECT * FROM team_stats_weekly")
        for row in cursor.fetchall():
            team_tricode = row['team_tricode']
            if team_tricode in team_stats_map:
                team_stats_map[team_tricode].update(dict(row))
        # --- [END] MODIFICATION ---

        cursor.execute("SELECT player_id FROM waiver_players")
        waiver_player_ids = [row['player_id'] for row in cursor.fetchall()]
        # --- MODIFIED: Pass only team_stats_map ---
        waiver_players = _get_ranked_players(cursor, waiver_player_ids, all_cat_rank_columns, raw_stat_columns, target_week, team_stats_map, sourcing)

        cursor.execute("SELECT player_id FROM free_agents")
        free_agent_ids = [row['player_id'] for row in cursor.fetchall()]
        # --- MODIFIED: Pass only team_stats_map ---
        free_agents = _get_ranked_players(cursor, free_agent_ids, all_cat_rank_columns, raw_stat_columns, target_week, team_stats_map, sourcing)

        # Recalculate total_cat_rank
        for player_list in [waiver_players, free_agents]:
            for player in player_list:
                total_rank = 0
                for cat in all_scoring_categories:
                    rank_key = f"{cat}_cat_rank"
                    rank_value = player.get(rank_key)
                    if rank_value is not None:
                        if cat in unchecked_categories:
                            total_rank += rank_value / 2.0
                        else:
                            total_rank += rank_value
                player['total_cat_rank'] = round(total_rank, 2)

        # Calculate Unused Roster Spots
        unused_roster_spots = None
        team_ranked_roster = []
        days_in_week_data = []
        selected_team_name = request_data.get('team_name')
        simulated_moves = request_data.get('simulated_moves', [])

        if selected_team_name:
            cursor.execute("SELECT team_id FROM teams WHERE CAST(name AS TEXT) = ?", (selected_team_name,))
            team_row = cursor.fetchone()
            if team_row:
                team_id = team_row['team_id']
                cursor.execute("SELECT start_date, end_date FROM weeks WHERE week_num = ?", (target_week,))
                week_dates = cursor.fetchone()
                if week_dates:
                    start_date_obj = datetime.strptime(week_dates['start_date'], '%Y-%m-%d').date()
                    end_date_obj = datetime.strptime(week_dates['end_date'], '%Y-%m-%d').date()
                    days_in_week = [(start_date_obj + timedelta(days=i)) for i in range((end_date_obj - start_date_obj).days + 1)]

                    today_obj = date.today()
                    for day in days_in_week:
                        if day >= today_obj:
                            days_in_week_data.append(day.isoformat())

                    cursor.execute("SELECT position, position_count FROM lineup_settings WHERE position NOT IN ('BN', 'IR', 'IR+')")
                    lineup_settings = {row['position']: row['position_count'] for row in cursor.fetchall()}

                    # --- MODIFIED: Pass only team_stats_map ---
                    team_ranked_roster = _get_ranked_roster_for_week(cursor, team_id, target_week, team_stats_map, sourcing)

                    unused_roster_spots = _calculate_unused_spots(days_in_week, team_ranked_roster, lineup_settings, simulated_moves)

        return jsonify({
            'waiver_players': waiver_players,
            'free_agents': free_agents,
            'scoring_categories': all_scoring_categories,
            'skater_categories': skater_categories,
            'goalie_categories': goalie_categories,
            'ranked_categories': all_scoring_categories,
            'checked_categories': checked_categories,
            'unused_roster_spots': unused_roster_spots,
            'team_roster': [dict(p) for p in team_ranked_roster],
            'week_dates': days_in_week_data
        })

    except Exception as e:
        logging.error(f"Error fetching free agent data: {e}", exc_info=True)
        return jsonify({'error': f"An error occurred: {e}"}), 500
    finally:
        if conn:
            conn.close()


def _get_team_goalie_stats(cursor, team_id, start_date_str, end_date_str):
    # 1. Get Aggregated Live Stats
    goalie_categories = ['W', 'L', 'GA', 'SV', 'SA', 'SHO', 'TOI/G']

    cursor.execute(f"""
        SELECT category, SUM(stat_value) as total
        FROM daily_player_stats
        WHERE date_ >= ? AND date_ <= ? AND team_id = ?
        AND category IN ({','.join('?' for _ in goalie_categories)})
        GROUP BY category
    """, (start_date_str, end_date_str, team_id, *goalie_categories))

    live_stats_raw = cursor.fetchall()
    live_stats_decoded = decode_dict_values([dict(row) for row in live_stats_raw])

    live_stats = {cat: 0 for cat in goalie_categories}
    for row in live_stats_decoded:
        if row['category'] in live_stats:
            live_stats[row['category']] = row.get('total', 0)

    if 'SHO' in live_stats and live_stats['SHO'] > 0:
        live_stats['TOI/G'] += (live_stats['SHO'] * 60)

    # 2. Get Individual Goalie Starts
    cursor.execute("""
        SELECT
            d.player_id,
            p.player_name,
            d.date_,
            d.category,
            d.stat_value
        FROM daily_player_stats d
        JOIN players p ON d.player_id = p.player_id
        WHERE d.team_id = ? AND d.date_ >= ? AND d.date_ <= ?
        AND d.category IN ('W', 'L', 'GA', 'SV', 'SA', 'SHO', 'TOI/G')
        ORDER BY d.date_, p.player_name
    """, (team_id, start_date_str, end_date_str))

    raw_starts = cursor.fetchall()

    starts_data = defaultdict(lambda: defaultdict(float))
    for row in raw_starts:
        key = (row['player_id'], row['player_name'], row['date_'])
        starts_data[key][row['category']] = row['stat_value']

    individual_starts = []
    for (player_id, player_name, date_), stats in starts_data.items():
        if stats.get('SA', 0) > 0:
            start_record = {
                "player_id": player_id,
                "player_name": player_name,
                "date": date_,
                **stats
            }

            toi = stats.get('TOI/G', 0)
            if stats.get('SHO', 0) > 0:
                toi += 60
                start_record['TOI/G'] = toi

            start_record['GAA'] = (stats.get('GA', 0) * 60) / toi if toi > 0 else 0
            start_record['SV%'] = stats.get('SV', 0) / stats.get('SA', 0) if stats.get('SA', 0) > 0 else 0

            individual_starts.append(start_record)

    goalie_starts = len(individual_starts)

    return {
        'live_stats': live_stats,
        'goalie_starts': goalie_starts,
        'individual_starts': individual_starts
    }


@app.route('/api/goalie_planning_stats', methods=['POST'])
@requires_premium
def get_goalie_planning_stats():
    league_id = session.get('league_id')
    data = request.get_json()
    week_num = data.get('week')
    your_team_name = data.get('your_team_name')
    opponent_team_name = data.get('opponent_team_name')

    conn, error_msg = get_db_connection_for_league(league_id)
    if not conn:
        return jsonify({'error': error_msg}), 404

    try:
        cursor = conn.cursor()

        # Get Team IDs
        cursor.execute("SELECT team_id FROM teams WHERE CAST(name AS TEXT) = ?", (your_team_name,))
        your_team_id_row = cursor.fetchone()

        cursor.execute("SELECT team_id FROM teams WHERE CAST(name AS TEXT) = ?", (opponent_team_name,))
        opponent_team_id_row = cursor.fetchone()

        if not your_team_id_row:
            return jsonify({'error': f'Team not found: {your_team_name}'}), 404
        if not opponent_team_id_row:
            return jsonify({'error': f'Team not found: {opponent_team_name}'}), 404

        your_team_id = your_team_id_row['team_id']
        opponent_team_id = opponent_team_id_row['team_id']

        # Get week dates
        cursor.execute("SELECT start_date, end_date FROM weeks WHERE week_num = ?", (week_num,))
        week_dates = cursor.fetchone()
        if not week_dates:
            return jsonify({'error': f'Week not found: {week_num}'}), 404
        start_date_str = week_dates['start_date']
        end_date_str = week_dates['end_date']

        # Get stats for both teams using the helper
        your_team_stats = _get_team_goalie_stats(cursor, your_team_id, start_date_str, end_date_str)
        opponent_team_stats = _get_team_goalie_stats(cursor, opponent_team_id, start_date_str, end_date_str)

        return jsonify({
            'your_team_stats': your_team_stats,
            'opponent_team_stats': opponent_team_stats
        })

    except Exception as e:
        logging.error(f"Error fetching goalie planning stats: {e}", exc_info=True)
        return jsonify({'error': f"An error occurred: {e}"}), 500
    finally:
        if conn:
            conn.close()


@app.route('/stream')
def stream():
    def event_stream():
        while True:
            message = log_queue.get()
            if message is None:
                break
            yield f"data: {message}\n\n"
    return Response(event_stream(), mimetype='text/event-stream')

#def update_db_in_background(yq, lg, league_id, data_dir, capture_lineups, skip_static_info, skip_available_players):
def update_db_in_background(yq, lg, league_id, data_dir, capture_lineups):
    """Function to run in a separate thread."""
    try:
        db_builder.update_league_db(
            yq, lg, league_id, data_dir,
            capture_lineups=capture_lineups#,
#            skip_static_info=skip_static_info,
#            skip_available_players=skip_available_players
        )
        log_queue.put("SUCCESS: Database update complete.")
    except Exception as e:
        logging.error(f"Error in background DB update: {e}", exc_info=True)
        log_queue.put(f"ERROR: {e}")
    finally:
        # Signal the end of the stream
        log_queue.put(None)

@app.route('/api/update_db', methods=['POST'])
def update_db_route():
    if session.get('dev_mode'):
        return jsonify({'success': False, 'error': 'Database updates are disabled in dev mode.'}), 403

    yq = get_yfpy_instance()
    lg = get_yfa_lg_instance()
    if not yq or not lg:
        return jsonify({"error": "Authentication failed. Please log in again."}), 401

    league_id = session.get('league_id')
    if not league_id:
        return jsonify({'success': False, 'error': 'League ID not found in session.'}), 400

    data = request.get_json() or {}
    capture_lineups = data.get('capture_lineups', False)
#    skip_static_info = data.get('skip_static_info', False)
#    skip_available_players = data.get('skip_available_players', False)

    # Run the database update in a background thread
    thread = threading.Thread(
        target=update_db_in_background,
        args=(yq, lg, league_id, DATA_DIR, capture_lineups)#, skip_static_info, skip_available_players)
    )
    thread.start()

    return jsonify({'success': True, 'message': 'Database update started.'})


@app.route('/api/download_db')
def download_db():
    if session.get('use_test_db'):
        logging.info(f"Downloading test database: {TEST_DB_FILENAME}")
        if not os.path.exists(TEST_DB_PATH):
            return jsonify({'error': 'Test database file not found in /server directory.'}), 404
        return send_from_directory(SERVER_DIR, TEST_DB_FILENAME, as_attachment=True)

    league_id = session.get('league_id')
    if not league_id:
        return jsonify({'error': 'Not logged in or session expired.'}), 401

    # --- MODIFICATION START: GCS Download Logic ---
    if not gcs_bucket:
        logging.error("GCS_BUCKET_NAME not set or GCS client failed to init.")
        return jsonify({'error': 'GCS is not configured on the server.'}), 500

    db_filename_prefix = f'yahoo-{league_id}-'
    remote_path_prefix = f'league-dbs/{db_filename_prefix}'

    logging.info(f"Searching GCS bucket for download: {remote_path_prefix}")

    blob_to_download = None
    db_filename = None
    try:
        # Find the database file in GCS
        for blob in gcs_bucket.list_blobs(prefix=remote_path_prefix):
            blob_to_download = blob
            db_filename = blob.name.split('/')[-1]  # Get the actual filename
            break  # Found the first match
    except Exception as e:
        logging.error(f"Error listing GCS blobs: {e}", exc_info=True)
        return jsonify({'error': 'Could not search GCS for the file.'}), 500

    if not blob_to_download or not db_filename:
        logging.warning(f"No DB file found in GCS with prefix: {remote_path_prefix}")
        return jsonify({'error': 'Database file not found in cloud storage. Please run a build.'}), 404

    try:
        logging.info(f"Streaming file {db_filename} from GCS...")
        # Download the blob's content as bytes
        file_bytes = blob_to_download.download_as_bytes()

        # Create a Flask Response to stream the file to the user
        return Response(
            file_bytes,
            mimetype='application/octet-stream',  # A generic type for file downloads
            headers={
                "Content-Disposition": f"attachment; filename={db_filename}",
                "Content-Length": len(file_bytes)
            }
        )
    except Exception as e:
        logging.error(f"Error downloading/streaming blob {db_filename}: {e}", exc_info=True)
        return jsonify({'error': 'An error occurred while trying to download the file from GCS.'}), 500
    # --- MODIFICATION END ---

@app.route('/api/settings', methods=['GET', 'POST'])
def api_settings():
    if request.method == 'GET':
        return jsonify({
            'use_test_db': session.get('use_test_db', False),
            'test_db_exists': os.path.exists(TEST_DB_PATH)
        })
    elif request.method == 'POST':
        data = request.get_json()
        use_test_db = data.get('use_test_db', False)

        # Dev mode forces the test DB on, don't let it be turned off
        if session.get('dev_mode'):
             session['use_test_db'] = True
        else:
            session['use_test_db'] = use_test_db

        logging.info(f"Test DB mode set to: {session['use_test_db']}")
        return jsonify({'success': True, 'use_test_db': session['use_test_db']})

@app.route('/api/db_timestamp')
def db_timestamp():
    league_id = session.get('league_id')
    conn, error_msg = get_db_connection_for_league(league_id)
    if not conn:
        return jsonify({'error': error_msg or "Database not found."}), 404

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM db_metadata WHERE key = 'last_updated_timestamp'")
        row = cursor.fetchone()
        timestamp = row['value'] if row else None
        return jsonify({'timestamp': timestamp})
    except Exception as e:
        logging.error(f"Error fetching timestamp: {e}", exc_info=True)
        return jsonify({'error': 'Could not retrieve timestamp.'}), 500
    finally:
        if conn:
            conn.close()

@app.route('/api/available_players_timestamp')
def available_players_timestamp():
    league_id = session.get('league_id')
    conn, error_msg = get_db_connection_for_league(league_id)
    if not conn:
        return jsonify({'error': error_msg or "Database not found."}), 404

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM db_metadata WHERE key = 'available_players_last_updated_timestamp'")
        row = cursor.fetchone()
        timestamp = row['value'] if row else None
        return jsonify({'timestamp': timestamp})
    except Exception as e:
        logging.error(f"Error fetching available players timestamp: {e}", exc_info=True)
        return jsonify({'error': 'Could not retrieve timestamp.'}), 500
    finally:
        if conn:
            conn.close()

@app.route('/pages/<path:page_name>')
def serve_page(page_name):
    return render_template(f"pages/{page_name}")

@app.route('/api/db_status')
def db_status():
    if session.get('use_test_db'):
        db_exists = os.path.exists(TEST_DB_PATH)
        timestamp = os.path.getmtime(TEST_DB_PATH) if db_exists else None
        return jsonify({
            'db_exists': db_exists,
            'league_name': f"TEST DB: {TEST_DB_FILENAME}",
            'timestamp': int(timestamp) if timestamp else None,
            'is_test_db': True
        })

    league_id = session.get('league_id')
    if not league_id:
        return jsonify({'db_exists': False, 'error': 'Not logged in.', 'is_test_db': False})

    db_path = None
    league_name = "[Unknown]"
    timestamp = None
    db_exists = False
    db_filename = None

    for filename in os.listdir(DATA_DIR):
        if filename.startswith(f"yahoo-{league_id}-") and filename.endswith(".db"):
            db_path = os.path.join(DATA_DIR, filename)
            db_exists = True
            db_filename = filename
            break

    if db_exists:
        try:
            match = re.search(f"yahoo-{league_id}-(.*)\\.db", db_filename)
            if match:
                league_name = match.group(1)
            timestamp = os.path.getmtime(db_path)
        except Exception as e:
            logging.error(f"Could not parse DB file info: {e}")
            return jsonify({'db_exists': False, 'error': 'Could not read database file details.', 'is_test_db': False})

    return jsonify({
        'db_exists': db_exists,
        'league_name': league_name,
        'timestamp': int(timestamp) if timestamp else None,
        'is_test_db': False
    })


redis_conn = redis.from_url(os.environ.get('REDIS_URL'))
job_queue = Queue('default', connection=redis_conn)

@app.route('/api/db_action', methods=['POST'])
def db_action():
    if not session.get('yahoo_token'):
        return jsonify({'error': 'Not authenticated'}), 401

    league_id = session.get('league_id')
    if not league_id:
        return jsonify({'error': 'No league selected'}), 400

    global db_build_status
    with db_build_status_lock:
        active_job_id = db_build_status.get("current_build_id")
        if active_job_id:
            try:
                job = job_queue.fetch_job(active_job_id)
                if job and job.get_status() not in ['finished', 'failed', 'canceled']:
                    # --- MODIFIED: Use logging ---
                    logging.warning(f"Build {active_job_id} already running with status: {job.get_status()}")
                    # --- END MODIFIED ---
                    return jsonify({
                        'error': 'A build is already in progress.',
                        'build_id': active_job_id
                    }), 409
            except Exception as e:
                # --- MODIFIED: Use logging ---
                logging.warning(f"Could not fetch job {active_job_id}, proceeding. Error: {e}")
                # --- END MODIFIED ---

        build_id = str(uuid.uuid4())
        log_file_path = os.path.join(tempfile.gettempdir(), f"{build_id}.log")

        db_build_status = {"running": True, "error": None, "current_build_id": build_id}

    data = request.get_json()
    options = {
        'capture_lineups': data.get('capture_lineups', False),
        'roster_updates_only': data.get('roster_updates_only', False),
        'skip_static': data.get('skip_static', False),
        'skip_players': data.get('skip_players', False)
    }

    thread_data = {
        "league_id": league_id,
        "token": session.get('yahoo_token'),
        "consumer_key": session.get('consumer_key'),
        "consumer_secret": session.get('consumer_secret'),
        "dev_mode": session.get('dev_mode', False)
    }

    try:
        # --- MODIFIED: Use logging ---
        logging.info(f"Enqueuing job {build_id} to db_builder.run_task")
        # --- END MODIFIED ---
        job = job_queue.enqueue(
            'db_builder.run_task',
            args=(build_id, log_file_path, options, thread_data),
            job_id=build_id,
            job_timeout=1800
        )
        return jsonify({'success': True, 'build_id': job.id})

    except Exception as e:
        # --- MODIFIED: Use logging ---
        logging.error(f"Failed to enqueue job: {e}", exc_info=True)
        # --- END MODIFIED ---
        with db_build_status_lock:
            db_build_status = {"running": False, "error": str(e), "current_build_id": None}
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/db_log_stream')
def db_log_stream():
    build_id = request.args.get('build_id')
    if not build_id:
        return Response("data: ERROR: No build_id provided.\n\ndata: __DONE__\n\n", mimetype='text/event-stream')

    def generate():
        try:
            job = job_queue.fetch_job(build_id)
            if not job:
                yield f"data: ERROR: Job {build_id} not found in queue.\n\n"
                yield 'data: __DONE__\n\n'
                return

            yield f"data: Job {build_id} found. Waiting for worker...\n\n"

            # --- MODIFIED: Added a counter ---
            running_counter = 0

            while True:
                job.refresh()
                status = job.get_status()

                if status == 'queued':
                    yield "data: Job is in the queue, waiting for a worker...\n\n"
                elif status == 'started':
                    # --- MODIFIED: Increment counter and update message ---
                    running_counter += 1
                    yield f"data: Job is running... (Check {running_counter})\n\n"
                elif status == 'finished':
                    yield "data: Job finished successfully.\n\n"
                    yield "data: __DONE__\n\n"
                    break
                elif status == 'failed':
                    yield "data: --- JOB FAILED ---\n\n"
                    yield f"data: ERROR: {job.exc_info}\n\n"
                    yield "data: __DONE__\n\n"
                    break

                # --- MODIFIED: Slowed down the polling interval ---
                time.sleep(10)

        except Exception as e:
            logging.error(f"Error in log stream generator for {build_id}: {e}")
            yield f"data: ERROR: Log stream failed. {e}\n\n"
            yield 'data: __DONE__\n\n'

    return Response(generate(), mimetype='text/event-stream')

#MOBILE ROUTES






if __name__ == '__main__':
    os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
    app.run(debug=True, port=5001)
