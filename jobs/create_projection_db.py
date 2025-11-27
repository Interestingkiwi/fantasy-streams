"""
Processes and joins data in the fantasy hockey database.
Refactored for PostgreSQL: Logic Preserved, Storage Changed.

Author: Jason Druckenmiller
Created: 11/02/2025
Updated: 11/26/2025
"""

import pandas as pd
import sys
import os
import re
import csv
import json
import requests
import time
import unicodedata
from datetime import date, timedelta
from collections import defaultdict, Counter
import numpy as np

# Add parent dir to path so we can import database
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import get_db_connection

# --- Constants ---
SEED_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "seed_data")

PROJ1_SKATER_FILE = os.path.join(SEED_DATA_DIR, 'proj1s.csv')
PROJ1_GOALIE_FILE = os.path.join(SEED_DATA_DIR, 'proj1g.csv')
PROJ2_SKATER_FILE = os.path.join(SEED_DATA_DIR, 'proj2s.csv')
PROJ2_GOALIE_FILE = os.path.join(SEED_DATA_DIR, 'proj2g.csv')

START_DATE = date(2025, 10, 7)
END_DATE = date(2026, 4, 17)
NHL_TEAM_COUNT = 32

TEAM_TRICODE_MAP = {
    "TB": "TBL", "NJ": "NJD", "SJ": "SJS", "LA": "LAK", "T.B": "TBL",
    "N.J": "NJD", "S.J": "SJS", "L.A": "LAK", "MON": "MTL", "WAS": "WSH"
}

# --- Helper Functions (Logic Preserved) ---

def normalize_name(name):
    if not name: return ""
    nfkd_form = unicodedata.normalize('NFKD', name.lower())
    ascii_name = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    return re.sub(r'[^a-z0-9]', '', ascii_name)

def sanitize_header(header_list):
    sanitized = []
    stat_mapping = {
        'goals': 'G', 'g':'G', 'assists': 'A', 'a':'A', 'points': 'P', 'p':'P',
        'pp_points': 'PPP', 'ppp':'PPP', 'hits': 'HIT', 'hit':'HIT', 'sog': 'SOG',
        'blk': 'BLK', 'w': 'W', 'so': 'SHO', 'sho':'SHO', 'sv%': 'SVpct', 'svpct': 'SVpct',
        'ga': 'GA', 'plus_minus': 'plus_minus', 'shg': 'SHG', 'sha': 'SHA', 'shp': 'SHP',
        'pim': 'PIM', 'fow': 'FOW', 'fol': 'FOL', 'ppg': 'PPG', 'ppa': 'PPA',
        'gaa': 'GAA', 'gs': 'GS', 'sv': 'SV', 'sa': 'SA', 'qs': 'QS', 'l':'L'
    }
    for h in header_list:
        clean_h = h.strip().lower()
        if clean_h == '"+/-"': clean_h = 'plus_minus'
        else: clean_h = re.sub(r'[^a-z0-9_%]', '', clean_h.replace(' ', '_'))
        sanitized.append(stat_mapping.get(clean_h, clean_h))
    return sanitized

def calculate_per_game_stats(row, gp_index, stat_indices):
    try:
        games_played = float(row[gp_index])
    except (ValueError, IndexError):
        games_played = 0.0

    if games_played == 0:
        for i in stat_indices:
            if i < len(row): row[i] = 0.0
        return row

    for i in stat_indices:
        if i < len(row):
            try:
                stat_value = float(row[i])
                row[i] = round(stat_value / games_played, 4)
            except (ValueError, IndexError, TypeError):
                row[i] = 0.0
    return row

def calculate_and_add_category_ranks(player_data):
    new_rank_columns = []
    # Skaters
    skater_stats_to_rank = [
        'G', 'A', 'P', 'PPG', 'PPA', 'PPP', 'SHG', 'SHA', 'SHP',
        'HIT', 'BLK', 'PIM', 'FOW', 'SOG', 'plus_minus'
    ]
    skaters = {name: data for name, data in player_data.items() if 'G' not in data.get('positions', '')}
    num_skaters = len(skaters)

    if num_skaters > 0:
        for stat in skater_stats_to_rank:
            new_col_name = f"{stat}_cat_rank"
            new_rank_columns.append(new_col_name)
            stat_values = []
            for name, data in skaters.items():
                try: value = float(data.get(stat, 0.0))
                except: value = 0.0
                stat_values.append((name, value))
            stat_values.sort(key=lambda x: x[1], reverse=True)
            for i, (name, value) in enumerate(stat_values):
                percentile = (i + 1) / num_skaters
                # Logic Preserved:
                if percentile <= 0.05: rank = 1
                elif percentile <= 0.10: rank = 2
                elif percentile <= 0.15: rank = 3
                elif percentile <= 0.20: rank = 4
                elif percentile <= 0.25: rank = 5
                elif percentile <= 0.30: rank = 6
                elif percentile <= 0.35: rank = 7
                elif percentile <= 0.40: rank = 8
                elif percentile <= 0.45: rank = 9
                elif percentile <= 0.50: rank = 10
                elif percentile <= 0.75: rank = 15
                else: rank = 20
                player_data[name][new_col_name] = rank

    # Goalies
    goalie_stats_to_rank = {
        'GS': False, 'W': False, 'L': True, 'GA': True, 'SA': False,
        'SV': False, 'SVpct': False, 'GAA': True, 'SHO': False, 'QS': False
    }
    goalies = {name: data for name, data in player_data.items() if 'G' in data.get('positions', '')}
    num_goalies = len(goalies)

    if num_goalies > 0:
        for stat, is_inverse in goalie_stats_to_rank.items():
            new_col_name = f"{stat}_cat_rank"
            new_rank_columns.append(new_col_name)
            stat_values = []
            for name, data in goalies.items():
                try: value = float(data.get(stat, 0.0))
                except: value = 0.0
                stat_values.append((name, value))
            stat_values.sort(key=lambda x: x[1], reverse=not is_inverse)
            for i, (name, value) in enumerate(stat_values):
                percentile = (i + 1) / num_goalies
                # Logic Preserved
                if percentile <= 0.05: rank = 1
                elif percentile <= 0.10: rank = 2
                elif percentile <= 0.15: rank = 3
                elif percentile <= 0.20: rank = 4
                elif percentile <= 0.25: rank = 5
                elif percentile <= 0.30: rank = 6
                elif percentile <= 0.35: rank = 7
                elif percentile <= 0.40: rank = 8
                elif percentile <= 0.45: rank = 9
                elif percentile <= 0.50: rank = 10
                elif percentile <= 0.75: rank = 15
                else: rank = 20
                player_data[name][new_col_name] = rank

    return player_data, new_rank_columns


def df_to_postgres(df, table_name, conn, if_exists='replace'):
    """
    Writes DataFrame to Postgres using efficient COPY or INSERT.
    Replaces df.to_sql which requires SQLAlchemy.
    """
    cursor = conn.cursor()

    if if_exists == 'replace':
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")

    # Generate CREATE TABLE
    cols = []
    for col, dtype in df.dtypes.items():
        pg_type = 'TEXT'
        if pd.api.types.is_integer_dtype(dtype): pg_type = 'INTEGER'
        elif pd.api.types.is_float_dtype(dtype): pg_type = 'REAL'

        # Specific primary key logic if known
        if col == 'player_name_normalized':
            cols.append(f"{col} TEXT PRIMARY KEY")
        else:
            cols.append(f'"{col}" {pg_type}')

    create_sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(cols)})"
    cursor.execute(create_sql)

    # Insert Data
    columns = list(df.columns)
    placeholders = ",".join(["%s"] * len(columns))
    col_names = ",".join([f'"{c}"' for c in columns])

    insert_sql = f"INSERT INTO {table_name} ({col_names}) VALUES ({placeholders}) ON CONFLICT DO NOTHING"

    # Convert to list of tuples, handling NaNs
    data = [tuple(None if pd.isna(x) else x for x in row) for row in df.to_numpy()]

    cursor.executemany(insert_sql, data)
    conn.commit()


def process_separate_files_to_table(cursor, skater_csv_file, goalie_csv_file, target_table_name):
    """ Reads CSVs, calculates stats, and upserts to Postgres table. """
    print(f"\n--- Processing {target_table_name} ---")
    player_data = {}

    # 1. Skaters
    if os.path.exists(skater_csv_file):
        print(f"Reading {skater_csv_file}...")
        with open(skater_csv_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            header_raw = next(reader)
            header_lower = [h.strip().lower() for h in header_raw]
            headers_sanitized = sanitize_header(header_raw)

            try:
                p_name_idx = header_lower.index('player name')
                gp_idx = header_lower.index('gp')
                pos_idx = header_lower.index('positions')
            except ValueError as e: raise ValueError(f"Missing column in {skater_csv_file}: {e}")

            stats_exclude = ['player name', 'age', 'positions', 'team', 'salary', 'gp org', 'gp', 'toi org es', 'toi org pp', 'toi org pk', 'toi es', 'toi pp', 'toi pk', 'total toi', 'rank', 'playerid', 'fantasy team']
            stat_indices = [i for i, h in enumerate(header_lower) if h not in stats_exclude and h.strip() != '']

            for row in reader:
                if not row or (pos_idx < len(row) and 'G' in row[pos_idx]): continue
                calculate_per_game_stats(row, gp_idx, stat_indices)
                player_name = row[p_name_idx]
                if not player_name: continue
                norm = normalize_name(player_name)
                data_dict = {headers_sanitized[i]: val for i, val in enumerate(row)}

                team = data_dict.get('team', '').upper()
                data_dict['team'] = TEAM_TRICODE_MAP.get(team, team)

                player_data[norm] = data_dict
                player_data[norm]['player_name_normalized'] = norm

    # 2. Goalies
    if os.path.exists(goalie_csv_file):
        print(f"Reading {goalie_csv_file}...")
        with open(goalie_csv_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            header_raw = next(reader)
            header_lower = [h.strip().lower() for h in header_raw]
            headers_sanitized = sanitize_header(header_raw)

            try:
                p_name_idx = header_lower.index('player name')
                gp_idx = header_lower.index('gs')
            except ValueError as e: raise ValueError(f"Missing column in {goalie_csv_file}: {e}")

            stats_exclude = ['player name', 'team', 'age', 'position', 'salary', 'gs', 'sv%', 'gaa', 'rank', 'playerid', 'fantasy team']
            stat_indices = [i for i, h in enumerate(header_lower) if h not in stats_exclude and h.strip() != '']

            for row in reader:
                if not row: continue
                calculate_per_game_stats(row, gp_idx, stat_indices)
                player_name = row[p_name_idx]
                if not player_name: continue
                norm = normalize_name(player_name)
                data_dict = {headers_sanitized[i]: val for i, val in enumerate(row)}

                team = data_dict.get('team', '').upper()
                data_dict['team'] = TEAM_TRICODE_MAP.get(team, team)
                if 'positions' not in data_dict: data_dict['positions'] = 'G'

                if norm in player_data:
                    player_data[norm].update(data_dict)
                else:
                    data_dict['player_name_normalized'] = norm
                    player_data[norm] = data_dict

    # 3. Ranks
    player_data, new_rank_cols = calculate_and_add_category_ranks(player_data)

    # 4. Create Table (Postgres Syntax)
    # Need to merge sanitized headers from both files
    # For simplicity, I'll extract all keys from player_data since they include sanitized + ranks
    if not player_data: return

    all_keys = set()
    for p in player_data.values():
        all_keys.update(p.keys())

    final_headers = list(all_keys)
    # Ensure required columns are present for types
    if 'player_name_normalized' in final_headers: final_headers.remove('player_name_normalized')

    cols_def = []
    cols_def.append('player_name_normalized TEXT PRIMARY KEY')

    # Map types
    for col in final_headers:
        if col in ['player_name', 'positions', 'team', 'playerid', 'fantasy_team']:
            cols_def.append(f'"{col}" TEXT')
        else:
            cols_def.append(f'"{col}" DOUBLE PRECISION')

    cursor.execute(f"DROP TABLE IF EXISTS {target_table_name}")
    cursor.execute(f"CREATE TABLE {target_table_name} ({', '.join(cols_def)})")

    # 5. Insert
    insert_headers = ['player_name_normalized'] + final_headers
    placeholders = ", ".join(['%s'] * len(insert_headers))
    # Using double quotes for column names to handle cases like "plus_minus" or special chars
    col_list = ", ".join(f'"{h}"' for h in insert_headers)

    insert_sql = f'INSERT INTO {target_table_name} ({col_list}) VALUES ({placeholders})'

    rows_to_insert = []
    for norm, data in player_data.items():
        # Ensure None is passed for missing keys
        rows_to_insert.append(tuple(data.get(h, None) for h in insert_headers))

    cursor.executemany(insert_sql, rows_to_insert)
    print(f"Populated {target_table_name} with {len(rows_to_insert)} rows.")

def create_averaged_projections(conn, cursor):
    print("\n--- Creating Final Averaged Projections ---")
    # Read into Pandas directly from Postgres
    df1 = pd.read_sql_query("SELECT * FROM proj1", conn)
    df2 = pd.read_sql_query("SELECT * FROM proj2", conn)

    merged = pd.merge(df1, df2, on='player_name_normalized', how='outer', suffixes=('_p1', '_p2'))

    final = pd.DataFrame()
    final['player_name_normalized'] = merged['player_name_normalized']

    COALESCE_COLS = ['player_name', 'positions', 'team', 'playerid', 'fantasy_team']
    for col in COALESCE_COLS:
        c1, c2 = f'{col}_p1', f'{col}_p2'
        if c1 in merged: final[col] = merged[c1].fillna(merged.get(c2))
        elif c2 in merged: final[col] = merged[c2]

    # Handle nhlplayerid logic (from your file)
    if 'playerid_p2' in merged:
        final['nhlplayerid'] = pd.to_numeric(merged['playerid_p2'], errors='coerce').astype('Int64')
    elif 'playerid' in merged.columns and 'playerid_p1' not in merged.columns:
        final['nhlplayerid'] = pd.to_numeric(merged['playerid'], errors='coerce').astype('Int64')
    else:
        final['nhlplayerid'] = pd.NA

    # Average Stats
    ignore = set(COALESCE_COLS + ['player_name_normalized', 'nhlplayerid'])
    stat_cols = (set(df1.columns) | set(df2.columns)) - ignore

    for col in stat_cols:
        c1, c2 = f'{col}_p1', f'{col}_p2'
        in1, in2 = c1 in merged, c2 in merged

        if in1 and in2:
            val1 = pd.to_numeric(merged[c1], errors='coerce')
            val2 = pd.to_numeric(merged[c2], errors='coerce')
            final[col] = pd.concat([val1, val2], axis=1).mean(axis=1)
        elif in1:
            final[col] = pd.to_numeric(merged[c1], errors='coerce')
        elif in2:
            final[col] = pd.to_numeric(merged[c2], errors='coerce')

    # Save to Postgres
    # Use 'replace' to let Pandas/SQLAlchemy handle the CREATE TABLE schema generation
    df_to_postgres(final, 'projections', conn)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_proj_norm ON projections(player_name_normalized)")

def join_yahoo_ids(conn, cursor):
    print("\n--- Joining Yahoo Player ID Data ---")
    df_proj = pd.read_sql_query("SELECT * FROM projections", conn)

    # Read directly from GLOBAL players table
    df_yahoo = pd.read_sql_query("SELECT player_name_normalized, player_id, positions, status FROM players", conn)

    if 'positions' in df_proj.columns: df_proj = df_proj.drop(columns=['positions'])

    df_final = pd.merge(df_proj, df_yahoo, on='player_name_normalized', how='left')

    if 'nhlplayerid' in df_final.columns:
        df_final['nhlplayerid'] = pd.to_numeric(df_final['nhlplayerid'], errors='coerce').astype('Int64')

    # Overwrite table
    df_to_postgres(df_final, 'projections', conn)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_proj_norm ON projections(player_name_normalized)")

def get_full_nhl_schedule(start_date, end_date):
    all_games = {}
    curr = start_date
    print("Fetching NHL Schedule...")

    while curr <= end_date:
        url = f"https://api-web.nhle.com/v1/schedule/{curr.strftime('%Y-%m-%d')}"
        try:
            resp = requests.get(url, timeout=10)
            if resp.ok:
                data = resp.json()
                for week in data.get('gameWeek', []):
                    gdate = week.get('date')
                    for g in week.get('games', []):
                        home = g.get('homeTeam', {}).get('abbrev')
                        away = g.get('awayTeam', {}).get('abbrev')
                        key = f"{gdate}-{home}-{away}"
                        all_games[key] = {'date': gdate, 'home_team': home, 'away_team': away}
        except Exception as e:
            print(f"Error fetching schedule: {e}")

        curr += timedelta(days=7)
        time.sleep(0.1)

    return list(all_games.values())

def setup_schedule_tables(cursor, games):
    if not games: return

    cursor.execute("DROP TABLE IF EXISTS schedule")
    cursor.execute("CREATE TABLE schedule (game_id SERIAL PRIMARY KEY, game_date TEXT, home_team TEXT, away_team TEXT)")
    cursor.executemany("INSERT INTO schedule (game_date, home_team, away_team) VALUES (%s, %s, %s)",
                       [(g['date'], g['home_team'], g['away_team']) for g in games])

    cursor.execute("DROP TABLE IF EXISTS team_schedules")
    cursor.execute("CREATE TABLE team_schedules (team_tricode TEXT PRIMARY KEY, schedule_json TEXT)")
    sched_map = defaultdict(list)
    for g in games:
        sched_map[g['home_team']].append(g['date'])
        sched_map[g['away_team']].append(g['date'])

    cursor.executemany("INSERT INTO team_schedules VALUES (%s, %s)",
                       [(t, json.dumps(sorted(d))) for t, d in sched_map.items()])

    cursor.execute("DROP TABLE IF EXISTS off_days")
    cursor.execute("CREATE TABLE off_days (off_day_date TEXT PRIMARY KEY)")
    counts = Counter(g['date'] for g in games)
    off_days = [(d,) for d, c in counts.items() if c * 4 < NHL_TEAM_COUNT]
    cursor.executemany("INSERT INTO off_days VALUES (%s)", sorted(off_days))

def run():
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            process_separate_files_to_table(cursor, PROJ1_SKATER_FILE, PROJ1_GOALIE_FILE, 'proj1')
            process_separate_files_to_table(cursor, PROJ2_SKATER_FILE, PROJ2_GOALIE_FILE, 'proj2')
            create_averaged_projections(conn, cursor)
            join_yahoo_ids(conn, cursor)
            games = get_full_nhl_schedule(START_DATE, END_DATE)
            setup_schedule_tables(cursor, games)
            conn.commit()

if __name__ == "__main__":
    run()
