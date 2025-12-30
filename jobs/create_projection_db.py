"""
Processes and joins data in the fantasy hockey database.
Refactored for PostgreSQL.
"""

import pandas as pd
import sys
import os
import re
import csv
import unicodedata
import requests
import time
import json
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

# --- Helper Functions ---

def normalize_name(name):
    if not name: return ""
    nfkd = unicodedata.normalize('NFKD', name.lower())
    ascii_name = "".join([c for c in nfkd if not unicodedata.combining(c)])
    return re.sub(r'[^a-z0-9]', '', ascii_name)

def sanitize_header(header_list):
    sanitized = []
    stat_mapping = {
        'goals': 'G', 'g':'G', 'assists': 'A', 'a':'A', 'points': 'P', 'p':'P',
        'pp_points': 'PPP', 'ppp':'PPP', 'pp_goals': 'PPG', 'pp_assists': 'PPA',
        'hits': 'HIT', 'hit':'HIT', 'sog': 'SOG',
        'blk': 'BLK', 'w': 'W', 'so': 'SHO', 'sho':'SHO', 'sv%': 'SVpct', 'svpct': 'SVpct',
        'ga': 'GA', 'plus_minus': 'plus_minus',
        'shg': 'SHG', 'sha': 'SHA', 'shp': 'SHP',
        'sh_goals': 'SHG', 'sh_assists': 'SHA', 'sh_points': 'SHP',
        'pim': 'PIM', 'fow': 'FW', 'fol': 'FOL', 'ppg': 'PPG', 'ppa': 'PPA',
        'gaa': 'GAA', 'gs': 'GS', 'sv': 'SV', 'sa': 'SA', 'qs': 'QS', 'l':'L',
        'toi/g': 'TOI/G', 'timeonice': 'TOI/G', 'total_toi': 'Total TOI', 'gwg': 'GWG'
    }
    for h in header_list:
        clean_h = h.strip().lower()

        if clean_h == '"+/-"':
            clean_h = 'plus_minus'

        if clean_h in stat_mapping:
            sanitized.append(stat_mapping[clean_h])
            continue

        clean_h = clean_h.replace('%', 'pct').replace(' ', '_')
        clean_h = re.sub(r'[^a-z0-9_]', '', clean_h)

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
    skater_stats_to_rank = ['G', 'A', 'P', 'PPG', 'PPA', 'PPP', 'SHG', 'SHA', 'SHP', 'HIT', 'BLK', 'PIM', 'FW', 'SOG', 'plus_minus', 'GWG']

    skaters = {
        name: data for name, data in player_data.items()
        if 'G' not in str(data.get('positions', ''))
    }
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

    goalie_stats_to_rank = {'GS': False, 'W': False, 'L': True, 'GA': True, 'SA': False, 'SV': False, 'SVpct': False, 'GAA': True, 'SHO': False, 'QS': False}

    goalies = {
        name: data for name, data in player_data.items()
        if 'G' in str(data.get('positions', ''))
    }
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

# --- DB HELPER ---

def read_sql_postgres(query, conn):
    with conn.cursor() as cursor:
        cursor.execute(query)
        if cursor.description:
            columns = [desc[0] for desc in cursor.description]
            data = cursor.fetchall()
            return pd.DataFrame(data, columns=columns)
        return pd.DataFrame()

def df_to_postgres(df, table_name, conn, if_exists='replace'):
    cursor = conn.cursor()
    if if_exists == 'replace':
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")

    cols = []
    for col, dtype in df.dtypes.items():
        if not col or col.strip() == "": continue

        pg_type = 'TEXT'
        if pd.api.types.is_integer_dtype(dtype): pg_type = 'INTEGER'
        elif pd.api.types.is_float_dtype(dtype): pg_type = 'DOUBLE PRECISION'

        if col == 'player_name_normalized':
            cols.append(f"{col} TEXT PRIMARY KEY")
        else:
            cols.append(f'"{col}" {pg_type}')

    cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(cols)})")

    valid_cols = [c for c in df.columns if c and c.strip() != ""]
    df_clean = df[valid_cols]

    columns = list(df_clean.columns)
    placeholders = ",".join(["%s"] * len(columns))
    col_names = ",".join([f'"{c}"' for c in columns])

    data = [tuple(None if pd.isna(x) else x for x in row) for row in df_clean.to_numpy()]

    cursor.executemany(f"INSERT INTO {table_name} ({col_names}) VALUES ({placeholders}) ON CONFLICT DO NOTHING", data)
    conn.commit()

# --- Processing Functions ---

def process_separate_files_to_table(cursor, skater_csv_file, goalie_csv_file, target_table_name):
    print(f"\n--- Processing {target_table_name} ---")
    player_data = {}

    # 1. Skaters
    if os.path.exists(skater_csv_file):
        with open(skater_csv_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            header_raw = next(reader)
            header_lower = [h.strip().lower() for h in header_raw]
            headers_sanitized = sanitize_header(header_raw)

            try:
                p_name_idx = header_lower.index('player name')
                gp_idx = header_lower.index('gp')
            except ValueError as e: raise ValueError(f"Missing column in {skater_csv_file}: {e}")

            stats_exclude = ['player name', 'age', 'positions', 'position', 'team', 'salary', 'gp org', 'gp', 'toi org es', 'toi org pp', 'toi org pk', 'toi es', 'toi pp', 'toi pk', 'total toi', 'rank', 'playerid', 'fantasy team']
            stat_indices = [i for i, h in enumerate(header_lower) if h not in stats_exclude and h.strip() != '']

            for row in reader:
                pos_idx = -1
                if 'positions' in header_lower: pos_idx = header_lower.index('positions')
                elif 'position' in header_lower: pos_idx = header_lower.index('position')

                if not row or (pos_idx != -1 and pos_idx < len(row) and 'G' in row[pos_idx]): continue

                calculate_per_game_stats(row, gp_idx, stat_indices)
                player_name = row[p_name_idx]
                if not player_name: continue
                norm = normalize_name(player_name)
                data_dict = {headers_sanitized[i]: val for i, val in enumerate(row)}

                team = data_dict.get('team', '').upper()
                data_dict['team'] = TEAM_TRICODE_MAP.get(team, team)

                # --- FIX: Calculate PPA (Power Play Assists) ---
                if 'PPA' not in data_dict and 'PPP' in data_dict and 'PPG' in data_dict:
                    try:
                        ppp = float(data_dict['PPP'])
                        ppg = float(data_dict['PPG'])
                        data_dict['PPA'] = round(ppp - ppg, 4)
                    except: pass

                # --- FIX: Calculate SHA (Shorthanded Assists) ---
                if 'SHA' not in data_dict and 'SHP' in data_dict and 'SHG' in data_dict:
                    try:
                        shp = float(data_dict['SHP'])
                        shg = float(data_dict['SHG'])
                        data_dict['SHA'] = round(shp - shg, 4)
                    except: pass

                # --- FIX: Calculate TOI/G for Skaters ---
                if 'TOI/G' not in data_dict and 'Total TOI' in data_dict:
                    try:
                         toi = float(data_dict['Total TOI'])
                         gp = float(row[gp_index])
                         if gp > 0: data_dict['TOI/G'] = round(toi / gp, 2)
                    except: pass
                # ----------------------------------------

                player_data[norm] = data_dict
                player_data[norm]['player_name_normalized'] = norm

    # 2. Goalies
    if os.path.exists(goalie_csv_file):
        with open(goalie_csv_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            header_raw = next(reader)
            header_lower = [h.strip().lower() for h in header_raw]
            headers_sanitized = sanitize_header(header_raw)

            try:
                p_name_idx = header_lower.index('player name')
                gp_idx = header_lower.index('gs')
            except ValueError as e: raise ValueError(f"Missing column in {goalie_csv_file}: {e}")

            stats_exclude = ['player name', 'team', 'age', 'position', 'positions', 'salary', 'gs', 'sv%', 'gaa', 'rank', 'playerid', 'fantasy team']
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

                # --- FIX: Calculate TOI/G for Goalies ---
                if 'TOI/G' not in data_dict and 'GA' in data_dict and 'GAA' in data_dict:
                    try:
                        ga = float(data_dict['GA'])
                        gaa = float(data_dict['GAA'])
                        # Attempt to find GP in row
                        gp = float(row[gp_idx]) if gp_idx < len(row) else 0

                        if gaa > 0 and gp > 0:
                            total_mins = (ga * 60) / gaa
                            data_dict['TOI/G'] = round(total_mins / gp, 2)
                    except: pass
                # ----------------------------------------

                if norm in player_data:
                    player_data[norm].update(data_dict)
                else:
                    data_dict['player_name_normalized'] = norm
                    player_data[norm] = data_dict

    player_data, new_rank_cols = calculate_and_add_category_ranks(player_data)

    if not player_data: return

    all_keys = set()
    for p in player_data.values(): all_keys.update(p.keys())

    final_headers = [k for k in list(all_keys) if k and k.strip() != ""]
    if 'player_name_normalized' in final_headers: final_headers.remove('player_name_normalized')

    text_cols = ['player_name', 'positions', 'position', 'team', 'playerid', 'fantasy_team', 'salary', 'age', 'rank', 'gp_org', 'gp', 'total_toi']

    cols_def = []
    cols_def.append('player_name_normalized TEXT PRIMARY KEY')

    for col in final_headers:
        if col in text_cols:
            cols_def.append(f'"{col}" TEXT')
        else:
            cols_def.append(f'"{col}" DOUBLE PRECISION')

    cursor.execute(f"DROP TABLE IF EXISTS {target_table_name}")
    cursor.execute(f"CREATE TABLE {target_table_name} ({', '.join(cols_def)})")

    insert_headers = ['player_name_normalized'] + final_headers
    placeholders = ", ".join(['%s'] * len(insert_headers))
    col_list = ", ".join(f'"{h}"' for h in insert_headers)

    insert_sql = f'INSERT INTO {target_table_name} ({col_list}) VALUES ({placeholders})'

    text_cols = [
        'player_name_normalized', 'player_name', 'positions', 'position',
        'team', 'playerid', 'fantasy_team', 'salary', 'age', 'rank',
        'gp_org', 'gp', 'total_toi'
    ]

    rows_to_insert = []
    for norm, data in player_data.items():
        clean_row = []
        for h in insert_headers:
            val = data.get(h, None)

            # 1. Handle explicit NULLs or empty strings
            if val is None or str(val).strip() == "" or str(val).lower() in ["#div/0!", "#n/a", "nan"]:
                val = None

            # 2. If the column is numeric (not in text_cols), force conversion to float
            elif h not in text_cols:
                try:
                    val = float(str(val).replace('%', '').replace('$', '').replace(',', ''))
                except (ValueError, TypeError):
                    val = None

            clean_row.append(val)
        rows_to_insert.append(tuple(clean_row))

    cursor.executemany(insert_sql, rows_to_insert)
    print(f"Populated {target_table_name} with {len(rows_to_insert)} rows.")

def create_averaged_projections(conn, cursor):
    print("\n--- Creating Final Averaged Projections ---")
    df1 = read_sql_postgres("SELECT * FROM proj1", conn)
    df2 = read_sql_postgres("SELECT * FROM proj2", conn)

    merged = pd.merge(df1, df2, on='player_name_normalized', how='outer', suffixes=('_p1', '_p2'))

    final = pd.DataFrame()
    final['player_name_normalized'] = merged['player_name_normalized']

    COALESCE_COLS = ['player_name', 'positions', 'team', 'playerid', 'fantasy_team']
    for col in COALESCE_COLS:
        c1, c2 = f'{col}_p1', f'{col}_p2'
        if c1 in merged: final[col] = merged[c1].fillna(merged.get(c2))
        elif c2 in merged: final[col] = merged[c2]
        # --- FIX: Handle columns that exist in only one file (no suffix) ---
        elif col in merged: final[col] = merged[col]

    if 'playerid_p2' in merged:
        final['nhlplayerid'] = pd.to_numeric(merged['playerid_p2'], errors='coerce').astype('Int64')
    elif 'playerid' in merged.columns:
        final['nhlplayerid'] = pd.to_numeric(merged['playerid'], errors='coerce').astype('Int64')
    else:
        final['nhlplayerid'] = pd.NA

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
        # --- FIX: Handle columns that exist in only one file (no suffix) ---
        elif col in merged.columns:
             final[col] = pd.to_numeric(merged[col], errors='coerce')

    df_to_postgres(final, 'projections', conn)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_proj_norm ON projections(player_name_normalized)")

def join_yahoo_ids(conn, cursor):
    print("\n--- Joining Yahoo Player ID Data ---")
    df_proj = read_sql_postgres("SELECT * FROM projections", conn)
    df_yahoo = read_sql_postgres("SELECT player_name_normalized, player_id, positions, status FROM players", conn)

    if 'positions' in df_proj.columns: df_proj = df_proj.drop(columns=['positions'])

    df_final = pd.merge(df_proj, df_yahoo, on='player_name_normalized', how='left')

    missing_mask = df_final['player_id'].isnull()
    df_missing = df_final[missing_mask][['player_name', 'player_name_normalized', 'team']]
    if not df_missing.empty:
        print(f"WARNING: {len(df_missing)} players did not match a Yahoo ID.")
        df_to_postgres(df_missing, 'missing_id', conn)

    if 'nhlplayerid' in df_final.columns:
        df_final['nhlplayerid'] = pd.to_numeric(df_final['nhlplayerid'], errors='coerce').astype('Int64')

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

def create_projections_db():
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
    create_projections_db()
