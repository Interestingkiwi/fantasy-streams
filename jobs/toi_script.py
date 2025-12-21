"""
Fetches NHL TOI/Special Teams stats and updates the Global Postgres DB.
Refactored for Postgres.
"""

import requests
import pandas as pd
from datetime import date, timedelta, datetime
import time
import os
import sys
import unicodedata
import re
import pytz
import numpy as np
import json

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import get_db_connection

# --- Constants ---
DEBUG_DUMP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static')
os.makedirs(DEBUG_DUMP_DIR, exist_ok=True)

FRANCHISE_TO_TRICODE_MAP = {
    "Anaheim Ducks": "ANA", "Boston Bruins": "BOS", "Buffalo Sabres": "BUF",
    "Calgary Flames": "CGY", "Carolina Hurricanes": "CAR", "Chicago Blackhawks": "CHI",
    "Colorado Avalanche": "COL", "Columbus Blue Jackets": "CBJ", "Dallas Stars": "DAL",
    "Detroit Red Wings": "DET", "Edmonton Oilers": "EDM", "Florida Panthers": "FLA",
    "Los Angeles Kings": "LAK", "Minnesota Wild": "MIN", "Montréal Canadiens": "MTL",
    "Nashville Predators": "NSH", "New Jersey Devils": "NJD", "New York Islanders": "NYI",
    "New York Rangers": "NYR", "Ottawa Senators": "OTT", "Philadelphia Flyers": "PHI",
    "Pittsburgh Penguins": "PIT", "San Jose Sharks": "SJS", "Seattle Kraken": "SEA",
    "St. Louis Blues": "STL", "Tampa Bay Lightning": "TBL", "Toronto Maple Leafs": "TOR",
    "Utah Mammoth": "UTA", "Vancouver Canucks": "VAN", "Vegas Golden Knights": "VGK",
    "Washington Capitals": "WSH", "Winnipeg Jets": "WPG", "Utah Hockey Club": "UTA"
}

def normalize_name(name):
    if not name: return ""
    nfkd = unicodedata.normalize('NFKD', name.lower())
    ascii = "".join([c for c in nfkd if not unicodedata.combining(c)])
    return re.sub(r'[^a-z0-9]', '', ascii)

# --- DATABASE HELPERS ---

def read_sql_postgres(query, conn):
    with conn.cursor() as cursor:
        cursor.execute(query)
        if cursor.description:
            columns = [desc[0] for desc in cursor.description]
            data = cursor.fetchall()
            return pd.DataFrame(data, columns=columns)
        return pd.DataFrame()

def df_to_postgres(df, table_name, conn, if_exists='replace', lowercase_columns=True, primary_key=None):
    if df.empty:
        print(f"Warning: DataFrame for {table_name} is empty. Skipping write.")
        return

    if lowercase_columns:
        df.columns = [c.lower() for c in df.columns]
        if primary_key: primary_key = primary_key.lower()

    cursor = conn.cursor()
    if if_exists == 'replace':
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")

    cols = []
    for col, dtype in df.dtypes.items():
        pg_type = 'TEXT'
        if pd.api.types.is_integer_dtype(dtype): pg_type = 'INTEGER'
        elif pd.api.types.is_float_dtype(dtype): pg_type = 'DOUBLE PRECISION'

        if primary_key and col == primary_key:
            cols.append(f'"{col}" {pg_type} PRIMARY KEY')
        else:
            cols.append(f'"{col}" {pg_type}')

    create_sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(cols)})"
    cursor.execute(create_sql)

    columns = list(df.columns)
    placeholders = ",".join(["%s"] * len(columns))
    col_names = ",".join([f'"{c}"' for c in columns])

    data = [tuple(None if pd.isna(x) else x for x in row) for row in df.to_numpy()]

    conflict_clause = f"ON CONFLICT (\"{primary_key}\") DO NOTHING" if primary_key else ""

    insert_sql = f"INSERT INTO {table_name} ({col_names}) VALUES ({placeholders}) {conflict_clause}"
    cursor.executemany(insert_sql, data)
    conn.commit()
    print(f"Successfully wrote {len(data)} rows to {table_name}.")

def run_database_cleanup(target_start_date):
    target_str = target_start_date.strftime("%Y-%m-%d")
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT to_regclass('public.powerplay_stats')")
            if cursor.fetchone()[0]:
                cursor.execute("DELETE FROM powerplay_stats WHERE date_ < %s", (target_str,))
        conn.commit()

def get_last_run_end_date():
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT to_regclass('public.table_metadata')")
            if cursor.fetchone()[0] is None: return None
            cursor.execute("SELECT end_date FROM table_metadata WHERE id = 1")
            res = cursor.fetchone()
            if res: return date.fromisoformat(res[0])
    return None

def update_metadata(start, end):
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO table_metadata (id, start_date, end_date) VALUES (1, %s, %s)
                ON CONFLICT (id) DO UPDATE SET start_date = EXCLUDED.start_date, end_date = EXCLUDED.end_date
            """, (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")))
            conn.commit()

def create_global_tables():
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS powerplay_stats (
                    date_ TEXT, nhlplayerid INTEGER, skaterfullname TEXT, teamabbrevs TEXT,
                    pptimeonice INTEGER, pptimeonicepctpergame REAL, ppassists INTEGER, ppgoals INTEGER,
                    PRIMARY KEY (date_, nhlplayerid)
                );
                CREATE TABLE IF NOT EXISTS table_metadata (id INTEGER PRIMARY KEY DEFAULT 1, start_date TEXT, end_date TEXT);
                CREATE TABLE IF NOT EXISTS unmatched_players (
                    run_date TEXT, source_table TEXT, nhlplayerid INTEGER, player_name TEXT, team TEXT
                );
                CREATE TABLE IF NOT EXISTS debug_dumps (
                    filename TEXT PRIMARY KEY,
                    content TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.commit()

def save_debug_to_db(filename, data):
    try:
        json_str = json.dumps(data)
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO debug_dumps (filename, content, created_at)
                    VALUES (%s, %s, NOW())
                    ON CONFLICT (filename) DO UPDATE
                    SET content = EXCLUDED.content, created_at = NOW()
                """, (filename, json_str))
            conn.commit()
    except Exception as e:
        print(f"Failed to save debug dump: {e}")

def log_unmatched_players(conn, df_unmatched, source_table_name):
    if df_unmatched.empty: return

    log_df = pd.DataFrame()
    log_df['nhlplayerid'] = df_unmatched.get('nhlplayerid', pd.NA)
    if 'player_name_normalized' in df_unmatched.columns:
        log_df['player_name'] = df_unmatched['player_name_normalized']
    elif 'skaterfullname' in df_unmatched.columns:
        log_df['player_name'] = df_unmatched['skaterfullname']
    elif 'goaliefullname' in df_unmatched.columns:
        log_df['player_name'] = df_unmatched['goaliefullname']
    else:
        log_df['player_name'] = 'Unknown'

    log_df['team'] = df_unmatched.get('team', df_unmatched.get('teamabbrevs', 'Unknown'))
    log_df['source_table'] = source_table_name
    log_df['run_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with conn.cursor() as cursor:
        data = [tuple(x) for x in log_df.to_numpy()]
        cursor.executemany("""
            INSERT INTO unmatched_players (nhlplayerid, player_name, team, source_table, run_date)
            VALUES (%s, %s, %s, %s, %s)
        """, data)
    conn.commit()

# --- FETCH FUNCTIONS ---

def fetch_daily_pp_stats():
#   API url
#   https://api.nhle.com/stats/rest/en/skater/powerplay?isAggregate=false&sort=[{"property":"ppTimeOnIce","direction":"DESC"},{"property":"playerId","direction":"ASC"}]&start=0&limit=100&cayenneExp=gameDate>="2025-10-27" and gameDate<="2025-10-27" and gameTypeId=2
    print("\n--- Fetching Daily PP Stats ---")
    est = pytz.timezone('US/Eastern')
    today = datetime.now(est).date()
    target_end_date = today - timedelta(days=1)
    target_start_date = today - timedelta(days=7)

    run_database_cleanup(target_start_date)

    print(f"Self-Healing: Clearing records from {target_start_date} to {target_end_date}...")
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "DELETE FROM powerplay_stats WHERE date_ >= %s AND date_ <= %s",
                (target_start_date.strftime("%Y-%m-%d"), target_end_date.strftime("%Y-%m-%d"))
            )
        conn.commit()

    dates = []
    curr = target_start_date
    while curr <= target_end_date:
        dates.append(curr.strftime("%Y-%m-%d"))
        curr += timedelta(days=1)

    all_data = []
    has_errors = False
    BASE_URL = "https://api.nhle.com/stats/rest/en/skater/powerplay"

    for d in dates:
        print(f"Querying {d}...")
        idx = 0
        try:
            while True:
                params = {
                    "isAggregate": "false",
                    "sort": '[{"property":"ppTimeOnIce","direction":"DESC"},{"property":"playerId","direction":"ASC"}]',
                    "start": idx, "limit": 100,
                    "cayenneExp": f'gameDate>="{d}" and gameDate<="{d}" and gameTypeId=2'
                }
                resp = requests.get(BASE_URL, params=params, timeout=10)
                data = resp.json().get("data", [])
                if not data: break

                for p in data:
                    pp_pct = p.get("ppTimeOnIcePctPerGame")
                    if pp_pct is None: continue

                    rec = {
                        "date_": d, "nhlplayerid": p.get("playerId"), "skaterfullname": p.get("skaterFullName"),
                        "teamabbrevs": p.get("teamAbbrevs"), "pptimeonice": p.get("ppTimeOnIce"),
                        "pptimeonicepctpergame": pp_pct,
                        "ppassists": p.get("ppAssists"),
                        "ppgoals": p.get("ppGoals")
                    }
                    all_data.append(rec)
                idx += 100
                time.sleep(0.1)
        except Exception as e:
            has_errors = True
            print(f"Error fetching {d}: {e}")

    if all_data:
        df = pd.DataFrame(all_data)
        df.drop_duplicates(subset=['date_', 'nhlplayerid'], inplace=True)

        with get_db_connection() as conn:
            df.columns = [c.lower() for c in df.columns]

            with conn.cursor() as cursor:
                vals = [tuple(x) for x in df[['date_', 'nhlplayerid', 'skaterfullname', 'teamabbrevs', 'pptimeonice', 'pptimeonicepctpergame', 'ppassists', 'ppgoals']].to_numpy()]
                cursor.executemany("""
                    INSERT INTO powerplay_stats (date_, nhlplayerid, skaterfullname, teamabbrevs, pptimeonice, pptimeonicepctpergame, ppassists, ppgoals)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (date_, nhlplayerid) DO UPDATE SET
                        pptimeonice = EXCLUDED.pptimeonice,
                        pptimeonicepctpergame = EXCLUDED.pptimeonicepctpergame,
                        ppassists = EXCLUDED.ppassists,
                        ppgoals = EXCLUDED.ppgoals
                """, vals)
                conn.commit()

        if not has_errors:
            update_metadata(target_start_date, target_end_date)
        return True
    return False

def create_last_game_pp_table():
    print("\n--- Creating 'last_game_pp' ---")
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("DROP TABLE IF EXISTS last_game_pp")
            cursor.execute("""
                CREATE TABLE last_game_pp AS
                SELECT
                    t1.nhlplayerid,
                    t1.pptimeonice as "lg_ppTimeOnIce",
                    COALESCE(t1.pptimeonicepctpergame, 0) as "lg_ppTimeOnIcePctPerGame",
                    t1.ppassists as "lg_ppAssists",
                    t1.ppgoals as "lg_ppGoals"
                FROM powerplay_stats t1
                INNER JOIN (
                    SELECT teamabbrevs, MAX(date_) as max_date
                    FROM powerplay_stats GROUP BY teamabbrevs
                ) t2 ON t1.teamabbrevs = t2.teamabbrevs AND t1.date_ = t2.max_date
            """)
            conn.commit()

def create_last_week_pp_table():
    print("\n--- Creating 'last_week_pp' ---")
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("DROP TABLE IF EXISTS last_week_pp")
            cursor.execute("""
                CREATE TABLE last_week_pp AS
                WITH team_game_counts AS (
                    SELECT teamabbrevs, COUNT(DISTINCT date_) as team_games_played
                    FROM powerplay_stats GROUP BY teamabbrevs
                ),
                player_sums AS (
                    SELECT nhlplayerid, teamabbrevs, MAX(skaterfullname) as skaterfullname,
                    SUM(pptimeonice) as total_pptimeonice,
                    COALESCE(SUM(pptimeonicepctpergame), 0) as total_pptimeonicepctpergame,
                    SUM(ppassists) as total_ppassists,
                    SUM(ppgoals) as total_ppgoals,
                    COUNT(date_) as player_games_played
                    FROM powerplay_stats GROUP BY nhlplayerid, teamabbrevs
                )
                SELECT ps.nhlplayerid, ps.skaterfullname, ps.teamabbrevs,
                    CAST(ps.total_pptimeonice AS REAL) / tgc.team_games_played AS "avg_ppTimeOnIce",
                    CAST(ps.total_pptimeonicepctpergame AS REAL) / tgc.team_games_played AS "avg_ppTimeOnIcePctPerGame",
                    ps.total_ppassists as "total_ppAssists",
                    ps.total_ppgoals as "total_ppGoals",
                    ps.player_games_played as "player_games_played",
                    tgc.team_games_played as "team_games_played"
                FROM player_sums ps
                JOIN team_game_counts tgc ON ps.teamabbrevs = tgc.teamabbrevs
            """)
            conn.commit()

# --- TEAM STATS ---

def fetch_team_standings():
    print("\n--- Fetching Team Standings ---")
    url = f"https://api-web.nhle.com/v1/standings/{date.today()}"
    try:
        data = requests.get(url).json().get("standings", [])
        rows = []
        for t in data:
            abbrev = t.get("teamAbbrev", {}).get("default")
            pts = t.get("pointPctg")
            ga = t.get("goalAgainst")
            gp = t.get("gamesPlayed")
            if abbrev:
                val_ga = ga/gp if gp and gp > 0 else 0
                rows.append((abbrev, pts, val_ga, gp))

        if rows:
            with get_db_connection() as conn:
                df = pd.DataFrame(rows, columns=['team_tricode', 'point_pct', 'goals_against_per_game', 'games_played'])
                df_to_postgres(df, 'team_standings', conn)
    except Exception as e:
        print(f"Error: {e}")

def fetch_team_stats_summary():
    print("\n--- Fetching Team Stats Summary ---")
    try:
        params = {"isAggregate":"false", "isGame":"false", "start":0, "limit":50, "cayenneExp":"seasonId=20252026 and gameTypeId=2"}
        data = requests.get("https://api.nhle.com/stats/rest/en/team/summary", params=params).json().get("data", [])
        rows = []
        for t in data:
            code = FRANCHISE_TO_TRICODE_MAP.get(t.get("teamFullName"))
            if code:
                rows.append((code, t.get("powerPlayPct"), t.get("penaltyKillPct"), t.get("goalsForPerGame"), t.get("goalsAgainstPerGame"), t.get("shotsForPerGame"), t.get("shotsAgainstPerGame")))

        if rows:
            with get_db_connection() as conn:
                df = pd.DataFrame(rows, columns=['team_tricode', 'pp_pct', 'pk_pct', 'gf_gm', 'ga_gm', 'sogf_gm', 'soga_gm'])
                df_to_postgres(df, 'team_stats_summary', conn)
    except Exception as e: print(f"Error: {e}")

def fetch_team_stats_weekly():
    print("\n--- Fetching Team Stats Weekly ---")
    end = date.today() - timedelta(days=1)
    start = date.today() - timedelta(days=7)
    try:
        params = {"isAggregate":"true", "isGame":"false", "start":0, "limit":50, "cayenneExp":f'gameDate>="{start}" and gameDate<="{end}" and gameTypeId=2'}
        data = requests.get("https://api.nhle.com/stats/rest/en/team/summary", params=params).json().get("data", [])
        rows = []
        for t in data:
            code = FRANCHISE_TO_TRICODE_MAP.get(t.get("franchiseName"))
            if code:
                rows.append((code, t.get("powerPlayPct"), t.get("penaltyKillPct"), t.get("goalsForPerGame"), t.get("goalsAgainstPerGame"), t.get("shotsForPerGame"), t.get("shotsAgainstPerGame")))

        if rows:
            with get_db_connection() as conn:
                df = pd.DataFrame(rows, columns=['team_tricode', 'pp_pct_weekly', 'pk_pct_weekly', 'gf_gm_weekly', 'ga_gm_weekly', 'sogf_gm_weekly', 'soga_gm_weekly'])
                df_to_postgres(df, 'team_stats_weekly', conn)
    except Exception as e: print(f"Error: {e}")

# --- PLAYER FETCHERS ---

def fetch_and_update_scoring_to_date():
    print("Fetching Scoring to Date...")
    season_id = "20252026"
    base_url = "https://api.nhle.com/stats/rest/en/skater/summary"
    all_data = []
    start = 0
    total_expected = 0

    try:
        while True:
            params = {
                "isAggregate": "false",
                "cayenneExp": f"gameTypeId=2 and seasonId={season_id}",
                "sort": '[{"property":"points","direction":"DESC"},{"property":"playerId","direction":"ASC"}]',
                "start": start,
                "limit": 100
            }

            success = False
            for attempt in range(3):
                try:
                    r = requests.get(base_url, params=params, timeout=15)
                    if r.status_code == 200:
                        success = True
                        break
                except requests.RequestException:
                    time.sleep(2)

            if not success: break

            resp_json = r.json()
            if start == 0: total_expected = resp_json.get('total', 0)

            data = resp_json.get('data', [])
            if not data: break

            all_data.extend(data)
            start += 100
            time.sleep(0.1)

        print(f"Fetched {len(all_data)} scoring records.")

        if all_data:
            df = pd.DataFrame(all_data)
            df.drop_duplicates(subset=['playerId'], inplace=True)
            if 'skaterFullName' in df.columns:
                df['player_name_normalized'] = df['skaterFullName'].apply(normalize_name)

            df['playerId'] = pd.to_numeric(df['playerId'], errors='coerce')
            df.loc[df['playerId'] == 8480012, 'player_name_normalized'] = 'eliaspetterssonf'
            df.loc[df['playerId'] == 8478427, 'player_name_normalized'] = 'sebastianahof'

            # --- UPDATED: Added gameWinningGoals to numeric conversion ---
            numeric_cols = ['gamesPlayed', 'goals', 'assists', 'points', 'plusMinus',
                            'penaltyMinutes', 'ppGoals', 'ppPoints', 'shGoals', 'shPoints', 'shots', 'gameWinningGoals']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

            # --- UPDATED: Added gameWinningGoals to averaging loop ---
            cols_to_average = ['goals', 'assists', 'points', 'plusMinus', 'penaltyMinutes',
                               'ppGoals', 'ppPoints', 'shGoals', 'shPoints', 'shots', 'gameWinningGoals']
            df['gamesPlayed'] = df['gamesPlayed'].astype(float)
            for col in cols_to_average:
                df[col] = np.where(df['gamesPlayed'] > 0, df[col] / df['gamesPlayed'], 0.0)
                df[col] = df[col].round(3)

            # --- UPDATED: Calculate Derived Assists ---
            df['ppAssists'] = df['ppPoints'] - df['ppGoals']
            df['shAssists'] = df['shPoints'] - df['shGoals']

            cols = {
                'playerId': 'nhlplayerid', 'skaterFullName': 'skaterfullname', 'teamAbbrevs': 'teamabbrevs',
                'gamesPlayed': 'gamesplayed', 'goals': 'goals', 'assists': 'assists', 'points': 'points',
                'plusMinus': 'plusminus', 'penaltyMinutes': 'penaltyminutes',
                'ppGoals': 'ppgoals', 'ppPoints': 'pppoints',
                'shGoals': 'shgoals', 'shPoints': 'shpoints',
                'gameWinningGoals': 'gwgoals', # New Column Mapping
                'shootingPct': 'shootingpct', 'timeOnIcePerGame': 'toi/g',
                'shots': 'shots', 'ppAssists': 'ppassists', 'shAssists': 'shassists'
            }
            cols['timeOnIcePerGame'] = 'toi/g'

            keep_cols = list(cols.keys()) + ['player_name_normalized']
            df_final = df[keep_cols].rename(columns=cols)

            with get_db_connection() as conn:
                df_to_postgres(df_final, 'scoring_to_date', conn, primary_key='nhlplayerid')

    except Exception as e:
        print(f"Error scoring: {e}")

def fetch_and_update_bangers_stats():
    print("Fetching Bangers...")
    season_id = "20252026"
    base_url = "https://api.nhle.com/stats/rest/en/skater/scoringpergame"
    all_data = []
    start = 0
    try:
        while True:
            params = {
                "isAggregate": "false",
                "cayenneExp": f"gameTypeId=2 and seasonId={season_id}",
                "sort": '[{"property":"pointsPerGame","direction":"DESC"},{"property":"playerId","direction":"ASC"}]',
                "start": start,
                "limit": 100
            }

            success = False
            for attempt in range(3):
                try:
                    r = requests.get(base_url, params=params, timeout=15)
                    if r.status_code == 200:
                        success = True
                        break
                except requests.RequestException:
                    time.sleep(2)

            if not success: break

            data = r.json().get('data', [])
            if not data: break
            all_data.extend(data)
            start += 100
            time.sleep(0.1)

        print(f"Fetched {len(all_data)} bangers records.")
        # save_debug_to_db('debug_bangers.json', all_data)

        if all_data:
            df = pd.DataFrame(all_data)
            df.drop_duplicates(subset=['playerId'], inplace=True)
            if 'skaterFullName' in df.columns:
                df['player_name_normalized'] = df['skaterFullName'].apply(normalize_name)

            df['playerId'] = pd.to_numeric(df['playerId'], errors='coerce')
            df.loc[df['playerId'] == 8480012, 'player_name_normalized'] = 'eliaspetterssonf'
            df.loc[df['playerId'] == 8478427, 'player_name_normalized'] = 'sebastianahof'

            cols = {'playerId': 'nhlplayerid', 'skaterFullName': 'skaterfullname', 'teamAbbrevs': 'teamabbrevs', 'blocksPerGame': 'BLK', 'hitsPerGame': 'HIT'}

            keep_cols = list(cols.keys()) + ['player_name_normalized']
            df_final = df[keep_cols].rename(columns=cols)

            with get_db_connection() as conn:
                df_to_postgres(df_final, 'bangers_to_date', conn, primary_key='nhlplayerid')

    except Exception as e: print(f"Error bangers: {e}")

def fetch_and_update_faceoff_stats():
#   API URL
#    https://api.nhle.com/stats/rest/en/skater/faceoffwins?isAggregate=false&isGame=false&sort=%5B%7B%22property%22:%22totalFaceoffWins%22,%22direction%22:%22DESC%22%7D,%7B%22property%22:%22faceoffWinPct%22,%22direction%22:%22DESC%22%7D%5D&start=0&limit=50&cayenneExp=gameTypeId=2%20and%20seasonId=20252026
    print("Fetching Faceoffs...")
    season_id = "20252026"
    base_url = "https://api.nhle.com/stats/rest/en/skater/faceoffwins"
    all_data = []
    start = 0

    try:
        while True:
            params = {
                "isAggregate": "false",
                "cayenneExp": f"gameTypeId=2 and seasonId={season_id}",
                "sort": '[{"property":"totalFaceoffWins","direction":"DESC"},{"property":"faceoffWinPct","direction":"DESC"},{"property":"playerId","direction":"ASC"}]',
                "start": start,
                "limit": 100
            }

            success = False
            for attempt in range(3):
                try:
                    r = requests.get(base_url, params=params, timeout=15)
                    if r.status_code == 200:
                        success = True
                        break
                except requests.RequestException:
                    time.sleep(2)

            if not success: break

            data = r.json().get('data', [])
            if not data: break
            all_data.extend(data)
            start += 100
            time.sleep(0.1)

        print(f"Fetched {len(all_data)} faceoff records.")

        if all_data:
            df = pd.DataFrame(all_data)
            df.drop_duplicates(subset=['playerId'], inplace=True)
            if 'skaterFullName' in df.columns:
                df['player_name_normalized'] = df['skaterFullName'].apply(normalize_name)

            df['playerId'] = pd.to_numeric(df['playerId'], errors='coerce')
            df.loc[df['playerId'] == 8480012, 'player_name_normalized'] = 'eliaspetterssonf'
            df.loc[df['playerId'] == 8478427, 'player_name_normalized'] = 'sebastianahof'

            # --- Calculate Per Game Stats ---
            df['gamesPlayed'] = pd.to_numeric(df['gamesPlayed'], errors='coerce').fillna(0)
            df['totalFaceoffWins'] = pd.to_numeric(df['totalFaceoffWins'], errors='coerce').fillna(0)
            df['totalFaceoffLosses'] = pd.to_numeric(df['totalFaceoffLosses'], errors='coerce').fillna(0)

            df['totalfaceoffwins'] = np.where(df['gamesPlayed'] > 0, df['totalFaceoffWins'] / df['gamesPlayed'], 0.0).round(3)
            df['totalfaceofflosses'] = np.where(df['gamesPlayed'] > 0, df['totalFaceoffLosses'] / df['gamesPlayed'], 0.0).round(3)

            cols = {
                'playerId': 'nhlplayerid',
                'skaterFullName': 'skaterfullname',
                'teamAbbrevs': 'teamabbrevs',
                'faceoffWinPct': 'faceoffwinpct',
                'totalfaceoffwins': 'totalfaceoffwins', # Mapped to calculated per-game col
                'totalfaceofflosses': 'totalfaceofflosses' # Mapped to calculated per-game col
            }

            keep_cols = list(cols.keys()) + ['player_name_normalized']
            df_final = df[keep_cols].rename(columns=cols)

            with get_db_connection() as conn:
                df_to_postgres(df_final, 'faceoff_to_date', conn, primary_key='nhlplayerid')

    except Exception as e:
        print(f"Error faceoffs: {e}")

def fetch_and_update_goalie_stats():
    print("Fetching Goalies...")
    season_id = "20252026"
    base_url = "https://api.nhle.com/stats/rest/en/goalie/summary"
    all_data = []
    start = 0
    try:
        while True:
            params = {
                "isAggregate": "false",
                "cayenneExp": f"gameTypeId=2 and seasonId={season_id}",
                "sort": '[{"property":"wins","direction":"DESC"},{"property":"playerId","direction":"ASC"}]',
                "start": start,
                "limit": 100
            }

            r = requests.get(base_url, params=params).json()
            data = r.get('data', [])
            if not data: break
            all_data.extend(data)
            start += 100
            time.sleep(0.1)

        print(f"Fetched {len(all_data)} goalie records.")

        if all_data:
            df = pd.DataFrame(all_data)
            df.drop_duplicates(subset=['playerId'], inplace=True)
            if 'goalieFullName' in df.columns:
                df['player_name_normalized'] = df['goalieFullName'].apply(normalize_name)

            numeric_cols = ['gamesPlayed', 'wins', 'losses', 'saves', 'shotsAgainst', 'goalsAgainst', 'shutouts', 'goalsAgainstAverage']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

            total_ga = df['goalsAgainst']
            gaa = df['goalsAgainstAverage']
            gp = df['gamesPlayed']

            df['total_mins'] = np.where(gaa > 0, (total_ga * 60) / gaa, 0)
            df['TOI/G'] = np.where(gp > 0, df['total_mins'] / gp, 0.0).round(2)

            for col in ['wins', 'losses', 'saves', 'shotsAgainst', 'goalsAgainst', 'shutouts']:
                df[col] = np.where(gp > 0, df[col] / gp, 0.0).round(3)

            cols = {
                'playerId': 'nhlplayerid', 'goalieFullName': 'goaliefullname', 'teamAbbrevs': 'teamabbrevs',
                'gamesStarted': 'gamesstarted', 'gamesPlayed': 'gamesplayed', 'goalsAgainstAverage': 'goalsagainstaverage',
                'losses': 'losses', 'savePct': 'savepct', 'saves': 'saves', 'shotsAgainst': 'shotsagainst',
                'shutouts': 'shutouts', 'wins': 'wins', 'goalsAgainst': 'goalsagainst',
                'TOI/G': 'toi/g'
            }

            # FIX: Include player_name_normalized in the column selection
            keep_cols = list(cols.keys()) + ['player_name_normalized']
            df_final = df[keep_cols].rename(columns=cols)

            with get_db_connection() as conn:
                std = read_sql_postgres("SELECT team_tricode, games_played as team_gp FROM team_standings", conn)
                if not std.empty:
                    df_final = pd.merge(df_final, std, left_on='teamabbrevs', right_on='team_tricode', how='left')
                    df_final['startpct'] = np.where(df_final['team_gp']>0, df_final['gamesstarted']/df_final['team_gp'], 0)
                else:
                    df_final['startpct'] = 0

                if 'team_tricode' in df_final.columns: del df_final['team_tricode']
                if 'team_gp' in df_final.columns: del df_final['team_gp']

                df_to_postgres(df_final, 'goalie_to_date', conn, primary_key='nhlplayerid')

    except Exception as e: print(f"Error goalies: {e}")


def fetch_daily_historical_stats():
#TOI API
#https://api.nhle.com/stats/rest/en/skater/timeonice?isAggregate=false&isGame=true&sort=[{%22property%22:%22timeOnIce%22,%22direction%22:%22DESC%22}]&start=0&limit=50&cayenneExp=gameDate%3E=%222025-10-07%22%20and%20gameDate%3C=%222025-10-07%22%20and%20gameTypeId=2

    print("\n--- Fetching Daily Historical Stats (Time On Ice) ---")
    est = pytz.timezone('US/Eastern')
    today = datetime.now(est).date()
    target_end_date = today - timedelta(days=1)
    target_start_date = today - timedelta(days=7)

    print(f"Self-Healing: Checking schema and clearing records from {target_start_date} to {target_end_date}...")
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            # 1. Create the base table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS historical_stats (
                    gameid INTEGER,
                    gamedate DATE,
                    nhlplayerid INTEGER,
                    skaterfullname TEXT,
                    player_name_normalized TEXT,
                    teamabbrev TEXT,
                    opponentteamabbrev TEXT,
                    homeroad TEXT,
                    timeonice INTEGER,
                    evtimeonice INTEGER,
                    pptimeonice INTEGER,
                    shtimeonice INTEGER,
                    ottimeonice INTEGER,
                    shifts INTEGER,
                    timeonicepershift INTEGER,
                    PRIMARY KEY (gamedate, nhlplayerid)
                )
            """)

            # 2. Delete only the specific range to allow for re-insertion/updates
            cursor.execute(
                "DELETE FROM historical_stats WHERE gamedate >= %s AND gamedate <= %s",
                (target_start_date.strftime("%Y-%m-%d"), target_end_date.strftime("%Y-%m-%d"))
            )
        conn.commit()

    dates = []
    curr = target_start_date
    while curr <= target_end_date:
        dates.append(curr.strftime("%Y-%m-%d"))
        curr += timedelta(days=1)

    all_data = []
    BASE_URL = "https://api.nhle.com/stats/rest/en/skater/timeonice"

    for d in dates:
        print(f"Querying {d}...")
        idx = 0
        try:
            while True:
                params = {
                    "isAggregate": "false",
                    "isGame": "true",
                    "sort": '[{"property":"timeOnIce","direction":"DESC"},{"property":"playerId","direction":"ASC"}]',
                    "start": idx, "limit": 100,
                    "cayenneExp": f'gameDate>="{d}" and gameDate<="{d}" and gameTypeId=2'
                }
                resp = requests.get(BASE_URL, params=params, timeout=10)
                data = resp.json().get("data", [])
                if not data: break

                for p in data:
                    # Map API fields to your requested column names
                    rec = {
                        "gameid": p.get("gameId"), # FIX: API uses camelCase "gameId"
                        "gamedate": d,
                        "nhlplayerid": p.get("playerId"),
                        "skaterfullname": p.get("skaterFullName"),
                        "teamabbrev": p.get("teamAbbrev"),
                        "opponentteamabbrev": p.get("opponentTeamAbbrev"),
                        "homeroad": p.get("homeRoad"),
                        "timeonice": p.get("timeOnIce"),
                        "evtimeonice": p.get("evTimeOnIce"),
                        "pptimeonice": p.get("ppTimeOnIce"),
                        "shtimeonice": p.get("shTimeOnIce"),
                        "ottimeonice": p.get("otTimeOnIce"),
                        "shifts": p.get("shifts"),
                        "timeonicepershift": p.get("timeOnIcePerShift")
                    }
                    all_data.append(rec)
                idx += 100
                time.sleep(0.1)
        except Exception as e:
            print(f"Error fetching {d}: {e}")

    if all_data:
        df = pd.DataFrame(all_data)

        # --- Type Conversion & ID Logic ---
        df['nhlplayerid'] = pd.to_numeric(df['nhlplayerid'], errors='coerce')

        # Normalize Names
        if 'skaterfullname' in df.columns:
            df['player_name_normalized'] = df['skaterfullname'].apply(normalize_name)

        # Specific Player ID / Name Fixes
        df.loc[df['nhlplayerid'] == 8480012, 'player_name_normalized'] = 'eliaspetterssonf'
        df.loc[df['nhlplayerid'] == 8478427, 'player_name_normalized'] = 'sebastianahof'

        # Dedup based on Date + PlayerID (Primary Key)
        df.drop_duplicates(subset=['gamedate', 'nhlplayerid'], inplace=True)

        with get_db_connection() as conn:
            # Ensure columns are lowercased for Postgres mapping
            df.columns = [c.lower() for c in df.columns]

            # Define the exact column order for the INSERT
            # FIX: Used 'gameid' (lowercase) to match df.columns
            cols_to_insert = [
                'gameid', 'gamedate', 'nhlplayerid', 'skaterfullname', 'player_name_normalized',
                'teamabbrev', 'opponentteamabbrev', 'homeroad',
                'timeonice', 'evtimeonice', 'pptimeonice', 'shtimeonice', 'ottimeonice',
                'shifts', 'timeonicepershift'
            ]

            vals = [tuple(x) for x in df[cols_to_insert].to_numpy()]

            with conn.cursor() as cursor:
                cursor.executemany("""
                    INSERT INTO historical_stats (
                        gameid, gamedate, nhlplayerid, skaterfullname, player_name_normalized,
                        teamabbrev, opponentteamabbrev, homeroad,
                        timeonice, evtimeonice, pptimeonice, shtimeonice, ottimeonice,
                        shifts, timeonicepershift
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (gamedate, nhlplayerid) DO UPDATE SET
                        gameid = EXCLUDED.gameid,
                        timeonice = EXCLUDED.timeonice,
                        evtimeonice = EXCLUDED.evtimeonice,
                        pptimeonice = EXCLUDED.pptimeonice,
                        shtimeonice = EXCLUDED.shtimeonice,
                        ottimeonice = EXCLUDED.ottimeonice,
                        shifts = EXCLUDED.shifts,
                        timeonicepershift = EXCLUDED.timeonicepershift
                """, vals)
                conn.commit()

        return True
    return False

def fetch_daily_summary_stats():
    # SUMMARY API
#https://api.nhle.com/stats/rest/en/skater/summary?isAggregate=false&isGame=true&sort=%5B%7B%22property%22:%22points%22,%22direction%22:%22DESC%22%7D,%7B%22property%22:%22goals%22,%22direction%22:%22DESC%22%7D,%7B%22property%22:%22assists%22,%22direction%22:%22DESC%22%7D,%7B%22property%22:%22playerId%22,%22direction%22:%22ASC%22%7D%5D&start=0&limit=50&factCayenneExp=gamesPlayed%3E=1&cayenneExp=gameDate%3E=%222025-10-07%22%20and%20gameDate%3C=%222025-10-07%22%20and%20gameTypeId=2

    print("\n--- Fetching Daily Summary Stats (Goals, Assists, Points) ---")
    est = pytz.timezone('US/Eastern')
    today = datetime.now(est).date()
    target_end_date = today - timedelta(days=1)
    target_start_date = today - timedelta(days=7)

    # 1. Ensure columns exist
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            # We assume historical_stats exists (created by previous function)
            # but we must ensure these specific scoring columns exist.
            new_cols = [
                "assists", "evgoals", "evpoints", "gamewinninggoals",
                "goals", "otgoals", "penaltyminutes", "plusminus",
                "points", "ppgoals", "pppoints", "shgoals",
                "shpoints", "shootingpct", "shots"
            ]
            for col in new_cols:
                # Most are INTEGER, shootingpct is likely REAL
                dtype = "REAL" if col == "shootingpct" else "INTEGER"
                cursor.execute(f"ALTER TABLE historical_stats ADD COLUMN IF NOT EXISTS {col} {dtype}")
        conn.commit()

    dates = []
    curr = target_start_date
    while curr <= target_end_date:
        dates.append(curr.strftime("%Y-%m-%d"))
        curr += timedelta(days=1)

    all_data = []
    BASE_URL = "https://api.nhle.com/stats/rest/en/skater/summary"

    for d in dates:
        print(f"Querying {d}...")
        idx = 0
        try:
            while True:
                params = {
                    "isAggregate": "false",
                    "isGame": "true",
                    "sort": '[{"property":"points","direction":"DESC"}]',
                    "start": idx, "limit": 100,
                    "factCayenneExp": "gamesPlayed>=1",
                    "cayenneExp": f'gameDate>="{d}" and gameDate<="{d}" and gameTypeId=2'
                }
                resp = requests.get(BASE_URL, params=params, timeout=10)
                data = resp.json().get("data", [])
                if not data: break

                for p in data:
                    # Map API fields to your requested column names
                    rec = {
                        "gameid": p.get("gameId"),
                        "gamedate": d,
                        "nhlplayerid": p.get("playerId"),
                        "skaterfullname": p.get("skaterFullName"), # kept for safe keeping

                        # New Columns
                        "assists": p.get("assists"),
                        "evgoals": p.get("evGoals"),
                        "evpoints": p.get("evPoints"),
                        "gamewinninggoals": p.get("gameWinningGoals"),
                        "goals": p.get("goals"),
                        "otgoals": p.get("otGoals"),
                        "penaltyminutes": p.get("penaltyMinutes"),
                        "plusminus": p.get("plusMinus"),
                        "points": p.get("points"),
                        "ppgoals": p.get("ppGoals"),
                        "pppoints": p.get("ppPoints"),
                        "shgoals": p.get("shGoals"),
                        "shpoints": p.get("shPoints"),
                        "shootingpct": p.get("shootingPct"),
                        "shots": p.get("shots")
                    }
                    all_data.append(rec)
                idx += 100
                time.sleep(0.1)
        except Exception as e:
            print(f"Error fetching {d}: {e}")

    if all_data:
        df = pd.DataFrame(all_data)

        # --- Type Conversion & ID Logic ---
        df['nhlplayerid'] = pd.to_numeric(df['nhlplayerid'], errors='coerce')

        # We don't need name normalization here because we are relying on ID matching
        # from the row created in the previous function, but strictly speaking,
        # we still drop duplicates based on the primary key.
        df.drop_duplicates(subset=['gamedate', 'nhlplayerid'], inplace=True)

        with get_db_connection() as conn:
            # Ensure columns are lowercased for Postgres mapping
            # (Though I manually lowercased keys in the dict above)

            # Define the exact column order for the INSERT/UPDATE
            cols_to_insert = [
                'gameid', 'gamedate', 'nhlplayerid',
                'assists', 'evgoals', 'evpoints', 'gamewinninggoals',
                'goals', 'otgoals', 'penaltyminutes', 'plusminus',
                'points', 'ppgoals', 'pppoints', 'shgoals',
                'shpoints', 'shootingpct', 'shots'
            ]

            vals = [tuple(x) for x in df[cols_to_insert].to_numpy()]

            with conn.cursor() as cursor:
                # We use INSERT ... ON CONFLICT DO UPDATE.
                # If the row exists (from TOI function), it updates the stats.
                # If the row doesn't exist (weird edge case), it creates it.
                cursor.executemany("""
                    INSERT INTO historical_stats (
                        gameid, gamedate, nhlplayerid,
                        assists, evgoals, evpoints, gamewinninggoals,
                        goals, otgoals, penaltyminutes, plusminus,
                        points, ppgoals, pppoints, shgoals,
                        shpoints, shootingpct, shots
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (gamedate, nhlplayerid) DO UPDATE SET
                        assists = EXCLUDED.assists,
                        evgoals = EXCLUDED.evgoals,
                        evpoints = EXCLUDED.evpoints,
                        gamewinninggoals = EXCLUDED.gamewinninggoals,
                        goals = EXCLUDED.goals,
                        otgoals = EXCLUDED.otgoals,
                        penaltyminutes = EXCLUDED.penaltyminutes,
                        plusminus = EXCLUDED.plusminus,
                        points = EXCLUDED.points,
                        ppgoals = EXCLUDED.ppgoals,
                        pppoints = EXCLUDED.pppoints,
                        shgoals = EXCLUDED.shgoals,
                        shpoints = EXCLUDED.shpoints,
                        shootingpct = EXCLUDED.shootingpct,
                        shots = EXCLUDED.shots
                """, vals)
                conn.commit()

        return True
    return False


def fetch_daily_realtime_stats():
    # REALTIME (MISCELLANEOUS) API
#https://api.nhle.com/stats/rest/en/skater/realtime?isAggregate=false&isGame=true&sort=%5B%7B%22property%22:%22hits%22,%22direction%22:%22DESC%22%7D%5D&start=0&limit=50&factCayenneExp=gamesPlayed%3E=1&cayenneExp=gameDate%3E=%222025-10-07%22%20and%20gameDate%3C=%222025-10-07%22%20and%20gameTypeId=2
    print("\n--- Fetching Daily Realtime Stats (Hits, Blocks, etc.) ---")
    est = pytz.timezone('US/Eastern')
    today = datetime.now(est).date()
    target_end_date = today - timedelta(days=1)
    target_start_date = today - timedelta(days=7)

    # 1. Ensure columns exist
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            new_cols = [
                "hits", "blockedshots", "giveaways",
                "takeaways", "shotattemptsblocked", "missedshots"
            ]
            for col in new_cols:
                cursor.execute(f"ALTER TABLE historical_stats ADD COLUMN IF NOT EXISTS {col} INTEGER")
        conn.commit()

    dates = []
    curr = target_start_date
    while curr <= target_end_date:
        dates.append(curr.strftime("%Y-%m-%d"))
        curr += timedelta(days=1)

    all_data = []
    BASE_URL = "https://api.nhle.com/stats/rest/en/skater/realtime"

    for d in dates:
        print(f"Querying {d}...")
        idx = 0
        try:
            while True:
                params = {
                    "isAggregate": "false",
                    "isGame": "true",
                    "sort": '[{"property":"hits","direction":"DESC"}]',
                    "start": idx, "limit": 100,
                    "factCayenneExp": "gamesPlayed>=1",
                    "cayenneExp": f'gameDate>="{d}" and gameDate<="{d}" and gameTypeId=2'
                }
                resp = requests.get(BASE_URL, params=params, timeout=10)
                data = resp.json().get("data", [])
                if not data: break

                for p in data:
                    rec = {
                        "gameid": p.get("gameId"),
                        "gamedate": d,
                        "nhlplayerid": p.get("playerId"),
                        "skaterfullname": p.get("skaterFullName"),

                        # New Columns for Realtime Report
                        "hits": p.get("hits"),
                        "blockedshots": p.get("blockedShots"), # API uses camelCase
                        "giveaways": p.get("giveaways"),
                        "takeaways": p.get("takeaways"),
                        "shotattemptsblocked": p.get("shotAttemptsBlocked"),
                        "missedshots": p.get("missedShots")
                    }
                    all_data.append(rec)
                idx += 100
                time.sleep(0.1)
        except Exception as e:
            print(f"Error fetching {d}: {e}")

    if all_data:
        df = pd.DataFrame(all_data)

        # --- Type Conversion & ID Logic ---
        df['nhlplayerid'] = pd.to_numeric(df['nhlplayerid'], errors='coerce')
        df.drop_duplicates(subset=['gamedate', 'nhlplayerid'], inplace=True)

        with get_db_connection() as conn:

            # Define columns to extract from DataFrame
            cols_to_insert = [
                'gameid', 'gamedate', 'nhlplayerid',
                'hits', 'blockedshots', 'giveaways', 'takeaways',
                'shotattemptsblocked', 'missedshots'
            ]

            vals = [tuple(x) for x in df[cols_to_insert].to_numpy()]

            with conn.cursor() as cursor:
                # UPDATE existing rows found by (gamedate, nhlplayerid)
                # FIX: Removed the trailing comma after 'missedshots' in the INSERT line below
                cursor.executemany("""
                    INSERT INTO historical_stats (
                        gameid, gamedate, nhlplayerid,
                        hits, blockedshots, giveaways, takeaways,
                        shotattemptsblocked, missedshots
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (gamedate, nhlplayerid) DO UPDATE SET
                        hits = EXCLUDED.hits,
                        blockedshots = EXCLUDED.blockedshots,
                        giveaways = EXCLUDED.giveaways,
                        takeaways = EXCLUDED.takeaways,
                        shotattemptsblocked = EXCLUDED.shotattemptsblocked,
                        missedshots = EXCLUDED.missedshots
                """, vals)
                conn.commit()

        return True
    return False


def fetch_game_shifts():
    print("\n--- Fetching Shift Data (Raw Line Info) ---")
#api URL
#https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId=2025020407
    est = pytz.timezone('US/Eastern')
    today = datetime.now(est).date()
    target_date = today - timedelta(days=1)
    date_str = target_date.strftime("%Y-%m-%d")

    # 1. Create table if not exists
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS game_shifts (
                    gameid INTEGER,
                    nhlplayerid INTEGER,
                    period INTEGER,
                    starttime TEXT,
                    endtime TEXT,
                    duration TEXT,
                    start_seconds INTEGER,
                    end_seconds INTEGER,
                    PRIMARY KEY (gameid, nhlplayerid, period, starttime)
                )
            """)
        conn.commit()

    # 2. Get the Game IDs for this date first
    game_ids = []
    print(f"Identifying Game IDs for {date_str}...")

    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT DISTINCT gameid FROM historical_stats WHERE gamedate = %s", (date_str,))
            rows = cursor.fetchall()
            game_ids = [r[0] for r in rows]

    if not game_ids:
        print("No game IDs found to fetch shifts for.")
        return

    # 3. Fetch Shifts for each Game
    base_url = "https://api.nhle.com/stats/rest/en/shiftcharts"

    all_shifts = []

    for gid in game_ids:
        print(f"Fetching shifts for Game {gid}...")
        try:
            params = {
                "cayenneExp": f"gameId={gid}"
            }
            resp = requests.get(base_url, params=params, timeout=10)
            data = resp.json().get("data", [])

            for s in data:
                # FIX: Added Try/Except to handle cases where time is None or malformed
                try:
                    start_min, start_sec = map(int, s.get("startTime").split(':'))
                    end_min, end_sec = map(int, s.get("endTime").split(':'))

                    start_seconds = (start_min * 60) + start_sec
                    end_seconds = (end_min * 60) + end_sec
                except (ValueError, AttributeError):
                    start_seconds = 0
                    end_seconds = 0

                rec = {
                    "gameid": s.get("gameId"),
                    "nhlplayerid": s.get("playerId"),
                    "period": s.get("period"),
                    "starttime": s.get("startTime"),
                    "endtime": s.get("endTime"),
                    "duration": s.get("duration"),
                    "start_seconds": start_seconds,
                    "end_seconds": end_seconds
                }
                all_shifts.append(rec)

            time.sleep(0.1)

        except Exception as e:
            print(f"Error fetching shifts for {gid}: {e}")

    # 4. Insert into Database
    if all_shifts:
        print(f"Saving {len(all_shifts)} shift records...")

        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                sql = """
                    INSERT INTO game_shifts (
                        gameid, nhlplayerid, period, starttime, endtime,
                        duration, start_seconds, end_seconds
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (gameid, nhlplayerid, period, starttime) DO NOTHING
                """

                vals = []
                for row in all_shifts:
                    vals.append((
                        row["gameid"], row["nhlplayerid"], row["period"],
                        row["starttime"], row["endtime"], row["duration"],
                        row["start_seconds"], row["end_seconds"]
                    ))

                cursor.executemany(sql, vals)
            conn.commit()
            print("Shifts saved.")


def fetch_daily_historical_goalie_stats():
    # GOALIE SUMMARY API
    # https://api.nhle.com/stats/rest/en/goalie/summary?isAggregate=false&isGame=true&sort=%5B%7B%22property%22:%22wins%22,%22direction%22:%22DESC%22%7D,%7B%22property%22:%22savePct%22,%22direction%22:%22DESC%22%7D%5D&start=0&limit=50&factCayenneExp=gamesPlayed%3E=1&cayenneExp=gameDate%3E=%222025-10-07%22%20and%20gameDate%3C=%222025-10-07%22%20and%20gameTypeId=2

    print("\n--- Fetching Daily Historical Goalie Stats ---")
    est = pytz.timezone('US/Eastern')
    today = datetime.now(est).date()
    target_end_date = today - timedelta(days=1)
    target_start_date = today - timedelta(days=7)

    print(f"Self-Healing: Clearing goalie records from {target_start_date} to {target_end_date}...")
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            # Create the table if it doesn't exist yet
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS historical_goalie_stats (
                    gameid INTEGER,
                    gamedate DATE,
                    nhlplayerid INTEGER,
                    goaliefullname TEXT,
                    player_name_normalized TEXT,
                    teamabbrev TEXT,
                    opponentteamabbrev TEXT,
                    homeroad TEXT,
                    gamesstarted INTEGER,
                    wins INTEGER,
                    losses INTEGER,
                    overtimelosses INTEGER,
                    shotsagainst INTEGER,
                    saves INTEGER,
                    goalsagainst INTEGER,
                    timeonice INTEGER,
                    goals INTEGER,
                    assists INTEGER,
                    points INTEGER,
                    penaltyminutes INTEGER,
                    shutouts INTEGER,
                    PRIMARY KEY (gamedate, nhlplayerid)
                )
            """)

            # Delete the specific range to allow for re-insertion
            cursor.execute(
                "DELETE FROM historical_goalie_stats WHERE gamedate >= %s AND gamedate <= %s",
                (target_start_date.strftime("%Y-%m-%d"), target_end_date.strftime("%Y-%m-%d"))
            )
        conn.commit()

    dates = []
    curr = target_start_date
    while curr <= target_end_date:
        dates.append(curr.strftime("%Y-%m-%d"))
        curr += timedelta(days=1)

    all_data = []
    BASE_URL = "https://api.nhle.com/stats/rest/en/goalie/summary"

    for d in dates:
        print(f"Querying Goalies for {d}...")
        idx = 0
        try:
            while True:
                params = {
                    "isAggregate": "false",
                    "isGame": "true",
                    "sort": '[{"property":"wins","direction":"DESC"},{"property":"savePct","direction":"DESC"}]',
                    "start": idx,
                    "limit": 100,
                    "cayenneExp": f'gameDate>="{d}" and gameDate<="{d}" and gameTypeId=2'
                }
                resp = requests.get(BASE_URL, params=params, timeout=10)
                data = resp.json().get("data", [])
                if not data: break

                for p in data:
                    rec = {
                        "gameid": p.get("gameId"),
                        "gamedate": d,
                        "nhlplayerid": p.get("playerId"),
                        "goaliefullname": p.get("goalieFullName"),
                        "teamabbrev": p.get("teamAbbrev"),
                        "opponentteamabbrev": p.get("opponentTeamAbbrev"),
                        "homeroad": p.get("homeRoad"),

                        # Stats you requested
                        "gamesstarted": p.get("gamesStarted"),
                        "wins": p.get("wins"),
                        "losses": p.get("losses"),
                        "overtimelosses": p.get("otLosses"), # API usually calls this otLosses
                        "shotsagainst": p.get("shotsAgainst"),
                        "saves": p.get("saves"),
                        "goalsagainst": p.get("goalsAgainst"),
                        "timeonice": p.get("timeOnIce"),
                        "goals": p.get("goals"),
                        "assists": p.get("assists"),
                        "points": p.get("points"),
                        "penaltyminutes": p.get("penaltyMinutes"),
                        "shutouts": p.get("shutouts")
                    }
                    all_data.append(rec)
                idx += 100
                time.sleep(0.1)
        except Exception as e:
            print(f"Error fetching goalies for {d}: {e}")

    if all_data:
        df = pd.DataFrame(all_data)

        # --- Type Conversion & ID Logic ---
        df['nhlplayerid'] = pd.to_numeric(df['nhlplayerid'], errors='coerce')

        # Normalize Names
        if 'goaliefullname' in df.columns:
            df['player_name_normalized'] = df['goaliefullname'].apply(normalize_name)

        # Dedup based on Date + PlayerID (Primary Key)
        df.drop_duplicates(subset=['gamedate', 'nhlplayerid'], inplace=True)

        with get_db_connection() as conn:
            # Ensure columns are lowercased
            df.columns = [c.lower() for c in df.columns]

            cols_to_insert = [
                'gameid', 'gamedate', 'nhlplayerid', 'goaliefullname', 'player_name_normalized',
                'teamabbrev', 'opponentteamabbrev', 'homeroad', 'gamesstarted',
                'wins', 'losses', 'overtimelosses', 'shotsagainst', 'saves',
                'goalsagainst', 'timeonice', 'goals', 'assists', 'points',
                'penaltyminutes', 'shutouts'
            ]

            vals = [tuple(x) for x in df[cols_to_insert].to_numpy()]

            with conn.cursor() as cursor:
                cursor.executemany("""
                    INSERT INTO historical_goalie_stats (
                        gameid, gamedate, nhlplayerid, goaliefullname, player_name_normalized,
                        teamabbrev, opponentteamabbrev, homeroad, gamesstarted,
                        wins, losses, overtimelosses, shotsagainst, saves,
                        goalsagainst, timeonice, goals, assists, points,
                        penaltyminutes, shutouts
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (gamedate, nhlplayerid) DO UPDATE SET
                        gameid = EXCLUDED.gameid,
                        gamesstarted = EXCLUDED.gamesstarted,
                        wins = EXCLUDED.wins,
                        losses = EXCLUDED.losses,
                        overtimelosses = EXCLUDED.overtimelosses,
                        shotsagainst = EXCLUDED.shotsagainst,
                        saves = EXCLUDED.saves,
                        goalsagainst = EXCLUDED.goalsagainst,
                        timeonice = EXCLUDED.timeonice,
                        goals = EXCLUDED.goals,
                        assists = EXCLUDED.assists,
                        points = EXCLUDED.points,
                        penaltyminutes = EXCLUDED.penaltyminutes,
                        shutouts = EXCLUDED.shutouts
                """, vals)
                conn.commit()

        return True
    return False


def fetch_daily_goalie_advanced_stats():
    # GOALIE ADVANCED REPORT API
    # https://api.nhle.com/stats/rest/en/goalie/advanced?isAggregate=false&isGame=true&sort=%5B%7B%22property%22:%22goalsFor%22,%22direction%22:%22DESC%22%7D%5D&start=0&limit=100&cayenneExp=gameDate%3E=%222025-10-07%22%20and%20gameDate%3C=%222025-10-07%22%20and%20gameTypeId=2

    print("\n--- Fetching Daily Goalie Advanced Stats (Goals For) ---")
    est = pytz.timezone('US/Eastern')
    today = datetime.now(est).date()
    target_end_date = today - timedelta(days=1)
    target_start_date = today - timedelta(days=7)

    # 1. Ensure the 'goalsfor' column exists in the table
    print("Verifying schema...")
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            # We use a safe check to add the column if it doesn't exist
            cursor.execute("""
                ALTER TABLE historical_goalie_stats
                ADD COLUMN IF NOT EXISTS goalsfor INTEGER
            """)
        conn.commit()

    # 2. Setup Dates and Loop
    dates = []
    curr = target_start_date
    while curr <= target_end_date:
        dates.append(curr.strftime("%Y-%m-%d"))
        curr += timedelta(days=1)

    all_data = []
    BASE_URL = "https://api.nhle.com/stats/rest/en/goalie/advanced"

    for d in dates:
        print(f"Querying Advanced Goalie Stats for {d}...")
        idx = 0
        try:
            while True:
                params = {
                    "isAggregate": "false",
                    "isGame": "true",
                    "sort": '[{"property":"goalsFor","direction":"DESC"}]',
                    "start": idx,
                    "limit": 100,
                    "cayenneExp": f'gameDate>="{d}" and gameDate<="{d}" and gameTypeId=2'
                }
                resp = requests.get(BASE_URL, params=params, timeout=10)
                data = resp.json().get("data", [])
                if not data: break

                for p in data:
                    rec = {
                        "gameid": p.get("gameId"),
                        "nhlplayerid": p.get("playerId"),
                        # The specific stat you requested
                        "goalsfor": p.get("goalsFor")
                    }
                    all_data.append(rec)
                idx += 100
                time.sleep(0.1)
        except Exception as e:
            print(f"Error fetching advanced stats for {d}: {e}")

    # 3. Update the Database
    if all_data:
        print(f"Updating {len(all_data)} records with Advanced Stats...")
        df = pd.DataFrame(all_data)

        # Ensure IDs are numeric
        df['nhlplayerid'] = pd.to_numeric(df['nhlplayerid'], errors='coerce')

        # Prepare data for bulk update
        vals = df[['goalsfor', 'gameid', 'nhlplayerid']].values.tolist()

        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                # Update existing rows based on GameID and PlayerID
                cursor.executemany("""
                    UPDATE historical_goalie_stats
                    SET goalsfor = %s
                    WHERE gameid = %s AND nhlplayerid = %s
                """, vals)
            conn.commit()
            print("Advanced stats updated.")
        return True

    print("No advanced data found to update.")
    return False



def fetch_daily_goalie_days_rest_stats():
    # GOALIE DAYS REST REPORT API
    # https://api.nhle.com/stats/rest/en/goalie/daysrest

    print("\n--- Fetching Daily Goalie Days Rest Stats ---")
    est = pytz.timezone('US/Eastern')
    today = datetime.now(est).date()
    target_end_date = today - timedelta(days=1)
    target_start_date = today - timedelta(days=7)

    # 1. Ensure the 'daysrest' column exists in the table
    print("Verifying schema...")
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                ALTER TABLE historical_goalie_stats
                ADD COLUMN IF NOT EXISTS daysrest INTEGER
            """)
        conn.commit()

    # 2. Setup Dates and Loop
    dates = []
    curr = target_start_date
    while curr <= target_end_date:
        dates.append(curr.strftime("%Y-%m-%d"))
        curr += timedelta(days=1)

    all_data = []
    BASE_URL = "https://api.nhle.com/stats/rest/en/goalie/daysrest"

    for d in dates:
        print(f"Querying Days Rest Stats for {d}...")
        idx = 0
        try:
            while True:
                params = {
                    "isAggregate": "false",
                    "isGame": "true",
                    "sort": '[{"property":"wins","direction":"DESC"}]', # Sort doesn't matter much for fetching all
                    "start": idx,
                    "limit": 100,
                    "cayenneExp": f'gameDate>="{d}" and gameDate<="{d}" and gameTypeId=2'
                }
                resp = requests.get(BASE_URL, params=params, timeout=10)
                data = resp.json().get("data", [])
                if not data: break

                for p in data:
                    # Logic to determine the single 'daysrest' integer from the buckets
                    rest_val = 0 # Default
                    if p.get("gamesPlayedDaysRest0") == 1:
                        rest_val = 0
                    elif p.get("gamesPlayedDaysRest1") == 1:
                        rest_val = 1
                    elif p.get("gamesPlayedDaysRest2") == 1:
                        rest_val = 2
                    elif p.get("gamesPlayedDaysRest3") == 1:
                        rest_val = 3
                    elif p.get("gamesPlayedDaysRest4Plus") == 1:
                        rest_val = 4 # Represents 4 or more days

                    rec = {
                        "gameid": p.get("gameId"),
                        "nhlplayerid": p.get("playerId"),
                        "daysrest": rest_val
                    }
                    all_data.append(rec)
                idx += 100
                time.sleep(0.1)
        except Exception as e:
            print(f"Error fetching days rest for {d}: {e}")

    # 3. Update the Database
    if all_data:
        print(f"Updating {len(all_data)} records with Days Rest...")
        df = pd.DataFrame(all_data)

        # Ensure IDs are numeric
        df['nhlplayerid'] = pd.to_numeric(df['nhlplayerid'], errors='coerce')

        # Prepare data for bulk update
        vals = df[['daysrest', 'gameid', 'nhlplayerid']].values.tolist()

        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                # Update existing rows based on GameID and PlayerID
                cursor.executemany("""
                    UPDATE historical_goalie_stats
                    SET daysrest = %s
                    WHERE gameid = %s AND nhlplayerid = %s
                """, vals)
            conn.commit()
            print("Days rest updated.")
        return True

    print("No days rest data found to update.")
    return False


# --- MERGING FUNCTIONS ---

def join_special_teams_data():
    # ... (No changes here) ...
    print("\n--- Joining Special Teams into Projections ---")
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT to_regclass('public.projections')")
            if not cursor.fetchone()[0]:
                print("Table 'projections' missing. Skipping special teams join.")
                return

        df_proj = read_sql_postgres("SELECT * FROM projections", conn)
        if df_proj.empty: return

        df_lg = read_sql_postgres("""
            SELECT nhlplayerid,
                   "lg_ppTimeOnIce", "lg_ppTimeOnIcePctPerGame",
                   "lg_ppAssists", "lg_ppGoals"
            FROM last_game_pp
        """, conn)

        df_lw = read_sql_postgres("""
            SELECT nhlplayerid,
                   "avg_ppTimeOnIce", "avg_ppTimeOnIcePctPerGame",
                   "total_ppAssists", "total_ppGoals",
                   "player_games_played", "team_games_played"
            FROM last_week_pp
        """, conn)

        df_final = pd.merge(df_proj, df_lg, on='nhlplayerid', how='left')
        df_final = pd.merge(df_final, df_lw, on='nhlplayerid', how='left')

        if 'nhlplayerid' in df_final.columns:
            df_final['nhlplayerid'] = pd.to_numeric(df_final['nhlplayerid'], errors='coerce').astype('Int64')

        df_to_postgres(df_final, 'projections', conn, lowercase_columns=False, primary_key='player_name_normalized')

        with conn.cursor() as cursor:
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_proj_norm ON projections(player_name_normalized)")
        conn.commit()

def perform_smart_join(base_df, merge_df, merge_cols, source_name, conn):
    # ... (No changes here) ...
    if 'teamabbrevs' in merge_df.columns and 'team' not in merge_df.columns:
        merge_df = merge_df.rename(columns={'teamabbrevs': 'team'})

    print(f"  Smart Join: {source_name}")

    base_df['mc_key'] = base_df['nhlplayerid'].astype(str) + "_" + base_df['team'].astype(str)
    merge_df['mc_key'] = merge_df['nhlplayerid'].astype(str) + "_" + merge_df['team'].astype(str)

    mask_exact = merge_df['mc_key'].isin(base_df['mc_key'])
    df_exact = merge_df[mask_exact].copy()
    df_remain = merge_df[~mask_exact].copy()

    df_name = pd.DataFrame()
    if 'player_name_normalized' in df_remain.columns:
        mask_name = df_remain['player_name_normalized'].isin(base_df['player_name_normalized'])
        df_name = df_remain[mask_name].copy()
        df_unmatched = df_remain[~mask_name].copy()
    else:
        df_unmatched = df_remain.copy()

    log_unmatched_players(conn, df_unmatched, source_name)

    cols_found = []
    for c in merge_cols:
        if c in merge_df.columns:
            cols_found.append(c)
        elif c.lower() in merge_df.columns:
            merge_df.rename(columns={c.lower(): c}, inplace=True)
            df_exact.rename(columns={c.lower(): c}, inplace=True)
            if not df_name.empty:
                df_name.rename(columns={c.lower(): c}, inplace=True)
            cols_found.append(c)

    cols_to_select = ['nhlplayerid', 'team'] + cols_found
    cols_exact = [c for c in cols_to_select if c in df_exact.columns]

    df_merged = pd.merge(base_df, df_exact[cols_exact], on=['nhlplayerid', 'team'], how='left', suffixes=('', '_new'))

    if not df_name.empty:
        cols_name = list(set(['player_name_normalized', 'nhlplayerid'] + cols_exact))
        cols_name = [c for c in cols_name if c in df_name.columns]
        df_merged = pd.merge(df_merged, df_name[cols_name], on='player_name_normalized', how='left', suffixes=('', '_name'))

        if 'nhlplayerid_name' in df_merged.columns:
            df_merged['nhlplayerid'] = np.where(df_merged['nhlplayerid_name'].notna(), df_merged['nhlplayerid_name'], df_merged['nhlplayerid'])

    for col in merge_cols:
        c_new = f"{col}_new"
        c_name = f"{col}_name"

        if col in df_merged.columns and col not in base_df.columns:
            final_col = df_merged[col]
        else:
            final_col = pd.Series(np.nan, index=df_merged.index)

        if c_new in df_merged.columns: final_col = final_col.fillna(df_merged[c_new])
        if c_name in df_merged.columns: final_col = final_col.fillna(df_merged[c_name])

        df_merged[col] = final_col

    drop_cols = [c for c in df_merged.columns if c.endswith('_new') or c.endswith('_name') or c == 'mc_key']
    return df_merged.drop(columns=drop_cols)

def create_stats_to_date_table():
    print("\n--- Creating stats_to_date ---")
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("TRUNCATE TABLE unmatched_players")
        conn.commit()

        df_proj = read_sql_postgres("SELECT * FROM projections", conn)
        if 'gp' in df_proj.columns: df_proj.rename(columns={'gp': 'GP'}, inplace=True)
        df_proj['nhlplayerid'] = pd.to_numeric(df_proj['nhlplayerid'], errors='coerce').fillna(0).astype(int)
        df_proj.drop_duplicates(subset=['nhlplayerid'], inplace=True)

        drop_rank_cols = [c for c in df_proj.columns if c.endswith('_cat_rank')]
        if drop_rank_cols:
            df_proj.drop(columns=drop_rank_cols, inplace=True)

        df_sc = read_sql_postgres("SELECT * FROM scoring_to_date", conn)
        # --- UPDATED MAPPING: Added gwgoals to GWG ---
        sc_map = {
            'gamesplayed': 'GPskater', 'goals': 'G', 'assists': 'A', 'points': 'P', 'plusminus': 'plus_minus',
            'penaltyminutes': 'PIM', 'ppgoals': 'PPG', 'ppassists': 'PPA', 'pppoints': 'PPP',
            'shgoals': 'SHG', 'shassists': 'SHA', 'shpoints': 'SHP',
            'gwgoals': 'GWG',  # Added GWG mapping
            'shootingpct': 'shootingPct', 'toi/g': 'TOI/G', 'shots': 'SOG'
        }
        df_sc.rename(columns=sc_map, inplace=True)
        df_sc['nhlplayerid'] = pd.to_numeric(df_sc['nhlplayerid'], errors='coerce').fillna(0).astype(int)

        df_merged = perform_smart_join(df_proj, df_sc, list(sc_map.values()), 'scoring', conn)

        df_bn = read_sql_postgres("SELECT * FROM bangers_to_date", conn)
        bn_map = {'blockspergame': 'BLK', 'hitspergame': 'HIT'}
        df_bn.rename(columns=bn_map, inplace=True)
        df_bn['nhlplayerid'] = pd.to_numeric(df_bn['nhlplayerid'], errors='coerce').fillna(0).astype(int)

        df_merged = perform_smart_join(df_merged, df_bn, list(bn_map.values()), 'bangers', conn)

        # --- NEW: Join Faceoff Stats ---
        df_fo = read_sql_postgres("SELECT * FROM faceoff_to_date", conn)
        fo_map = {'totalfaceoffwins': 'FW', 'totalfaceofflosses': 'FL', 'faceoffwinpct': 'FOpct'}
        df_fo.rename(columns=fo_map, inplace=True)
        df_fo['nhlplayerid'] = pd.to_numeric(df_fo['nhlplayerid'], errors='coerce').fillna(0).astype(int)

        df_merged = perform_smart_join(df_merged, df_fo, list(fo_map.values()), 'faceoffs', conn)

        # --- FIX for GOALIE JOIN ---
        df_gl = read_sql_postgres("SELECT * FROM goalie_to_date", conn)
        gl_map = {
            'gamesstarted': 'GS', 'gamesplayed': 'GP', 'goalsagainstaverage': 'GAA', 'losses': 'L',
            'savepct': 'SVpct', 'saves': 'SV', 'shotsagainst': 'SA', 'shutouts': 'SHO', 'wins': 'W',
            'goalsagainst': 'GA', 'startpct': 'startpct', 'toi/g': 'TOI/G'
        }
        df_gl.rename(columns=gl_map, inplace=True)
        df_gl['nhlplayerid'] = pd.to_numeric(df_gl['nhlplayerid'], errors='coerce').fillna(0).astype(int)

        # Use Name-Only Match for Goalies (overwrites missing IDs in Projections)
        print(f"  Smart Join: goalies (Name Only)")
        cols_to_use = list(gl_map.values()) + ['player_name_normalized', 'nhlplayerid']
        cols_to_use = [c for c in cols_to_use if c in df_gl.columns]

        df_merged = pd.merge(df_merged, df_gl[cols_to_use], on='player_name_normalized', how='left', suffixes=('', '_gl'))

        # Coalesce Stats
        for col in gl_map.values():
            if f"{col}_gl" in df_merged.columns:
                 df_merged[col] = df_merged[col].fillna(df_merged[f"{col}_gl"])

        # Fix Broken IDs from Projections using correct API ID
        if 'nhlplayerid_gl' in df_merged.columns:
             # If projection ID is missing (0) but we found a name match, update it
             df_merged['nhlplayerid'] = np.where(
                 df_merged['nhlplayerid'] == 0,
                 df_merged['nhlplayerid_gl'].fillna(0),
                 df_merged['nhlplayerid']
             )

        df_to_postgres(df_merged, 'stats_to_date', conn, lowercase_columns=False, primary_key='player_name_normalized')

def calculate_and_save_to_date_ranks():
    # ... (No changes here) ...
    print("\n--- Calculating Ranks ---")
    with get_db_connection() as conn:
        df = read_sql_postgres("SELECT * FROM stats_to_date", conn)
        if df.empty: return

        # --- UPDATED: Added GWG, FW, FL, FOpct to list of ranked stats ---
        skater_stats = ['G', 'A', 'P', 'PPG', 'PPA', 'PPP', 'SHG', 'SHA', 'SHP', 'GWG', 'HIT', 'BLK', 'PIM', 'FOW', 'SOG', 'plus_minus', 'FW', 'FL', 'FOpct']
        goalie_stats = {'GS': False, 'W': False, 'L': True, 'GA': True, 'SA': False, 'SV': False, 'SVpct': False, 'GAA': True, 'SHO': False, 'QS': False}

        if 'positions' not in df.columns: return

        mask_skater = ~df['positions'].str.contains('G', na=False)
        num_skaters = mask_skater.sum()
        if num_skaters > 0:
            for stat in skater_stats:
                col = f"{stat}_cat_rank"
                s_key = stat if stat in df.columns else stat.lower()

                if s_key in df.columns:
                    df[s_key] = pd.to_numeric(df[s_key], errors='coerce').fillna(0)
                    ranks = df.loc[mask_skater, s_key].rank(method='first', ascending=False)
                    pct = ranks / num_skaters

                    cond = [pct<=0.05, pct<=0.10, pct<=0.15, pct<=0.20, pct<=0.25, pct<=0.30, pct<=0.35, pct<=0.40, pct<=0.45, pct<=0.50, pct<=0.75]
                    choice = [1,2,3,4,5,6,7,8,9,10,15]
                    df.loc[mask_skater, col] = np.select(cond, choice, default=20)

        mask_goalie = df['positions'].str.contains('G', na=False)
        num_goalies = mask_goalie.sum()
        if num_goalies > 0:
            for stat, is_inv in goalie_stats.items():
                col = f"{stat}_cat_rank"
                s_key = stat if stat in df.columns else stat.lower()
                if stat == 'TOI/G': s_key = 'TOI/G' if 'TOI/G' in df.columns else 'toi/g'

                if s_key in df.columns:
                    df[s_key] = pd.to_numeric(df[s_key], errors='coerce').fillna(0)
                    ranks = df.loc[mask_goalie, s_key].rank(method='first', ascending=is_inv)
                    pct = ranks / num_goalies
                    cond = [pct<=0.05, pct<=0.10, pct<=0.15, pct<=0.20, pct<=0.25, pct<=0.30, pct<=0.35, pct<=0.40, pct<=0.45, pct<=0.50, pct<=0.75]
                    choice = [1,2,3,4,5,6,7,8,9,10,15]
                    df.loc[mask_goalie, col] = np.select(cond, choice, default=20)

        df_to_postgres(df, 'stats_to_date', conn, lowercase_columns=False, primary_key='player_name_normalized')

def create_combined_projections():
    # ... (No changes here) ...
    print("\n--- Creating Combined Projections ---")
    with get_db_connection() as conn:
        df_proj = read_sql_postgres("SELECT * FROM projections", conn)
        df_stats = read_sql_postgres("SELECT * FROM stats_to_date", conn)

        if 'gp' in df_proj.columns: df_proj.rename(columns={'gp': 'GP'}, inplace=True)

        df_merged = pd.merge(df_proj, df_stats, on='nhlplayerid', how='outer', suffixes=('_proj', '_stats'))

        # 1. Init with ID
        df_final_ids = df_merged[['nhlplayerid']].copy()

        # 2. Collect new columns in a dict (Performance Optimization)
        final_processed_data = {}

        id_cols = ['player_name_normalized', 'player_name', 'team', 'age', 'player_id', 'positions', 'status',
                   'lg_ppTimeOnIce', 'lg_ppTimeOnIcePctPerGame', 'lg_ppAssists', 'lg_ppGoals',
                   'avg_ppTimeOnIce', 'avg_ppTimeOnIcePctPerGame', 'total_ppAssists', 'total_ppGoals',
                   'player_games_played', 'team_games_played']

        for col in id_cols:
            col_lower = col.lower()
            c_proj, c_stat = f"{col}_proj", f"{col}_stats"
            c_proj_l, c_stat_l = f"{col_lower}_proj", f"{col_lower}_stats"

            def get_series(name_list):
                for n in name_list:
                    if n in df_merged: return df_merged[n]
                return None

            s_proj = get_series([c_proj, c_proj_l, col, col_lower])
            s_stat = get_series([c_stat, c_stat_l])

            if s_stat is not None and s_proj is not None:
                final_processed_data[col] = s_stat.fillna(s_proj)
            elif s_stat is not None:
                final_processed_data[col] = s_stat
            elif s_proj is not None:
                final_processed_data[col] = s_proj

        skip = set([c.lower() for c in id_cols] + ['nhlplayerid'])
        all_data_cols = (set(df_proj.columns) | set(df_stats.columns)) - skip

        for col in all_data_cols:
            if col.lower() in [c.lower() for c in id_cols]: continue

            def get_numeric(base, suffix):
                keys = [f"{base}{suffix}", f"{base.lower()}{suffix}", base, base.lower()]
                for k in keys:
                    if k in df_merged: return pd.to_numeric(df_merged[k], errors='coerce')
                return pd.Series(0.0, index=df_merged.index)

            s_proj = get_numeric(col, "_proj")
            s_stat = get_numeric(col, "_stats")

            s_proj = s_proj.fillna(0)
            s_stat = s_stat.fillna(0)

            has_p = (col in df_proj.columns) or (col.lower() in df_proj.columns)
            has_s = (col in df_stats.columns) or (col.lower() in df_stats.columns)

            if has_p and has_s:
                final_processed_data[col] = (s_proj + s_stat) / 2
            elif has_s:
                final_processed_data[col] = s_stat
            elif has_p:
                final_processed_data[col] = s_proj

        # 3. Create DataFrame and Concat
        df_data = pd.DataFrame(final_processed_data)
        df_final = pd.concat([df_final_ids, df_data], axis=1)

        df_to_postgres(df_final, 'combined_projections', conn, lowercase_columns=False, primary_key='player_name_normalized')


def create_player_lines_table():
    print("\n--- Creating 'player_lines' Table (Canonical Ranks + Alt Lines + PP Units) ---")

    # 1. EXPLICITLY DROP THE TABLE FIRST
    print("Clearing existing 'player_lines' table...")
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("DROP TABLE IF EXISTS player_lines")
        conn.commit()

    # 2. Fetch Data
    with get_db_connection() as conn:
        print("Fetching most recent games and checking active roster status...")

        df_players = read_sql_postgres("""
            WITH player_max_dates AS (
                SELECT nhlplayerid, MAX(gamedate) as player_last_date
                FROM historical_stats
                GROUP BY nhlplayerid
            ),
            team_max_dates AS (
                SELECT teamabbrev, MAX(gamedate) as team_last_date
                FROM historical_stats
                GROUP BY teamabbrev
            )
            SELECT
                p.player_name_normalized,
                p.team,
                p.nhlplayerid,
                p.positions,
                h.gameid,
                h.gamedate as player_game_date,
                h.timeonice,
                h.skaterfullname, -- [NEW] Fetch the pretty name
                tm.team_last_date
            FROM stats_to_date p
            JOIN player_max_dates pmd ON p.nhlplayerid = pmd.nhlplayerid
            JOIN historical_stats h ON h.nhlplayerid = pmd.nhlplayerid AND h.gamedate = pmd.player_last_date
            JOIN team_max_dates tm ON h.teamabbrev = tm.teamabbrev
        """, conn)

        if df_players.empty:
            print("No player data found.")
            return

        # Active = Player's last game is the same as the Team's last game
        df_active = df_players[df_players['player_game_date'] == df_players['team_last_date']].copy()
        df_inactive = df_players[df_players['player_game_date'] < df_players['team_last_date']].copy()

        if df_active.empty:
            print("No active players found for recent games.")
            return

        unique_game_ids = df_active['gameid'].unique().tolist()

        print(f"Loading shift data for {len(unique_game_ids)} active games...")
        game_id_str = ",".join(str(gid) for gid in unique_game_ids)

        query_shifts = f"""
            SELECT gameid, nhlplayerid, period, start_seconds, end_seconds
            FROM game_shifts
            WHERE gameid IN ({game_id_str})
        """
        df_shifts = read_sql_postgres(query_shifts, conn)

        # --- NEW: Fetch PP Data ---
        print("Loading PP Data...")
        df_pp = read_sql_postgres('SELECT nhlplayerid, "lg_ppTimeOnIce" FROM last_game_pp', conn)

        if df_shifts.empty:
            print("No shift data found. Run 'fetch_game_shifts' first.")
            return

    # 3. Process Logic: Calculate ALL Combinations for Active Players
    print("Calculating line combinations...")

    player_positions = {}
    player_names = {}

    for _, row in df_active.iterrows():
        pid = row['nhlplayerid']
        pos_str = str(row['positions']).upper()

        # [FIX] Use the pretty name (skaterfullname) if available, fallback to normalized
        pretty_name = row['skaterfullname']
        if not pretty_name:
            pretty_name = row['player_name_normalized']

        player_names[pid] = pretty_name

        if 'D' in pos_str:
            player_positions[pid] = 'D'
        elif any(x in pos_str for x in ['C', 'L', 'R']):
            player_positions[pid] = 'F'
        else:
            player_positions[pid] = 'G' if 'G' in pos_str else 'F'

    def get_overlap(start1, end1, start2, end2):
        return max(0, min(end1, end2) - max(start1, start2))

    # Collection of every line instance
    raw_line_instances = []

    games = df_shifts.groupby('gameid')

    for game_id, group in games:
        p_shifts = group.groupby('nhlplayerid')
        valid_players = [p for p in p_shifts.groups.keys() if p in player_positions]

        for pid in valid_players:
            my_pos = player_positions.get(pid)
            if my_pos == 'G': continue

            # Optimization: Pre-filter teammates to only same team
            try:
                my_team = df_players.loc[df_players['nhlplayerid'] == pid, 'team'].values[0]
            except: continue

            my_shift_list = group[group['nhlplayerid'] == pid].to_dict('records')
            teammates = {}

            for teammate_id in valid_players:
                if pid == teammate_id: continue

                # Check Team
                try:
                    t_team = df_players.loc[df_players['nhlplayerid'] == teammate_id, 'team'].values[0]
                    if my_team != t_team: continue
                except: continue

                mate_shift_list = group[group['nhlplayerid'] == teammate_id].to_dict('records')
                total_time = 0

                for s1 in my_shift_list:
                    for s2 in mate_shift_list:
                        if s1['period'] == s2['period']:
                            total_time += get_overlap(s1['start_seconds'], s1['end_seconds'],
                                                      s2['start_seconds'], s2['end_seconds'])
                if total_time > 0:
                    teammates[teammate_id] = total_time

            sorted_mates = sorted(teammates.items(), key=lambda item: item[1], reverse=True)

            final_mates = []
            if my_pos == 'F':
                count = 0
                for mate_id, time_tog in sorted_mates:
                    if player_positions.get(mate_id) == 'F':
                        final_mates.append(mate_id)
                        count += 1
                    if count >= 2: break
            elif my_pos == 'D':
                count = 0
                for mate_id, time_tog in sorted_mates:
                    if player_positions.get(mate_id) == 'D':
                        final_mates.append(mate_id)
                        count += 1
                    if count >= 1: break

            full_line_ids = final_mates + [pid]
            # Names are now "Cale Makar", "Devon Toews", etc.
            full_line_names = sorted([player_names.get(x, str(x)) for x in full_line_ids])
            line_id_str = ",".join(sorted([str(x) for x in full_line_ids]))
            line_comb = " - ".join(full_line_names)
            linemates_str = " | ".join([player_names.get(mid, str(mid)) for mid in final_mates])

            raw_line_instances.append({
                "nhlplayerid": pid,
                "gameid": game_id,
                "team": my_team,
                "linemates": linemates_str,
                "full_line": line_comb,
                "line_ids": line_id_str,
                "pos_type": my_pos,
                "weight": sum(teammates.get(m, 0) for m in final_mates)
            })

    if not raw_line_instances:
        print("No lines calculated.")
        return

    df_lines_raw = pd.DataFrame(raw_line_instances)

    # 4. PASS 1: Identify "Canonical" Lines (The Ranked Ones)
    print("Identifying Canonical Lines...")

    line_summary = df_lines_raw.groupby(['gameid', 'team', 'pos_type', 'full_line', 'line_ids'])['weight'].mean().reset_index()
    line_summary.rename(columns={'weight': 'avg_toi_proxy'}, inplace=True)
    line_summary.sort_values(by=['gameid', 'team', 'pos_type', 'avg_toi_proxy'], ascending=[True, True, True, False], inplace=True)

    canonical_assignments = {}

    for keys, group in line_summary.groupby(['gameid', 'team', 'pos_type']):
        _, _, pos_type = keys
        lines = group.to_dict('records')

        rank_counter = 1

        for line in lines:
            if not line['line_ids']: continue
            p_ids = line['line_ids'].split(',')

            # Check if any player is already taken by a higher ranked line
            is_blocked = False
            for p_id in p_ids:
                if (line['gameid'], int(p_id)) in canonical_assignments:
                    is_blocked = True
                    break

            if not is_blocked:
                # Assign Rank!
                label = 'Depth'
                if pos_type == 'F' and rank_counter <= 4:
                    label = str(rank_counter)
                elif pos_type == 'D' and rank_counter <= 3:
                    label = str(rank_counter)

                # Register players
                for p_id in p_ids:
                    canonical_assignments[(line['gameid'], int(p_id))] = {
                        'rank': label,
                        'full_line': line['full_line']
                    }

                rank_counter += 1

    # 5. PASS 2: Construct Final Player Rows (Canonical + Alt Lines)
    print("Consolidating Player Data...")

    final_rows = []

    for _, player in df_active.iterrows():
        pid = player['nhlplayerid']
        gid = player['gameid']

        p_lines = df_lines_raw[(df_lines_raw['nhlplayerid'] == pid) & (df_lines_raw['gameid'] == gid)].sort_values('weight', ascending=False)

        if p_lines.empty:
            continue

        assignment = canonical_assignments.get((gid, pid))

        if assignment:
            final_rank = assignment['rank']
            final_full_line = assignment['full_line']
            match = p_lines[p_lines['full_line'] == final_full_line]
            final_linemates = match.iloc[0]['linemates'] if not match.empty else ""
        else:
            top_line = p_lines.iloc[0]
            final_rank = 'Depth'
            final_full_line = top_line['full_line']
            final_linemates = top_line['linemates']

        # Determine Alt Lines
        ALT_THRESHOLD = 45

        alt_line_list = []
        for _, row in p_lines.iterrows():
            if row['full_line'] != final_full_line and row['weight'] > ALT_THRESHOLD:
                alt_line_list.append(row['full_line'])

        unique_alts = list(dict.fromkeys(alt_line_list))
        alt_lines_str = ", ".join(unique_alts) if unique_alts else None

        final_rows.append({
            "nhlplayerid": pid,
            "gameid": gid,
            "team": player['team'],
            "player_name_normalized": player['player_name_normalized'],
            "positions": player['positions'],
            "timeonice": player['timeonice'],
            "linemates": final_linemates,
            "full_line": final_full_line,
            "line_number": final_rank,
            "alt_lines": alt_lines_str,
            "pos_type": p_lines.iloc[0]['pos_type']
        })

    df_final_active = pd.DataFrame(final_rows)

    # --- NEW: CALCULATE PP UNITS ---
    print("Calculating PP Units (PP1 vs PP2)...")
    if not df_pp.empty:
        df_final_active = pd.merge(df_final_active, df_pp, on='nhlplayerid', how='left')
        df_final_active['lg_ppTimeOnIce'] = df_final_active['lg_ppTimeOnIce'].fillna(0)

        df_final_active['pp_rank_int'] = df_final_active.groupby('team')['lg_ppTimeOnIce'].rank(method='first', ascending=False)

        def assign_pp_unit(row):
            if row['lg_ppTimeOnIce'] <= 0: return None
            if row['pp_rank_int'] <= 5: return "PP1"
            if row['pp_rank_int'] <= 10: return "PP2"
            return None

        df_final_active['pp_unit'] = df_final_active.apply(assign_pp_unit, axis=1)

        pp_combinations = {}
        pp_players = df_final_active[df_final_active['pp_unit'].notna()]

        for keys, group in pp_players.groupby(['team', 'pp_unit']):
            team_key, unit_key = keys
            # [FIX] Use the player map to get pretty names for PP lines too
            # We can't use player_names dict directly here because df_final_active iterrows is cleaner
            # But we can just grab the ID and look it up!

            # Helper to get name from ID in the dataframe
            # Note: df_final_active doesn't have the 'skaterfullname' column directly,
            # but we can look up the ID in our existing player_names dict

            member_names = []
            for pid in group['nhlplayerid']:
                name = player_names.get(pid, str(pid))
                member_names.append(name)

            names = sorted(member_names)
            pp_line_str = " - ".join(names)
            pp_combinations[(team_key, unit_key)] = pp_line_str

        def assign_pp_line(row):
            if not row['pp_unit']: return None
            return pp_combinations.get((row['team'], row['pp_unit']))

        df_final_active['pp_line'] = df_final_active.apply(assign_pp_line, axis=1)
        df_final_active.drop(columns=['lg_ppTimeOnIce', 'pp_rank_int'], inplace=True, errors='ignore')

    else:
        df_final_active['pp_unit'] = None
        df_final_active['pp_line'] = None

    # 6. Handle INACTIVE Players
    if not df_inactive.empty:
        df_inactive_out = df_inactive.copy()

        # Zero out TOI for scratched players so modal doesn't show old data
        df_inactive_out['timeonice'] = 0

        df_inactive_out['linemates'] = 'Scratched/Injured'
        df_inactive_out['full_line'] = ''
        df_inactive_out['line_number'] = 'Depth'
        df_inactive_out['alt_lines'] = None
        df_inactive_out['pos_type'] = ''
        df_inactive_out['pp_unit'] = None
        df_inactive_out['pp_line'] = None

        cols = ['nhlplayerid', 'gameid', 'team', 'player_name_normalized', 'positions',
                'timeonice', 'linemates', 'full_line', 'line_number', 'alt_lines', 'pos_type', 'pp_unit', 'pp_line']

        for c in cols:
            if c not in df_final_active.columns: df_final_active[c] = None

        df_total = pd.concat([df_final_active[cols], df_inactive_out[cols]], ignore_index=True)
    else:
        df_total = df_final_active

    # 7. Write to Postgres
    with get_db_connection() as conn:
        df_to_postgres(df_total, 'player_lines', conn, if_exists='replace', primary_key='nhlplayerid')

    print("Done. 'player_lines' table created.")


def verify_shift_coverage():
    print("\n--- Verifying Shift Data Integrity (Last 7 Days) ---")

    with get_db_connection() as conn:
        # 1. Get list of missing games
        df_missing = read_sql_postgres("""
            SELECT DISTINCT h.gameid, h.gamedate
            FROM historical_stats h
            WHERE h.gamedate >= CURRENT_DATE - INTERVAL '7 days'
            AND NOT EXISTS (
                SELECT 1
                FROM game_shifts s
                WHERE s.gameid = h.gameid
            )
            ORDER BY h.gamedate
        """, conn)

        # 2. Report Results
        if df_missing.empty:
            print("SUCCESS: All recent games in 'historical_stats' have corresponding shift data.")
        else:
            print(f"WARNING: Found {len(df_missing)} games with MISSING shift data:")
            print(df_missing)

            # Optional: Return the list of missing IDs if you want to auto-fix them
            return df_missing['gameid'].tolist()

    return []


def create_player_trends_table():
    """
    Ensures the player_trends table exists and syncs new players from stats_to_date.
    Does NOT drop the table or wipe existing data.
    """
    print("Verifying player_trends schema and syncing roster...")

    # 1. Generate the 5-game block column definitions
    block_columns = []

    # Generate 1-5, 6-10 ... 71-75
    for i in range(0, 75, 5):
        start = i + 1
        end = i + 5
        col_name = f"block_{start}_{end}"
        block_columns.append(f"{col_name} JSONB")

    # Add the final block (76-82)
    block_columns.append("block_76_82 JSONB")

    blocks_sql = ",\n        ".join(block_columns)

    # 2. Create Table (Only if it doesn't exist)
    create_query = f"""
    CREATE TABLE IF NOT EXISTS player_trends (
        player_name_normalized TEXT PRIMARY KEY,
        team TEXT,
        nhlplayerid INTEGER,

        -- Game Blocks
        {blocks_sql},

        -- Trend Columns
        l5_trend JSONB,
        l10_trend JSONB,
        l20_trend JSONB,
        home_trend JSONB,
        away_trend JSONB,

        -- Splits (Averages per Game)
        home_stats JSONB,
        away_stats JSONB,

        -- Splits (Running Totals - Hidden)
        home_total JSONB,
        away_total JSONB,

        -- Season Aggregates
        season_total JSONB,
        season_avg JSONB,

        -- Tracking
        last_date_used DATE
    );

    -- Create indexes
    CREATE INDEX IF NOT EXISTS idx_trends_team ON player_trends(team);
    CREATE INDEX IF NOT EXISTS idx_trends_last_date ON player_trends(last_date_used);
    """

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                # 1. Create structure if missing
                cursor.execute(create_query)

                # 2. Insert NEW players only (Safe Sync)
                # Uses 'DO NOTHING' on conflict to preserve existing 'last_date_used' and stats
                init_query = """
                INSERT INTO player_trends (player_name_normalized, team, nhlplayerid, last_date_used)
                SELECT DISTINCT player_name_normalized, team, nhlplayerid, '2000-01-01'::DATE
                FROM stats_to_date
                WHERE player_name_normalized IS NOT NULL
                ON CONFLICT (player_name_normalized) DO NOTHING;
                """
                cursor.execute(init_query)

                # 3. Update metadata for EXISTING players (e.g. Traded players or ID fixes)
                # This keeps teams/IDs current without wiping statistical history
                update_meta_query = """
                UPDATE player_trends pt
                SET team = s.team,
                    nhlplayerid = s.nhlplayerid
                FROM stats_to_date s
                WHERE pt.player_name_normalized = s.player_name_normalized
                  AND (pt.team IS DISTINCT FROM s.team OR pt.nhlplayerid IS DISTINCT FROM s.nhlplayerid);
                """
                cursor.execute(update_meta_query)

            conn.commit()
            print("player_trends table verified and roster synced.")

    except Exception as e:
        print(f"Error ensuring player_trends table: {e}")
        # raise e  # Uncomment if you want the script to stop on error


def update_player_trends():
    """
    Iterates through players, fetches new games, chunks them,
    aggregates stats (Season, Home, Away), and updates player_trends.
    HARDENED VERSION: Commits per-player and logs activity.
    """
    print("Starting player_trends update (Debug Mode)...")

    # 1. Define Columns
    SUM_COLS = [
        'missedshots', 'shotattemptsblocked', 'takeaways', 'giveaways',
        'blockedshots', 'hits', 'shots', 'shpoints', 'shgoals', 'pppoints',
        'ppgoals', 'points', 'plusminus', 'penaltyminutes', 'otgoals', 'goals',
        'gamewinninggoals', 'evpoints', 'evgoals', 'assists', 'shifts',
        'ottimeonice', 'shtimeonice', 'pptimeonice', 'evtimeonice', 'timeonice'
    ]

    BLOCK_KEYS = [f"block_{i+1}_{i+5}" for i in range(0, 75, 5)]
    BLOCK_KEYS.append("block_76_82")

    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:

            # 2. Get players and current state
            cursor.execute(f"""
                SELECT player_name_normalized, nhlplayerid, last_date_used,
                       season_total, home_total, away_total,
                       {', '.join(BLOCK_KEYS)}
                FROM player_trends
            """)
            players = cursor.fetchall()
            print(f"Found {len(players)} players to check.")

            for p in players:
                try:
                    p_name = p['player_name_normalized']
                    pid = p['nhlplayerid']
                    last_date = p['last_date_used'] or '2000-01-01'

                    # Initialize totals if None
                    season_total = p['season_total'] or {col: 0 for col in SUM_COLS}
                    home_total = p['home_total'] or {col: 0 for col in SUM_COLS}
                    away_total = p['away_total'] or {col: 0 for col in SUM_COLS}

                    for t in [season_total, home_total, away_total]:
                        if 'games_played' not in t: t['games_played'] = 0

                    # 3. Find first empty block
                    current_block_col = None
                    for key in BLOCK_KEYS:
                        if p.get(key) is None:
                            current_block_col = key
                            break

                    if not current_block_col: continue

                    # 4. Fetch NEW games
                    cols_sql = ", ".join(SUM_COLS)
                    # Use a new cursor for the inner query to keep the outer loop clean
                    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as inner_cursor:
                        inner_cursor.execute(f"""
                            SELECT gamedate, homeroad, {cols_sql}
                            FROM historical_stats
                            WHERE nhlplayerid = %s AND gamedate > %s
                            ORDER BY gamedate ASC
                        """, (pid, last_date))
                        new_games = inner_cursor.fetchall()

                    if not new_games: continue

                    # DEBUG LOGGING for the player in question
                    # (You can remove this condition later, or set it to the specific player name you are debugging)
                    if len(new_games) >= 5:
                        print(f"  -> Processing {p_name}: Found {len(new_games)} new games since {last_date}")

                    # 5. Chunking Logic
                    games_to_process = []

                    if current_block_col == "block_76_82":
                        chunk = new_games[:7]
                        if chunk: games_to_process.append((current_block_col, chunk))
                    else:
                        idx = 0
                        while idx + 5 <= len(new_games):
                            if not current_block_col: break
                            chunk = new_games[idx : idx+5]
                            games_to_process.append((current_block_col, chunk))
                            idx += 5
                            curr_idx = BLOCK_KEYS.index(current_block_col)
                            current_block_col = BLOCK_KEYS[curr_idx + 1] if curr_idx + 1 < len(BLOCK_KEYS) else None

                    if not games_to_process: continue

                    # 6. Aggregation Loop
                    for block_name, games in games_to_process:
                        block_stats = {col: 0 for col in SUM_COLS}
                        block_stats['games_played'] = len(games)

                        def parse_val(v, col_name):
                            if v is None: return 0
                            # Handle string TOI if necessary (though historical_stats should be int)
                            if 'timeonice' in col_name and isinstance(v, str) and ':' in v:
                                try:
                                    m, s = v.split(':')
                                    return int(m) * 60 + int(s)
                                except: return 0
                            return v

                        for game in games:
                            is_home = (game.get('homeroad') == 'H')
                            target_split = home_total if is_home else away_total

                            for col in SUM_COLS:
                                val = parse_val(game.get(col), col)
                                # Ensure val is numeric before adding
                                if isinstance(val, (int, float)):
                                    block_stats[col] += val
                                    season_total[col] = season_total.get(col, 0) + val
                                    target_split[col] = target_split.get(col, 0) + val

                            target_split['games_played'] += 1

                        season_total['games_played'] += len(games)

                        # Helper for Derived Stats
                        def calc_derived(stats_dict):
                            g, s = stats_dict.get('goals', 0), stats_dict.get('shots', 0)
                            stats_dict['shootingpct'] = round((g / s) * 100, 1) if s > 0 else 0.0
                            toi, shifts = stats_dict.get('timeonice', 0), stats_dict.get('shifts', 0)
                            stats_dict['timeonicepershift'] = round(toi / shifts) if shifts > 0 else 0

                        # Helper for Averages
                        def calc_avg(total_dict, scale=1.0):
                            avg_dict = {}
                            gp = total_dict.get('games_played', 0)
                            if gp > 0:
                                for k, v in total_dict.items():
                                    if k == 'games_played': continue
                                    avg_dict[k] = round((v / gp) * scale, 2)
                                calc_derived(avg_dict)
                                avg_dict['games_played'] = 1 if scale == 1 else 5
                            return avg_dict

                        calc_derived(block_stats)

                        season_avg = calc_avg(season_total, scale=5.0)
                        home_stats = calc_avg(home_total, scale=1.0)
                        away_stats = calc_avg(away_total, scale=1.0)

                        new_last_date = games[-1]['gamedate']

                        # Use a dedicated cursor for the UPDATE
                        with conn.cursor() as update_cursor:
                            update_cursor.execute(f"""
                                UPDATE player_trends
                                SET {block_name} = %s,
                                    season_total = %s, season_avg = %s,
                                    home_total = %s, home_stats = %s,
                                    away_total = %s, away_stats = %s,
                                    last_date_used = %s
                                WHERE player_name_normalized = %s
                            """, (
                                json.dumps(block_stats),
                                json.dumps(season_total), json.dumps(season_avg),
                                json.dumps(home_total), json.dumps(home_stats),
                                json.dumps(away_total), json.dumps(away_stats),
                                new_last_date,
                                p_name
                            ))
                            print(f"    Updated {p_name}: {block_name} (New Last Date: {new_last_date})")

                    # COMMIT PER PLAYER to save progress and isolate errors
                    conn.commit()

                except Exception as e:
                    print(f"ERROR updating {p.get('player_name_normalized', 'Unknown')}: {e}")
                    conn.rollback() # Rollback only this failed player
                    continue

    print("Player trends update complete.")

def calculate_trend_metrics(recent_stats, season_avg_stats, groups_config, anomaly_checks):
    """
    Generic trend calculator.
    Compares recent performance vs Season Average.
    """
    trends = {}
    anomalies = []

    # --- NORMALIZE RECENT STATS ---
    # Scale counts to "Per 5 Games" for comparison with Season Avg
    recent_gp = recent_stats.get('games_played', 5)
    if recent_gp == 0: recent_gp = 1
    scale_factor = 5.0 / recent_gp

    # Rates that should NOT be scaled
    RATE_STATS = ['shootingpct', 'timeonicepershift', 'savepct', 'gaa']

    normalized_recent = {}
    for k, v in recent_stats.items():
        if k in RATE_STATS:
            normalized_recent[k] = v
        else:
            normalized_recent[k] = (v or 0) * scale_factor

    # --- CALCULATE GROUPS ---
    for group_name, subsets in groups_config.items():
        total_variance = 0.0
        stat_count = 0

        # Inverse Stats (Lower is Better)
        for stat in subsets.get('inverse', []):
            s_val = season_avg_stats.get(stat, 0) or 0
            r_val = normalized_recent.get(stat, 0)

            if s_val < 0.5 and r_val < 0.5: continue

            if s_val == 0:
                diff = -1.0 if r_val > 0 else 0.0
            else:
                diff = (s_val - r_val) / s_val

            diff = max(-1.0, min(1.0, diff))
            total_variance += diff
            stat_count += 1

        # Standard Stats (Higher is Better)
        for stat in subsets.get('standard', []):
            s_val = season_avg_stats.get(stat, 0) or 0
            r_val = normalized_recent.get(stat, 0)

            if s_val < 0.5 and r_val < 0.5: continue

            if s_val == 0:
                diff = 1.0 if r_val > 0 else 0.0
            else:
                diff = (r_val - s_val) / s_val

            diff = max(-1.0, min(1.0, diff))
            total_variance += diff
            stat_count += 1

        if stat_count > 0:
            trends[group_name] = round(total_variance / stat_count, 3)
        else:
            trends[group_name] = 0.0

    # --- DETECT ANOMALIES ---
    if anomaly_checks:
        for check in anomaly_checks:
            s_total = 0
            r_total = 0

            for stat in check['stats']:
                s_total += (season_avg_stats.get(stat, 0) or 0)
                r_total += (normalized_recent.get(stat, 0) or 0)

            # --- NEW LOGIC: Check for Absolute vs Percentage ---
            threshold = check['threshold']
            is_absolute = check.get('is_absolute', False)

            if is_absolute:
                # Additive Difference (e.g. 16.0 - 6.0 = 10.0)
                diff = r_total - s_total
                if abs(diff) > threshold:
                    direction = "High" if diff > 0 else "Low"
                    anomalies.append(f"{check['name']} ({direction})")
            else:
                # Percentage Variance (Standard)
                if s_total == 0:
                    if r_total > 1:
                        anomalies.append(f"{check['name']} (High Spike)")
                else:
                    variance = abs((r_total - s_total) / s_total)
                    if variance > threshold:
                        direction = "High" if r_total > s_total else "Low"
                        anomalies.append(f"{check['name']} ({direction})")

    trends['anomalies'] = anomalies
    return trends


def generate_player_trends():
    """ Calculates Trends for SKATERS """
    print("Generating SKATER trends analysis...")

    # Skater Config
    SKATER_GROUPS = {
        'luck': {
            'inverse': ['missedshots', 'shotattemptsblocked', 'giveaways'],
            'standard': ['takeaways', 'plusminus', 'gamewinninggoals', 'shootingpct']
        },
        'bangers': { 'standard': ['blockedshots', 'hits', 'penaltyminutes'] },
        'scoring': { 'standard': ['shots', 'shootingpct', 'points', 'goals', 'evpoints', 'evgoals', 'assists'] },
        'special': { 'standard': ['shpoints', 'shgoals', 'pppoints', 'ppgoals', 'otgoals', 'ottimeonice', 'shtimeonice', 'pptimeonice'] },
        'utilization': { 'standard': ['timeonicepershift', 'shifts', 'evtimeonice', 'timeonice', 'ottimeonice', 'shtimeonice', 'pptimeonice'] }
    }

    SKATER_ANOMALIES = [
        {'name': 'Bad Luck', 'stats': ['missedshots', 'shotattemptsblocked'], 'threshold': 0.30},

        # PIM: 300% variance to catch major spikes
        {'name': 'PIM Spike', 'stats': ['penaltyminutes'], 'threshold': 3.00},

        # Shooting %: Now checking for ADDITIVE difference of 10.0 (e.g. 6% vs 16%)
        {'name': 'Shooting %', 'stats': ['shootingpct'], 'threshold': 10.0, 'is_absolute': True},

        {'name': 'Heavy Workload', 'stats': ['timeonice'], 'threshold': 0.40},
        {'name': 'Special Teams', 'stats': ['pptimeonice', 'ottimeonice', 'shtimeonice'], 'threshold': 0.25}
    ]

    BLOCK_KEYS = [f"block_{i+1}_{i+5}" for i in range(0, 75, 5)] + ["block_76_82"]

    # Skater Sum Cols
    SUM_COLS = [
        'missedshots', 'shotattemptsblocked', 'takeaways', 'giveaways', 'blockedshots', 'hits', 'shots',
        'shpoints', 'shgoals', 'pppoints', 'ppgoals', 'points', 'plusminus', 'penaltyminutes', 'otgoals',
        'goals', 'gamewinninggoals', 'evpoints', 'evgoals', 'assists', 'shifts', 'ottimeonice', 'shtimeonice',
        'pptimeonice', 'evtimeonice', 'timeonice', 'games_played'
    ]

    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                # Fetch only rows that look like skaters
                query_cols = ", ".join(BLOCK_KEYS)
                cursor.execute(f"""
                    SELECT player_name_normalized, season_avg, home_stats, away_stats, {query_cols}
                    FROM player_trends
                    WHERE season_avg->>'shifts' IS NOT NULL
                       OR season_avg->>'shots' IS NOT NULL
                """)
                players = cursor.fetchall()

                for p in players:
                    s_avg = p['season_avg']
                    if not s_avg: continue

                    # 1. Block Trends
                    valid_blocks = [p[k] for k in BLOCK_KEYS if p.get(k)]
                    periods = {
                        'l5_trend': valid_blocks[-1:] if valid_blocks else [],
                        'l10_trend': valid_blocks[-2:] if valid_blocks else [],
                        'l20_trend': valid_blocks[-4:] if valid_blocks else []
                    }
                    updates = {}

                    for col_name, blocks in periods.items():
                        if not blocks:
                            updates[col_name] = None
                            continue

                        agg_stats = {k: 0 for k in SUM_COLS}
                        for b in blocks:
                            for k in SUM_COLS:
                                agg_stats[k] += (b.get(k, 0) or 0)

                        # Recalc Skater Rates
                        g, s = agg_stats.get('goals', 0), agg_stats.get('shots', 0)
                        agg_stats['shootingpct'] = round((g/s)*100, 1) if s > 0 else 0.0
                        t, sh = agg_stats.get('timeonice', 0), agg_stats.get('shifts', 0)
                        agg_stats['timeonicepershift'] = round(t/sh) if sh > 0 else 0

                        updates[col_name] = json.dumps(calculate_trend_metrics(agg_stats, s_avg, SKATER_GROUPS, SKATER_ANOMALIES))

                    # 2. Split Trends
                    for split_col, target_col in [('home_stats', 'home_trend'), ('away_stats', 'away_trend')]:
                        split_data = p.get(split_col)
                        if split_data:
                            split_data['games_played'] = 1
                            updates[target_col] = json.dumps(calculate_trend_metrics(split_data, s_avg, SKATER_GROUPS, SKATER_ANOMALIES))
                        else:
                            updates[target_col] = None

                    # 3. Update DB
                    cursor.execute("""
                        UPDATE player_trends
                        SET l5_trend = %s, l10_trend = %s, l20_trend = %s, home_trend = %s, away_trend = %s
                        WHERE player_name_normalized = %s
                    """, (
                        updates.get('l5_trend'), updates.get('l10_trend'), updates.get('l20_trend'),
                        updates.get('home_trend'), updates.get('away_trend'),
                        p['player_name_normalized']
                    ))

            conn.commit()
            print("Skater trends analysis complete.")
    except Exception as e:
        print(f"Error generating skater trends: {e}")

def update_goalie_trends():
    """
    Updates player_trends for GOALIES using historical_goalie_stats.
    """
    print("Starting GOALIE trends update...")

    # Goalie Cols (Summable)
    SUM_COLS = [
        'wins', 'losses', 'overtimelosses', 'shotsagainst', 'saves',
        'goalsagainst', 'shutouts', 'goalsfor', 'gamesstarted', 'timeonice'
    ]

    BLOCK_KEYS = [f"block_{i+1}_{i+5}" for i in range(0, 75, 5)] + ["block_76_82"]

    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:

                # Fetch players
                cursor.execute(f"""
                    SELECT player_name_normalized, nhlplayerid, last_date_used,
                           season_total, home_total, away_total,
                           {', '.join(BLOCK_KEYS)}
                    FROM player_trends
                """)
                players = cursor.fetchall()

                for p in players:
                    p_name = p['player_name_normalized']
                    pid = p['nhlplayerid']
                    last_date = p['last_date_used'] or '2000-01-01'

                    # 1. Fetch Goalie Games
                    cols_sql = ", ".join(SUM_COLS)
                    cursor.execute(f"""
                        SELECT gamedate, homeroad, {cols_sql}
                        FROM historical_goalie_stats
                        WHERE nhlplayerid = %s AND gamedate > %s
                        ORDER BY gamedate ASC
                    """, (pid, last_date))

                    new_games = cursor.fetchall()
                    if not new_games: continue

                    # Load/Init Totals
                    season_total = p['season_total'] or {col: 0 for col in SUM_COLS}
                    home_total = p['home_total'] or {col: 0 for col in SUM_COLS}
                    away_total = p['away_total'] or {col: 0 for col in SUM_COLS}
                    for t in [season_total, home_total, away_total]:
                        if 'games_played' not in t: t['games_played'] = 0

                    # Find Block
                    current_block_col = None
                    for key in BLOCK_KEYS:
                        if p.get(key) is None:
                            current_block_col = key
                            break
                    if not current_block_col: continue

                    # Chunking
                    games_to_process = []
                    if current_block_col == "block_76_82":
                        chunk = new_games[:7]
                        if chunk: games_to_process.append((current_block_col, chunk))
                    else:
                        idx = 0
                        while idx + 5 <= len(new_games):
                            if not current_block_col: break
                            chunk = new_games[idx : idx+5]
                            games_to_process.append((current_block_col, chunk))
                            idx += 5
                            curr_idx = BLOCK_KEYS.index(current_block_col)
                            current_block_col = BLOCK_KEYS[curr_idx + 1] if curr_idx + 1 < len(BLOCK_KEYS) else None

                    if not games_to_process: continue

                    # Aggregation
                    for block_name, games in games_to_process:
                        block_stats = {col: 0 for col in SUM_COLS}
                        block_stats['games_played'] = len(games)

                        def parse_toi(v):
                            if isinstance(v, str) and ':' in v:
                                m, s = v.split(':')
                                return int(m) * 60 + int(s)
                            return v or 0

                        for game in games:
                            is_home = (game.get('homeroad') == 'H')
                            target_split = home_total if is_home else away_total

                            for col in SUM_COLS:
                                val = parse_toi(game.get(col)) if col == 'timeonice' else (game.get(col) or 0)
                                block_stats[col] += val
                                season_total[col] = season_total.get(col, 0) + val
                                target_split[col] = target_split.get(col, 0) + val

                            target_split['games_played'] += 1

                        season_total['games_played'] += len(games)

                        # Derived Stats Helper
                        def calc_derived(stats_dict):
                            sv = stats_dict.get('saves', 0)
                            sa = stats_dict.get('shotsagainst', 0)
                            stats_dict['savepct'] = round(sv / sa, 3) if sa > 0 else 0.0

                            ga = stats_dict.get('goalsagainst', 0)
                            toi = stats_dict.get('timeonice', 0)
                            # GAA = (GA * 3600) / TOI (Seconds)
                            stats_dict['gaa'] = round((ga * 3600) / toi, 2) if toi > 0 else 0.0

                        # Avg Helper
                        def calc_avg(total_dict, scale=1.0):
                            avg_dict = {}
                            gp = total_dict.get('games_played', 0)
                            if gp > 0:
                                for k, v in total_dict.items():
                                    if k == 'games_played': continue
                                    # For counts, scale. For TOI, scale.
                                    avg_dict[k] = round((v / gp) * scale, 2)

                                # For derived stats (GAA/Sv%), calculate from the averaged totals
                                # (Mathematically same as calc_derived on totals, but clean)
                                calc_derived(avg_dict)
                                avg_dict['games_played'] = 1 if scale == 1 else 5
                            return avg_dict

                        calc_derived(block_stats)
                        calc_derived(season_total) # Recalc total derived

                        season_avg = calc_avg(season_total, scale=5.0)
                        home_stats = calc_avg(home_total, scale=1.0)
                        away_stats = calc_avg(away_total, scale=1.0)

                        new_last_date = games[-1]['gamedate']

                        cursor.execute(f"""
                            UPDATE player_trends
                            SET {block_name} = %s,
                                season_total = %s, season_avg = %s,
                                home_total = %s, home_stats = %s,
                                away_total = %s, away_stats = %s,
                                last_date_used = %s
                            WHERE player_name_normalized = %s
                        """, (
                            json.dumps(block_stats),
                            json.dumps(season_total), json.dumps(season_avg),
                            json.dumps(home_total), json.dumps(home_stats),
                            json.dumps(away_total), json.dumps(away_stats),
                            new_last_date,
                            p_name
                        ))

            conn.commit()
            print("Goalie trends update complete.")

    except Exception as e:
        print(f"Error updating goalie trends: {e}")


def generate_goalie_trends():
    """ Calculates Trends for GOALIES """
    print("Generating GOALIE trends analysis...")

    GOALIE_GROUPS = {
        'goalieperformance': {
            'inverse': ['losses', 'overtimelosses', 'goalsagainst', 'gaa'],
            'standard': ['wins', 'savepct', 'timeonice', 'shutouts', 'gamesstarted']
        },
        'teamperformance': {
            'standard': ['shotsagainst', 'goalsfor']
        }
    }
    # No specific anomalies requested for goalies yet, passing empty list
    GOALIE_ANOMALIES = []

    BLOCK_KEYS = [f"block_{i+1}_{i+5}" for i in range(0, 75, 5)] + ["block_76_82"]

    SUM_COLS = [
        'wins', 'losses', 'overtimelosses', 'shotsagainst', 'saves',
        'goalsagainst', 'shutouts', 'goalsfor', 'gamesstarted', 'timeonice',
        'games_played'
    ]

    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                # Fetch only rows that look like goalies (have 'wins' in season_avg)
                query_cols = ", ".join(BLOCK_KEYS)
                cursor.execute(f"""
                    SELECT player_name_normalized, season_avg, home_stats, away_stats, {query_cols}
                    FROM player_trends
                    WHERE season_avg->>'wins' IS NOT NULL
                """)
                players = cursor.fetchall()

                for p in players:
                    s_avg = p['season_avg']
                    if not s_avg: continue

                    valid_blocks = [p[k] for k in BLOCK_KEYS if p.get(k)]
                    periods = {
                        'l5_trend': valid_blocks[-1:] if valid_blocks else [],
                        'l10_trend': valid_blocks[-2:] if valid_blocks else [],
                        'l20_trend': valid_blocks[-4:] if valid_blocks else []
                    }
                    updates = {}

                    for col_name, blocks in periods.items():
                        if not blocks:
                            updates[col_name] = None
                            continue

                        agg_stats = {k: 0 for k in SUM_COLS}
                        for b in blocks:
                            for k in SUM_COLS:
                                agg_stats[k] += (b.get(k, 0) or 0)

                        # Recalc Rates
                        sv, sa = agg_stats.get('saves', 0), agg_stats.get('shotsagainst', 0)
                        agg_stats['savepct'] = round(sv/sa, 3) if sa > 0 else 0.0
                        ga, toi = agg_stats.get('goalsagainst', 0), agg_stats.get('timeonice', 0)
                        agg_stats['gaa'] = round((ga*3600)/toi, 2) if toi > 0 else 0.0

                        updates[col_name] = json.dumps(calculate_trend_metrics(agg_stats, s_avg, GOALIE_GROUPS, GOALIE_ANOMALIES))

                    # Split Trends
                    for split_col, target_col in [('home_stats', 'home_trend'), ('away_stats', 'away_trend')]:
                        split_data = p.get(split_col)
                        if split_data:
                            split_data['games_played'] = 1
                            updates[target_col] = json.dumps(calculate_trend_metrics(split_data, s_avg, GOALIE_GROUPS, GOALIE_ANOMALIES))
                        else:
                            updates[target_col] = None

                    cursor.execute("""
                        UPDATE player_trends
                        SET l5_trend = %s, l10_trend = %s, l20_trend = %s, home_trend = %s, away_trend = %s
                        WHERE player_name_normalized = %s
                    """, (
                        updates.get('l5_trend'), updates.get('l10_trend'), updates.get('l20_trend'),
                        updates.get('home_trend'), updates.get('away_trend'),
                        p['player_name_normalized']
                    ))

            conn.commit()
            print("Goalie trends analysis complete.")
    except Exception as e:
        print(f"Error generating goalie trends: {e}")


def update_toi_stats():
    """ Orchestrator Function """
    create_global_tables()

    fetch_team_standings()
    fetch_team_stats_summary()
    fetch_team_stats_weekly()

    fetch_and_update_scoring_to_date()
    fetch_and_update_bangers_stats()
    fetch_and_update_faceoff_stats() # New Function
    fetch_and_update_goalie_stats()

    fetch_daily_historical_stats()
    fetch_daily_summary_stats()
    fetch_daily_realtime_stats()
    fetch_game_shifts()
    fetch_daily_historical_goalie_stats()
    fetch_daily_goalie_advanced_stats()
    fetch_daily_goalie_days_rest_stats()

    fetch_daily_pp_stats()

    create_last_game_pp_table()
    create_last_week_pp_table()

    join_special_teams_data()
    create_player_lines_table()
    verify_shift_coverage()

    create_stats_to_date_table()
    calculate_and_save_to_date_ranks()
    create_combined_projections()
    create_player_trends_table()
    update_player_trends()
    update_goalie_trends()
    generate_player_trends()
    generate_goalie_trends()

    print("TOI Script Complete.")

if __name__ == "__main__":
    update_toi_stats()
