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

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import get_db_connection

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

def df_to_postgres(df, table_name, conn, if_exists='replace', lowercase_columns=True):
    """
    Writes DataFrame to Postgres.
    lowercase_columns=True: Forces all columns to lowercase (Good for raw stats).
    lowercase_columns=False: Preserves CamelCase (Required for projections table used by App).
    """
    if df.empty:
        print(f"Warning: DataFrame for {table_name} is empty. Skipping write.")
        return

    # 1. Handle Case Sensitivity
    if lowercase_columns:
        df.columns = [c.lower() for c in df.columns]

    cursor = conn.cursor()
    if if_exists == 'replace':
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")

    # 2. Create Table Schema
    cols = []
    for col, dtype in df.dtypes.items():
        pg_type = 'TEXT'
        if pd.api.types.is_integer_dtype(dtype): pg_type = 'INTEGER'
        elif pd.api.types.is_float_dtype(dtype): pg_type = 'DOUBLE PRECISION'

        # Always quote column names to handle special chars and case
        if col == 'player_name_normalized':
            cols.append(f'"{col}" TEXT PRIMARY KEY')
        elif col == 'nhlplayerid':
            cols.append(f'"{col}" INTEGER')
        else:
            cols.append(f'"{col}" {pg_type}')

    create_sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(cols)})"
    cursor.execute(create_sql)

    # 3. Insert Data
    columns = list(df.columns)
    placeholders = ",".join(["%s"] * len(columns))
    col_names = ",".join([f'"{c}"' for c in columns]) # Quoted!

    data = [tuple(None if pd.isna(x) else x for x in row) for row in df.to_numpy()]

    insert_sql = f"INSERT INTO {table_name} ({col_names}) VALUES ({placeholders}) ON CONFLICT DO NOTHING"
    cursor.executemany(insert_sql, data)
    conn.commit()
    print(f"Successfully wrote {len(data)} rows to {table_name}.")

def run_database_cleanup(target_start_date):
    """Deletes records from powerplay_stats older than the target start date."""
    target_str = target_start_date.strftime("%Y-%m-%d")
    print(f"Cleaning up powerplay_stats before {target_str}...")

    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT to_regclass('public.powerplay_stats')")
            if cursor.fetchone()[0]:
                cursor.execute("DELETE FROM powerplay_stats WHERE date_ < %s", (target_str,))
                deleted_count = cursor.rowcount
                print(f"Deleted {deleted_count} old records.")
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
            """)
            conn.commit()

def log_unmatched_players(conn, df_unmatched, source_table_name):
    if df_unmatched.empty: return
    print(f"    -> Found {len(df_unmatched)} unmatched records from '{source_table_name}'. Logging.")

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
    print("\n--- Fetching Daily PP Stats ---")
    est = pytz.timezone('US/Eastern')
    today = datetime.now(est).date()
    target_end_date = today - timedelta(days=1)
    target_start_date = today - timedelta(days=7)

    # FIX: Always run cleanup for older records first
    run_database_cleanup(target_start_date)

    # FIX: Clean the current window to ensure self-healing
    # This removes records we are about to re-fetch, ensuring that if a player
    # is skipped (due to bad data), their old record doesn't persist.
    print(f"Self-Healing: Clearing records from {target_start_date} to {target_end_date}...")
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "DELETE FROM powerplay_stats WHERE date_ >= %s AND date_ <= %s",
                (target_start_date.strftime("%Y-%m-%d"), target_end_date.strftime("%Y-%m-%d"))
            )
        conn.commit()

    dates = []
    curr = target_start_date # FIX: Always start from 7 days ago
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
                    "isAggregate": "false", "sort": '[{"property":"ppTimeOnIce","direction":"DESC"}]',
                    "start": idx, "limit": 100,
                    "cayenneExp": f'gameDate>="{d}" and gameDate<="{d}" and gameTypeId=2'
                }
                resp = requests.get(BASE_URL, params=params, timeout=10)
                data = resp.json().get("data", [])
                if not data: break

                for p in data:
                    pp_pct = p.get("ppTimeOnIcePctPerGame")

                    # FIX: Skip games where the team had 0 PP time (null/None pct)
                    if pp_pct is None:
                        continue

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
                    t1.pptimeonicepctpergame as "lg_ppTimeOnIcePctPerGame",
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
            # Added COALESCE to handle any edge case NULLs
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
    try:
        while True:
            params = {"isAggregate": "false", "cayenneExp": f"gameTypeId=2 and seasonId={season_id}", "start": start, "limit": 100}
            r = requests.get(base_url, params=params).json()
            data = r.get('data', [])
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

            # Handle Elias Pettersson Duplicate (Forward)
            df['playerId'] = pd.to_numeric(df['playerId'], errors='coerce')
            df.loc[df['playerId'] == 8480012, 'player_name_normalized'] = 'eliaspetterssonf'

            # Convert to numeric
            numeric_cols = ['gamesPlayed', 'goals', 'assists', 'points', 'plusMinus', 'penaltyMinutes', 'ppGoals', 'ppPoints', 'shots']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

            # Per Game Division
            cols_to_average = ['goals', 'assists', 'points', 'plusMinus', 'penaltyMinutes', 'ppGoals', 'ppPoints', 'shots']
            df['gamesPlayed'] = df['gamesPlayed'].astype(float)
            for col in cols_to_average:
                df[col] = np.where(df['gamesPlayed'] > 0, df[col] / df['gamesPlayed'], 0.0)
                df[col] = df[col].round(3)

            df['ppAssists'] = df['ppPoints'] - df['ppGoals']

            cols = {
                'playerId': 'nhlplayerid', 'skaterFullName': 'skaterfullname', 'teamAbbrevs': 'teamabbrevs',
                'gamesPlayed': 'gamesplayed', 'goals': 'goals', 'assists': 'assists', 'points': 'points',
                'plusMinus': 'plusminus', 'penaltyMinutes': 'penaltyminutes', 'ppGoals': 'ppgoals',
                'ppPoints': 'pppoints', 'shootingPct': 'shootingpct', 'timeOnIcePerGame': 'toi/g',
                'shots': 'shots', 'ppAssists': 'ppassists'
            }
            cols['timeOnIcePerGame'] = 'toi/g'

            keep_cols = list(cols.keys()) + ['player_name_normalized']
            df_final = df[keep_cols].rename(columns=cols)

            with get_db_connection() as conn:
                df_to_postgres(df_final, 'scoring_to_date', conn)

    except Exception as e:
        print(f"Error scoring: {e}")

def fetch_and_update_bangers_stats():
    print("Fetching Bangers...")
    # Reverted to scoringpergame endpoint
    season_id = "20252026"
    base_url = "https://api.nhle.com/stats/rest/en/skater/scoringpergame"
    all_data = []
    start = 0
    try:
        while True:
            params = {"isAggregate": "false", "cayenneExp": f"gameTypeId=2 and seasonId={season_id}", "start": start, "limit": 100}
            r = requests.get(base_url, params=params).json()
            data = r.get('data', [])
            if not data: break
            all_data.extend(data)
            start += 100
            time.sleep(0.1)

        print(f"Fetched {len(all_data)} bangers records.")

        if all_data:
            df = pd.DataFrame(all_data)
            df.drop_duplicates(subset=['playerId'], inplace=True)
            if 'skaterFullName' in df.columns:
                df['player_name_normalized'] = df['skaterFullName'].apply(normalize_name)

            # Handle Elias Pettersson Duplicate (Forward)
            df['playerId'] = pd.to_numeric(df['playerId'], errors='coerce')
            df.loc[df['playerId'] == 8480012, 'player_name_normalized'] = 'eliaspetterssonf'

            cols = {'playerId': 'nhlplayerid', 'skaterFullName': 'skaterfullname', 'teamAbbrevs': 'teamabbrevs', 'blocksPerGame': 'BLK', 'hitsPerGame': 'HIT'}

            keep_cols = list(cols.keys()) + ['player_name_normalized']
            df_final = df[keep_cols].rename(columns=cols)

            with get_db_connection() as conn:
                df_to_postgres(df_final, 'bangers_to_date', conn)

    except Exception as e: print(f"Error bangers: {e}")

def fetch_and_update_goalie_stats():
    print("Fetching Goalies...")
    season_id = "20252026"
    base_url = "https://api.nhle.com/stats/rest/en/goalie/summary"
    all_data = []
    start = 0
    try:
        while True:
            params = {"isAggregate": "false", "cayenneExp": f"gameTypeId=2 and seasonId={season_id}", "start": start, "limit": 100}
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

                df_to_postgres(df_final, 'goalie_to_date', conn)

    except Exception as e: print(f"Error goalies: {e}")

# --- MERGING FUNCTIONS ---

def join_special_teams_data():
    """
    Joins data from last_game_pp and last_week_pp into the main projections table.
    """
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

        # Use lowercase=False to PRESERVE MixedCase for Projections table
        df_to_postgres(df_final, 'projections', conn, lowercase_columns=False)

        with conn.cursor() as cursor:
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_proj_norm ON projections(player_name_normalized)")
        conn.commit()


def perform_smart_join(base_df, merge_df, merge_cols, source_name, conn):
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

        # DROP RANK COLUMNS from projections to avoid duplication later
        drop_rank_cols = [c for c in df_proj.columns if c.endswith('_cat_rank')]
        if drop_rank_cols:
            df_proj.drop(columns=drop_rank_cols, inplace=True)

        df_sc = read_sql_postgres("SELECT * FROM scoring_to_date", conn)
        sc_map = {
            'gamesplayed': 'GPskater', 'goals': 'G', 'assists': 'A', 'points': 'P', 'plusminus': 'plus_minus',
            'penaltyminutes': 'PIM', 'ppgoals': 'PPG', 'ppassists': 'PPA', 'pppoints': 'PPP',
            'shootingpct': 'shootingPct', 'toi/g': 'TOI/G', 'shots': 'SOG'
        }
        df_sc.rename(columns=sc_map, inplace=True)
        df_sc['nhlplayerid'] = pd.to_numeric(df_sc['nhlplayerid'], errors='coerce').fillna(0).astype(int)

        df_merged = perform_smart_join(df_proj, df_sc, list(sc_map.values()), 'scoring', conn)

        df_bn = read_sql_postgres("SELECT * FROM bangers_to_date", conn)

        # FIX: Map from lowercase 'blockspergame' to uppercase 'BLK'
        bn_map = {'blockspergame': 'BLK', 'hitspergame': 'HIT'}
        df_bn.rename(columns=bn_map, inplace=True)
        df_bn['nhlplayerid'] = pd.to_numeric(df_bn['nhlplayerid'], errors='coerce').fillna(0).astype(int)

        # Merge with Uppercase keys (BLK, HIT)
        df_merged = perform_smart_join(df_merged, df_bn, list(bn_map.values()), 'bangers', conn)

        df_gl = read_sql_postgres("SELECT * FROM goalie_to_date", conn)
        gl_map = {
            'gamesstarted': 'GS', 'gamesplayed': 'GP', 'goalsagainstaverage': 'GAA', 'losses': 'L',
            'savepct': 'SVpct', 'saves': 'SV', 'shotsagainst': 'SA', 'shutouts': 'SHO', 'wins': 'W',
            'goalsagainst': 'GA', 'startpct': 'startpct', 'toi/g': 'TOI/G'
        }
        df_gl.rename(columns=gl_map, inplace=True)
        df_gl['nhlplayerid'] = pd.to_numeric(df_gl['nhlplayerid'], errors='coerce').fillna(0).astype(int)

        df_merged = perform_smart_join(df_merged, df_gl, list(gl_map.values()), 'goalies', conn)

        # FIX: lowercase=False to preserve BLK/HIT/TOI/G keys
        df_to_postgres(df_merged, 'stats_to_date', conn, lowercase_columns=False)

def calculate_and_save_to_date_ranks():
    print("\n--- Calculating Ranks ---")
    with get_db_connection() as conn:
        df = read_sql_postgres("SELECT * FROM stats_to_date", conn)
        if df.empty: return

        skater_stats = ['G', 'A', 'P', 'PPG', 'PPA', 'PPP', 'SHG', 'SHA', 'SHP', 'HIT', 'BLK', 'PIM', 'FOW', 'SOG', 'plus_minus']
        goalie_stats = {'GS': False, 'W': False, 'L': True, 'GA': True, 'SA': False, 'SV': False, 'SVpct': False, 'GAA': True, 'SHO': False, 'QS': False}

        if 'positions' not in df.columns: return

        mask_skater = ~df['positions'].str.contains('G', na=False)
        num_skaters = mask_skater.sum()
        if num_skaters > 0:
            for stat in skater_stats:
                # FIX: Create rank columns as UpperCase to match standard
                col = f"{stat}_cat_rank"

                # Check both Mixed and Lower
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

        # FIX: Preserve mixed case
        df_to_postgres(df, 'stats_to_date', conn, lowercase_columns=False)

def create_combined_projections():
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
            # Check Mixed, Lower, and Suffixed
            col_lower = col.lower()

            def get_series(base, suffix):
                keys = [f"{base}{suffix}", f"{base.lower()}{suffix}", base, base.lower()]
                for k in keys:
                    if k in df_merged: return df_merged[k]
                return None

            s_proj = get_series(col, "_proj")
            s_stat = get_series(col, "_stats")

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

            # Logic: If column exists in original PROJ df, use average.
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

        # Use lowercase=False to preserve MixedCase names
        df_to_postgres(df_final, 'combined_projections', conn, lowercase_columns=False)


def update_toi_stats():
    """ Orchestrator Function """
    create_global_tables()

    fetch_team_standings()
    fetch_team_stats_summary()
    fetch_team_stats_weekly()

    fetch_and_update_scoring_to_date()
    fetch_and_update_bangers_stats()
    fetch_and_update_goalie_stats()

    fetch_daily_pp_stats()

    create_last_game_pp_table()
    create_last_week_pp_table()

    join_special_teams_data()

    create_stats_to_date_table()
    calculate_and_save_to_date_ranks()
    create_combined_projections()
    print("TOI Script Complete.")

if __name__ == "__main__":
    update_toi_stats()
