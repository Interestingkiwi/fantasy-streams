"""
Fetches NHL TOI/Special Teams stats and updates the DB.
Refactored for Postgres.

Author: Jason Druckenmiller
Created: 11/02/2025
Updated: 11/26/2025
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
    "Utah Hockey Club": "UTA", "Vancouver Canucks": "VAN", "Vegas Golden Knights": "VGK",
    "Washington Capitals": "WSH", "Winnipeg Jets": "WPG"
}

def normalize_name(name):
    """
    Normalizes a player name by converting to lowercase, removing diacritics,
    and removing all non-alphanumeric characters.
    """
    if not name: return ""
    nfkd = unicodedata.normalize('NFKD', name.lower())
    ascii = "".join([c for c in nfkd if not unicodedata.combining(c)])
    return re.sub(r'[^a-z0-9]', '', ascii)


def df_to_postgres(df, table_name, conn, if_exists='replace'):
    cursor = conn.cursor()
    if if_exists == 'replace': cursor.execute(f"DROP TABLE IF EXISTS {table_name}")

    cols = []
    for col, dtype in df.dtypes.items():
        pg_type = 'TEXT'
        if pd.api.types.is_integer_dtype(dtype): pg_type = 'INTEGER'
        elif pd.api.types.is_float_dtype(dtype): pg_type = 'REAL'

        if col == 'nhlplayerid' or col == 'date_':
            cols.append(f'"{col}" {pg_type}') # Don't force PK here, handle manually if needed
        else:
            cols.append(f'"{col}" {pg_type}')

    create_sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(cols)})"
    cursor.execute(create_sql)

    columns = list(df.columns)
    placeholders = ",".join(["%s"] * len(columns))
    col_names = ",".join([f'"{c}"' for c in columns])
    insert_sql = f"INSERT INTO {table_name} ({col_names}) VALUES ({placeholders}) ON CONFLICT DO NOTHING"

    data = [tuple(None if pd.isna(x) else x for x in row) for row in df.to_numpy()]
    cursor.executemany(insert_sql, data)
    conn.commit()


def log_unmatched_players(conn, df_unmatched, source_table_name):
    """
    Writes unmatched rows to the 'unmatched_players' table in projections.db.
    """
    if df_unmatched.empty: return
    print(f"    -> Found {len(df_unmatched)} unmatched records from '{source_table_name}'. Logging.")

    log_df = pd.DataFrame()
    log_df['nhlplayerid'] = df_unmatched.get('nhlplayerid', pd.NA)

    if 'player_name_normalized' in df_unmatched.columns:
        log_df['player_name'] = df_unmatched['player_name_normalized']
    elif 'skaterFullName' in df_unmatched.columns:
        log_df['player_name'] = df_unmatched['skaterFullName']
    else:
        log_df['player_name'] = 'Unknown'

    log_df['team'] = df_unmatched.get('team', df_unmatched.get('teamAbbrevs', 'Unknown'))
    log_df['source_table'] = source_table_name
    log_df['run_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Postgres Write
    with conn.cursor() as cursor:
        data = [tuple(x) for x in log_df.to_numpy()]
        cursor.executemany("""
            INSERT INTO unmatched_players (nhlplayerid, player_name, team, source_table, run_date)
            VALUES (%s, %s, %s, %s, %s)
        """, data)



def perform_smart_join(base_df, merge_df, merge_cols_data, source_name, conn):
    """
    Joins merge_df into base_df using the Two-Step Priority Logic:
    1. Match on (nhlplayerid AND team)
    2. Match remaining on (player_name_normalized) -> OVERWRITES nhlplayerid from merge_df
    3. Log residuals to unmatched_players
    """
    if 'teamAbbrevs' in merge_df.columns and 'team' not in merge_df.columns:
        merge_df = merge_df.rename(columns={'teamAbbrevs': 'team'})

    required_base = ['nhlplayerid', 'team', 'player_name_normalized']
    required_merge = ['nhlplayerid', 'team']
    if 'player_name_normalized' in merge_df.columns: required_merge.append('player_name_normalized')

    if not all(c in base_df.columns for c in required_base): return base_df
    if not all(c in merge_df.columns for c in required_merge): return base_df

    print(f"    Performing Smart Join for {source_name}...")

    # 1. Exact Match
    base_df['mc_id_team'] = base_df['nhlplayerid'].astype(str) + "_" + base_df['team'].astype(str)
    merge_df['mc_id_team'] = merge_df['nhlplayerid'].astype(str) + "_" + merge_df['team'].astype(str)

    mask_id = merge_df['mc_id_team'].isin(base_df['mc_id_team'])
    df_merge_id = merge_df[mask_id].copy()
    df_merge_rem = merge_df[~mask_id].copy()

    # 2. Name Match
    df_merge_name = pd.DataFrame()
    if 'player_name_normalized' in df_merge_rem.columns:
        mask_name = df_merge_rem['player_name_normalized'].isin(base_df['player_name_normalized'])
        df_merge_name = df_merge_rem[mask_name].copy()
        df_unmatched = df_merge_rem[~mask_name].copy()
    else:
        df_unmatched = df_merge_rem.copy()

    log_unmatched_players(conn, df_unmatched, source_name)

    # 3. Merge
    cols_use = list(set(['nhlplayerid', 'team'] + list(merge_cols_data)))
    cols_use = [c for c in cols_use if c in df_merge_id.columns]

    df_stage1 = pd.merge(base_df, df_merge_id[cols_use], on=['nhlplayerid', 'team'], how='left', suffixes=('', '_new'))

    if not df_merge_name.empty:
        cols_use_name = list(set(['player_name_normalized', 'nhlplayerid'] + list(merge_cols_data)))
        cols_use_name = [c for c in cols_use_name if c in df_merge_name.columns]

        df_stage2 = pd.merge(df_stage1, df_merge_name[cols_use_name], on='player_name_normalized', how='left', suffixes=('', '_namejoin'))

        if 'nhlplayerid_namejoin' in df_stage2.columns:
            df_stage2['nhlplayerid'] = np.where(
                df_stage2['nhlplayerid_namejoin'].notna(),
                df_stage2['nhlplayerid_namejoin'],
                df_stage2['nhlplayerid']
            )
            df_stage2['nhlplayerid'] = pd.to_numeric(df_stage2['nhlplayerid'], errors='coerce').fillna(0).astype(int)
    else:
        df_stage2 = df_stage1

    # 4. Coalesce
    for col in merge_cols_data:
        col_new = f"{col}_new"
        col_name = f"{col}_namejoin"

        sources = []
        if col_new in df_stage2.columns: sources.append(col_new)
        elif col in df_stage2.columns and col not in base_df.columns: sources.append(col)

        if col_name in df_stage2.columns: sources.append(col_name)

        if sources:
            s = pd.Series(np.nan, index=df_stage2.index)
            if len(sources) > 0 and sources[0] in df_stage2: s = s.fillna(df_stage2[sources[0]])
            if len(sources) > 1 and sources[1] in df_stage2: s = s.fillna(df_stage2[sources[1]])
            df_stage2[col] = s

    cols_drop = [c for c in df_stage2.columns if c.endswith('_new') or c.endswith('_namejoin') or c.startswith('mc_')]
    return df_stage2.drop(columns=cols_drop, errors='ignore')


def create_global_tables():
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS powerplay_stats (
                    date_ TEXT, nhlplayerid INTEGER, skaterFullName TEXT, teamAbbrevs TEXT,
                    ppTimeOnIce INTEGER, ppTimeOnIcePctPerGame REAL, ppAssists INTEGER, ppGoals INTEGER,
                    PRIMARY KEY (date_, nhlplayerid)
                );
                CREATE TABLE IF NOT EXISTS table_metadata (id INTEGER PRIMARY KEY DEFAULT 1, start_date TEXT, end_date TEXT);
                CREATE TABLE IF NOT EXISTS team_standings (
                    team_tricode TEXT PRIMARY KEY, point_pct TEXT, goals_against_per_game REAL, games_played INTEGER
                );
                CREATE TABLE IF NOT EXISTS team_stats_summary (
                    team_tricode TEXT PRIMARY KEY, pp_pct REAL, pk_pct REAL, gf_gm REAL, ga_gm REAL, sogf_gm REAL, soga_gm REAL
                );
                CREATE TABLE IF NOT EXISTS team_stats_weekly (
                    team_tricode TEXT PRIMARY KEY, pp_pct_weekly REAL, pk_pct_weekly REAL, gf_gm_weekly REAL,
                    ga_gm_weekly REAL, sogf_gm_weekly REAL, soga_gm_weekly REAL
                );
                CREATE TABLE IF NOT EXISTS scoring_to_date (
                    nhlplayerid INTEGER PRIMARY KEY, skaterFullName TEXT, teamAbbrevs TEXT, gamesPlayed INTEGER,
                    goals INTEGER, assists INTEGER, points INTEGER, plusMinus TEXT, penaltyMinutes INTEGER,
                    ppGoals INTEGER, ppAssists INTEGER, ppPoints INTEGER, shootingPct REAL, timeonIcePerGame REAL, shots INTEGER
                );
                CREATE TABLE IF NOT EXISTS bangers_to_date (
                    nhlplayerid INTEGER PRIMARY KEY, skaterFullName TEXT, teamAbbrevs TEXT, blocksPerGame INTEGER, hitsPerGame INTEGER
                );
                CREATE TABLE IF NOT EXISTS goalie_to_date (
                    nhlplayerid INTEGER PRIMARY KEY, goalieFullName TEXT, teamAbbrevs TEXT, gamesStarted INTEGER, gamesPlayed INTEGER,
                    goalsAgainstAverage REAL, losses INTEGER, savePct REAL, saves INTEGER, shotsAgainst INTEGER,
                    shutouts INTEGER, wins INTEGER, goalsAgainst INTEGER, startpct INTEGER
                );
                CREATE TABLE IF NOT EXISTS unmatched_players (
                    run_date TEXT, source_table TEXT, nhlplayerid INTEGER, player_name TEXT, team TEXT
                );
            """)
            conn.commit()


def get_last_run_end_date():
    """Fetches the last successfully recorded end_date from metadata."""
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT end_date FROM table_metadata WHERE id = 1")
            res = cursor.fetchone()
            if res: return date.fromisoformat(res[0])
    return None


def run_database_cleanup(target_start_date):
    """Deletes records from powerplay_stats older than the target start date."""
    target_str = target_start_date.strftime("%Y-%m-%d")
    print(f"Cleaning up powerplay_stats before {target_str}...")

    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            # Postgres uses %s
            cursor.execute("DELETE FROM powerplay_stats WHERE date_ < %s", (target_str,))
            # rowcount works the same in psycopg2
            deleted_count = cursor.rowcount
            print(f"Deleted {deleted_count} old records.")
        conn.commit()


def update_metadata(start_date, end_date):
    """Updates the metadata table with the new start and end dates of the data window."""
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO table_metadata (id, start_date, end_date) VALUES (1, %s, %s)
                ON CONFLICT (id) DO UPDATE SET start_date = EXCLUDED.start_date, end_date = EXCLUDED.end_date
            """, (start, end))
            conn.commit()


def fetch_daily_pp_stats():
    """
    Creates/replaces the 'powerplay stats' table with all player rows from
    the most recent game for each team.
    """
    print("\n--- Fetching Daily PP Stats ---")
    est = pytz.timezone('US/Eastern')
    today = datetime.now(est).date()
    target_end = today - timedelta(days=1)
    target_start = today - timedelta(days=7)

    last_run = get_last_run_end_date()
    query_start = last_run + timedelta(days=1) if last_run else target_start
    if query_start < target_start: query_start = target_start
    run_database_cleanup(target_start_date)
    if query_start > target_end:
        print("Data up to date.")
        return False

    dates = []
    curr = query_start
    while curr <= target_end:
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
                    rec = {
                        "date_": d, "nhlplayerid": p.get("playerId"), "skaterFullName": p.get("skaterFullName"),
                        "teamAbbrevs": p.get("teamAbbrevs"), "ppTimeOnIce": p.get("ppTimeOnIce"),
                        "ppTimeOnIcePctPerGame": p.get("ppTimeOnIcePctPerGame"), "ppAssists": p.get("ppAssists"),
                        "ppGoals": p.get("ppGoals")
                    }
                    all_data.append(rec)
                idx += 100
                time.sleep(0.1)
        except:
            has_errors = True
            print(f"Error fetching {d}")

    if all_data:
        df = pd.DataFrame(all_data)
        df.drop_duplicates(subset=['date_', 'nhlplayerid'], inplace=True)

        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                # Manual Upsert logic for cleaner Postgres handling
                vals = [tuple(x) for x in df[['date_', 'nhlplayerid', 'skaterFullName', 'teamAbbrevs', 'ppTimeOnIce', 'ppTimeOnIcePctPerGame', 'ppAssists', 'ppGoals']].to_numpy()]
                cursor.executemany("""
                    INSERT INTO powerplay_stats (date_, nhlplayerid, skaterFullName, teamAbbrevs, ppTimeOnIce, ppTimeOnIcePctPerGame, ppAssists, ppGoals)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (date_, nhlplayerid) DO UPDATE SET
                        ppTimeOnIce = EXCLUDED.ppTimeOnIce,
                        ppTimeOnIcePctPerGame = EXCLUDED.ppTimeOnIcePctPerGame,
                        ppAssists = EXCLUDED.ppAssists,
                        ppGoals = EXCLUDED.ppGoals
                """, vals)
                conn.commit()

        if not has_errors: update_metadata(target_start, target_end)
        return True
    return False


def create_last_game_pp_table():
    print("\n--- Creating 'last_game_pp' ---")
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("DROP TABLE IF EXISTS last_game_pp")
            cursor.execute("""
                CREATE TABLE last_game_pp AS
                SELECT t1.* FROM powerplay_stats t1
                INNER JOIN (
                    SELECT teamAbbrevs, MAX(date_) as max_date
                    FROM powerplay_stats GROUP BY teamAbbrevs
                ) t2 ON t1.teamAbbrevs = t2.teamAbbrevs AND t1.date_ = t2.max_date
            """)
            conn.commit()


def create_last_week_pp_table(db_file):
    """
    Creates/replaces the 'last_week_pp' table with aggregated 7-day stats
    for each player, using team total games as the divisor for averages.
    """
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("DROP TABLE IF EXISTS last_week_pp")
            cursor.execute("""
                CREATE TABLE last_week_pp AS
                WITH team_game_counts AS (
                    SELECT teamAbbrevs, COUNT(DISTINCT date_) as team_games_played
                    FROM powerplay_stats GROUP BY teamAbbrevs
                ),
                player_sums AS (
                    SELECT nhlplayerid, teamAbbrevs, MAX(skaterFullName) as skaterFullName,
                    SUM(ppTimeOnIce) as total_ppTimeOnIce,
                    SUM(ppTimeOnIcePctPerGame) as total_ppTimeOnIcePctPerGame,
                    SUM(ppAssists) as total_ppAssists,
                    SUM(ppGoals) as total_ppGoals,
                    COUNT(date_) as player_games_played
                    FROM powerplay_stats GROUP BY nhlplayerid, teamAbbrevs
                )
                SELECT ps.nhlplayerid, ps.skaterFullName, ps.teamAbbrevs,
                    CAST(ps.total_ppTimeOnIce AS REAL) / tgc.team_games_played AS avg_ppTimeOnIce,
                    CAST(ps.total_ppTimeOnIcePctPerGame AS REAL) / tgc.team_games_played AS avg_ppTimeOnIcePctPerGame,
                    ps.total_ppAssists, ps.total_ppGoals, ps.player_games_played, tgc.team_games_played
                FROM player_sums ps
                JOIN team_game_counts tgc ON ps.teamAbbrevs = tgc.teamAbbrevs
            """)
            conn.commit()


def fetch_team_stats_summary():
    """
    Fetches Team PP%, PK%, GF, GA, SF, and SA from the NHL Stats API and stores them
    in the 'team_stats_summary' table.
    """
    print("\n--- Fetching Team Stats Summary ---")
    try:
        params = {"isAggregate":"false", "isGame":"false", "start":0, "limit":50, "cayenneExp":"seasonId=20252026 and gameTypeId=2"}
        data = requests.get("https://api.nhle.com/stats/rest/en/team/summary", params=params).json().get("data", [])
        rows = []
        for t in data:
            code = FRANCHISE_TO_TRICODE_MAP.get(t.get("teamFullName"))
            if code:
                rows.append((code, t.get("powerPlayPct"), t.get("penaltyKillPct"), t.get("goalsForPerGame"), t.get("goalsAgainstPerGame"), t.get("shotsForPerGame"), t.get("shotsAgainstPerGame")))

        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM team_stats_summary")
                cursor.executemany("INSERT INTO team_stats_summary VALUES (%s, %s, %s, %s, %s, %s, %s)", rows)
            conn.commit()
    except Exception as e: print(f"Error: {e}")


def fetch_team_stats_weekly():
    """
    Fetches Team PP% and PK% for the last 7 days (7 days ago to yesterday)
    and stores them in the 'team_stats_weekly' table.
    """
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

        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM team_stats_weekly")
                cursor.executemany("INSERT INTO team_stats_weekly VALUES (%s, %s, %s, %s, %s, %s, %s)", rows)
            conn.commit()
    except Exception as e: print(f"Error: {e}")


def join_special_teams_data():
    """
    Joins data from last_game_pp and last_week_pp (from special_teams.db)
    into the main projections table (in projections.db).
    """
    print("\n--- Joining Special Teams into Projections ---")
    with get_db_connection() as conn:
        # Load Projections
        df_proj = pd.read_sql_query("SELECT * FROM projections", conn)
        if df_proj.empty: return

        # Load Last Game
        df_lg = pd.read_sql_query("SELECT nhlplayerid, ppTimeOnIce as lg_ppTimeOnIce, ppTimeOnIcePctPerGame as lg_ppTimeOnIcePctPerGame, ppAssists as lg_ppAssists, ppGoals as lg_ppGoals FROM last_game_pp", conn)

        # Load Last Week
        df_lw = pd.read_sql_query("SELECT nhlplayerid, avg_ppTimeOnIce, avg_ppTimeOnIcePctPerGame, total_ppAssists, total_ppGoals, player_games_played, team_games_played FROM last_week_pp", conn)

        # Merge
        df_final = pd.merge(df_proj, df_lg, on='nhlplayerid', how='left')
        df_final = pd.merge(df_final, df_lw, on='nhlplayerid', how='left')

        # Clean Types
        if 'nhlplayerid' in df_final.columns:
            df_final['nhlplayerid'] = pd.to_numeric(df_final['nhlplayerid'], errors='coerce').astype('Int64')

        # Overwrite Projections Table
        # Using replace is safe here because `create_projection_db` runs first and sets up the table structure
        df_to_postgres(df_merged, 'projections', conn)

        with conn.cursor() as cursor:
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_proj_norm ON projections(player_name_normalized)")
        conn.commit()


def fetch_team_standings():
    """
    Fetches the current team standings, clears the 'team_standings' table,
    and inserts the new data.
    """
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
                rows.append((abbrev, pts, ga/gp if gp else 0, gp))

        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM team_standings")
                cursor.executemany("INSERT INTO team_standings (team_tricode, point_pct, goals_against_per_game, games_played) VALUES (%s, %s, %s, %s)", rows)
            conn.commit()
    except Exception as e:
        print(f"Error: {e}")


def fetch_and_update_scoring_to_date():
    """
    Fetches the current season's to-date summary stats for all skaters
    from the NHL API, calculates per-game stats, and writes to
    the 'scoring_to_date' table in the special_teams.db.
    """
    print("\n--- Starting NHL To-Date Skater Stats Fetch ---")
    base_url = "https://api.nhle.com/stats/rest/en/skater/summary"
    all_players_data = []
    start = 0
    limit = 100

    current_year = date.today().year
    season_end_year = current_year if date.today().month < 7 else current_year + 1
    season_start_year = season_end_year - 1
    season_id = f"{season_start_year}{season_end_year}"
    print(f"Fetching data for season: {season_id}")

    while True:
        try:
            params = {
                "isAggregate": "false",
                "cayenneExp": f"gameTypeId=2 and seasonId={season_id}",
                "sort": '[{"property":"points","direction":"DESC"}]',
                "start": start,
                "limit": limit
            }
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            players_list = data.get('data', [])

            if not players_list: break
            all_players_data.extend(players_list)
            print(f"Retrieved {len(players_list)} players... (Total: {len(all_players_data)})")
            start += limit
            time.sleep(0.2)
        except Exception as e:
            print(f"Error fetching data: {e}")
            return

    if not all_players_data: return

    try:
        df = pd.DataFrame(all_players_data)
        df.drop_duplicates(subset=['playerId'], keep='first', inplace=True)
        if 'skaterFullName' in df.columns:
             df['player_name_normalized'] = df['skaterFullName'].apply(normalize_name)

        required_cols = [
            'playerId', 'skaterFullName', 'player_name_normalized', 'teamAbbrevs', 'gamesPlayed',
            'goals', 'assists', 'points', 'plusMinus', 'penaltyMinutes',
            'ppGoals', 'ppPoints', 'shootingPct', 'timeOnIcePerGame', 'shots'
        ]
        df = df[[c for c in required_cols if c in df.columns]].copy()

        # Logic: Calc Per Game
        df['gamesPlayed'] = pd.to_numeric(df['gamesPlayed'], errors='coerce').fillna(0)
        df['ppPoints'] = pd.to_numeric(df['ppPoints'], errors='coerce').fillna(0)
        df['ppGoals'] = pd.to_numeric(df['ppGoals'], errors='coerce').fillna(0)
        df['ppAssists'] = df['ppPoints'] - df['ppGoals']

        cols_to_divide = ['goals', 'assists', 'points', 'plusMinus', 'penaltyMinutes', 'ppGoals', 'ppPoints', 'ppAssists', 'timeOnIcePerGame', 'shots']
        for col in cols_to_divide:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                df[col] = np.where(df['gamesPlayed'] > 0, df[col] / df['gamesPlayed'], 0)

        df = df.rename(columns={'playerId': 'nhlplayerid'})

        # Upsert to Postgres
        with get_db_connection() as conn:
            print(f"Writing {len(df)} records to 'scoring_to_date' table...")
            df_to_postgres(df, 'scoring_to_date', conn)
            with conn.cursor() as cursor:
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_scoring_norm ON scoring_to_date(player_name_normalized)")
            conn.commit()

    except Exception as e:
        print(f"Error processing scoring stats: {e}")


def fetch_and_update_bangers_stats():
    """
    Fetches the current season's 'scoringpergame' report for all skaters
    from the NHL API, selects 'banger' stats, and writes them to
    the 'bangers_to_date' table in special_teams.db.
    """
    print("\n--- Starting NHL 'Bangers' Stats Fetch ---")
    base_url = "https://api.nhle.com/stats/rest/en/skater/scoringpergame"
    all_players_data = []
    start = 0
    limit = 100

    current_year = date.today().year
    season_end_year = current_year if date.today().month < 7 else current_year + 1
    season_start_year = season_end_year - 1
    season_id = f"{season_start_year}{season_end_year}"

    while True:
        try:
            params = {
                "isAggregate": "false", "cayenneExp": f"gameTypeId=2 and seasonId={season_id}",
                "sort": '[{"property":"hitsPerGame","direction":"DESC"}]', "start": start, "limit": limit
            }
            response = requests.get(base_url, params=params, timeout=10)
            data = response.json()
            players_list = data.get('data', [])
            if not players_list: break
            all_players_data.extend(players_list)
            start += limit
            time.sleep(0.2)
        except Exception as e:
            print(f"Error fetching bangers: {e}")
            return

    if not all_players_data: return

    try:
        df = pd.DataFrame(all_players_data)
        df.drop_duplicates(subset=['playerId'], keep='first', inplace=True)
        if 'skaterFullName' in df.columns:
             df['player_name_normalized'] = df['skaterFullName'].apply(normalize_name)

        required_cols = ['playerId', 'skaterFullName', 'player_name_normalized', 'teamAbbrevs', 'blocksPerGame', 'hitsPerGame']
        df = df[[c for c in required_cols if c in df.columns]].copy()
        df = df.rename(columns={'playerId': 'nhlplayerid'})

        with get_db_connection() as conn:
            print(f"Writing {len(df)} records to 'bangers_to_date' table...")
            df_to_postgres(df, 'bangers_to_date', conn)
            conn.commit()

    except Exception as e:
        print(f"Error processing bangers stats: {e}")


def fetch_and_update_goalie_stats():
    """
    Fetches the current season's 'summary' report for all goalies
    from the NHL API, calculates per-game stats, joins team games_played,
    calculates startpct, and writes them to the 'goalie_to_date'
    table in special_teams.db.
    """
    print("\n--- Starting NHL Goalie Stats Fetch ---")
    base_url = "https://api.nhle.com/stats/rest/en/goalie/summary"
    all_goalie_data = []
    start = 0
    limit = 100

    current_year = date.today().year
    season_end_year = current_year if date.today().month < 7 else current_year + 1
    season_start_year = season_end_year - 1
    season_id = f"{season_start_year}{season_end_year}"

    while True:
        try:
            params = {
                "isAggregate": "false", "cayenneExp": f"gameTypeId=2 and seasonId={season_id}",
                "sort": '[{"property":"wins","direction":"DESC"}]', "start": start, "limit": limit
            }
            response = requests.get(base_url, params=params, timeout=10)
            data = response.json()
            goalie_list = data.get('data', [])
            if not goalie_list: break
            all_goalie_data.extend(goalie_list)
            start += limit
            time.sleep(0.2)
        except Exception as e:
            print(f"Error fetching goalies: {e}")
            return

    if not all_goalie_data: return

    try:
        df = pd.DataFrame(all_goalie_data)
        df.drop_duplicates(subset=['playerId'], keep='first', inplace=True)
        if 'goalieFullName' in df.columns:
             df['player_name_normalized'] = df['goalieFullName'].apply(normalize_name)

        required = ['playerId', 'goalieFullName', 'player_name_normalized', 'teamAbbrevs', 'gamesStarted', 'gamesPlayed', 'goalsAgainstAverage', 'losses', 'savePct', 'saves', 'shotsAgainst', 'shutouts', 'wins', 'goalsAgainst']
        df = df[[c for c in required if c in df.columns]].copy()

        # Math Logic
        df['gamesPlayed'] = pd.to_numeric(df['gamesPlayed'], errors='coerce').fillna(0)
        stats_convert = ['saves', 'shotsAgainst', 'wins', 'losses', 'shutouts', 'gamesStarted', 'goalsAgainst']
        for col in stats_convert:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        df['win_total'] = df['wins']
        # Convert counts to per-game
        for col in ['wins', 'shutouts', 'saves', 'shotsAgainst', 'goalsAgainst']:
            df[col] = np.where(df['gamesPlayed'] > 0, df[col] / df['gamesPlayed'], 0)

        df = df.rename(columns={'playerId': 'nhlplayerid'})

        # Join with Standings (Now in Postgres)
        with get_db_connection() as conn:
            df_standings = pd.read_sql_query("SELECT team_tricode, games_played AS team_games_played FROM team_standings", conn)

            if not df_standings.empty:
                df = pd.merge(df, df_standings, left_on='teamAbbrevs', right_on='team_tricode', how='left')
                df['team_games_played'] = pd.to_numeric(df['team_games_played'], errors='coerce').fillna(0)
                df['startpct'] = np.where(df['team_games_played'] > 0, df['gamesStarted'] / df['team_games_played'], 0)
                df.drop(columns=['team_tricode', 'team_games_played'], errors='ignore', inplace=True)
            else:
                df['startpct'] = 0

            print(f"Writing {len(df)} records to 'goalie_to_date' table...")
            df_to_postgres(df, 'goalie_to_date', conn)
            conn.commit()

    except Exception as e:
        print(f"Error processing goalie stats: {e}")



def create_stats_to_date_table():
    """
    Joins 'projections' with 'scoring', 'bangers', 'goalies' using Smart Join logic.
    Saves to 'stats_to_date'.
    """
    print(f"\n--- Creating 'stats_to_date' table ---")
    with get_db_connection() as conn:
        # 1. Reset Log
        with conn.cursor() as cursor:
            cursor.execute("DROP TABLE IF EXISTS unmatched_players")
            cursor.execute("CREATE TABLE IF NOT EXISTS unmatched_players (run_date TEXT, source_table TEXT, nhlplayerid INTEGER, player_name TEXT, team TEXT)")
            conn.commit()

        # 2. Read Tables
        try:
            df_proj = pd.read_sql_query("SELECT * FROM projections", conn)
            if 'gp' in df_proj.columns: df_proj.rename(columns={'gp': 'GP'}, inplace=True)
            df_proj['nhlplayerid'] = pd.to_numeric(df_proj['nhlplayerid'], errors='coerce').fillna(0).astype(int)
            df_proj.drop_duplicates(subset=['nhlplayerid'], inplace=True)

            # SCORING
            df_scoring = pd.read_sql_query("SELECT * FROM scoring_to_date", conn)
            df_scoring['nhlplayerid'] = pd.to_numeric(df_scoring['nhlplayerid'], errors='coerce').fillna(0).astype(int)
            scoring_map = {'gamesPlayed': 'GPskater', 'goals': 'G', 'assists': 'A', 'points': 'P', 'plusMinus': 'plus_minus', 'penaltyMinutes': 'PIM', 'ppGoals': 'PPG', 'ppAssists': 'PPA', 'ppPoints': 'PPP', 'shootingPct': 'shootingPct', 'timeOnIcePerGame': 'timeOnIcePerGame', 'shots': 'SOG'}
            df_scoring.rename(columns=scoring_map, inplace=True)

            # Smart Join 1
            df_merged = perform_smart_join(df_proj, df_scoring, list(scoring_map.values()), 'scoring_to_date', conn)

            # BANGERS
            df_bang = pd.read_sql_query("SELECT * FROM bangers_to_date", conn)
            df_bang['nhlplayerid'] = pd.to_numeric(df_bang['nhlplayerid'], errors='coerce').fillna(0).astype(int)
            bang_map = {'blocksPerGame': 'BLK', 'hitsPerGame': 'HIT'}
            df_bang.rename(columns=bang_map, inplace=True)

            # Smart Join 2
            df_merged = perform_smart_join(df_merged, df_bang, list(bang_map.values()), 'bangers_to_date', conn)

            # GOALIES
            df_goal = pd.read_sql_query("SELECT * FROM goalie_to_date", conn)
            df_goal['nhlplayerid'] = pd.to_numeric(df_goal['nhlplayerid'], errors='coerce').fillna(0).astype(int)
            goal_map = {'gamesStarted': 'GS', 'gamesPlayed': 'GP', 'goalsAgainstAverage': 'GAA', 'losses': 'L', 'savePct': 'SVpct', 'saves': 'SV', 'shotsAgainst': 'SA', 'shutouts': 'SHO', 'wins': 'W', 'win_total': 'win_total', 'goalsAgainst': 'GA', 'startpct': 'startpct'}
            df_goal.rename(columns=goal_map, inplace=True)

            # Smart Join 3
            df_merged = perform_smart_join(df_merged, df_goal, list(goal_map.values()), 'goalie_to_date', conn)

            # 3. Write
            print(f"Writing {len(df_merged)} records to 'stats_to_date'...")
            df_to_postgres(df_merged, 'stats_to_date', conn)
            with conn.cursor() as cursor:
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_stats_date_norm ON stats_to_date(player_name_normalized)")
            conn.commit()

        except Exception as e:
            print(f"Error creating stats_to_date: {e}")


def calculate_and_save_to_date_ranks():
    """
    Reads the 'stats_to_date' table from projections.db, calculates percentile-based
    category ranks for existing stats, and saves the updated table.
    """
    print("\n--- Calculating Ranks for 'stats_to_date' ---")
    with get_db_connection() as conn:
        try:
            df = pd.read_sql_query("SELECT * FROM stats_to_date", conn)
            if df.empty: return

            # Ranking Logic
            new_rank_cols = []
            skater_stats = ['G', 'A', 'P', 'PPG', 'PPA', 'PPP', 'SHG', 'SHA', 'SHP', 'HIT', 'BLK', 'PIM', 'FOW', 'SOG', 'plus_minus']

            # Skaters
            mask_skater = ~df['positions'].str.contains('G', na=False)
            if mask_skater.any():
                num = mask_skater.sum()
                for stat in skater_stats:
                    if stat in df.columns:
                        col = f"{stat}_cat_rank"
                        new_rank_cols.append(col)
                        df[stat] = pd.to_numeric(df[stat], errors='coerce').fillna(0)
                        ranks = df.loc[mask_skater, stat].rank(method='first', ascending=False)
                        pct = ranks / num
                        # Binning logic from original
                        cond = [pct<=0.05, pct<=0.10, pct<=0.15, pct<=0.20, pct<=0.25, pct<=0.30, pct<=0.35, pct<=0.40, pct<=0.45, pct<=0.50, pct<=0.75]
                        choice = [1,2,3,4,5,6,7,8,9,10,15]
                        df.loc[mask_skater, col] = np.select(cond, choice, default=20)

            # Goalies
            goalie_stats = {'GS': False, 'W': False, 'L': True, 'GA': True, 'SA': False, 'SV': False, 'SVpct': False, 'GAA': True, 'SHO': False, 'QS': False}
            mask_goalie = df['positions'].str.contains('G', na=False)
            if mask_goalie.any():
                num = mask_goalie.sum()
                for stat, inv in goalie_stats.items():
                    if stat in df.columns:
                        col = f"{stat}_cat_rank"
                        new_rank_cols.append(col)
                        df[stat] = pd.to_numeric(df[stat], errors='coerce').fillna(0)
                        ranks = df.loc[mask_goalie, stat].rank(method='first', ascending=inv)
                        pct = ranks / num
                        cond = [pct<=0.05, pct<=0.10, pct<=0.15, pct<=0.20, pct<=0.25, pct<=0.30, pct<=0.35, pct<=0.40, pct<=0.45, pct<=0.50, pct<=0.75]
                        choice = [1,2,3,4,5,6,7,8,9,10,15]
                        df.loc[mask_goalie, col] = np.select(cond, choice, default=20)

            # Clean NaNs in rank cols
            for col in new_rank_cols:
                df[col] = df[col].fillna(0).astype(int)

            print(f"Saving ranks to 'stats_to_date'...")
            df_to_postgres(df_merged, 'stats_to_date', conn)
            conn.commit()

        except Exception as e:
            print(f"Error ranking stats_to_date: {e}")


def create_combined_projections():
    """
    Creates a new table 'combined_projections' by merging 'projections'
    and 'stats_to_date'.

    - Identity columns are carried over (prioritizing 'stats_to_date').
    - Data columns existing in both tables are averaged.
    - Data columns existing in only one table are carried over.
    """
    print("\n--- Creating Combined Projections ---")
    with get_db_connection() as conn:
        try:
            df_proj = pd.read_sql_query("SELECT * FROM projections", conn)
            df_stats = pd.read_sql_query("SELECT * FROM stats_to_date", conn)

            if 'gp' in df_proj.columns: df_proj.rename(columns={'gp': 'GP'}, inplace=True)

            # Merge
            df_merged = pd.merge(df_proj, df_stats, on='nhlplayerid', how='outer', suffixes=('_proj', '_stats'))

            # Base ID frame
            df_final = df_merged[['nhlplayerid']].dropna().astype(int).copy()

            # Coalesce Identity
            identity_cols = ['player_name_normalized', 'player_name', 'team', 'age', 'player_id', 'positions', 'status', 'lg_ppTimeOnIce', 'lg_ppTimeOnIcePctPerGame', 'lg_ppAssists', 'lg_ppGoals', 'avg_ppTimeOnIce', 'avg_ppTimeOnIcePctPerGame', 'total_ppAssists', 'total_ppGoals', 'player_games_played', 'team_games_played']

            for col in identity_cols:
                c_proj, c_stats = f"{col}_proj", f"{col}_stats"
                if c_stats in df_merged and c_proj in df_merged:
                    df_final[col] = df_merged[c_stats].fillna(df_merged[c_proj])
                elif c_stats in df_merged: df_final[col] = df_merged[c_stats]
                elif c_proj in df_merged: df_final[col] = df_merged[c_proj]
                elif col in df_merged: df_final[col] = df_merged[col]

            # Average Data Cols
            skip = set(identity_cols + ['nhlplayerid'])
            all_cols = (set(df_proj.columns) | set(df_stats.columns)) - skip

            for col in all_cols:
                c_proj, c_stats = f"{col}_proj", f"{col}_stats"

                # Helpers to get numeric series
                s_proj = pd.to_numeric(df_merged.get(c_proj, df_merged.get(col) if col in df_proj.columns else None), errors='coerce')
                s_stats = pd.to_numeric(df_merged.get(c_stats, df_merged.get(col) if col in df_stats.columns else None), errors='coerce')

                # Logic: If overlap -> Average. Else -> Carry over.
                # Note: This is simplified; original logic checks explicit overlap.
                # Pandas arithmetic (s1 + s2) / 2 handles alignment.
                # We use fillna(0) cautiously only if we want 0 instead of NaN.

                # Replicating specific 'safe_get_and_convert' logic from original
                s_proj = s_proj.fillna(0)
                s_stats = s_stats.fillna(0)

                has_proj = (c_proj in df_merged) or (col in df_proj.columns)
                has_stat = (c_stats in df_merged) or (col in df_stats.columns)

                if has_proj and has_stat:
                    df_final[col] = (s_proj + s_stats) / 2
                elif has_stat:
                    df_final[col] = s_stats
                elif has_proj:
                    df_final[col] = s_proj

            print(f"Saving {len(df_final)} records to 'combined_projections'...")
            df_to_postgres(df_merged, 'combined_projections', conn)
            conn.commit()

        except Exception as e:
            print(f"Error combined projections: {e}")


def update_toi_stats():
    """ Orchestrator Function """
    create_global_tables()
    fetch_team_standings()
    fetch_team_stats_summary()
    fetch_team_stats_weekly()

    # Note: These funcs need the body code restored (omitted for length, but logic is identical to original)
    fetch_and_update_scoring_to_date()
    fetch_and_update_bangers_stats()
    fetch_and_update_goalie_stats()

    fetch_daily_pp_stats()

    create_last_game_pp_table()
    create_last_week_pp_table()

    join_special_teams_data()

    # No need for "copy" functions anymore, we are in the same DB

    create_stats_to_date_table()
    calculate_and_save_to_date_ranks()
    create_combined_projections()
    print("TOI Script Complete.")

if __name__ == "__main__":
    update_toi_stats()
