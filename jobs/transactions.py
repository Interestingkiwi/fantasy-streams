import logging
import time
import json
import os
import tempfile
import uuid
from datetime import datetime
from database import get_db_connection
import yahoo_fantasy_api as yfa
from yahoo_oauth import OAuth2

logger = logging.getLogger(__name__)

def refresh_user_token(cursor, user_guid, user_row):
    """Helper to refresh token using stored credentials."""
    creds = {
        "consumer_key": user_row['consumer_key'],
        "consumer_secret": user_row['consumer_secret'],
        "access_token": user_row['access_token'],
        "refresh_token": user_row['refresh_token'],
        "token_type": user_row['token_type'],
        "token_time": user_row['token_time']
    }

    # Write temp file for oauth lib
    temp_dir = os.path.join(tempfile.gettempdir(), 'temp_creds')
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}.json")

    try:
        with open(temp_file_path, 'w') as f:
            json.dump(creds, f)

        sc = OAuth2(None, None, from_file=temp_file_path)
        if not sc.token_is_valid():
            logger.info(f"Refreshing token for user {user_guid}")
            sc.refresh_access_token()

            # Read new creds
            with open(temp_file_path, 'r') as f:
                new_creds = json.load(f)

            # Update DB
            cursor.execute("""
                UPDATE users SET
                    access_token = %s, refresh_token = %s, token_time = %s
                WHERE guid = %s
            """, (new_creds['access_token'], new_creds['refresh_token'], new_creds['token_time'], user_guid))

        return sc
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def process_due_transactions():
    """Main job function called by Scheduler."""
    logger.info("Checking for due transactions...")

    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            # 1. Fetch Due Transactions
            cursor.execute("""
                SELECT id, user_guid, league_id, team_key, add_player_id, add_player_name, drop_player_id, drop_player_name
                FROM scheduled_transactions
                WHERE status = 'pending' AND scheduled_time <= NOW()
            """)
            due_moves = cursor.fetchall()

            if not due_moves:
                return

            logger.info(f"Found {len(due_moves)} transactions to execute.")

            for move in due_moves:
                trans_id = move[0]
                user_guid = move[1]
                team_key = move[3]
                add_id = move[4]
                drop_id = move[6]

                try:
                    # 2. Get User Credentials
                    cursor.execute("SELECT * FROM users WHERE guid = %s", (user_guid,))
                    cols = [desc[0] for desc in cursor.description]
                    user_row_tuple = cursor.fetchone()
                    user_row = dict(zip(cols, user_row_tuple))

                    if not user_row:
                        raise Exception("User credentials not found.")

                    # 3. Authenticate & Refresh
                    sc = refresh_user_token(cursor, user_guid, user_row)

                    # 4. Perform Move
                    # FIX: Instantiate Team directly instead of using Game.to_team (which doesn't exist)
                    tm = yfa.Team(sc, team_key)

                    logger.info(f"Executing: Add {add_id} / Drop {drop_id} for {team_key}")
                    tm.add_and_drop_players(add_id, drop_id)

                    # 5. Success Update
                    cursor.execute("""
                        UPDATE scheduled_transactions
                        SET status = 'executed', result_message = 'Success'
                        WHERE id = %s
                    """, (trans_id,))

                except Exception as e:
                    logger.error(f"Transaction {trans_id} failed: {e}")
                    cursor.execute("""
                        UPDATE scheduled_transactions
                        SET status = 'failed', result_message = %s
                        WHERE id = %s
                    """, (str(e), trans_id))

            conn.commit()
