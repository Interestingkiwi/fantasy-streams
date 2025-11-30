from flask import Blueprint, jsonify
import logging
import psycopg2.extras
from database import get_db_connection

# We import requires_auth from app.
# Since app.py defines requires_auth BEFORE importing api_v1, this circular import is safe.
from app import requires_auth

api = Blueprint('api_v1', __name__, url_prefix='/api/v1')
logger = logging.getLogger(__name__)

@api.route("/league/<league_id>/database-status")
@requires_auth
def api_get_database_status(league_id):
    """
    Checks Postgres for a league's status.
    Replaces the old GCS blob check.
    """
    if not league_id:
        return jsonify({"error": "League ID is required."}), 400

    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:

                # 1. Check League Info (Does the league exist?)
                cursor.execute("""
                    SELECT value
                    FROM league_info
                    WHERE league_id = %s AND key = 'league_name'
                """, (league_id,))
                name_row = cursor.fetchone()

                if not name_row:
                    # League not found in Postgres
                    return jsonify({
                        "exists": False,
                        "message": "No database found. Please build one on the website."
                    }), 404

                league_name = name_row['value']

                # 2. Check Timestamp (When was it last updated?)
                cursor.execute("""
                    SELECT value
                    FROM db_metadata
                    WHERE league_id = %s AND key = 'last_updated_timestamp'
                """, (league_id,))
                time_row = cursor.fetchone()

                last_updated_utc = time_row['value'] if time_row else None

                # 3. Construct Response
                # We maintain the same JSON structure so the mobile app doesn't break
                return jsonify({
                    "exists": True,
                    "league_id": league_id,
                    "league_name": league_name,
                    "filename": f"yahoo-{league_id}.db", # Legacy placeholder
                    "last_updated_utc": last_updated_utc,
                    "size_bytes": 0 # Legacy placeholder
                })

    except Exception as e:
        logger.error(f"API Error checking DB status: {e}", exc_info=True)
        return jsonify({"error": "Internal Server Error"}), 500
