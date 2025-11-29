"""
Orchestrator for Global Data Updates.
Runs: Player IDs -> Projections -> TOI -> Schedule

Author: Jason Druckenmiller
Created: 11/26/2025
Updated: 11/28/2025
"""

import os
import logging
import sys

# Add parent path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jobs.fetch_player_ids import fetch_and_store_players, initialize_yahoo_query
from jobs.create_projection_db import create_projections_db
from jobs.toi_script import update_toi_stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("global_updater")

def update_global_data():
    logger.info("--- Starting Global Data Update ---")

    # 1. Fetch Players (System League ID)
    # FIX: Corrected typo here
    system_league_id = os.environ.get("LEAGUE_ID")

    if system_league_id:
        try:
            logger.info(f"Fetching Players using League ID: {system_league_id}")
            yq = initialize_yahoo_query(int(system_league_id))
            if yq:
                fetch_and_store_players(yq)
            else:
                logger.error("Failed to initialize Yahoo Query.")
        except Exception as e:
            logger.error(f"Player fetch failed: {e}")
    else:
        logger.warning("Skipping Player Fetch: LEAGUE_ID env var not set.")

    # 2. Update Projections & Schedule
    try:
        logger.info("Running Projections & Schedule Update...")
        create_projections_db()
    except Exception as e:
        logger.error(f"Projections update failed: {e}")

    # 3. Update TOI Stats
    try:
        logger.info("Running TOI Stats Update...")
        update_toi_stats()
    except Exception as e:
        logger.error(f"TOI update failed: {e}")

    logger.info("--- Global Update Complete ---")

if __name__ == "__main__":
    update_global_data()
