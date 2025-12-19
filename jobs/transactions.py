@app.route('/tools/scheduled-transactions')
@requires_auth
def scheduled_transactions_page():
    return render_template('pages/scheduled_add_drops.html')

@app.route('/api/tools/search_players')
@requires_auth
def search_players_tool():
    """Smart search for the 'Add' field."""
    query = request.args.get('q', '').strip()
    if len(query) < 3:
        return jsonify([])

    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                # Search by name, limit results
                cursor.execute("""
                    SELECT player_id, player_name, player_team, positions
                    FROM players
                    WHERE player_name ILIKE %s
                    ORDER BY player_name ASC
                    LIMIT 10
                """, (f'%{query}%',))
                players = cursor.fetchall()
        return jsonify(players)
    except Exception as e:
        logging.error(f"Search error: {e}")
        return jsonify([])

@app.route('/api/tools/my_droppable_players')
@requires_auth
def get_droppable_players():
    """Fetches the user's current roster for the 'Drop' dropdown."""
    league_id = session.get('league_id')
    if not league_id:
        return jsonify({'error': 'No league selected'}), 400

    try:
        # 1. Get User's Team Key/ID via YFA (safest way to link user->team)
        lg = get_yfa_lg_instance()
        if not lg:
             return jsonify({'error': 'Could not connect to Yahoo.'}), 401

        user_team_key = lg.team_key() # e.g. 419.l.12345.t.2
        team_id = user_team_key.split('.')[-1] # e.g. 2

        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                # 2. Fetch roster from DB for this team
                cursor.execute("""
                    SELECT p.player_id, p.player_name, p.positions, rp.eligible_positions
                    FROM rosters_tall rp
                    JOIN players p ON CAST(rp.player_id AS TEXT) = p.player_id
                    WHERE rp.league_id = %s AND rp.team_id = %s
                    ORDER BY p.player_name
                """, (league_id, team_id))
                roster = cursor.fetchall()

        return jsonify({'roster': roster, 'team_key': user_team_key})
    except Exception as e:
        logging.error(f"Roster fetch error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/tools/schedule_transaction', methods=['POST'])
@requires_auth
def schedule_transaction():
    data = request.get_json()
    league_id = session.get('league_id')
    user_guid = session['yahoo_token'].get('xoauth_yahoo_guid')

    add_id = data.get('add_player_id')
    add_name = data.get('add_player_name')
    drop_id = data.get('drop_player_id')
    drop_name = data.get('drop_player_name')
    sched_time = data.get('scheduled_time') # Expecting ISO string (UTC recommended)
    team_key = data.get('team_key')

    if not all([add_id, drop_id, sched_time, team_key]):
        return jsonify({'success': False, 'error': 'Missing required fields'}), 400

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO scheduled_transactions
                    (user_guid, league_id, team_key, add_player_id, add_player_name, drop_player_id, drop_player_name, scheduled_time)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (user_guid, league_id, team_key, add_id, add_name, drop_id, drop_name, sched_time))
            conn.commit()

        return jsonify({'success': True})
    except Exception as e:
        logging.error(f"Scheduling error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/tools/get_scheduled_transactions')
@requires_auth
def get_scheduled_transactions():
    league_id = session.get('league_id')
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT * FROM scheduled_transactions
                    WHERE league_id = %s
                    ORDER BY scheduled_time ASC
                """, (league_id,))
                rows = cursor.fetchall()
        return jsonify(rows)
    except Exception as e:
        return jsonify([])
