(async function() {
    await new Promise(resolve => setTimeout(resolve, 0));

    const errorDiv = document.getElementById('db-error-message');
    const controlsDiv = document.getElementById('lineup-controls');
    const tableContainer = document.getElementById('roster-table-container');
    const optimalLineupContainer = document.getElementById('optimal-lineup-container');
    const unusedRosterSpotsContainer = document.getElementById('unused-roster-spots-container');
    const weekSelect = document.getElementById('week-select');
    const checkboxesContainer = document.getElementById('category-checkboxes-container');
    const yourTeamSelect = document.getElementById('your-team-select');
    const simLogContainer = document.getElementById('simulated-moves-log');
    const SIMULATION_KEY = 'simulationCache';
    const CATEGORY_PREF_KEY = 'lineupCategoryPreferences';

    let pageData = null;
    let allScoringCategories = [];
    let checkedCategories = [];
    let simulatedMoves = [];
    let skaterCategories = [];
    let goalieCategories = [];
    let currentRosterData = null;

    // --- Helper Functions ---
    function formatPercentage(decimal) { if (decimal == null) return 'N/A'; try { const num = parseFloat(decimal); if (isNaN(num)) return 'N/A'; return (num * 100).toFixed(1) + '%'; } catch (e) { return 'N/A'; } }
    function formatSecondsToMMSS(seconds) { if (seconds == null) return 'N/A'; try { const s = parseInt(seconds, 10); if (isNaN(s)) return 'N/A'; const m = Math.floor(s / 60); const rs = s % 60; return `${m}:${rs < 10 ? '0' : ''}${rs}`; } catch (e) { return 'N/A'; } }
    function formatNullable(value) { return value ?? 'N/A'; }
    function formatNumber(value, decimals, defaultVal = 'N/A') { if (value == null) return defaultVal; try { const num = parseFloat(value); if (isNaN(num)) return defaultVal; return num.toFixed(decimals); } catch (e) { return defaultVal; } }
    function getHeatmapColor(rank) { if (rank == null || rank === '-') return ''; const min = 1, max = 20; const clamped = Math.max(min, Math.min(rank, max)); const pct = (clamped - min) / (max - min); const hue = (1 - pct) * 120; return `hsl(${hue}, 65%, 75%)`; }

    async function init() {
        // --- 1. Inject Local Modals (PP & Opponent) ---
        if (!document.getElementById('pp-stats-modal')) {
            const modalsHTML = `
            <div id="pp-stats-modal" class="fixed inset-0 bg-gray-900 bg-opacity-75 flex items-center justify-center z-50 hidden" style="backdrop-filter: blur(2px);">
                <div class="bg-gray-800 rounded-lg shadow-xl p-6 w-full max-w-lg relative border border-gray-700">
                    <button id="pp-modal-close" class="absolute top-3 right-3 text-gray-400 hover:text-white text-2xl leading-none">&times;</button>
                    <h3 id="pp-modal-title" class="text-xl font-bold text-white mb-4">Player PP Stats</h3>
                    <div id="pp-modal-content" class="text-gray-300"></div>
                </div>
            </div>
            <div id="opponent-stats-modal" class="fixed inset-0 bg-gray-900 bg-opacity-75 flex items-center justify-center z-50 hidden" style="backdrop-filter: blur(2px);">
                <div class="bg-gray-800 rounded-lg shadow-xl p-6 w-full max-w-2xl relative border border-gray-700">
                    <button id="opponent-modal-close" class="absolute top-3 right-3 text-gray-400 hover:text-white text-2xl leading-none">&times;</button>
                    <h3 id="opponent-modal-title" class="text-xl font-bold text-white mb-4">Opponent Stats</h3>
                    <div id="opponent-modal-content" class="text-gray-300 overflow-x-auto"></div>
                </div>
            </div>`;
            document.body.insertAdjacentHTML('beforeend', modalsHTML);
        }

        // --- 2. Global Modal Click Listener (Delegation) ---
        // Attaching to Body ensures we catch clicks even after table re-renders
        document.body.addEventListener('click', (e) => {
            // Close Buttons
            if (e.target.closest('#pp-modal-close') || e.target.id === 'pp-stats-modal') document.getElementById('pp-stats-modal').classList.add('hidden');
            if (e.target.closest('#opponent-modal-close') || e.target.id === 'opponent-stats-modal') document.getElementById('opponent-stats-modal').classList.add('hidden');

            // 1. PP Util Cell
            const ppCell = e.target.closest('.pp-util-cell');
            if (ppCell) {
                const data = ppCell.dataset;
                document.getElementById('pp-modal-title').textContent = `${data.playerName} - PP Stats`;
                document.getElementById('pp-modal-content').innerHTML = `
                <div class="space-y-4">
                    <div><h4 class="text-md font-semibold text-white mb-2">Last Game</h4><dl class="grid grid-cols-2 sm:grid-cols-4 gap-x-4 gap-y-2"><div class="bg-gray-700 p-2 rounded"><dt class="text-xs text-gray-400">PP TOI</dt><dd class="text-sm font-medium">${formatSecondsToMMSS(data.lgPpToi)}</dd></div><div class="bg-gray-700 p-2 rounded"><dt class="text-xs text-gray-400">PP %</dt><dd class="text-sm font-medium">${formatPercentage(data.lgPpPct)}</dd></div><div class="bg-gray-700 p-2 rounded"><dt class="text-xs text-gray-400">PPA</dt><dd class="text-sm font-medium">${formatNullable(data.lgPpa)}</dd></div><div class="bg-gray-700 p-2 rounded"><dt class="text-xs text-gray-400">PPG</dt><dd class="text-sm font-medium">${formatNullable(data.lgPpg)}</dd></div></dl></div>
                    <div><h4 class="text-md font-semibold text-white mb-2">Last Week</h4><dl class="grid grid-cols-2 sm:grid-cols-5 gap-x-4 gap-y-2"><div class="bg-gray-700 p-2 rounded"><dt class="text-xs text-gray-400">Avg PP TOI</dt><dd class="text-sm font-medium">${formatSecondsToMMSS(data.lwPpToi)}</dd></div><div class="bg-gray-700 p-2 rounded"><dt class="text-xs text-gray-400">Avg PP %</dt><dd class="text-sm font-medium">${formatPercentage(data.lwPpPct)}</dd></div><div class="bg-gray-700 p-2 rounded"><dt class="text-xs text-gray-400">Tot PPA</dt><dd class="text-sm font-medium">${formatNullable(data.lwPpa)}</dd></div><div class="bg-gray-700 p-2 rounded"><dt class="text-xs text-gray-400">Tot PPG</dt><dd class="text-sm font-medium">${formatNullable(data.lwPpg)}</dd></div><div class="bg-gray-700 p-2 rounded"><dt class="text-xs text-gray-400">GP</dt><dd class="text-sm font-medium">${formatNullable(data.lwGp)}</dd></div></dl></div>
                </div>`;
                document.getElementById('pp-stats-modal').classList.remove('hidden');
                return;
            }
            const pill = e.target.closest('.line-info-pill');
            if (pill && window.openLineInfoModal && currentRosterData) {
                const pid = String(pill.dataset.playerId);
                const player = currentRosterData.players.find(p => String(p.player_id) === pid);
                if (player) {
                    window.openLineInfoModal(player);
                }
                return;
            }
            const gPill = e.target.closest('.goalie-info-pill');
            if (gPill && window.openGoalieInfoModal && (rosterData.players || allWaiverPlayers)) {
                const pid = String(gPill.dataset.playerId);
                // Search available lists
                let player;
                if (typeof rosterData !== 'undefined') player = rosterData.players.find(p => String(p.player_id) === pid);
                if (!player && typeof allWaiverPlayers !== 'undefined') player = [...allWaiverPlayers, ...allFreeAgents].find(p => String(p.player_id) === pid);

                if (player) window.openGoalieInfoModal(player);
                return;
            }
            // 2. Opponent Stats Cell
            const oppCell = e.target.closest('.opponent-stats-cell');
            if (oppCell) {
                const data = oppCell.dataset;
                const isGoalie = data.isGoalie === 'true';
                const stats = JSON.parse(data.opponentStats || '[]');
                document.getElementById('opponent-modal-title').textContent = `${data.playerName} - Opponent Stats`;

                let headers, statKeys, totalAvgs;
                if (isGoalie) {
                    headers = ["Date", "Opp", "GF/G (Szn)", "GF/G (Last Wk)", "SOG/G (Szn)", "SOG/G (Last Wk)"];
                    statKeys = ["gf_gm", "gf_gm_weekly", "sogf_gm", "sogf_gm_weekly"];
                    totalAvgs = { gf_gm: 0, sogf_gm: 0, gf_gm_weekly: 0, sogf_gm_weekly: 0, count: 0 };
                } else {
                    headers = ["Date", "Opp", "GA/G (Szn)", "GA/G (Last Wk)", "SOGA/G (Szn)", "SOGA/G (Last Wk)", "PK% (Szn)", "PK% (Last Wk)"];
                    statKeys = ["ga_gm", "ga_gm_weekly", "soga_gm", "soga_gm_weekly", "pk_pct", "pk_pct_weekly"];
                    totalAvgs = { ga_gm: 0, soga_gm: 0, ga_gm_weekly: 0, soga_gm_weekly: 0, pk_pct: 0, pk_pct_weekly: 0, count: 0 };
                }

                let tableHtml = `<table class="min-w-full divide-y divide-gray-700"><thead class="bg-gray-700/50"><tr>${headers.map(h => `<th class="px-2 py-1 text-left text-xs font-bold text-gray-300 uppercase tracking-wider">${h}</th>`).join('')}</tr></thead><tbody class="bg-gray-800 divide-y divide-gray-700">`;
                if (stats.length === 0) {
                    tableHtml += `<tr><td colspan="${headers.length}" class="text-center text-gray-400 py-3">No opponent data available for this week.</td></tr>`;
                } else {
                    stats.forEach(game => {
                        tableHtml += `<tr class="hover:bg-gray-700/50"><td class="px-2 py-1 whitespace-nowrap text-sm text-gray-300">${game.game_date}</td><td class="px-2 py-1 whitespace-nowrap text-sm text-gray-300">${game.opponent_tricode}</td>`;
                        statKeys.forEach(key => {
                            const isWhole = key.includes('sog');
                            const isPct = key.includes('pk_');
                            const val = parseFloat(game[key]);
                            if (!isNaN(val)) totalAvgs[key] += val;
                            let formattedVal = isPct ? formatPercentage(game[key]) : formatNumber(game[key], isWhole ? 0 : 2);
                            tableHtml += `<td class="px-2 py-1 whitespace-nowrap text-sm text-gray-300">${formattedVal}</td>`;
                        });
                        tableHtml += `</tr>`;
                    });
                    const numGames = stats.length || 1;
                    tableHtml += `<tr class="bg-gray-700 font-bold"><td class="px-2 py-1 text-sm text-white" colspan="2">Average</td>`;
                    statKeys.forEach(key => {
                        const isWhole = key.includes('sog');
                        const isPct = key.includes('pk_');
                        const avgVal = totalAvgs[key] / numGames;
                        let formattedAvg = isPct ? formatPercentage(avgVal) : formatNumber(avgVal, isWhole ? 0 : 2);
                        tableHtml += `<td class="px-2 py-1 whitespace-nowrap text-sm text-white">${formattedAvg}</td>`;
                    });
                    tableHtml += `</tr>`;
                }
                tableHtml += `</tbody></table>`;
                document.getElementById('opponent-modal-content').innerHTML = tableHtml;
                document.getElementById('opponent-stats-modal').classList.remove('hidden');
            }

            // 3. Handle Cat Rank Cell (Calls Global Modal)
            const rankCell = e.target.closest('.cat-rank-cell');
            if (rankCell && currentRosterData) {
                const pid = String(rankCell.dataset.playerId); // String for safety
                const player = currentRosterData.players.find(p => String(p.player_id) === pid);

                if (player && window.openCatRankModal) {
                    const isGoalie = (player.eligible_positions || '').includes('G');
                    const cats = isGoalie ? goalieCategories : skaterCategories;
                    window.openCatRankModal(player, cats);
                }
                return;
            }
        });

        try {
            const response = await fetch('/api/lineup_page_data');
            const data = await response.json();
            if (!response.ok || !data.db_exists) throw new Error(data.error || 'Database has not been initialized.');

            pageData = data;
            populateDropdowns();
            const savedCategories = localStorage.getItem(CATEGORY_PREF_KEY);
            if (savedCategories) checkedCategories = JSON.parse(savedCategories);

            setupEventListeners();
            await fetchAndRenderTable();
            controlsDiv.classList.remove('hidden');
        } catch (error) {
            console.error('Initialization error:', error);
            errorDiv.classList.remove('hidden');
            controlsDiv.classList.add('hidden');
            tableContainer.classList.add('hidden');
        }
    }

    window.addEventListener('rawDataToggled', (e) => {
        if (currentRosterData) {
            renderTable(currentRosterData.players, currentRosterData.daily_optimal_lineups);
        }
    });

    function populateDropdowns() {
        weekSelect.innerHTML = pageData.weeks.map(w => `<option value="${w.week_num}">Week ${w.week_num} (${w.start_date} to ${w.end_date})</option>`).join('');
        yourTeamSelect.innerHTML = pageData.teams.map(t => `<option value="${t.name}">${t.name}</option>`).join('');

        const savedTeam = localStorage.getItem('selectedTeam');
        if (savedTeam) yourTeamSelect.value = savedTeam;

        if (!sessionStorage.getItem('fantasySessionStarted')) {
            weekSelect.value = pageData.current_week;
            localStorage.setItem('selectedWeek', pageData.current_week);
            sessionStorage.setItem('fantasySessionStarted', 'true');
        } else {
            const savedWeek = localStorage.getItem('selectedWeek');
            if (savedWeek) weekSelect.value = savedWeek;
            else weekSelect.value = pageData.current_week;
        }
    }

    async function fetchAndRenderTable() {
        const selectedWeek = weekSelect.value;
        const yourTeamName = yourTeamSelect.value;
        if (!selectedWeek || !yourTeamName) return;

        tableContainer.innerHTML = '<p class="text-gray-400">Loading roster...</p>';

        const categoryCheckboxes = document.querySelectorAll('#category-checkboxes-container input[name="category"]:checked');
        let categoriesToSend = categoryCheckboxes.length > 0 ? Array.from(categoryCheckboxes).map(cb => cb.value) : checkedCategories;

        const cachedSim = localStorage.getItem(SIMULATION_KEY);
        simulatedMoves = cachedSim ? JSON.parse(cachedSim) : [];
        const selectedSourcing = localStorage.getItem('selectedStatSourcing') || 'projected';

        try {
            const response = await fetch('/api/roster_data', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ week: selectedWeek, team_name: yourTeamName, categories: categoriesToSend, simulated_moves: simulatedMoves, sourcing: selectedSourcing })
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Failed to fetch roster.');

            currentRosterData = data;

            if (allScoringCategories.length === 0) {
                allScoringCategories = data.scoring_categories;
                if (localStorage.getItem(CATEGORY_PREF_KEY) === null) checkedCategories = data.checked_categories;
                renderCategoryCheckboxes();
            }
            skaterCategories = data.skater_categories;
            goalieCategories = data.goalie_categories;

            renderTable(data.players, data.daily_optimal_lineups);
            renderOptimalLineups(data.daily_optimal_lineups, data.lineup_settings);
            renderUnusedRosterSpotsTable(data.unused_roster_spots);
            renderSimulatedMovesLog();

        } catch(error) {
            console.error('Error fetching roster:', error);
            tableContainer.innerHTML = `<p class="text-red-400">Error: ${error.message}</p>`;
        }
    }

    function renderTable(roster, dailyLineups) {
        const playerStartsByDay = {};
        const dayAbbrMap = { 'Monday': 'Mon', 'Tuesday': 'Tue', 'Wednesday': 'Wed', 'Thursday': 'Thu', 'Friday': 'Fri', 'Saturday': 'Sat', 'Sunday': 'Sun' };
        for (const dayString in dailyLineups) {
            const lineup = dailyLineups[dayString];
            const dayName = dayString.split(',')[0];
            const dayAbbr = dayAbbrMap[dayName];
            if (dayAbbr) {
                for (const position in lineup) {
                    lineup[position].forEach(player => {
                        if (!playerStartsByDay[player.player_name]) playerStartsByDay[player.player_name] = new Set();
                        playerStartsByDay[player.player_name].add(dayAbbr);
                    });
                }
            }
        }

        const skaters = roster.filter(p => !(p.eligible_positions || p.positions || '').includes('G'));
        const goalies = roster.filter(p => (p.eligible_positions || p.positions || '').includes('G'));

        const positionOrder = ['C', 'LW', 'RW', 'F', 'W', 'D', 'Util', 'G', 'IR', 'IR+'];
        const sortFn = (a, b) => {
            const getBestPosIndex = (posArr) => {
                let minIndex = Infinity;
                let isIR = false;
                posArr.forEach(p => {
                    if (p.includes('IR')) isIR = true;
                    const idx = positionOrder.indexOf(p);
                    if (idx !== -1 && idx < minIndex) minIndex = idx;
                });
                if (isIR) return 100;
                if (minIndex === Infinity) return 99;
                return minIndex;
            };
            const posA = (a.eligible_positions || a.positions || '').split(',').map(p => p.trim());
            const posB = (b.eligible_positions || b.positions || '').split(',').map(p => p.trim());
            return getBestPosIndex(posA) - getBestPosIndex(posB);
        };

        skaters.sort(sortFn);
        goalies.sort(sortFn);

        tableContainer.innerHTML = buildPlayerTable('Skaters', skaters, skaterCategories, playerStartsByDay) +
                                   buildPlayerTable('Goalies', goalies, goalieCategories, playerStartsByDay);
    }

    function buildPlayerTable(title, players, categories, playerStartsByDay) {
        const showRaw = localStorage.getItem('showRawData') === 'true';

        let tableHtml = `
            <div class="bg-gray-900 rounded-lg shadow ${title === 'Goalies' ? 'mt-6' : ''}">
                <h2 class="text-xl font-bold text-white p-3 bg-gray-800 rounded-t-lg">${title}</h2>
                <div class="overflow-x-auto">
                    <table class="min-w-full divide-y divide-gray-700">
                        <thead class="bg-gray-700/50">
                            <tr>
                                <th class="px-2 py-1 text-left text-xs font-bold text-gray-300 uppercase">Player Name</th>
                                <th class="px-2 py-1 text-left text-xs font-bold text-gray-300 uppercase">Team</th>
                                <th class="px-2 py-1 text-left text-xs font-bold text-gray-300 uppercase">Positions</th>
                                <th class="px-2 py-1 text-left text-xs font-bold text-gray-300 uppercase">This Week</th>
                                <th class="px-2 py-1 text-left text-xs font-bold text-gray-300 uppercase">Opponents</th>
                                <th class="px-2 py-1 text-left text-xs font-bold text-gray-300 uppercase">#</th>
                                <th class="px-2 py-1 text-left text-xs font-bold text-gray-300 uppercase">Starts</th>
                                <th class="px-2 py-1 text-left text-xs font-bold text-gray-300 uppercase">Next Wk</th>
                                <th class="px-2 py-1 text-left text-xs font-bold text-gray-300 uppercase">PP Util</th>
                                <th class="px-2 py-1 text-center text-xs font-bold text-gray-300 uppercase">Total Rank</th>
        `;
        (categories || []).forEach(cat => {
            tableHtml += `<th class="px-2 py-1 text-center text-xs font-bold text-gray-300 uppercase">${cat}</th>`;
        });
        tableHtml += `</tr></thead><tbody class="bg-gray-800 divide-y divide-gray-700">`;

        if (players.length === 0) {
            const colspan = 10 + (categories || []).length;
            tableHtml += `<tr><td colspan="${colspan}" class="text-center py-4 text-gray-400">No ${title.toLowerCase()} found on roster.</td></tr>`;
        }

        players.forEach(player => {
            const gamesThisWeekHtml = (player.games_this_week || []).map(day => (playerStartsByDay[player.player_name]?.has(day) ? `<strong class="text-yellow-300">${day}</strong>` : day)).join(', ');
            const statusHtml = (player.status && player.status !== 'FA') ? `<span class="text-red-400 ml-1">(${player.status})</span>` : '';

            // --- LINE INFO PILL ---
            let pillHtml = '';
            const isGoalie = (p.positions || p.eligible_positions || '').includes('G');
            const lineVal = player.line_number ? `L${player.line_number}` : 'N/A';
            const ppVal = player.pp_unit ? `${player.pp_unit}` : 'N/A';
            if (isGoalie) {
            // GOALIE PILL
            // Expecting player.goalie_data from backend
            const gd = p.goalie_data || { l10_start_pct: 'N/A', days_rest: 'N/A', next_loc: 'N/A' };
            const pct = gd.l10_start_pct !== 'N/A' ? `${gd.l10_start_pct}%` : 'N/A';

            pillHtml = `
                <span class="ml-2 px-2 py-0.5 rounded text-[10px] font-bold bg-blue-900 text-blue-200 border border-blue-700 cursor-pointer hover:bg-blue-800 goalie-info-pill"
                      data-player-id="${p.player_id}">
                    ${pct} | Rest: ${gd.days_rest} | ${gd.next_loc}
                </span>`;
        } else {
            // SKATER PILL (Existing Logic)
            let lineVal = p.line_number;
            if (!lineVal || lineVal === 'Depth') lineVal = 'N/A';
            else lineVal = `L${lineVal}`;

            let ppVal = p.pp_unit;
            if (!ppVal || ppVal === 'Depth') ppVal = 'N/A';

            pillHtml = `
                <span class="ml-2 px-2 py-0.5 rounded text-[10px] font-bold bg-red-900 text-red-200 border border-red-700 cursor-pointer hover:bg-red-800 line-info-pill"
                      data-player-id="${p.player_id}">
                    ${lineVal} | ${ppVal}
                </span>`;
        }
            // ----------------------

            const opponentsList = (player.opponents_list || []).join(', ');
            const opponentStatsJson = JSON.stringify(player.opponent_stats_this_week || []);
            const isGoalie = (player.eligible_positions || player.positions || '').includes('G');

            let catRankSum = 0, validRanks = 0;
            (categories || []).forEach(cat => {
                const r = player[cat + '_cat_rank'];
                if (r != null) { catRankSum += r; validRanks++; }
            });

            tableHtml += `
                <tr class="hover:bg-gray-700/50">
                    <td class="px-2 py-1 text-sm font-medium text-gray-300 flex items-center">
                        ${player.player_name}${statusHtml}
                        ${pillHtml}
                    </td>
                    <td class="px-2 py-1 text-sm text-gray-300">${player.team || player.player_team}</td>
                    <td class="px-2 py-1 text-sm text-gray-300">${player.eligible_positions}</td>
                    <td class="px-2 py-1 text-sm text-gray-300">${gamesThisWeekHtml}</td>
                    <td class="px-2 py-1 text-sm text-gray-300 cursor-pointer hover:bg-gray-700 opponent-stats-cell"
                        data-player-name="${player.player_name}"
                        data-is-goalie="${isGoalie}"
                        data-opponent-stats='${opponentStatsJson}'>${opponentsList}</td>
                    <td class="px-2 py-1 text-sm text-gray-300">${(player.games_this_week || []).length}</td>
                    <td class="px-2 py-1 text-sm text-gray-300">${player.starts_this_week}</td>
                    <td class="px-2 py-1 text-sm text-gray-300">${(player.games_next_week || []).join(', ')}</td>

                    <td class="px-2 py-1 text-sm text-gray-300 cursor-pointer hover:bg-gray-700 pp-util-cell"
                        data-player-name="${player.player_name}"
                        data-avg-pp-pct="${player.avg_ppTimeOnIcePctPerGame}"
                        data-lg-pp-toi="${player.lg_ppTimeOnIce}"
                        data-lg-pp-pct="${player.lg_ppTimeOnIcePctPerGame}"
                        data-lg-ppa="${player.lg_ppAssists}"
                        data-lg-ppg="${player.lg_ppGoals}"
                        data-lw-pp-toi="${player.avg_ppTimeOnIce}"
                        data-lw-pp-pct="${player.avg_ppTimeOnIcePctPerGame}"
                        data-lw-ppa="${player.total_ppAssists}"
                        data-lw-ppg="${player.total_ppGoals}"
                        data-lw-gp="${player.team_games_played}">
                        ${formatPercentage(player.avg_ppTimeOnIcePctPerGame)}
                    </td>

                    <td class="px-2 py-1 text-center font-bold text-blue-400 cursor-pointer hover:text-blue-300 cat-rank-cell" data-player-id="${player.player_id}">
                        ${validRanks > 0 ? Math.round(catRankSum) : '-'}
                    </td>
            `;

            // --- MODIFIED: Use rank color in both Raw and Rank views ---
            (categories || []).forEach(cat => {
                const rank = player[cat + '_cat_rank'];
                const heatColor = getHeatmapColor(rank);
                let displayValue = '-';

                if (showRaw) {
                    const val = player[cat];
                    displayValue = (val != null && !isNaN(val)) ? parseFloat(val).toFixed(2).replace(/[.,]00$/, "") : (val || '-');
                } else {
                    displayValue = (rank != null) ? Math.round(rank) : '-';
                }

                let cellStyle = '';
                let cellClass = 'px-2 py-1 text-center text-sm';

                if (heatColor) {
                    // Apply heat color with dark text for contrast
                    cellStyle = `background-color: ${heatColor}; color: #1f2937; font-weight: 600;`;
                } else {
                    // Default gray text if no rank available
                    cellClass += ' text-gray-400';
                }

                tableHtml += `<td class="${cellClass}" style="${cellStyle}">${displayValue}</td>`;
            });
            // --- END MODIFICATION ---

            tableHtml += `</tr>`;
        });

        tableHtml += `</tbody></table></div></div>`;
        return tableHtml;
    }

    function renderCategoryCheckboxes() { /* ... (Unchanged) ... */
        let checkboxHtml = `
            <div class="flex justify-between items-center mb-2">
                <label class="block text-sm font-medium text-gray-300">Update Lineup Priority Based On:</label>
                <div>
                    <button id="check-all-btn" class="text-xs bg-gray-600 hover:bg-gray-500 text-white py-1 px-2 rounded mr-2 transition-colors duration-150">Check All</button>
                    <button id="uncheck-all-btn" class="text-xs bg-gray-600 hover:bg-gray-500 text-white py-1 px-2 rounded transition-colors duration-150">Uncheck All</button>
                </div>
            </div>
            <div class="grid grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-2 mb-4">
        `;
        allScoringCategories.forEach(cat => {
            const isChecked = checkedCategories.includes(cat);
            checkboxHtml += `
                <div class="flex items-center">
                    <input id="cat-${cat}" name="category" type="checkbox" value="${cat}" ${isChecked ? 'checked' : ''} class="h-4 w-4 bg-gray-700 border-gray-600 text-indigo-600 focus:ring-indigo-500 rounded">
                    <label for="cat-${cat}" class="ml-2 block text-sm text-gray-300">${cat}</label>
                </div>
            `;
        });
        checkboxHtml += '</div>';

        checkboxHtml += `
            <button id="update-lineups-btn" class="w-full bg-indigo-600 hover:bg-indigo-500 text-white font-bold py-2 px-4 rounded transition-colors duration-150">
                Update Lineups
            </button>
        `;
        checkboxesContainer.innerHTML = checkboxHtml;

        document.getElementById('check-all-btn').addEventListener('click', () => {
            document.querySelectorAll('#category-checkboxes-container input[name="category"]').forEach(cb => cb.checked = true);
        });

        document.getElementById('uncheck-all-btn').addEventListener('click', () => {
            document.querySelectorAll('#category-checkboxes-container input[name="category"]').forEach(cb => cb.checked = false);
        });

        document.getElementById('update-lineups-btn').addEventListener('click', () => {
            const currentChecked = Array.from(
                document.querySelectorAll('#category-checkboxes-container input[name="category"]:checked')
            ).map(cb => cb.value);
            localStorage.setItem(CATEGORY_PREF_KEY, JSON.stringify(currentChecked));
            fetchAndRenderTable();
        });
    }

    function setupEventListeners() {
        weekSelect.addEventListener('change', fetchAndRenderTable);
        yourTeamSelect.addEventListener('change', fetchAndRenderTable);
    }

    function renderOptimalLineups(dailyLineups, lineupSettings) { /* ... (Unchanged) ... */
        let finalHtml = '<div class="flex flex-wrap gap-4 justify-center">';
        const positionOrder = ['C', 'LW', 'RW', 'F', 'W', 'D', 'Util', 'G'];

        const sortedDays = Object.keys(dailyLineups).sort((a, b) => {
            const currentYear = new Date().getFullYear();
            const dateA = new Date(`${a}, ${currentYear}`);
            const dateB = new Date(`${b}, ${currentYear}`);
            return dateA - dateB;
        });

        sortedDays.forEach(day => {
            const lineup = dailyLineups[day];
            let tableHtml = `
                <div class="bg-gray-900 rounded-lg shadow flex-grow" style="min-width: 300px;">
                    <h2 class="text-xl font-bold text-white p-3 bg-gray-800 rounded-t-lg">${day}</h2>
                    <table class="w-full divide-y divide-gray-700">
                        <thead class="bg-gray-700/50">
                            <tr>
                                <th scope="col" class="px-2 py-1 text-left text-xs font-bold text-gray-300 uppercase tracking-wider">Position</th>
                                <th scope="col" class="px-2 py-1 text-left text-xs font-bold text-gray-300 uppercase tracking-wider">Player Name</th>
                            </tr>
                        </thead>
                        <tbody class="bg-gray-800 divide-y divide-gray-700">
            `;
            positionOrder.forEach(pos => {
                const numSlots = lineupSettings[pos] || 0;
                const playersInPos = lineup[pos] || [];
                for (let i = 0; i < numSlots; i++) {
                    const player = playersInPos[i];
                    if (player) {
                        tableHtml += `<tr class="hover:bg-gray-700/50"><td class="px-2 py-1 whitespace-nowrap text-sm font-medium text-gray-300">${pos}</td><td class="px-2 py-1 whitespace-nowrap text-sm text-gray-300">${player.player_name}</td></tr>`;
                    } else {
                        tableHtml += `<tr class="hover:bg-gray-700/50"><td class="px-2 py-1 whitespace-nowrap text-sm font-medium text-gray-300">${pos}</td><td class="px-2 py-1 whitespace-nowrap text-sm text-gray-500 italic">(Empty)</td></tr>`;
                    }
                }
            });
            tableHtml += `</tbody></table></div>`;
            finalHtml += tableHtml;
        });

        if (sortedDays.length === 0) {
            optimalLineupContainer.innerHTML = '<p class="text-gray-400">No games scheduled for active players this week.</p>';
        } else {
            finalHtml += '</div>';
            optimalLineupContainer.innerHTML = finalHtml;
        }
    }

    function renderUnusedRosterSpotsTable(unusedSpotsData) { /* ... (Unchanged) ... */
        if (!unusedSpotsData) {
            unusedRosterSpotsContainer.innerHTML = '';
            return;
        }

        const dayOrder = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
        const sortedDays = Object.keys(unusedSpotsData).sort((a, b) => dayOrder.indexOf(a) - dayOrder.indexOf(b));

        const firstDayData = unusedSpotsData[sortedDays[0]];
        const availablePositions = Object.keys(firstDayData);

        const masterOrder = ['C', 'LW', 'RW', 'F', 'W', 'D', 'Util', 'G'];
        const positionOrder = masterOrder.filter(pos => availablePositions.includes(pos));

        let tableHtml = `
            <div class="bg-gray-900 rounded-lg shadow mt-6">
                <h2 class="text-xl font-bold text-white p-3 bg-gray-800 rounded-t-lg">Unused Roster Spots</h2>
                <div class="overflow-x-auto">
                    <table class="divide-y divide-gray-700">
                        <thead class="bg-gray-700/50">
                            <tr>
                                <th class="px-2 py-1 text-left text-xs font-bold text-gray-300 uppercase tracking-wider">Day</th>
                                ${positionOrder.map(pos => `<th class="px-2 py-1 text-center text-xs font-bold text-gray-300 uppercase tracking-wider">${pos}</th>`).join('')}
                            </tr>
                        </thead>
                        <tbody class="bg-gray-800 divide-y divide-gray-700">
        `;
        sortedDays.forEach(day => {
            tableHtml += `<tr class="hover:bg-gray-700/50">
                <td class="px-2 py-1 whitespace-nowrap text-sm font-medium text-gray-300">${day}</td>`;
            positionOrder.forEach(pos => {
                const value = unusedSpotsData[day][pos];
                const stringValue = String(value);
                const highlightClass = (stringValue !== '0') ? 'bg-green-200 text-gray-900' : 'text-gray-300';
                tableHtml += `<td class="px-2 py-1 whitespace-nowrap text-sm text-center ${highlightClass}">${value}</td>`;
            });
            tableHtml += `</tr>`;
        });
        tableHtml += `</tbody></table></div></div>`;
        unusedRosterSpotsContainer.innerHTML = tableHtml;
    }

    function renderSimulatedMovesLog() {
        if (!simLogContainer) return;
        if (simulatedMoves.length === 0) {
            simLogContainer.innerHTML = '';
            return;
        }
        const sortedMoves = [...simulatedMoves].sort((a, b) => (a.date < b.date) ? -1 : 1);
        let logHtml = `
            <p class="text-sm text-gray-400 italic mb-2">Lineups assume the below planned transactions are made.</p>
            <h4 class="text-lg font-semibold text-white mt-6 mb-2">Simulated Moves Log</h4>
            <div class="overflow-x-auto bg-gray-800 rounded-lg shadow">
                <table class="min-w-full divide-y divide-gray-700">
                    <thead class="bg-gray-700/50">
                        <tr>
                            <th class="px-3 py-2 text-left text-xs font-bold text-gray-300 uppercase tracking-wider">Date of Move</th>
                            <th class="px-3 py-2 text-left text-xs font-bold text-gray-300 uppercase tracking-wider">Player Added</th>
                            <th class="px-3 py-2 text-left text-xs font-bold text-gray-300 uppercase tracking-wider">Player Dropped</th>
                        </tr>
                    </thead>
                    <tbody class="bg-gray-800 divide-y divide-gray-700">
        `;
        sortedMoves.forEach(move => {
            // Handle null players
            const addedName = move.added_player ? move.added_player.player_name : '<span class="text-gray-500 italic">-</span>';
            const droppedName = move.dropped_player ? move.dropped_player.player_name : '<span class="text-gray-500 italic">-</span>';

            logHtml += `
                <tr class="hover:bg-gray-700/50">
                    <td class="px-3 py-2 whitespace-nowrap text-sm text-gray-300">${move.date}</td>
                    <td class="px-3 py-2 whitespace-nowrap text-sm text-green-400">${addedName}</td>
                    <td class="px-3 py-2 whitespace-nowrap text-sm text-red-400">${droppedName}</td>
                </tr>
            `;
        });
        logHtml += `</tbody></table></div>`;
        simLogContainer.innerHTML = logHtml;
    }

    init();
})();
