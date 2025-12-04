(async function() {
    // A short delay to ensure the page elements are in the DOM
    await new Promise(resolve => setTimeout(resolve, 0));

    const errorDiv = document.getElementById('db-error-message');
    const controlsDiv = document.getElementById('matchup-controls');
    const tableContainer = document.getElementById('table-container');
    const unusedRosterSpotsContainer = document.getElementById('unused-roster-spots-container');
    const gameCountsContainer = document.getElementById('game-counts-container');
    const weekSelect = document.getElementById('week-select');
    const yourTeamSelect = document.getElementById('your-team-select');
    const opponentSelect = document.getElementById('opponent-select');
    const simLogContainer = document.getElementById('simulated-moves-log');

    let pageData = null; // To store weeks, teams, and matchups
    const CATEGORY_PREF_KEY = 'lineupCategoryPreferences';
    const SIMULATION_KEY = 'simulationCache';
    let simulatedMoves = [];

    async function init() {
        try {
            const response = await fetch('/api/matchup_page_data');
            const data = await response.json();

            if (!response.ok || !data.db_exists) {
                throw new Error(data.error || 'Database has not been initialized.');
            }

            pageData = data;
            populateDropdowns();
            updateOpponentDropdown();
            setupEventListeners();

            // Initial data load
            await fetchAndRenderTable();

            controlsDiv.classList.remove('hidden');

        } catch (error) {
            console.error('Initialization error:', error);
            errorDiv.classList.remove('hidden');
            controlsDiv.classList.add('hidden');
            tableContainer.classList.add('hidden');
        }
    }

    function populateDropdowns() {
        // Populate Weeks
        weekSelect.innerHTML = pageData.weeks.map(week =>
            `<option value="${week.week_num}">
                Week ${week.week_num} (${week.start_date} to ${week.end_date})
            </option>`
        ).join('');

        // Populate Teams
        const teamOptions = pageData.teams.map(team =>
            `<option value="${team.name}">${team.name}</option>`
        ).join('');
        yourTeamSelect.innerHTML = teamOptions;
        opponentSelect.innerHTML = teamOptions;

        // Restore team selection from localStorage
        const savedTeam = localStorage.getItem('selectedTeam');
        if (savedTeam) {
            yourTeamSelect.value = savedTeam;
        }

        // Check if a session has started to handle the week selection
        if (!sessionStorage.getItem('fantasySessionStarted')) {
            // This is a new session. Default to the current week.
            const currentWeek = pageData.current_week;
            weekSelect.value = currentWeek;
            localStorage.setItem('selectedWeek', currentWeek);
            sessionStorage.setItem('fantasySessionStarted', 'true');
        } else {
            // A session is active. Restore from localStorage.
            const savedWeek = localStorage.getItem('selectedWeek');
            if (savedWeek) {
                weekSelect.value = savedWeek;
            } else {
                weekSelect.value = pageData.current_week;
            }
        }
    }

    function updateOpponentDropdown() {
        const selectedWeek = weekSelect.value;
        const yourTeamName = yourTeamSelect.value;

        // Find the matchup for the current week and your team
        const matchup = pageData.matchups.find(m =>
            m.week == selectedWeek && (m.team1 === yourTeamName || m.team2 === yourTeamName)
        );

        if (matchup) {
            const opponentName = matchup.team1 === yourTeamName ? matchup.team2 : matchup.team1;
            opponentSelect.value = opponentName;
        } else {
             // If no specific matchup, just pick the first team that isn't your team
            const firstOtherTeam = pageData.teams.find(t => t.name !== yourTeamName);
            if (firstOtherTeam) {
                opponentSelect.value = firstOtherTeam.name;
            }
        }
    }

    async function fetchAndRenderTable() {
        const selectedWeek = weekSelect.value;
        const yourTeamName = yourTeamSelect.value;
        const opponentName = opponentSelect.value;

        if (!selectedWeek || !yourTeamName || !opponentName) {
            tableContainer.innerHTML = '<p class="text-gray-400">Please make all selections.</p>';
            return;
        }

        tableContainer.innerHTML = '<p class="text-gray-400">Loading matchup stats...</p>';
        unusedRosterSpotsContainer.innerHTML = '';
        gameCountsContainer.innerHTML = '';

        const savedCategories = localStorage.getItem(CATEGORY_PREF_KEY);
        const categoriesToSend = savedCategories ? JSON.parse(savedCategories) : null;

        const cachedSim = localStorage.getItem(SIMULATION_KEY);
        simulatedMoves = cachedSim ? JSON.parse(cachedSim) : [];
        const selectedSourcing = localStorage.getItem('selectedStatSourcing') || 'projected';
        try {
          const response = await fetch('/api/matchup_team_stats', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    week: selectedWeek,
                    team1_name: yourTeamName,
                    team2_name: opponentName,
                    categories: categoriesToSend,
                    simulated_moves: simulatedMoves,
                    sourcing: selectedSourcing
                })
            });

            const stats = await response.json();
            if (!response.ok) throw new Error(stats.error || 'Failed to fetch stats.');

            renderTable(stats, yourTeamName, opponentName);
            renderUnusedRosterSpotsTable(stats.team1_unused_spots);
            renderGameCounts(stats.game_counts, yourTeamName, opponentName);
            renderSimulatedMovesLog();

        } catch(error) {
            console.error('Error fetching stats:', error);
            tableContainer.innerHTML = `<p class="text-red-400">Error: ${error.message}</p>`;
        }
    }

    function renderTable(stats, yourTeamName, opponentName) {
        // Define pastel colors
        const colors = {
            win: 'bg-green-800/50',
            loss: 'bg-red-800/50',
            tie: 'bg-yellow-800/50',
            losingCategory: 'bg-yellow-800/50'
        };

        let tableHtml = `
            <div class="bg-gray-900 rounded-lg shadow">
                <table class="divide-y divide-gray-700">
                    <thead class="bg-gray-700/50">
                        <tr>
                            <th scope="col" class="px-4 py-2 text-left text-xs font-bold text-gray-300 uppercase tracking-wider">Category</th>
                            <th scope="col" class="px-4 py-2 text-center text-xs font-bold text-gray-300 uppercase tracking-wider">${yourTeamName} (Live)</th>
                            <th scope="col" class="px-4 py-2 text-center text-xs font-bold text-gray-300 uppercase tracking-wider">${yourTeamName} (ROW)</th>
                            <th scope="col" class="px-4 py-2 text-center text-xs font-bold text-gray-300 uppercase tracking-wider">${opponentName} (Live)</th>
                            <th scope="col" class="px-4 py-2 text-center text-xs font-bold text-gray-300 uppercase tracking-wider">${opponentName} (ROW)</th>
                        </tr>
                    </thead>
                    <tbody class="bg-gray-800 divide-y divide-gray-700">
        `;

        // Define relationship between parent stats and sub-stats
        const goalieCats = { 'SVpct': ['SV', 'SA'], 'GAA': ['GA', 'TOI/G'] };

        // --- FIX: Dynamic Skip List ---
        // Only skip sub-stats (GA, SV) if their PARENT (GAA, SVpct) is actually active.
        const activeCategories = pageData.scoring_categories.map(c => c.category);
        const subCatsToSkip = [];
        if (activeCategories.includes('SVpct')) subCatsToSkip.push('SV', 'SA');
        if (activeCategories.includes('GAA')) subCatsToSkip.push('GA', 'TOI/G');

        pageData.scoring_categories.forEach(cat => {
            const category = cat.category;

            // Skip this category ONLY if it is being displayed as a sub-stat of an active parent
            if (subCatsToSkip.includes(category)) return;

            let t1_live_val, t2_live_val;

            if (category === 'SVpct') {
                const t1_sv = stats.team1.live['SV'] || 0;
                const t1_sa = stats.team1.live['SA'] || 0;
                t1_live_val = t1_sa > 0 ? (t1_sv / t1_sa) : 0;

                const t2_sv = stats.team2.live['SV'] || 0;
                const t2_sa = stats.team2.live['SA'] || 0;
                t2_live_val = t2_sa > 0 ? (t2_sv / t2_sa) : 0;
            } else if (category === 'GAA') {
                const t1_ga = stats.team1.live['GA'] || 0;
                const t1_toi = stats.team1.live['TOI/G'] || 0;
                t1_live_val = t1_toi > 0 ? ((t1_ga * 60) / t1_toi) : Infinity;

                const t2_ga = stats.team2.live['GA'] || 0;
                const t2_toi = stats.team2.live['TOI/G'] || 0;
                t2_live_val = t2_toi > 0 ? ((t2_ga * 60) / t2_toi) : Infinity;
            } else {
                t1_live_val = stats.team1.live[category] || 0;
                t2_live_val = stats.team2.live[category] || 0;
            }

            let t1_row_val = stats.team1.row[category] || 0;
            let t2_row_val = stats.team2.row[category] || 0;

            // Determine colors
            const isGaa = category === 'GAA';
            let live_t1_class = colors.tie, live_t2_class = colors.tie;
            let row_t1_class = colors.tie, row_t2_class = colors.tie;

            // Live comparison
            if (t1_live_val > t2_live_val) {
                live_t1_class = isGaa ? colors.loss : colors.win;
                live_t2_class = isGaa ? colors.win : colors.loss;
            } else if (t1_live_val < t2_live_val) {
                live_t1_class = isGaa ? colors.win : colors.loss;
                live_t2_class = isGaa ? colors.loss : colors.win;
            }

            // ROW comparison
            if (t1_row_val > t2_row_val) {
                row_t1_class = isGaa ? colors.loss : colors.win;
                row_t2_class = isGaa ? colors.win : colors.loss;
            } else if (t1_row_val < t2_row_val) {
                row_t1_class = isGaa ? colors.win : colors.loss;
                row_t2_class = isGaa ? colors.loss : colors.win;
            }

            const categoryClass = (row_t1_class === colors.loss) ? colors.losingCategory : '';

            // Format for display
            const display_t1_live = (category === 'SVpct') ? t1_live_val.toFixed(3) : (isGaa && t1_live_val === Infinity) ? '0.00' : (typeof t1_live_val === 'number' ? t1_live_val.toFixed(2) : t1_live_val);
            const display_t2_live = (category === 'SVpct') ? t2_live_val.toFixed(3) : (isGaa && t2_live_val === Infinity) ? '0.00' : (typeof t2_live_val === 'number' ? t2_live_val.toFixed(2) : t2_live_val);

            tableHtml += `
                <tr class="hover:bg-gray-700/50">
                    <td class="px-3 py-1 whitespace-nowrap text-sm font-bold text-gray-300 ${categoryClass}">${category}</td>
                    <td class="px-3 py-1 whitespace-nowrap text-sm text-center ${live_t1_class}">${display_t1_live}</td>
                    <td class="px-3 py-1 whitespace-nowrap text-sm text-center ${row_t1_class}">${t1_row_val}</td>
                    <td class="px-3 py-1 whitespace-nowrap text-sm text-center ${live_t2_class}">${display_t2_live}</td>
                    <td class="px-3 py-1 whitespace-nowrap text-sm text-center ${row_t2_class}">${t2_row_val}</td>
                </tr>
            `;

            // Render sub-stats (if this is a parent category)
            if (goalieCats[category]) {
                goalieCats[category].forEach(subCat => {
                    // Always render these rows if we are currently inside the Parent Category loop
                    tableHtml += `
                        <tr class="hover:bg-gray-700/50">
                            <td class="px-3 py-1 whitespace-nowrap text-sm font-normal text-gray-400 pl-8">${subCat}</td>
                            <td class="px-3 py-1 whitespace-nowrap text-sm text-center text-gray-300">${stats.team1.live[subCat] || 0}</td>
                            <td class="px-3 py-1 whitespace-nowrap text-sm text-center text-gray-300">${stats.team1.row[subCat] || 0}</td>
                            <td class="px-3 py-1 whitespace-nowrap text-sm text-center text-gray-300">${stats.team2.live[subCat] || 0}</td>
                            <td class="px-3 py-1 whitespace-nowrap text-sm text-center text-gray-300">${stats.team2.row[subCat] || 0}</td>
                        </tr>
                    `;
                });
            }
        });

        tableHtml += `
                    </tbody>
                </table>
            </div>
        `;
        tableContainer.innerHTML = tableHtml;
    }

    function renderUnusedRosterSpotsTable(unusedSpotsData) {
            if (!unusedSpotsData) {
                unusedRosterSpotsContainer.innerHTML = '';
                return;
            }

            const positionOrder = ['C', 'LW', 'RW', 'F', 'W', 'D', 'Util', 'G'];
            const dayOrder = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];

            const sortedDays = Object.keys(unusedSpotsData).sort((a, b) => {
                return dayOrder.indexOf(a) - dayOrder.indexOf(b);
            });

            let tableHtml = `
                <div class="bg-gray-900 rounded-lg shadow">
                    <h2 class="text-xl font-bold text-white p-3 bg-gray-800 rounded-t-lg">Unused Roster Spots</h2>
                    <table class="w-full divide-y divide-gray-700">
                        <thead class="bg-gray-700/50">
                            <tr>
                                <th class="px-2 py-1 text-left text-xs font-bold text-gray-300 uppercase tracking-wider">Day</th>
                                ${positionOrder.map(pos => `<th class="px-2 py-1 text-center text-xs font-bold text-gray-300 uppercase tracking-wider">${pos}</th>`).join('')}
                            </tr>
                        </thead>
                        <tbody class="bg-gray-800"> `;

            sortedDays.forEach(day => {
                tableHtml += `<tr class="hover:bg-gray-700/50">
                    <td class="px-2 py-1 whitespace-nowrap text-sm font-medium text-gray-300">${day}</td>`;
                positionOrder.forEach(pos => {
                    const value = unusedSpotsData[day][pos];
                    const stringValue = String(value);

                    const highlightClass = (stringValue !== '0')
                        ? 'bg-green-800/50 text-white font-bold'
                        : 'text-gray-300';

                    tableHtml += `<td class="px-2 py-1 whitespace-nowrap text-sm text-center ${highlightClass}">${value}</td>`;
                });
                tableHtml += `</tr>`;
            });

            tableHtml += `
                        </tbody>
                    </table>
                </div>
            `;

            unusedRosterSpotsContainer.innerHTML = tableHtml;
        }


        function renderGameCounts(gameCounts, yourTeamName, opponentName) {
                if (!gameCounts) {
                    gameCountsContainer.innerHTML = '';
                    return;
                }

                let tableHtml = `
                    <div class="bg-gray-900 rounded-lg shadow">
                        <h2 class="text-xl font-bold text-white p-3 bg-gray-800 rounded-t-lg">Total Player Starts</h2>
                        <table class="w-full divide-y divide-gray-700">
                            <thead class="bg-gray-700/50">
                                <tr>
                                    <th class="px-2 py-1 text-left text-xs font-bold text-gray-300 uppercase tracking-wider">Team</th>
                                    <th class="px-2 py-1 text-center text-xs font-bold text-gray-300 uppercase tracking-wider">Total</th>
                                    <th class="px-2 py-1 text-center text-xs font-bold text-gray-300 uppercase tracking-wider">Remaining</th>
                                </tr>
                            </thead>
                            <tbody class="bg-gray-800"> <tr class="hover:bg-gray-700/50">
                                    <td class="px-2 py-1 whitespace-nowrap text-sm font-medium text-gray-300">${yourTeamName}</td>
                                    <td class="px-2 py-1 whitespace-nowrap text-sm text-center text-gray-300">${gameCounts.team1_total}</td>
                                    <td class="px-2 py-1 whitespace-nowrap text-sm text-center text-gray-300">${gameCounts.team1_remaining}</td>
                                </tr>
                               <tr class="hover:bg-gray-700/50">
                                    <td class="px-2 py-1 whitespace-nowrap text-sm font-medium text-gray-300">${opponentName}</td>
                                    <td class="px-2 py-1 whitespace-nowrap text-sm text-center text-gray-300">${gameCounts.team2_total}</td>
                                    <td class="px-2 py-1 whitespace-nowrap text-sm text-center text-gray-300">${gameCounts.team2_remaining}</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                `;

                gameCountsContainer.innerHTML = tableHtml;
            }

    function renderSimulatedMovesLog() {
        if (!simLogContainer) return;

        if (simulatedMoves.length === 0) {
            simLogContainer.innerHTML = '';
            return;
        }

        const sortedMoves = [...simulatedMoves].sort((a, b) => {
            if (a.date < b.date) return -1;
            if (a.date > b.date) return 1;
            return 0;
        });

        let logHtml = `
            <p class="text-sm text-gray-400 italic mb-2">Projections assume the below planned transactions are made.</p>
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

        logHtml += `
                    </tbody>
                </table>
            </div>
        `;
        simLogContainer.innerHTML = logHtml;
    }


    function setupEventListeners() {
        weekSelect.addEventListener('change', async () => {
            updateOpponentDropdown();
            await fetchAndRenderTable();
        });
        yourTeamSelect.addEventListener('change', async () => {
            updateOpponentDropdown();
            await fetchAndRenderTable();
        });
        opponentSelect.addEventListener('change', fetchAndRenderTable);
    }

    init();
})();
