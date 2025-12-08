(async function() {
    const waiverSkatersContainer = document.getElementById('waiver-skaters-container');
    const waiverGoaliesContainer = document.getElementById('waiver-goalies-container');
    const freeAgentSkatersContainer = document.getElementById('free-agent-skaters-container');
    const freeAgentGoaliesContainer = document.getElementById('free-agent-goalies-container');
    const errorDiv = document.getElementById('db-error-message');
    const playerSearchInput = document.getElementById('player-search');
    const checkboxesContainer = document.getElementById('category-checkboxes-container');
    const positionFiltersContainer = document.getElementById('position-filters-container');
    const dayFiltersContainer = document.getElementById('day-filters-container');
    const injuryFiltersContainer = document.getElementById('injury-filters-container');
    const recalculateButton = document.getElementById('recalculate-button');
    const unusedRosterSpotsContainer = document.getElementById('unused-roster-spots-container');
    const timestampText = document.getElementById('available-players-timestamp-text');
    const playerDropDropdown = document.getElementById('player-drop-dropdown');
    const transactionDatePicker = document.getElementById('transaction-date-picker');
    const simulateButton = document.getElementById('simulate-add-drop-button');
    const resetButton = document.getElementById('reset-add-drops-button');
    const simLogContainer = document.getElementById('simulated-moves-log');

    const CACHE_KEY = 'freeAgentsCache';
    const SIMULATION_KEY = 'simulationCache';
    const CATEGORY_PREF_KEY = 'freeAgentCategoryPreferences';

    // --- Global State ---
    let allWaiverPlayers = [];
    let allFreeAgents = [];
    let allScoringCategories = [];
    let rankedCategories = [];
    let skaterCategories = [];
    let goalieCategories = [];
    let checkedCategories = [];
    let selectedPositions = [];
    let selectedDays = [];
    let injuryFilters = { hideDTD: true, hideIR: true }
    let currentUnusedSpots = null;
    let currentTeamRoster = [];
    let currentWeekDates = [];
    let simulatedMoves = [];
    let sortConfig = {
        waiverSkaters: { key: 'total_cat_rank', direction: 'ascending' },
        waiverGoalies: { key: 'total_cat_rank', direction: 'ascending' },
        freeAgentSkaters: { key: 'total_cat_rank', direction: 'ascending' },
        freeAgentGoalies: { key: 'total_cat_rank', direction: 'ascending' }
    };

    // --- Helper Functions ---
    function formatPercentage(decimal) {
        if (decimal === null || decimal === undefined) return 'N/A';
        try {
            const num = parseFloat(decimal);
            if (isNaN(num)) return 'N/A';
            return (num * 100).toFixed(1) + '%';
        } catch (e) {
            return 'N/A';
        }
    }

    function formatSecondsToMMSS(seconds) {
        if (seconds === null || seconds === undefined) return 'N/A';
        try {
            const s = parseInt(seconds, 10);
            if (isNaN(s)) return 'N/A';
            const minutes = Math.floor(s / 60);
            const remainingSeconds = s % 60;
            return `${minutes}:${remainingSeconds < 10 ? '0' : ''}${remainingSeconds}`;
        } catch (e) {
            return 'N/A';
        }
    }

    // --- [START] NEW: Number formatting helper ---
    function formatNumber(value, decimals, defaultVal = 'N/A') {
        if (value === null || value === undefined) return defaultVal;
        try {
            const num = parseFloat(value);
            if (isNaN(num)) return defaultVal;
            return num.toFixed(decimals);
        } catch (e) {
            return defaultVal;
        }
    }
    // --- [END] NEW ---

    function formatNullable(value) {
        return value ?? 'N/A';
    }

    const irSortFn = (a, b) => {
        const positionOrder = ['C', 'LW', 'RW', 'D', 'G', 'IR', 'IR+'];
        const posStrA = (a.eligible_positions || a.positions || '').toString();
        const posStrB = (b.eligible_positions || b.positions || '').toString();
        const posA = posStrA.split(',').map(p => p.trim());
        const posB = posStrB.split(',').map(p => p.trim());

        const getBestPosIndex = (posArr) => {
            let minIndex = Infinity;
            let isIR = false;
            posArr.forEach(p => {
                if (p.includes('IR')) {
                    isIR = true;
                }
                const idx = positionOrder.indexOf(p);
                if (idx !== -1 && idx < minIndex) {
                    minIndex = idx;
                }
            });
            if (isIR) return 100;
            if (minIndex === Infinity) return 99;
            return minIndex;
        };

        const bestPosA = getBestPosIndex(posA);
        const bestPosB = getBestPosIndex(posB);

        return bestPosA - bestPosB;
    };


    // --- Caching Functions ---
    function saveStateToCache() {
        try {
            const weekSelect = document.getElementById('week-select');
            const selectedWeek = weekSelect ? weekSelect.value : null;
            const selectedSourcing = localStorage.getItem('selectedStatSourcing') || 'projected';
            const state = {
                allWaiverPlayers,
                allFreeAgents,
                allScoringCategories,
                rankedCategories,
                skaterCategories,
                goalieCategories,
                checkedCategories,
                selectedPositions,
                selectedDays,
                injuryFilters: injuryFilters,
                sortConfig,
                unusedRosterSpotsHTML: unusedRosterSpotsContainer.innerHTML,
                unusedRosterSpotsData: currentUnusedSpots,
                currentTeamRoster: currentTeamRoster,
                currentWeekDates: currentWeekDates,
                selectedTeam: document.getElementById('your-team-select')?.value,
                searchTerm: playerSearchInput.value,
                timestamp: Date.now(),
                selectedWeek: selectedWeek,
                sourcing: selectedSourcing
            };
            localStorage.setItem(CACHE_KEY, JSON.stringify(state));
        } catch (error) {
            console.warn("Could not save state to local storage.", error);
        }
    }

    function loadStateFromCache() {
        try {
            const cachedSim = localStorage.getItem(SIMULATION_KEY);
            simulatedMoves = cachedSim ? JSON.parse(cachedSim) : [];

            const cachedJSON = localStorage.getItem(CACHE_KEY);
            if (!cachedJSON) return null;

            const cachedState = JSON.parse(cachedJSON);
            const CACHE_TTL_MS = 2 * 60 * 1000; // 2 minutes
            if (Date.now() - cachedState.timestamp > CACHE_TTL_MS) {
                localStorage.removeItem(CACHE_KEY);
                return null;
            }
            const weekSelect = document.getElementById('week-select');
            const currentSelectedWeek = weekSelect ? weekSelect.value : null;

            if (currentSelectedWeek && cachedState.selectedWeek !== currentSelectedWeek) {
                console.warn("Cache is for a different week. Discarding cache.");
                localStorage.removeItem(CACHE_KEY);
                return null;
            }
            const currentSourcing = localStorage.getItem('selectedStatSourcing') || 'projected';
            const cachedSourcing = cachedState.sourcing || 'projected'; // Default for old caches

            if (cachedSourcing !== currentSourcing) {
                console.log("Cache source mismatch (" + cachedSourcing + " vs " + currentSourcing + "). Discarding cache.");
                localStorage.removeItem(CACHE_KEY);
                return null;
            }
            skaterCategories = cachedState.skaterCategories || [];
            goalieCategories = cachedState.goalieCategories || [];
            const defaultSortConfig = {
                waiverSkaters: { key: 'total_cat_rank', direction: 'ascending' },
                waiverGoalies: { key: 'total_cat_rank', direction: 'ascending' },
                freeAgentSkaters: { key: 'total_cat_rank', direction: 'ascending' },
                freeAgentGoalies: { key: 'total_cat_rank', direction: 'ascending' }
            };
            sortConfig = { ...defaultSortConfig, ...(cachedState.sortConfig || {}) };

            currentUnusedSpots = cachedState.unusedRosterSpotsData;
            currentTeamRoster = cachedState.currentTeamRoster || [];
            currentWeekDates = cachedState.currentWeekDates || [];
            selectedPositions = cachedState.selectedPositions || [];
            selectedDays = cachedState.selectedDays || [];
            injuryFilters = cachedState.injuryFilters !== undefined
                ? cachedState.injuryFilters
                : { hideDTD: true, hideIR: true };
            return cachedState;
        } catch (error) {
            console.warn("Could not load state from local storage.", error);
            return null;
        }
    }

    function getHeatmapColor(rank) {
        if (rank === null || rank === undefined || rank === '-') return '';
        const minRank = 1, maxRank = 20;
        const clampedRank = Math.max(minRank, Math.min(rank, maxRank));
        const percentage = (clampedRank - minRank) / (maxRank - minRank);
        const hue = (1 - percentage) * 120;
        return `hsl(${hue}, 65%, 75%)`;
    }

    async function getTimestamp() {
        if (!timestampText) return;
        try {
            const response = await fetch('/api/available_players_timestamp');
            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Failed to fetch timestamp.');
            }

            if (data.timestamp) {
                timestampText.textContent = `Available player data was last refreshed at: ${data.timestamp}`;
            } else {
                timestampText.textContent = 'Available player data has not been updated. Please run an update from the League Database page.';
            }
        } catch (error) {
            console.error('Error setting timestamp:', error);
            timestampText.textContent = 'Could not load data status.';
        }
    }

    async function fetchData(selectedCategories = null) {
        waiverSkatersContainer.innerHTML = '<p class="text-gray-400">Loading waiver skaters...</p>';
        waiverGoaliesContainer.innerHTML = '<p class="text-gray-400">Loading waiver goalies...</p>';
        freeAgentSkatersContainer.innerHTML = '<p class="text-gray-400">Loading free agent skaters...</p>';
        freeAgentGoaliesContainer.innerHTML = '<p class="text-gray-400">Loading free agent goalies...</p>';
        unusedRosterSpotsContainer.innerHTML = '<p class="text-gray-400">Loading unused spots...</p>';

        const yourTeamSelect = document.getElementById('your-team-select');
        const selectedTeam = yourTeamSelect ? yourTeamSelect.value : null;
        const weekSelect = document.getElementById('week-select');
        const selectedWeek = weekSelect ? weekSelect.value : null;
        const selectedSourcing = localStorage.getItem('selectedStatSourcing') || 'projected';

        try {
            const payload = {
                team_name: selectedTeam,
                simulated_moves: simulatedMoves,
                selected_week: selectedWeek,
                sourcing: selectedSourcing
            };
            if (selectedCategories) payload.categories = selectedCategories;

            const response = await fetch('/api/free_agent_data', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Failed to fetch free agent data.');

            allWaiverPlayers = data.waiver_players;
            allFreeAgents = data.free_agents;
            rankedCategories = data.ranked_categories;
            checkedCategories = data.checked_categories || data.ranked_categories;
            currentUnusedSpots = data.unused_roster_spots;
            currentTeamRoster = data.team_roster;
            currentWeekDates = data.week_dates;
            skaterCategories = data.skater_categories;
            goalieCategories = data.goalie_categories;

            if (allScoringCategories.length === 0 && data.scoring_categories) {
                allScoringCategories = data.scoring_categories;
                renderCategoryCheckboxes();
            } else if (allScoringCategories.length > 0) {
                renderCategoryCheckboxes();
            }

            renderPositionFilters();
            renderDayFilters();
            renderInjuryFilters();
            populateDropPlayerDropdown();
            populateTransactionDatePicker(currentWeekDates);
            renderSimulatedMovesLog();
            renderUnusedRosterSpotsTable(currentUnusedSpots);

            filterAndSortPlayers();
            saveStateToCache();

        } catch (error) {
            console.error('Fetch error:', error);
            errorDiv.textContent = `Error: ${error.message}`;
            errorDiv.classList.remove('hidden');
            waiverSkatersContainer.innerHTML = ''; waiverGoaliesContainer.innerHTML = '';
            freeAgentSkatersContainer.innerHTML = ''; freeAgentGoaliesContainer.innerHTML = '';
            unusedRosterSpotsContainer.innerHTML = '';
        }
    }

    // --- NEW: Global Toggle Listener ---
    window.addEventListener('rawDataToggled', (e) => {
        filterAndSortPlayers(); // Re-render with new view mode
    });

    function renderCategoryCheckboxes() {
        let checkboxHtml = `
            <div class="flex justify-between items-center mb-2">
                <label class="block text-sm font-medium text-gray-300">Recalculate Rank Based On:</label>
                <div>
                    <button id="check-all-btn" class="text-xs bg-gray-600 hover:bg-gray-500 text-white py-1 px-2 rounded mr-2 transition-colors duration-150">Check All</button>
                    <button id="uncheck-all-btn" class="text-xs bg-gray-600 hover:bg-gray-500 text-white py-1 px-2 rounded transition-colors duration-150">Uncheck All</button>
                </div>
            </div>
            <div class="grid grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-2">
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
        checkboxesContainer.innerHTML = checkboxHtml;
    }

    function renderPositionFilters() {
            const POSITIONS = ['C', 'LW', 'RW', 'D', 'G'];
            let filterHtml = '';
            POSITIONS.forEach(pos => {
                const isChecked = selectedPositions.includes(pos);
                filterHtml += `
                    <div class="flex items-center">
                        <input id="pos-${pos}" name="position-filter" type="checkbox" value="${pos}" ${isChecked ? 'checked' : ''} class="h-4 w-4 bg-gray-700 border-gray-600 text-indigo-600 focus:ring-indigo-500 rounded">
                        <label for="pos-${pos}" class="ml-2 text-sm text-gray-300">${pos}</label>
                    </div>
                `;
            });
            positionFiltersContainer.innerHTML = filterHtml;
        }
    function renderInjuryFilters() {
            if (!injuryFiltersContainer) return;
            let filterHtml = `
                <div classflex items-center mr-4">
                    <input id="filter-hide-dtd" name="injury-filter" type="checkbox" value="DTD" ${injuryFilters.hideDTD ? 'checked' : ''} class="h-4 w-4 bg-gray-700 border-gray-600 text-indigo-600 focus:ring-indigo-500 rounded">
                    <label for="filter-hide-dtd" class="ml-2 text-sm text-gray-300">Hide DTD/O</label>
                </div>
                <div class="flex items-center">
                    <input id="filter-hide-ir" name="injury-filter" type="checkbox" value="IR" ${injuryFilters.hideIR ? 'checked' : ''} class="h-4 w-4 bg-gray-700 border-gray-600 text-indigo-600 focus:ring-indigo-500 rounded">
                    <label for="filter-hide-ir" class="ml-2 text-sm text-gray-300">Hide IR</label>
                </div>
            `;
            injuryFiltersContainer.innerHTML = filterHtml;
        }

    function renderDayFilters() {
            const DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
            let filterHtml = '';
            DAYS.forEach(day => {
                const isChecked = selectedDays.includes(day);
                filterHtml += `
                    <div class="flex items-center">
                        <input id="day-${day}" name="day-filter" type="checkbox" value="${day}" ${isChecked ? 'checked' : ''} class="h-4 w-4 bg-gray-700 border-gray-600 text-indigo-500 focus:ring-indigo-500 rounded">
                        <label for="day-${day}" class="ml-2 text-sm text-gray-300">${day}</label>
                    </div>
                `;
            });
            dayFiltersContainer.innerHTML = filterHtml;
        }

        function filterAndSortPlayers() {
            const searchTerm = playerSearchInput.value.toLowerCase();
            // ... (Filters same as before) ...
            const positionFilter = (player) => { if (selectedPositions.length === 0) return true; return selectedPositions.some(pos => (player.positions || '').includes(pos)); };
            const dayFilter = (player) => { if (selectedDays.length === 0) return true; return selectedDays.every(day => (player.games_this_week || []).includes(day)); };
            const searchFilter = (player) => player.player_name.toLowerCase().includes(searchTerm);
            const injuryFilter = (player) => {
                const status = player.status;
                if (injuryFilters.hideDTD && (status === 'DTD' || status === 'O')) return false;
                if (injuryFilters.hideIR && (status === 'IR' || status === 'IR-NR' || status === 'IR-LT')) return false;
                return true;
            };

            const waiverSkaters = allWaiverPlayers.filter(p => !(p.positions || '').includes('G'));
            const waiverGoalies = allWaiverPlayers.filter(p => (p.positions || '').includes('G'));
            const faSkaters = allFreeAgents.filter(p => !(p.positions || '').includes('G'));
            const faGoalies = allFreeAgents.filter(p => (p.positions || '').includes('G'));

            const filteredWaiverSkaters = waiverSkaters.filter(p => searchFilter(p) && positionFilter(p) && dayFilter(p) && injuryFilter(p));
            const filteredWaiverGoalies = waiverGoalies.filter(p => searchFilter(p) && positionFilter(p) && dayFilter(p) && injuryFilter(p));
            const filteredFaSkaters = faSkaters.filter(p => searchFilter(p) && positionFilter(p) && dayFilter(p) && injuryFilter(p));
            const filteredFaGoalies = faGoalies.filter(p => searchFilter(p) && positionFilter(p) && dayFilter(p) && injuryFilter(p));

            // Sort by IR then User Preference
            [filteredWaiverSkaters, filteredWaiverGoalies, filteredFaSkaters, filteredFaGoalies].forEach(list => list.sort(irSortFn));
            sortPlayers(filteredWaiverSkaters, sortConfig.waiverSkaters);
            sortPlayers(filteredWaiverGoalies, sortConfig.waiverGoalies);
            sortPlayers(filteredFaSkaters, sortConfig.freeAgentSkaters);
            sortPlayers(filteredFaGoalies, sortConfig.freeAgentGoalies);

            renderPlayerTable('Skaters', filteredWaiverSkaters, waiverSkatersContainer, 'waiverSkaters', skaterCategories, false);
            renderPlayerTable('Goalies', filteredWaiverGoalies, waiverGoaliesContainer, 'waiverGoalies', goalieCategories, false);
            renderPlayerTable('Skaters', filteredFaSkaters, freeAgentSkatersContainer, 'freeAgentSkaters', skaterCategories, true);
            renderPlayerTable('Goalies', filteredFaGoalies, freeAgentGoaliesContainer, 'freeAgentGoalies', goalieCategories, true);
        }

        function sortPlayers(players, config) {
            const getSortableValue = (value) => (value === null || value === undefined || value === 0 || value === '-') ? Infinity : value;
            players.sort((a, b) => {
                let valA = (config.key === 'player_name') ? String(a.player_name).toLowerCase() : getSortableValue(a[config.key]);
                let valB = (config.key === 'player_name') ? String(b.player_name).toLowerCase() : getSortableValue(b[config.key]);
                if (valA < valB) return config.direction === 'ascending' ? -1 : 1;
                if (valA > valB) return config.direction === 'ascending' ? 1 : -1;
                return 0;
            });
        }

        function renderPlayerTable(title, players, container, tableType, categories, shouldCap = false) {
        const showRaw = localStorage.getItem('showRawData') === 'true';
        if (!players) { container.innerHTML = `<h3 class="text-xl font-bold text-white mb-2">${title}</h3><p class="text-gray-400">No players found.</p>`; return; }
        const playersToDisplay = shouldCap ? players.slice(0, 100) : players;
        const totalColumns = 9 + categories.length;

        let tableHtml = `
            <div class="bg-gray-900 rounded-lg shadow mb-8">
                <h3 class="text-xl font-bold text-white p-4 bg-gray-800 rounded-t-lg flex justify-between items-center">${title} <span class="text-sm font-normal text-gray-400">${players.length} players</span></h3>
                <div class="overflow-x-auto">
                    <table class="min-w-full divide-y divide-gray-700">
                        <thead class="bg-gray-700/50">
                            <tr>
                                <th class="px-2 py-2 text-left text-xs font-bold text-gray-300 uppercase tracking-wider">Add</th>
                                <th class="px-2 py-2 text-left text-xs font-bold text-gray-300 uppercase tracking-wider sortable" data-sort-key="player_name" data-table-type="${tableType}">Player Name</th>
                                <th class="px-2 py-2 text-left text-xs font-bold text-gray-300 uppercase tracking-wider">Team</th>
                                <th class="px-2 py-2 text-left text-xs font-bold text-gray-300 uppercase tracking-wider">Positions</th>
                                <th class="px-2 py-2 text-left text-xs font-bold text-gray-300 uppercase tracking-wider">This Week</th>
                                <th class="px-2 py-2 text-left text-xs font-bold text-gray-300 uppercase tracking-wider">Opponents</th>
                                <th class="px-2 py-2 text-left text-xs font-bold text-gray-300 uppercase tracking-wider">Next Week</th>
                                <th class="px-2 py-2 text-left text-xs font-bold text-gray-300 uppercase tracking-wider">PP Utilization<span class="text-xs text-gray-400 font-light block">(Click)</span></th>
                                <th class="px-2 py-2 text-left text-xs font-bold text-gray-300 uppercase tracking-wider sortable" data-sort-key="total_cat_rank" data-table-type="${tableType}">Total Cat Rank</th>
        `;
        categories.forEach(cat => {
            const isChecked = checkedCategories.includes(cat);
            tableHtml += `<th class="px-2 py-2 text-center text-xs font-bold text-gray-300 uppercase tracking-wider sortable" data-sort-key="${cat}_cat_rank" data-table-type="${tableType}">${isChecked ? cat : cat + '*'}</th>`;
        });
        tableHtml += `</tr><tr><td colspan="${totalColumns}" class="text-center text-xs text-gray-500 py-1">Click headers to sort</td></tr></thead><tbody class="bg-gray-800 divide-y divide-gray-700">`;

        if (playersToDisplay.length === 0) {
            tableHtml += `<tr><td colspan="${totalColumns}" class="text-center py-4 text-gray-400">No players match the current filter.</td></tr>`;
        } else {
            playersToDisplay.forEach(player => {
                const isAlreadyAdded = simulatedMoves.some(m => m.added_player && m.added_player.player_id === player.player_id);
                const checkboxDisabled = isAlreadyAdded ? 'disabled' : '';
                const statusHtml = player.status ? ` <a href="https://sports.yahoo.com/nhl/players/${player.player_id}/news/" target="_blank" rel="noopener noreferrer" class="text-red-400 ml-1 hover:text-red-300 hover:underline">(${player.status})</a>` : '';

                // --- LINE INFO PILL LOGIC ---
                // Shows "L# | PP#" (e.g. "L1 | PP1" or "L2 | N/A")
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
                // ----------------------------

                const playerPositions = player.positions ? player.positions.split(',') : [];
                const gamesThisWeek = player.games_this_week || [];
                let gamesThisWeekHtml = gamesThisWeek.map(day => {
                    if (!currentUnusedSpots || !currentUnusedSpots[day]) return day;
                    for (const pos of playerPositions) {
                        if (String(currentUnusedSpots[day][pos.trim()] || 0) !== '0') return `<strong class="text-yellow-300">${day}</strong>`;
                    }
                    return day;
                }).join(', ');

                const opponentsList = (player.opponents_list || []).join(', ');
                const opponentStatsJson = JSON.stringify(player.opponent_stats_this_week || []);
                const isGoalie = (player.positions || '').includes('G');

                tableHtml += `
                    <tr class="hover:bg-gray-700/50">
                        <td class="px-2 py-2 whitespace-nowrap text-sm text-center"><input type="checkbox" name="player-to-add" class="h-4 w-4 bg-gray-700 border-gray-600 text-indigo-600 focus:ring-indigo-500 rounded" value="${player.player_id}" data-table="${tableType}" ${checkboxDisabled}></td>
                        <td class="px-2 py-2 whitespace-nowrap text-sm font-medium text-gray-300 flex items-center">
                            ${player.player_name}${statusHtml}
                            ${pillHtml}
                        </td>
                        <td class="px-2 py-2 whitespace-nowrap text-sm text-gray-300">${player.player_team}</td>
                        <td class="px-2 py-2 whitespace-nowrap text-sm text-gray-300">${player.positions}</td>
                        <td class="px-2 py-2 whitespace-nowrap text-sm text-gray-300">${gamesThisWeekHtml}</td>
                        <td class="px-2 py-2 whitespace-nowrap text-sm text-gray-300 cursor-pointer hover:bg-gray-700 opponent-stats-cell" data-player-name="${player.player_name}" data-is-goalie="${isGoalie}" data-opponent-stats='${opponentStatsJson}'>${opponentsList}</td>
                        <td class="px-2 py-2 whitespace-nowrap text-sm text-gray-300">${(player.games_next_week || []).join(', ')}</td>
                        <td class="px-2 py-2 whitespace-nowrap text-sm text-gray-300 cursor-pointer hover:bg-gray-700 pp-util-cell" data-player-name="${player.player_name}" data-avg-pp-pct="${player.avg_ppTimeOnIcePctPerGame}" data-lg-pp-toi="${player.lg_ppTimeOnIce}" data-lg-pp-pct="${player.lg_ppTimeOnIcePctPerGame}" data-lg-ppa="${player.lg_ppAssists}" data-lg-ppg="${player.lg_ppGoals}" data-lw-pp-toi="${player.avg_ppTimeOnIce}" data-lw-pp-pct="${player.avg_ppTimeOnIcePctPerGame}" data-lw-ppa="${player.total_ppAssists}" data-lw-ppg="${player.total_ppGoals}" data-lw-gp="${player.team_games_played}">${formatPercentage(player.avg_ppTimeOnIcePctPerGame)}</td>

                        <td class="px-2 py-2 whitespace-nowrap text-sm font-bold text-blue-400 cursor-pointer hover:text-blue-300 cat-rank-cell" data-player-id="${player.player_id}">${player.total_cat_rank}</td>
                `;

                (categories || []).forEach(cat => {
                    const rank = player[`${cat}_cat_rank`];
                    const heatColor = getHeatmapColor(rank);
                    let displayValue = '-';

                    if (showRaw) {
                        const val = player[cat];
                        displayValue = (val != null && !isNaN(val)) ? parseFloat(val).toFixed(2).replace(/[.,]00$/, "") : (val || '-');
                    } else {
                        displayValue = (rank != null) ? rank.toFixed(2) : '-';
                    }

                    let cellStyle = '';
                    let cellClass = 'px-2 py-1 whitespace-nowrap text-sm text-center font-semibold';

                    if (heatColor) {
                        cellStyle = `background-color: ${heatColor}; color: #1f2937; font-weight: 600;`;
                    } else {
                        cellClass += ' text-gray-400';
                    }

                    tableHtml += `<td class="${cellClass}" style="${cellStyle}">${displayValue}</td>`;
                });

                tableHtml += `</tr>`;
            });
        }
        tableHtml += `</tbody></table></div></div>`;
        container.innerHTML = tableHtml;

        // Re-attach Sort Listeners
        document.querySelectorAll(`[data-table-type="${tableType}"].sortable`).forEach(header => {
            header.classList.remove('sort-asc', 'sort-desc');
            if (header.dataset.sortKey === sortConfig[tableType].key) header.classList.add(sortConfig[tableType].direction === 'ascending' ? 'sort-desc' : 'sort-asc');
            header.removeEventListener('click', handleSortClick);
            header.addEventListener('click', handleSortClick);
        });
    }


    function handleSortClick(e) {
        const key = e.target.closest('[data-sort-key]').dataset.sortKey;
        const tableType = e.target.closest('[data-table-type]').dataset.tableType;

        if (!sortConfig[tableType]) {
             console.error(`Invalid tableType for sorting: ${tableType}`);
             return;
        }

        if (sortConfig[tableType].key === key) {
            sortConfig[tableType].direction = sortConfig[tableType].direction === 'ascending' ? 'descending' : 'ascending';
        } else {
            sortConfig[tableType].key = key;
            sortConfig[tableType].direction = 'ascending';
        }

        filterAndSortPlayers(); // This re-sorts and re-renders all tables
        saveStateToCache();
    }

    function handleRecalculateClick() {
        const selectedCategories = Array.from(document.querySelectorAll('#category-checkboxes-container input:checked')).map(cb => cb.value);
        localStorage.removeItem(CACHE_KEY);
        fetchData(selectedCategories);
    }

    // --- SIMULATION FUNCTIONS (Unchanged) ---
    function populateDropPlayerDropdown() {
        // Filter out moves that don't have a dropped player
        const droppedPlayerIds = new Set(
            simulatedMoves
                .filter(m => m.dropped_player)
                .map(m => m.dropped_player.player_id)
        );

        let optionsHtml = '<option selected value="">Select player to drop...</option>';

        currentTeamRoster.forEach(player => {
            if (!droppedPlayerIds.has(player.player_id)) {
                optionsHtml += `<option value="${player.player_id}" data-type="roster">${player.player_name} - ${player.eligible_positions}</option>`;
            }
        });

        simulatedMoves.forEach(move => {
            // Only allow dropping players who were actually added (not null)
            const player = move.added_player;
            if (player && !droppedPlayerIds.has(player.player_id)) {
                optionsHtml += `<option value="${player.player_id}" data-type="simulated" data-add-date="${move.date}">
                    ${player.player_name} - ${player.positions} (Added ${move.date})
                </option>`;
            }
        });
        playerDropDropdown.innerHTML = optionsHtml;
    }

    function populateTransactionDatePicker(weekDates) {
            if (!transactionDatePicker) return;

            const previouslySelected = transactionDatePicker.value;
            let html = '<option value="" disabled selected>Select Date</option>';

            if (weekDates && weekDates.length > 0) {
                weekDates.forEach(dateStr => {
                    // dateStr is "YYYY-MM-DD"
                    html += `<option value="${dateStr}">${dateStr}</option>`;
                });
            } else {
                html = '<option value="" disabled selected>No dates available</option>';
            }

            transactionDatePicker.innerHTML = html;

            // Attempt to restore previous selection or default to today if it's in the list
            if (previouslySelected && weekDates.includes(previouslySelected)) {
                transactionDatePicker.value = previouslySelected;
            } else {
                const today = new Date().toISOString().split('T')[0];
                if (weekDates.includes(today)) {
                    transactionDatePicker.value = today;
                }
            }
        }

    function renderSimulatedMovesLog() {
        if (simulatedMoves.length === 0) {
            simLogContainer.innerHTML = '';
            return;
        }
        const sortedMoves = [...simulatedMoves].sort((a, b) => (a.date < b.date) ? -1 : 1);
        let logHtml = `
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
        logHtml += `</tbody></table></div>`;
        simLogContainer.innerHTML = logHtml;
    }

    function handleSimulateClick() {
        const checkedBox = document.querySelector('input[name="player-to-add"]:checked');
        const droppedPlayerOption = playerDropDropdown.options[playerDropDropdown.selectedIndex];
        const transactionDate = transactionDatePicker.value;

        // 1. Require Date
        if (!transactionDate) return alert("Please select a transaction date.");

        // 2. Require EITHER an Add OR a Drop
        if (!checkedBox && !droppedPlayerOption.value) {
            return alert("Please select a player to ADD or a player to DROP (or both).");
        }

        // Validate Drop Date Logic (only if dropping a simulated player)
        if (droppedPlayerOption.value && droppedPlayerOption.dataset.type === 'simulated') {
            const addDate = droppedPlayerOption.dataset.addDate;
            if (transactionDate < addDate) {
                return alert(`Error: Cannot drop ${droppedPlayerOption.text.split('(')[0].trim()} on ${transactionDate} because they are not scheduled to be added until ${addDate}.`);
            }
        }

        let addedPlayer = null;
        if (checkedBox) {
            const addedPlayerId = checkedBox.value;
            const tableType = checkedBox.dataset.table;
            let sourceList = tableType.startsWith('waiver') ? allWaiverPlayers : allFreeAgents;
            addedPlayer = sourceList.find(p => p.player_id == addedPlayerId);
            if (!addedPlayer) return alert("An error occurred trying to find the player to add.");
        }

        let droppedPlayer = null;
        if (droppedPlayerOption.value) {
            const droppedPlayerId = droppedPlayerOption.value;
            if (droppedPlayerOption.dataset.type === 'roster') {
                droppedPlayer = currentTeamRoster.find(p => p.player_id == droppedPlayerId);
            } else {
                const sourceMove = simulatedMoves.find(m => m.added_player && m.added_player.player_id == droppedPlayerId);
                if (sourceMove) droppedPlayer = sourceMove.added_player;
            }
            if (!droppedPlayer) return alert("An error occurred trying to find the player to drop.");
        }

        simulatedMoves.push({
            date: transactionDate,
            added_player: addedPlayer,     // Can be null
            dropped_player: droppedPlayer  // Can be null
        });

        localStorage.setItem(SIMULATION_KEY, JSON.stringify(simulatedMoves));
        localStorage.removeItem(CACHE_KEY);

        const selectedCategories = Array.from(document.querySelectorAll('#category-checkboxes-container input:checked')).map(cb => cb.value);
        fetchData(selectedCategories);

        if (checkedBox) checkedBox.checked = false;
        // Reset dropdown logic happens in popuplateDropPlayerDropdown during fetchData
    }

    function handleResetClick() {
        if (confirm("Are you sure you want to reset all simulated moves?")) {
            simulatedMoves = [];
            localStorage.removeItem(SIMULATION_KEY);
            localStorage.removeItem(CACHE_KEY); // Invalidate cache
            fetchData();
        }
    }

    function renderUnusedRosterSpotsTable(unusedSpotsData) {
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
            <div class="bg-gray-900 rounded-lg shadow">
                <h2 class="text-xl font-bold text-white p-3 bg-gray-800 rounded-t-lg">Unused Roster Spots</h2>
                <div class="overflow-x-auto">
                    <table class="min-w-full divide-y divide-gray-700">
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

    function setupEventListeners() {
        playerSearchInput.addEventListener('input', () => { filterAndSortPlayers(); });
        recalculateButton.addEventListener('click', handleRecalculateClick);
        positionFiltersContainer.addEventListener('change', (e) => { if (e.target.name === 'position-filter') { selectedPositions = Array.from(document.querySelectorAll('#position-filters-container input:checked')).map(cb => cb.value); filterAndSortPlayers(); saveStateToCache(); } });
        dayFiltersContainer.addEventListener('change', (e) => { if (e.target.name === 'day-filter') { selectedDays = Array.from(document.querySelectorAll('#day-filters-container input:checked')).map(cb => cb.value); filterAndSortPlayers(); saveStateToCache(); } });
        if (injuryFiltersContainer) { injuryFiltersContainer.addEventListener('change', (e) => { if (e.target.name === 'injury-filter') { injuryFilters.hideDTD = document.getElementById('filter-hide-dtd')?.checked || false; injuryFilters.hideIR = document.getElementById('filter-hide-ir')?.checked || false; filterAndSortPlayers(); saveStateToCache(); } }); }
        checkboxesContainer.addEventListener('click', (e) => { const setAll = (val) => checkboxesContainer.querySelectorAll('input[name="category"]').forEach(cb => cb.checked = val); if (e.target.id === 'check-all-btn') setAll(true); if (e.target.id === 'uncheck-all-btn') setAll(false); });
        const weekSelect = document.getElementById('week-select');
        if (weekSelect) { weekSelect.addEventListener('change', () => { localStorage.removeItem(CACHE_KEY); localStorage.removeItem(SIMULATION_KEY); simulatedMoves = []; const selectedCategories = Array.from(document.querySelectorAll('#category-checkboxes-container input:checked')).map(cb => cb.value); fetchData(selectedCategories); }); }
        const yourTeamSelect = document.getElementById('your-team-select');
        if (yourTeamSelect) { yourTeamSelect.addEventListener('change', () => { localStorage.removeItem(CACHE_KEY); const selectedCategories = Array.from(document.querySelectorAll('#category-checkboxes-container input:checked')).map(cb => cb.value); fetchData(selectedCategories); }); }

        // --- Event Delegation ---
        document.body.addEventListener('click', (e) => {
            if (e.target.closest('#pp-modal-close') || e.target.id === 'pp-stats-modal') document.getElementById('pp-stats-modal').classList.add('hidden');
            if (e.target.closest('#opponent-modal-close') || e.target.id === 'opponent-stats-modal') document.getElementById('opponent-stats-modal').classList.add('hidden');

            const ppCell = e.target.closest('.pp-util-cell');
            if (ppCell) { /* ... (PP logic handled above) ... */ }
            const oppCell = e.target.closest('.opponent-stats-cell');
            if (oppCell) { /* ... (Opp logic handled above) ... */ }
            const pill = e.target.closest('.line-info-pill');
            if (pill && window.openLineInfoModal) {
                const pid = String(pill.dataset.playerId);
                // Search both arrays since we don't know which table triggered it
                const player = [...allWaiverPlayers, ...allFreeAgents].find(p => String(p.player_id) === pid);
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
            // --- [NEW] Cat Rank Logic ---
            const rankCell = e.target.closest('.cat-rank-cell');
            if (rankCell && window.openCatRankModal) {
                const pid = String(rankCell.dataset.playerId);
                // Search all lists
                const player = [...allWaiverPlayers, ...allFreeAgents].find(p => String(p.player_id) === pid);
                if (player) {
                    const isGoalie = (player.positions || '').includes('G');
                    const cats = isGoalie ? goalieCategories : skaterCategories;
                    window.openCatRankModal(player, cats);
                }
            }
        });
    }

    // --- Initial Load ---
    async function init() {
        // --- 1. Inject Local Modals (PP & Opponent) ---
        // Only inject if they don't exist to prevent duplicates
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

        // --- 2. Global Click Handler (Delegation for all tables) ---
        document.body.addEventListener('click', (e) => {
            // Close Buttons
            if (e.target.closest('#pp-modal-close') || e.target.id === 'pp-stats-modal') document.getElementById('pp-stats-modal').classList.add('hidden');
            if (e.target.closest('#opponent-modal-close') || e.target.id === 'opponent-stats-modal') document.getElementById('opponent-stats-modal').classList.add('hidden');

            // A. PP Util Cell
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

            // B. Opponent Stats Cell
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

            // C. Cat Rank Cell (Global Modal)
            const rankCell = e.target.closest('.cat-rank-cell');
            if (rankCell && window.openCatRankModal) {
                const pid = String(rankCell.dataset.playerId);
                // We search both pools since we don't know which table clicked
                let player = [...allWaiverPlayers, ...allFreeAgents].find(p => String(p.player_id) === pid);

                if (player) {
                    const isGoalie = (player.positions || '').includes('G');
                    const cats = isGoalie ? goalieCategories : skaterCategories;
                    window.openCatRankModal(player, cats);
                }
            }
        });

        // --- 3. Listen for Raw Data Toggle ---
        window.addEventListener('rawDataToggled', (e) => {
            // Re-render active tables
            filterAndSortPlayers();
        });

        // --- 4. Caching Logic ---
        // Always attempt to load simulation state first
        const cachedSim = localStorage.getItem(SIMULATION_KEY);
        if (cachedSim) {
            simulatedMoves = JSON.parse(cachedSim);
        }

        // Try to load full page state
        const cachedState = loadStateFromCache();

        if (cachedState) {
            // Restore state variables
            allWaiverPlayers = cachedState.allWaiverPlayers;
            allFreeAgents = cachedState.allFreeAgents;
            allScoringCategories = cachedState.allScoringCategories;
            skaterCategories = cachedState.skaterCategories;
            goalieCategories = cachedState.goalieCategories;
            checkedCategories = cachedState.checkedCategories;
            currentUnusedSpots = cachedState.unusedRosterSpotsData;
            currentTeamRoster = cachedState.currentTeamRoster; // Restore roster
            currentWeekDates = cachedState.currentWeekDates;

            // Restore UI elements
            const teamSelect = document.getElementById('your-team-select');
            if (teamSelect) teamSelect.value = cachedState.selectedTeam;
            playerSearchInput.value = cachedState.searchTerm;

            // Render everything from cache
            renderCategoryCheckboxes();
            renderPositionFilters();
            renderDayFilters();
            renderInjuryFilters();
            filterAndSortPlayers(); // This triggers the new renderTable
            populateDropPlayerDropdown();
            renderSimulatedMovesLog();
            populateTransactionDatePicker(currentWeekDates);
            renderUnusedRosterSpotsTable(currentUnusedSpots);
            setupEventListeners();
        } else {
            // No valid cache, fetch fresh data
            setupEventListeners();
            fetchData();
        }

        simulateButton.addEventListener('click', handleSimulateClick);
        resetButton.addEventListener('click', handleResetClick);
    }

    init();
})();
