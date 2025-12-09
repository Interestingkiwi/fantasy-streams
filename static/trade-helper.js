(async function() {
    await new Promise(resolve => setTimeout(resolve, 0));

    // --- Global Elements ---
    const loadingText = document.getElementById('trade-helper-loading');
    const yourTeamSelect = document.getElementById('your-team-select');
    const navPartners = document.getElementById('nav-partners');
    const navCompare = document.getElementById('nav-compare');
    const subpageContent = document.getElementById('subpage-content');

    // --- State Variables ---
    let currentSubPage = 'trade-partners';
    let userTeamName = localStorage.getItem('selectedTeam') || '';

    // Sorting/Filter State
    let activeFromCats = [];
    let activeToCats = [];
    let filterPartner = "";
    let filterNHL = [];
    let filterPos = [];
    let filterSearch = "";

    // --- Simulation State ---
    let selectedPlayerIds = new Set();
    let rosterData = { players: [], skaterCategories: [], goalieCategories: [], currentWeek: 1 };
    let categoryData = { skater_stats: [], goalie_stats: [], league_rank_matrix: {}, total_teams: 12 };

    const NHL_TEAMS = [
        "ANA", "BOS", "BUF", "CGY", "CAR", "CHI", "COL", "CBJ", "DAL", "DET",
        "EDM", "FLA", "LAK", "MIN", "MTL", "NSH", "NJD", "NYI", "NYR", "OTT",
        "PHI", "PIT", "SJS", "SEA", "STL", "TBL", "TOR", "UTA", "VAN", "VGK", "WSH", "WPG"
    ];

    function getHeatmapColor(rank) {
        if (rank === null || rank === undefined || rank === '-' || isNaN(rank)) return '';
        const minRank = 1; const maxRank = 20;
        const clamped = Math.max(minRank, Math.min(rank, maxRank));
        const percentage = (clamped - minRank) / (maxRank - minRank);
        const hue = (1 - percentage) * 120;
        return `hsl(${hue}, 65%, 75%)`;
    }
    function formatPercentage(decimal) { if (decimal == null) return 'N/A'; try { const num = parseFloat(decimal); if (isNaN(num)) return 'N/A'; return (num * 100).toFixed(1) + '%'; } catch (e) { return 'N/A'; } }
    function formatSecondsToMMSS(seconds) { if (seconds == null) return 'N/A'; try { const s = parseInt(seconds, 10); if (isNaN(s)) return 'N/A'; const m = Math.floor(s / 60); const rs = s % 60; return `${m}:${rs < 10 ? '0' : ''}${rs}`; } catch (e) { return 'N/A'; } }
    function formatNullable(value) { return value ?? 'N/A'; }


    async function init() {
        if (!yourTeamSelect || !subpageContent) {
            console.error('Trade Helper: Critical elements missing.');
            return;
        }

        // --- 1. Inject Local Modal ---
        if (!document.getElementById('pp-stats-modal')) {
            const ppModalHTML = `
            <div id="pp-stats-modal" class="fixed inset-0 bg-gray-900 bg-opacity-75 flex items-center justify-center z-50 hidden" style="backdrop-filter: blur(2px);">
                <div class="bg-gray-800 rounded-lg shadow-xl p-6 w-full max-w-lg relative border border-gray-700">
                    <button id="pp-modal-close" class="absolute top-3 right-3 text-gray-400 hover:text-white text-2xl leading-none">&times;</button>
                    <h3 id="pp-modal-title" class="text-xl font-bold text-white mb-4">Player PP Stats</h3>
                    <div id="pp-modal-content" class="text-gray-300"></div>
                </div>
            </div>`;
            document.body.insertAdjacentHTML('beforeend', ppModalHTML);
        }
        if (!document.getElementById('trending-stats-modal')) {
             const trendingModalHTML = `
            <div id="trending-stats-modal" class="fixed inset-0 bg-gray-900 bg-opacity-75 flex items-center justify-center z-50 hidden" style="backdrop-filter: blur(2px);">
                <div class="bg-gray-800 rounded-lg shadow-xl p-6 w-full max-w-sm relative border border-gray-700">
                    <button id="trending-modal-close" class="absolute top-3 right-3 text-gray-400 hover:text-white text-2xl leading-none">&times;</button>
                    <h3 id="trending-modal-title" class="text-xl font-bold text-white mb-4">Player Trending</h3>
                    <div id="trending-modal-content" class="text-gray-300"></div>
                </div>
            </div>`;
            document.body.insertAdjacentHTML('beforeend', trendingModalHTML);
        }
        // --- 2. Global Event Listeners ---
        document.body.addEventListener('click', (e) => {
            if (e.target.closest('#pp-modal-close') || e.target.id === 'pp-stats-modal') {
                document.getElementById('pp-stats-modal').classList.add('hidden');
            }
            if (e.target.closest('#trending-modal-close') || e.target.id === 'trending-stats-modal') document.getElementById('trending-stats-modal').classList.add('hidden');
            // A. PP Util Cell
            const ppCell = e.target.closest('.pp-util-cell');
            if (ppCell) {
                const data = ppCell.dataset;
                document.getElementById('pp-modal-title').textContent = `${data.playerName} - PP Stats`;
                document.getElementById('pp-modal-content').innerHTML = `
                <div class="space-y-4">
                    <div>
                        <h4 class="text-md font-semibold text-white mb-2">Last Game</h4>
                        <dl class="grid grid-cols-2 sm:grid-cols-4 gap-x-4 gap-y-2">
                            <div class="bg-gray-700 p-2 rounded"><dt class="text-xs text-gray-400">PP TOI</dt><dd class="text-sm font-medium">${formatSecondsToMMSS(data.lgPpToi)}</dd></div>
                            <div class="bg-gray-700 p-2 rounded"><dt class="text-xs text-gray-400">PP %</dt><dd class="text-sm font-medium">${formatPercentage(data.lgPpPct)}</dd></div>
                            <div class="bg-gray-700 p-2 rounded"><dt class="text-xs text-gray-400">PPA</dt><dd class="text-sm font-medium">${formatNullable(data.lgPpa)}</dd></div>
                            <div class="bg-gray-700 p-2 rounded"><dt class="text-xs text-gray-400">PPG</dt><dd class="text-sm font-medium">${formatNullable(data.lgPpg)}</dd></div>
                        </dl>
                    </div>
                    <div>
                        <h4 class="text-md font-semibold text-white mb-2">Last Week</h4>
                        <dl class="grid grid-cols-2 sm:grid-cols-5 gap-x-4 gap-y-2">
                            <div class="bg-gray-700 p-2 rounded"><dt class="text-xs text-gray-400">Avg PP TOI</dt><dd class="text-sm font-medium">${formatSecondsToMMSS(data.lwPpToi)}</dd></div>
                            <div class="bg-gray-700 p-2 rounded"><dt class="text-xs text-gray-400">Avg PP %</dt><dd class="text-sm font-medium">${formatPercentage(data.lwPpPct)}</dd></div>
                            <div class="bg-gray-700 p-2 rounded"><dt class="text-xs text-gray-400">Tot PPA</dt><dd class="text-sm font-medium">${formatNullable(data.lwPpa)}</dd></div>
                            <div class="bg-gray-700 p-2 rounded"><dt class="text-xs text-gray-400">Tot PPG</dt><dd class="text-sm font-medium">${formatNullable(data.lwPpg)}</dd></div>
                            <div class="bg-gray-700 p-2 rounded"><dt class="text-xs text-gray-400">GP</dt><dd class="text-sm font-medium">${formatNullable(data.lwGp)}</dd></div>
                        </dl>
                    </div>
                </div>`;
                document.getElementById('pp-stats-modal').classList.remove('hidden');
            }
            const pill = e.target.closest('.line-info-pill');
            if (pill && window.openLineInfoModal && rosterData.players) {
                const pid = String(pill.dataset.playerId);
                const player = rosterData.players.find(p => String(p.player_id) === pid);
                if (player) {
                    window.openLineInfoModal(player);
                }
                return;
            }
            const gPill = e.target.closest('.goalie-info-pill');
            if (gPill && window.openGoalieInfoModal) {
                const pid = String(gPill.dataset.playerId);
                const player = rosterData.players.find(p => String(p.player_id) === pid);
                if (player) window.openGoalieInfoModal(player);
                return;
            }
            const rankCell = e.target.closest('.cat-rank-cell');
            if (rankCell && rosterData.players && window.openCatRankModal) {
                const pid = String(rankCell.dataset.playerId);
                const player = rosterData.players.find(p => String(p.player_id) === pid);

                if (player) {
                    const isGoalie = (player.eligible_positions || '').includes('G');
                    const cats = isGoalie ? rosterData.goalieCategories : rosterData.skaterCategories;
                    window.openCatRankModal(player, cats);
                }
            }
            const trendIcon = e.target.closest('.trending-icon');
            if (trendIcon && rosterData.players) {
                const pid = String(trendIcon.dataset.playerId);
                const player = rosterData.players.find(p => String(p.player_id) === pid);
                if (player) {
                    document.getElementById('trending-modal-title').textContent = `${player.player_name} - Trend`;
                    document.getElementById('trending-modal-content').innerHTML = `<pre class="text-xs">${JSON.stringify(player.trend_details, null, 2)}</pre>`;
                    document.getElementById('trending-stats-modal').classList.remove('hidden');
                }
            }

            const simBtn = e.target.closest('#simulate-trade-btn');
            if (simBtn && !simBtn.disabled) loadSubPage('trade-results');
        });

        document.body.addEventListener('change', (e) => {
            if (e.target.classList.contains('trade-player-checkbox')) {
                const pid = String(e.target.value);
                if (e.target.checked) selectedPlayerIds.add(pid);
                else selectedPlayerIds.delete(pid);
                updateSimulateButtonState();
            }
        });

        window.addEventListener('rawDataToggled', (e) => {
            if (currentSubPage === 'trade-compare') renderComparePage();
        });

        yourTeamSelect.addEventListener('change', () => {
            userTeamName = yourTeamSelect.value;
            filterPartner = "";
            activeFromCats = []; activeToCats = [];
            selectedPlayerIds.clear();
            fetchAllData();
        });

        if(navPartners) navPartners.onclick = () => loadSubPage('trade-partners');
        if(navCompare) navCompare.onclick = () => loadSubPage('trade-compare');

        await loadSubPage('trade-partners');
        fetchAllData();
    }

    window.loadSubPage = async function(pageName) {
        currentSubPage = pageName;
        if (navPartners && navCompare) {
            const activeClass = 'border-blue-500 text-blue-400';
            const inactiveClass = 'border-transparent text-gray-400 hover:text-gray-300';
            if (pageName === 'trade-partners') {
                navPartners.className = navPartners.className.replace(inactiveClass, '').replace('border-transparent', '') + ` ${activeClass}`;
                navCompare.className = navCompare.className.replace(activeClass, '') + ` ${inactiveClass}`;
            } else {
                navCompare.className = navCompare.className.replace(inactiveClass, '').replace('border-transparent', '') + ` ${activeClass}`;
                navPartners.className = navPartners.className.replace(activeClass, '') + ` ${inactiveClass}`;
            }
        }
        try {
            const response = await fetch(`/pages/${pageName}.html`);
            if (!response.ok) throw new Error(`Failed to load ${pageName}`);
            subpageContent.innerHTML = await response.text();
            renderCurrentPage();
        } catch (err) { console.error(err); subpageContent.innerHTML = `<p class="text-red-400">Error loading content.</p>`; }
    };

    async function fetchAllData() {
        if (!userTeamName) { if(loadingText) loadingText.textContent = "Please select a team."; return; }
        if(loadingText) { loadingText.textContent = "Loading data..."; loadingText.classList.remove('hidden'); }
        await Promise.all([fetchCategoryStrengths(), fetchLeagueRosterData()]);
        if(loadingText) loadingText.textContent = "";
    }

    async function fetchCategoryStrengths() {
        try {
            const response = await fetch('/api/trade_helper_data', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ team_name: userTeamName, week: 'all' })
            });
            categoryData = await response.json();
            if (currentSubPage === 'trade-partners') renderCurrentPage();
        } catch (err) { console.error(err); }
    }

    async function fetchLeagueRosterData() {
        const selectedSourcing = localStorage.getItem('selectedStatSourcing') || 'projected';
        try {
            const response = await fetch('/api/trade_helper_league_roster_data', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ sourcing: selectedSourcing })
            });
            const data = await response.json();
            rosterData.players = data.players || [];
            rosterData.skaterCategories = data.skater_categories || [];
            rosterData.goalieCategories = data.goalie_categories || [];
            rosterData.currentWeek = data.current_week || 1;
            if (currentSubPage === 'trade-compare') renderCurrentPage();
        } catch (err) { console.error(err); }
    }

    function renderCurrentPage() {
        if (currentSubPage === 'trade-partners') renderPartnersPage();
        else if (currentSubPage === 'trade-compare') renderComparePage();
        else if (currentSubPage === 'trade-results') renderResultsPage();
    }

    function renderPartnersPage() {
        const skaterContainer = document.getElementById('skater-table-container');
        const goalieContainer = document.getElementById('goalie-table-container');
        const recContainer = document.getElementById('trade-recommendations-container');
        if (!skaterContainer || !goalieContainer) return;
        if (!categoryData.skater_stats) { skaterContainer.innerHTML = '<p class="text-gray-500">Loading stats...</p>'; return; }

        const headers = ['category', 'Rank', 'Average Delta', 'Total'];
        renderSimpleTable(skaterContainer, categoryData.skater_stats, headers);

        let hiddenSubCats = new Set();
        if (categoryData.goalie_stats) {
            categoryData.goalie_stats.forEach(row => {
                if (row.sub_rows && row.sub_rows.length > 0) row.sub_rows.forEach(sub => hiddenSubCats.add(sub.category));
            });
        }
        const filteredGoalieStats = categoryData.goalie_stats.filter(row => !hiddenSubCats.has(row.category));
        renderSimpleTable(goalieContainer, filteredGoalieStats, headers);

        if (recContainer && categoryData.league_rank_matrix) {
            const matches = findTradeMatches(categoryData.league_rank_matrix, categoryData.total_teams);
            renderTradeMatches(recContainer, matches);
        }
    }

    function findTradeMatches(matrix, totalTeams) {
        const myRanks = matrix[userTeamName];
        if (!myRanks) return [];
        const strengthThreshold = Math.ceil(totalTeams / 3);
        const weaknessThreshold = totalTeams - Math.floor(totalTeams / 3) + 1;
        const allCats = Object.keys(myRanks);
        const excludedCats = [];
        if (allCats.includes('GAA')) excludedCats.push('GA', 'TOI/G');
        let matches = [];
        for (const [teamName, teamRanks] of Object.entries(matrix)) {
            if (teamName === userTeamName) continue;
            let youGive = [], youGet = [];
            for (const cat of allCats) {
                if (excludedCats.includes(cat)) continue;
                const myRank = myRanks[cat];
                const theirRank = teamRanks[cat];
                if (myRank <= strengthThreshold && theirRank >= weaknessThreshold) youGive.push(cat);
                if (myRank >= weaknessThreshold && theirRank <= strengthThreshold) youGet.push(cat);
            }
            if (youGive.length > 0 && youGet.length > 0) {
                matches.push({ team: teamName, gives: youGive, gets: youGet, score: youGive.length + youGet.length });
            }
        }
        matches.sort((a, b) => b.score - a.score);
        return matches;
    }

    function renderTradeMatches(container, matches) {
        if (matches.length === 0) { container.innerHTML = `<div class="col-span-full text-center p-4 bg-gray-700/50 rounded-lg text-gray-300">No perfect matches found.</div>`; return; }
        const formatForDisplay = (cats) => {
            let display = [...cats];
            if (display.includes('SA') && display.includes('SV')) { display = display.filter(c => c !== 'SA' && c !== 'SV'); display.push('Goalie Vol'); }
            return display;
        };
        let html = '';
        matches.forEach(m => {
            const displayGives = formatForDisplay(m.gives);
            const displayGets = formatForDisplay(m.gets);
            html += `
            <div class="bg-gray-700/40 border border-gray-600 rounded-lg p-4 hover:bg-gray-700/70 transition duration-200 cursor-pointer"
                 onclick="window.selectTradeScenario('${m.team}', '${m.gives.join(',')}', '${m.gets.join(',')}')">
                <h4 class="text-lg font-bold text-white mb-3 border-b border-gray-600 pb-2 flex justify-between items-center">${m.team} <span class="text-xs font-normal text-blue-300 hover:underline">Compare &rarr;</span></h4>
                <div class="mb-3"><p class="text-xs uppercase text-gray-400 font-bold mb-1">You Give (Surplus):</p><div class="flex flex-wrap gap-2">${displayGives.map(cat => `<span class="px-2 py-1 text-xs font-bold rounded bg-green-900 text-green-200 border border-green-700">${cat}</span>`).join('')}</div></div>
                <div><p class="text-xs uppercase text-gray-400 font-bold mb-1">You Get (Need):</p><div class="flex flex-wrap gap-2">${displayGets.map(cat => `<span class="px-2 py-1 text-xs font-bold rounded bg-blue-900 text-blue-200 border border-blue-700">${cat}</span>`).join('')}</div></div>
            </div>`;
        });
        container.innerHTML = html;
    }

    window.selectTradeScenario = function(partnerName, givesStr, getsStr) {
        filterPartner = partnerName;
        activeFromCats = givesStr ? givesStr.split(',') : [];
        activeToCats = getsStr ? getsStr.split(',') : [];
        loadSubPage('trade-compare');
    }

    function renderSimpleTable(container, data, headers) {
        if (!data || data.length === 0) { container.innerHTML = `<p class="text-gray-400 p-4">No data.</p>`; return; }
        let html = `<table class="min-w-full divide-y divide-gray-700 bg-gray-800 text-sm text-white"><thead class="bg-gray-750"><tr>`;
        headers.forEach(h => html += `<th class="px-4 py-3 text-left font-medium text-gray-300 uppercase">${h.replace('_', ' ')}</th>`);
        html += `</tr></thead><tbody class="divide-y divide-gray-700">`;
        data.forEach(row => {
            html += `<tr class="hover:bg-gray-700">`;
            headers.forEach(h => {
                let val = row[h], classes = "px-4 py-3 whitespace-nowrap font-medium";
                if (h === 'Rank') {
                    const r = parseInt(val);
                    if (r <= 3) classes += ' text-green-400 font-bold'; else if (r >= 8) classes += ' text-red-400'; else classes += ' text-yellow-400';
                } else if (h === 'Average Delta') {
                    const d = parseFloat(val);
                    if (d > 0.5) classes += ' text-green-400'; else if (d < -0.5) classes += ' text-red-400';
                }
                html += `<td class="${classes}">${val}</td>`;
            });
            html += `</tr>`;
            if (row.sub_rows && row.sub_rows.length > 0) {
                row.sub_rows.forEach(sub => {
                    html += `<tr class="hover:bg-gray-700/50 bg-gray-800/40">`;
                    headers.forEach((h, index) => {
                        let val = sub[h];
                        html += index === 0 ? `<td class="px-4 py-2 text-xs text-gray-400 pl-8 border-l-2 border-gray-600">${val}</td>` : `<td class="px-4 py-2 text-xs whitespace-nowrap text-gray-500">${val}</td>`;
                    });
                    html += `</tr>`;
                });
            }
        });
        html += `</tbody></table>`;
        container.innerHTML = html;
    }

    function renderComparePage() {
        const userSkaterContainer = document.getElementById('roster-skater-table-container');
        const userGoalieContainer = document.getElementById('roster-goalie-table-container');
        const oppSkaterContainer = document.getElementById('opponent-skater-table-container');
        const oppGoalieContainer = document.getElementById('opponent-goalie-table-container');
        const rosterLoader = document.getElementById('roster-loading');
        const oppSkaterSection = document.getElementById('opponent-skater-section');
        const oppGoalieSection = document.getElementById('opponent-goalie-section');

        if (!userSkaterContainer) return;

        if (categoryData.all_scoring_categories) setupCompareSorting(categoryData.all_scoring_categories);
        if (rosterData.players.length > 0) setupCompareFilters();

        if (rosterData.players.length === 0) {
            if(rosterLoader) rosterLoader.textContent = "Loading rosters...";
            return;
        }
        if(rosterLoader) rosterLoader.textContent = "";

        const userPlayers = rosterData.players.filter(p => p.fantasy_team_name === userTeamName);
        const oppPlayers = rosterData.players.filter(p => {
            if (p.fantasy_team_name === userTeamName || p.fantasy_team_name === 'Free Agent') return false;
            if (filterPartner && p.fantasy_team_name !== filterPartner) return false;
            if (filterNHL.length > 0 && !filterNHL.includes(p.team)) return false;
            if (filterPos.length > 0) {
                const pPos = (p.eligible_positions || "").split(',').map(s => s.trim());
                if (!pPos.some(pos => filterPos.includes(pos))) return false;
            }
            if (filterSearch) {
                if (!p.player_name.toLowerCase().includes(filterSearch.toLowerCase())) return false;
            }
            return true;
        });

        const userSkaters = userPlayers.filter(p => !(p.eligible_positions || '').includes('G'));
        const userGoalies = userPlayers.filter(p => (p.eligible_positions || '').includes('G'));
        const oppSkaters = oppPlayers.filter(p => !(p.eligible_positions || '').includes('G'));
        const oppGoalies = oppPlayers.filter(p => (p.eligible_positions || '').includes('G'));

        const multiSortFn = (keys) => (a, b) => {
            if (!keys || keys.length === 0) return 0;
            for (let key of keys) {
                let rankKey = key + '_cat_rank';
                let rA = a[rankKey], rB = b[rankKey];
                if (rA == null || rA === 0) rA = 999;
                if (rB == null || rB === 0) rB = 999;
                if (rA !== rB) return rA - rB;
            }
            return 0;
        };

        userSkaters.sort(multiSortFn(activeFromCats));
        userGoalies.sort(multiSortFn(activeFromCats));
        oppSkaters.sort(multiSortFn(activeToCats));
        oppGoalies.sort(multiSortFn(activeToCats));

        renderRosterTable(userSkaterContainer, userSkaters, rosterData.skaterCategories, false);
        renderRosterTable(userGoalieContainer, userGoalies, rosterData.goalieCategories, false);
        renderRosterTable(oppSkaterContainer, oppSkaters, rosterData.skaterCategories, true);
        renderRosterTable(oppGoalieContainer, oppGoalies, rosterData.goalieCategories, true);

        const primaryToCat = activeToCats[0];
        const isGoalieStat = primaryToCat && rosterData.goalieCategories.includes(primaryToCat);
        if (oppSkaterSection && oppGoalieSection) {
            if (isGoalieStat) { oppGoalieSection.style.order = 1; oppSkaterSection.style.order = 2; }
            else { oppSkaterSection.style.order = 1; oppGoalieSection.style.order = 2; }
        }

        updateSimulateButtonState();
    }

    function updateSimulateButtonState() {
        const btn = document.getElementById('simulate-trade-btn');
        if (!btn) return;

        const selectedPlayers = rosterData.players.filter(p => selectedPlayerIds.has(String(p.player_id)));
        const userSelected = selectedPlayers.filter(p => p.fantasy_team_name === userTeamName);
        const oppSelected = selectedPlayers.filter(p => p.fantasy_team_name !== userTeamName);

        if (userSelected.length === 0 || oppSelected.length === 0) {
            btn.disabled = true;
            btn.classList.add('opacity-50', 'cursor-not-allowed');
            return;
        }

        const uniqueOppTeams = new Set(oppSelected.map(p => p.fantasy_team_name));
        if (uniqueOppTeams.size > 1) {
            btn.disabled = true;
            btn.classList.add('opacity-50', 'cursor-not-allowed');
            return;
        }

        btn.disabled = false;
        btn.classList.remove('opacity-50', 'cursor-not-allowed');
    }

    function setupCompareSorting(categories) {
        const tradeFromSelect = document.getElementById('trade-from-select');
        const tradeToSelect = document.getElementById('trade-to-select');
        const tradeFromTags = document.getElementById('trade-from-tags');
        const tradeToTags = document.getElementById('trade-to-tags');
        if (!tradeFromSelect || !tradeToSelect) return;
        if (tradeFromSelect.options.length === 1) {
            let visibleCategories = [...categories];
            if (categories.includes('GAA')) visibleCategories = visibleCategories.filter(c => c !== 'GA' && c !== 'TOI/G');
            if (categories.includes('SVpct')) visibleCategories = visibleCategories.filter(c => c !== 'SV' && c !== 'SA');
            visibleCategories.forEach(cat => {
                const opt1 = document.createElement('option'); opt1.value = cat; opt1.textContent = cat; tradeFromSelect.appendChild(opt1);
                const opt2 = document.createElement('option'); opt2.value = cat; opt2.textContent = cat; tradeToSelect.appendChild(opt2);
            });
            tradeFromSelect.addEventListener('change', (e) => { if (e.target.value) { addCategory('from', e.target.value); e.target.value = ""; } });
            tradeToSelect.addEventListener('change', (e) => { if (e.target.value) { addCategory('to', e.target.value); e.target.value = ""; } });
        }
        if (tradeFromTags && tradeFromTags.children.length === 0) renderTags(tradeFromTags, activeFromCats, 'from');
        if (tradeToTags && tradeToTags.children.length === 0) renderTags(tradeToTags, activeToCats, 'to');
    }

    function addCategory(type, category) {
        const targetArray = type === 'from' ? activeFromCats : activeToCats;
        if (!targetArray.includes(category)) {
            targetArray.push(category);
            const container = type === 'from' ? document.getElementById('trade-from-tags') : document.getElementById('trade-to-tags');
            if(container) renderTags(container, targetArray, type);
            renderCurrentPage();
        }
    }

    function removeCategory(type, category) {
        if (type === 'from') activeFromCats = activeFromCats.filter(c => c !== category);
        else activeToCats = activeToCats.filter(c => c !== category);
        const container = type === 'from' ? document.getElementById('trade-from-tags') : document.getElementById('trade-to-tags');
        const arr = type === 'from' ? activeFromCats : activeToCats;
        if (container) renderTags(container, arr, type);
        renderCurrentPage();
    }

    function renderTags(container, tags, type) {
        container.innerHTML = '';
        tags.forEach(tag => {
            const pill = document.createElement('div');
            pill.className = 'inline-flex items-center bg-blue-900 text-blue-200 text-xs font-medium px-2.5 py-1 rounded border border-blue-700';
            pill.innerHTML = `${tag}<button type="button" class="ml-1.5 inline-flex items-center justify-center w-4 h-4 rounded-full text-blue-200 hover:bg-blue-800 hover:text-white focus:outline-none"><span class="sr-only">Remove</span>&times;</button>`;
            pill.querySelector('button').addEventListener('click', () => removeCategory(type, tag));
            container.appendChild(pill);
        });
    }

    function setupCompareFilters() {
        const partnerSelect = document.getElementById('filter-partner');
        const nhlSelect = document.getElementById('filter-nhl');
        const posSelect = document.getElementById('filter-pos');
        const searchInput = document.getElementById('filter-search');
        const tagsNHL = document.getElementById('tags-nhl');
        const tagsPos = document.getElementById('tags-pos');
        const clearBtn = document.getElementById('clear-selections-btn');

        if (!partnerSelect || !nhlSelect || !posSelect || !searchInput) return;

        if (clearBtn) {
            const newBtn = clearBtn.cloneNode(true);
            clearBtn.parentNode.replaceChild(newBtn, clearBtn);
            newBtn.addEventListener('click', () => {
                selectedPlayerIds.clear();
                renderComparePage();
            });
        }

        if (partnerSelect.options.length === 1) {
            const teams = [...new Set(rosterData.players.map(p => p.fantasy_team_name))].sort();
            teams.forEach(team => {
                if (team !== userTeamName && team !== 'Free Agent') {
                    const opt = document.createElement('option'); opt.value = team; opt.textContent = team;
                    partnerSelect.appendChild(opt);
                }
            });
            partnerSelect.value = filterPartner;

            NHL_TEAMS.forEach(t => { const opt = document.createElement('option'); opt.value = t; opt.textContent = t; nhlSelect.appendChild(opt); });

            partnerSelect.addEventListener('change', (e) => { filterPartner = e.target.value; renderComparePage(); });
            nhlSelect.addEventListener('change', (e) => { if (e.target.value && !filterNHL.includes(e.target.value)) { filterNHL.push(e.target.value); renderFilterTags(tagsNHL, filterNHL, 'nhl'); renderComparePage(); } e.target.value = ""; });
            posSelect.addEventListener('change', (e) => { if (e.target.value && !filterPos.includes(e.target.value)) { filterPos.push(e.target.value); renderFilterTags(tagsPos, filterPos, 'pos'); renderComparePage(); } e.target.value = ""; });
            searchInput.addEventListener('input', (e) => { filterSearch = e.target.value; renderComparePage(); });
        }
        if (tagsNHL && tagsNHL.children.length === 0) renderFilterTags(tagsNHL, filterNHL, 'nhl');
        if (tagsPos && tagsPos.children.length === 0) renderFilterTags(tagsPos, filterPos, 'pos');
    }

    function renderFilterTags(container, tags, type) {
        container.innerHTML = '';
        tags.forEach(tag => {
            const pill = document.createElement('div');
            pill.className = 'inline-flex items-center bg-gray-600 text-gray-200 text-[10px] px-2 py-0.5 rounded border border-gray-500';
            pill.innerHTML = `${tag}<button class="ml-1 text-gray-400 hover:text-white font-bold">&times;</button>`;
            pill.querySelector('button').addEventListener('click', () => {
                if (type === 'nhl') filterNHL = filterNHL.filter(t => t !== tag); else filterPos = filterPos.filter(p => p !== tag);
                renderFilterTags(container, (type==='nhl'?filterNHL:filterPos), type);
                renderComparePage();
            });
            container.appendChild(pill);
        });
    }

    function renderRosterTable(container, players, categories, showTeamColumn) {
        if (!players || players.length === 0) { container.innerHTML = `<p class="text-gray-400 p-4">No players found matching criteria.</p>`; return; }

        const showRaw = localStorage.getItem('showRawData') === 'true';

        let html = `<div class="overflow-x-auto"><table class="min-w-full divide-y divide-gray-700 text-sm"><thead class="bg-gray-700/50"><tr>`;
        html += `<th class="px-2 py-1 text-center w-8">Select</th>`;
        html += `<th class="px-2 py-1 text-left font-bold text-gray-300">Player</th>`;
        if (showTeamColumn) html += `<th class="px-2 py-1 text-left font-bold text-gray-300">Team</th>`;
        html += `<th class="px-2 py-1 text-left font-bold text-gray-300">NHL Team</th><th class="px-2 py-1 text-left font-bold text-gray-300">Pos</th>`;
        html += `<th class="px-2 py-1 text-center font-bold text-gray-300" title="Sum of Category Ranks">Cat Rank</th>`;
        html += `<th class="px-2 py-1 text-center font-bold text-gray-300" title="Power Play Utilization">PP Util</th>`;
        html += `<th class="px-2 py-1 text-center font-bold text-gray-300"><div class="text-[10px] leading-tight">L20 | L10 | L5 | H/A</div></th>`;
        categories.forEach(cat => html += `<th class="px-2 py-1 text-center font-bold text-gray-300" title="${cat}">${cat}</th>`);
        html += `</tr></thead><tbody class="bg-gray-800 divide-y divide-gray-700">`;

        players.forEach(p => {
            try {
                const isChecked = selectedPlayerIds.has(String(p.player_id)) ? 'checked' : '';
                const teamClass = p.fantasy_team_name === userTeamName ? 'border-l-4 border-blue-500' : '';

                let pillHtml = '';
                const isGoalie = (p.positions || p.eligible_positions || '').includes('G');

                if (isGoalie) {
                     const gd = p.goalie_data || { l10_start_pct: 'N/A', days_rest: 'N/A', next_loc: 'N/A' };
                     const safeVal = (v) => (v === undefined || v === null) ? 'N/A' : v;
                     const pct = safeVal(gd.l10_start_pct) !== 'N/A' ? `${gd.l10_start_pct}%` : 'N/A';

                     pillHtml = `
                         <span class="ml-2 px-2 py-0.5 rounded text-[10px] font-bold bg-blue-900 text-blue-200 border border-blue-700 cursor-pointer hover:bg-blue-800 goalie-info-pill"
                               data-player-id="${p.player_id}">
                             ${pct} | Rest: ${safeVal(gd.days_rest)} | ${safeVal(gd.next_loc)}
                         </span>`;
                } else {
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

                // --- TRENDING COLUMN LOGIC ---
                const tSum = p.trend_summary || ['N/A', 'N/A', 'N/A', 'N/A'];
                const icons = {
                    'UP': `<span class="text-green-500 text-lg cursor-pointer trending-icon" data-player-id="${p.player_id}">&#9650;</span>`,
                    'DOWN': `<span class="text-red-500 text-lg cursor-pointer trending-icon" data-player-id="${p.player_id}">&#9660;</span>`,
                    'FLAT': `<span class="text-yellow-500 text-lg cursor-pointer trending-icon" data-player-id="${p.player_id}">&#8722;</span>`,
                    'N/A': `<span class="text-gray-600 text-[10px] cursor-pointer trending-icon" data-player-id="${p.player_id}">-</span>`
                };

                let haHtml = '';
                if (tSum[3].includes('_')) {
                    const [loc, color] = tSum[3].split('_');
                    const colorClass = color === 'GREEN' ? 'text-green-500' : (color === 'RED' ? 'text-red-500' : 'text-gray-400');
                    haHtml = `<span class="${colorClass} font-bold text-sm cursor-pointer trending-icon" data-player-id="${p.player_id}">${loc}</span>`;
                } else if (tSum[3] !== 'N/A') {
                    haHtml = `<span class="text-gray-400 font-bold text-sm cursor-pointer trending-icon" data-player-id="${p.player_id}">${tSum[3]}</span>`;
                } else {
                    haHtml = `<span class="text-gray-600 text-[10px] cursor-pointer trending-icon" data-player-id="${p.player_id}">-</span>`;
                }

                const trendHtml = `
                    <div class="flex items-center justify-center space-x-1">
                        ${icons[tSum[0]] || icons['N/A']}
                        <span class="text-gray-700 text-[10px]">|</span>
                        ${icons[tSum[1]] || icons['N/A']}
                        <span class="text-gray-700 text-[10px]">|</span>
                        ${icons[tSum[2]] || icons['N/A']}
                        <span class="text-gray-700 text-[10px]">|</span>
                        ${haHtml}
                    </div>
                `;

                html += `<tr class="hover:bg-gray-700/50 ${teamClass}">`;
                html += `<td class="px-2 py-1 text-center"><input type="checkbox" value="${p.player_id}" class="trade-player-checkbox form-checkbox h-4 w-4 text-blue-600 rounded bg-gray-700 border-gray-600" ${isChecked}></td>`;

                html += `<td class="px-2 py-1 whitespace-nowrap font-medium text-gray-300 flex items-center">
                            ${p.player_name}
                            ${pillHtml}
                         </td>`;

                if (showTeamColumn) html += `<td class="px-2 py-1 whitespace-nowrap text-yellow-300">${p.fantasy_team_name}</td>`;
                html += `<td class="px-2 py-1 whitespace-nowrap text-gray-300">${p.team}</td><td class="px-2 py-1 whitespace-nowrap text-gray-300">${p.eligible_positions}</td>`;

                let catSum = 0, validRanks = 0;
                categories.forEach(cat => { const r = p[cat + '_cat_rank']; if (r != null) { catSum += r; validRanks++; } });

                html += `<td class="px-2 py-1 text-center font-bold text-white cursor-pointer hover:text-blue-400 cat-rank-cell" data-player-id="${p.player_id}">${validRanks > 0 ? Math.round(catSum) : '-'}</td>`;

                if (p.avg_ppTimeOnIcePctPerGame !== undefined) {
                    html += `<td class="px-2 py-1 whitespace-nowrap text-sm text-gray-300 cursor-pointer hover:bg-gray-700 pp-util-cell" data-player-name="${p.player_name}" data-lg-pp-toi="${p.lg_ppTimeOnIce}" data-lg-pp-pct="${p.lg_ppTimeOnIcePctPerGame}" data-lg-ppa="${p.lg_ppAssists}" data-lg-ppg="${p.lg_ppGoals}" data-lw-pp-toi="${p.avg_ppTimeOnIce}" data-lw-pp-pct="${p.avg_ppTimeOnIcePctPerGame}" data-lw-ppa="${p.total_ppAssists}" data-lw-ppg="${p.total_ppGoals}" data-lw-gp="${p.team_games_played}">${formatPercentage(p.avg_ppTimeOnIcePctPerGame)}</td>`;
                } else { html += `<td class="px-2 py-1 text-center text-gray-500">-</td>`; }

                html += `<td class="px-2 py-1 text-center whitespace-nowrap">${trendHtml}</td>`;

                categories.forEach(cat => {
                    const rank = p[cat + '_cat_rank'];
                    const heatColor = getHeatmapColor(rank);
                    let displayValue = '-';

                    if (showRaw) {
                        const val = p[cat];
                        displayValue = (val != null && !isNaN(val)) ? parseFloat(val).toFixed(2).replace(/[.,]00$/, "") : (val || '-');
                    } else {
                        displayValue = (rank != null) ? Math.round(rank) : '-';
                    }

                    let cellStyle = '';
                    let cellClass = 'px-2 py-1 text-center font-semibold';

                    if (heatColor) {
                        cellStyle = `background-color: ${heatColor}; color: #1f2937; font-weight: 600;`;
                    } else {
                        cellClass += ' text-gray-400';
                    }

                    html += `<td class="${cellClass}" style="${cellStyle}">${displayValue}</td>`;
                });

                html += `</tr>`;
            } catch (err) {
                console.error("Error rendering player row:", p.player_name, err);
            }
        });
        html += `</tbody></table></div>`;
        container.innerHTML = html;
    }

    function renderResultsPage() {
        const userContainer = document.getElementById('results-user-container');
        const oppContainer = document.getElementById('results-opponent-container');
        if (!userContainer || !oppContainer) return;

        const selectedPlayers = rosterData.players.filter(p => selectedPlayerIds.has(String(p.player_id)));
        const userTradePlayers = selectedPlayers.filter(p => p.fantasy_team_name === userTeamName);
        const oppTradePlayers = selectedPlayers.filter(p => p.fantasy_team_name !== userTeamName);
        const oppTeamName = oppTradePlayers[0] ? oppTradePlayers[0].fantasy_team_name : "Opponent";

        const results = calculateTradeImpact(userTeamName, oppTeamName, userTradePlayers, oppTradePlayers);

        renderImpactTable(userContainer, results.userImpact);
        renderImpactTable(oppContainer, results.oppImpact);
    }

    function calculateTradeImpact(userTeam, oppTeam, userGiving, oppGiving) {
        const completedWeeks = Math.max(1, rosterData.currentWeek - 1);
        const leagueTotals = JSON.parse(JSON.stringify(categoryData.league_raw_stats));
        if (!leagueTotals || !leagueTotals[userTeam]) { console.error("Missing league_raw_stats data."); return { userImpact: [], oppImpact: [] }; }

        const adjustTeam = (teamName, outgoingPlayers, incomingPlayers) => {
            const totals = { ...leagueTotals[teamName] };
            const cats = [...rosterData.skaterCategories, ...rosterData.goalieCategories, 'GA', 'SV', 'SA', 'TOI/G'];

            cats.forEach(cat => {
                let change = 0;
                outgoingPlayers.forEach(p => {
                    const isGoalie = (p.eligible_positions || '').includes('G');
                    const rate = parseFloat(p[cat] || 0);
                    const factor = isGoalie ? 2.0 : 3.4;
                    change -= (rate * factor);
                });
                incomingPlayers.forEach(p => {
                    const isGoalie = (p.eligible_positions || '').includes('G');
                    const rate = parseFloat(p[cat] || 0);
                    const factor = isGoalie ? 2.0 : 3.4;
                    change += (rate * factor);
                });
                totals[cat] = Math.max(0, totals[cat] + (change * completedWeeks));
            });

            if (totals['TOI/G'] > 0) totals['GAA'] = (totals['GA'] * 60) / totals['TOI/G'];
            if (totals['SA'] > 0) totals['SVpct'] = totals['SV'] / totals['SA'];
            return totals;
        };

        const newLeagueTotals = JSON.parse(JSON.stringify(leagueTotals));
        newLeagueTotals[userTeam] = adjustTeam(userTeam, userGiving, oppGiving);
        newLeagueTotals[oppTeam] = adjustTeam(oppTeam, oppGiving, userGiving);

        const calculateRankShift = (targetTeam, categories) => {
            const result = [];
            const reverseCats = ['GA', 'GAA', 'L'];
            categories.forEach(cat => {
                if (cat === 'GA' || cat === 'SV' || cat === 'SA') return;
                const oldVals = Object.values(leagueTotals).map(t => t[cat] || 0);
                const isRev = reverseCats.includes(cat);
                oldVals.sort((a, b) => isRev ? a - b : b - a);
                const oldVal = leagueTotals[targetTeam][cat] || 0;
                const oldRank = oldVals.indexOf(oldVal) + 1;

                const newVals = Object.values(newLeagueTotals).map(t => t[cat] || 0);
                newVals.sort((a, b) => isRev ? a - b : b - a);
                const newVal = newLeagueTotals[targetTeam][cat] || 0;
                const newRank = newVals.indexOf(newVal) + 1;

                result.push({ category: cat, oldRank: oldRank, newRank: newRank, change: oldRank - newRank });
            });
            return result;
        };

        const allCatsToRank = [...rosterData.skaterCategories, ...rosterData.goalieCategories];
        return {
            userImpact: calculateRankShift(userTeam, allCatsToRank),
            oppImpact: calculateRankShift(oppTeam, allCatsToRank)
        };
    }

    function renderImpactTable(container, data) {
        let html = `<table class="min-w-full divide-y divide-gray-700 text-sm text-white"><thead class="bg-gray-700/50"><tr>
            <th class="px-4 py-3 text-left font-medium text-gray-300">Category</th>
            <th class="px-4 py-3 text-center font-medium text-gray-300">Old Rank</th>
            <th class="px-4 py-3 text-center font-medium text-gray-300">Change</th>
            <th class="px-4 py-3 text-center font-medium text-gray-300">New Rank</th>
        </tr></thead><tbody class="divide-y divide-gray-700">`;
        data.forEach(row => {
            let changeClass = "text-gray-400";
            let changeSym = "";
            if (row.change > 0) { changeClass = "text-green-400 font-bold"; changeSym = "+"; }
            else if (row.change < 0) { changeClass = "text-red-400 font-bold"; changeSym = ""; }
            html += `<tr class="hover:bg-gray-700/50">
                <td class="px-4 py-3 whitespace-nowrap font-medium">${row.category}</td>
                <td class="px-4 py-3 text-center">${row.oldRank}</td>
                <td class="px-4 py-3 text-center ${changeClass}">${changeSym}${row.change}</td>
                <td class="px-4 py-3 text-center font-bold text-white">${row.newRank}</td>
            </tr>`;
        });
        html += `</tbody></table>`;
        container.innerHTML = html;
    }

    init().catch(e => console.error("Init failed", e));
})();
