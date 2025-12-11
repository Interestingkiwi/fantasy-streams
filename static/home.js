document.addEventListener('DOMContentLoaded', () => {
    const logoutButton = document.getElementById('logout-button');
    const timestampText = document.getElementById('timestamp-text');
    const dropdownContainer = document.getElementById('dropdown-container');

    // --- [START] INJECT LINE INFO MODAL (SKATERS) ---
    if (!document.getElementById('line-info-modal')) {
        const lineModalHTML = `
        <div id="line-info-modal" class="fixed inset-0 bg-gray-900 bg-opacity-75 flex items-center justify-center z-50 hidden" style="backdrop-filter: blur(2px);">
            <div class="bg-gray-800 rounded-lg shadow-xl border border-gray-700 w-full max-w-lg p-6 relative">
                <button id="line-info-close" class="absolute top-3 right-3 text-gray-400 hover:text-white text-2xl leading-none">&times;</button>
                <h3 id="line-info-title" class="text-xl font-bold text-white mb-4"></h3>
                <div class="space-y-4 text-sm text-gray-300">
                    <div class="bg-gray-700/50 p-3 rounded border border-gray-600">
                        <span class="block text-[10px] text-gray-400 uppercase font-bold tracking-wider mb-1">Time On Ice</span>
                        <span id="line-info-toi" class="text-white font-mono text-lg font-semibold"></span>
                    </div>
                    <div class="bg-gray-700/50 p-3 rounded border border-gray-600">
                        <span class="block text-[10px] text-gray-400 uppercase font-bold tracking-wider mb-1">Primary Line</span>
                        <span id="line-info-primary" class="text-white block leading-relaxed"></span>
                    </div>
                    <div class="bg-gray-700/50 p-3 rounded border border-gray-600">
                        <span class="block text-[10px] text-gray-400 uppercase font-bold tracking-wider mb-1">Alternate Linemates</span>
                        <span id="line-info-alt" class="text-white block leading-relaxed italic"></span>
                    </div>
                    <div class="bg-gray-700/50 p-3 rounded border border-gray-600">
                        <span class="block text-[10px] text-gray-400 uppercase font-bold tracking-wider mb-1">Power Play Unit</span>
                        <span id="line-info-pp" class="text-white block leading-relaxed"></span>
                    </div>
                </div>
            </div>
        </div>`;
        document.body.insertAdjacentHTML('beforeend', lineModalHTML);

        // Close Logic
        document.body.addEventListener('click', (e) => {
            const modal = document.getElementById('line-info-modal');
            if (e.target.closest('#line-info-close') || e.target === modal) {
                modal.classList.add('hidden');
            }
        });
    }
    // --- [END] INJECT LINE INFO MODAL ---

    // --- [START] INJECT GOALIE INFO MODAL ---
    if (!document.getElementById('goalie-info-modal')) {
        const goalieModalHTML = `
        <div id="goalie-info-modal" class="fixed inset-0 bg-gray-900 bg-opacity-75 flex items-center justify-center z-50 hidden" style="backdrop-filter: blur(2px);">
            <div class="bg-gray-800 rounded-lg shadow-xl border border-gray-700 w-full max-w-lg p-6 relative">
                <button id="goalie-info-close" class="absolute top-3 right-3 text-gray-400 hover:text-white text-2xl leading-none">&times;</button>
                <h3 id="goalie-info-title" class="text-xl font-bold text-white mb-4"></h3>
                <div class="space-y-4 text-sm text-gray-300">
                    <div class="grid grid-cols-2 gap-4">
                        <div class="bg-gray-700/50 p-3 rounded border border-gray-600">
                            <span class="block text-[10px] text-gray-400 uppercase font-bold tracking-wider mb-1">Last 10 Start %</span>
                            <span id="goalie-l10-pct" class="text-white font-mono text-lg font-semibold"></span>
                        </div>
                        <div class="bg-gray-700/50 p-3 rounded border border-gray-600">
                            <span class="block text-[10px] text-gray-400 uppercase font-bold tracking-wider mb-1">All Season Start %</span>
                            <span id="goalie-season-pct" class="text-white font-mono text-lg font-semibold"></span>
                        </div>
                    </div>
                    <div class="bg-gray-700/50 p-3 rounded border border-gray-600">
                        <span id="goalie-rest-label" class="block text-[10px] text-gray-400 uppercase font-bold tracking-wider mb-1">Days Rest Splits</span>
                        <div class="flex justify-between items-center">
                            <span class="text-gray-400 text-xs">Win %</span>
                            <span id="goalie-rest-w" class="text-white font-mono font-bold"></span>
                            <span class="text-gray-600">|</span>
                            <span class="text-gray-400 text-xs">SV %</span>
                            <span id="goalie-rest-sv" class="text-white font-mono font-bold"></span>
                        </div>
                    </div>
                    <div class="bg-gray-700/50 p-3 rounded border border-gray-600">
                        <span id="goalie-loc-label" class="block text-[10px] text-gray-400 uppercase font-bold tracking-wider mb-1">Location Splits</span>
                        <div class="flex justify-between items-center">
                            <span class="text-gray-400 text-xs">Win %</span>
                            <span id="goalie-loc-w" class="text-white font-mono font-bold"></span>
                            <span class="text-gray-600">|</span>
                            <span class="text-gray-400 text-xs">SV %</span>
                            <span id="goalie-loc-sv" class="text-white font-mono font-bold"></span>
                        </div>
                    </div>
                </div>
            </div>
        </div>`;
        document.body.insertAdjacentHTML('beforeend', goalieModalHTML);

        document.body.addEventListener('click', (e) => {
            const modal = document.getElementById('goalie-info-modal');
            if (e.target.closest('#goalie-info-close') || e.target === modal) {
                modal.classList.add('hidden');
            }
        });
    }
    // --- [END] INJECT GOALIE INFO MODAL ---

    // --- [START] INJECT TRENDING INFO MODAL ---
    if (!document.getElementById('trending-stats-modal')) {
        const trendingModalHTML = `
        <div id="trending-stats-modal" class="fixed inset-0 bg-gray-900 bg-opacity-75 flex items-center justify-center z-50 hidden" style="backdrop-filter: blur(2px);">
            <div class="bg-gray-800 rounded-lg shadow-xl border border-gray-700 w-full max-w-4xl p-6 relative max-h-[90vh] overflow-y-auto">
                <button id="trending-modal-close" class="absolute top-3 right-3 text-gray-400 hover:text-white text-2xl leading-none">&times;</button>
                <h3 id="trending-modal-title" class="text-xl font-bold text-white mb-4">Player Trend Analysis</h3>
                <div id="trending-modal-content" class="text-gray-300"></div>
            </div>
        </div>`;
        document.body.insertAdjacentHTML('beforeend', trendingModalHTML);

        document.body.addEventListener('click', (e) => {
            const modal = document.getElementById('trending-stats-modal');
            if (e.target.closest('#trending-modal-close') || e.target === modal) {
                modal.classList.add('hidden');
            }
        });
    }
    // --- [END] INJECT TRENDING INFO MODAL ---

    // --- Global Function to Open Line Info Modal (Skaters) ---
    window.openLineInfoModal = function(player) {
        const modal = document.getElementById('line-info-modal');
        if (!modal) return;

        // Helper to format TOI
        const formatTOI = (seconds) => {
            if (seconds == null || seconds === undefined) return 'N/A';
            const s = parseInt(seconds, 10);
            if (isNaN(s)) return 'N/A';
            const m = Math.floor(s / 60);
            const rs = s % 60;
            return `${m}:${rs < 10 ? '0' : ''}${rs}`;
        };

        // Populate Data
        document.getElementById('line-info-title').textContent = `${player.player_name}: Last Game Lines`;
        document.getElementById('line-info-toi').textContent = formatTOI(player.timeonice);
        document.getElementById('line-info-primary').textContent = player.full_line || "N/A";
        document.getElementById('line-info-alt').textContent = player.alt_lines || "None";

        const ppElem = document.getElementById('line-info-pp');
        if (player.pp_line) {
            ppElem.textContent = player.pp_line;
            ppElem.classList.remove('text-gray-500', 'italic');
        } else {
            ppElem.textContent = `${player.player_name} did not record any PP Time in previous game`;
            ppElem.classList.add('text-gray-500', 'italic');
        }

        modal.classList.remove('hidden');
    };

    // --- Global Function to Open Goalie Info Modal ---
    window.openGoalieInfoModal = function(player) {
        const modal = document.getElementById('goalie-info-modal');
        if (!modal || !player.goalie_data) return;

        const gd = player.goalie_data;

        document.getElementById('goalie-info-title').textContent = `${player.player_name} Starts`;
        document.getElementById('goalie-l10-pct').textContent = gd.l10_start_pct !== 'N/A' ? `${gd.l10_start_pct}%` : 'N/A';
        document.getElementById('goalie-season-pct').textContent = gd.season_start_pct !== 'N/A' ? `${gd.season_start_pct}%` : 'N/A';

        document.getElementById('goalie-rest-label').textContent = `${gd.days_rest} Days Rest Splits`;
        document.getElementById('goalie-rest-w').textContent = gd.rest_split.w_pct !== 'N/A' ? `${gd.rest_split.w_pct}%` : 'N/A';
        document.getElementById('goalie-rest-sv').textContent = gd.rest_split.sv_pct;

        const locText = gd.next_loc === 'H' ? 'Home' : (gd.next_loc === 'A' ? 'Away' : 'Unknown');
        document.getElementById('goalie-loc-label').textContent = `${locText} Splits`;
        document.getElementById('goalie-loc-w').textContent = gd.loc_split.w_pct !== 'N/A' ? `${gd.loc_split.w_pct}%` : 'N/A';
        document.getElementById('goalie-loc-sv').textContent = gd.loc_split.sv_pct;

        modal.classList.remove('hidden');
    };

    // --- Global Function to Open Trending Modal ---
    window.renderTrendingModal = function(player, trendDetails) {
        const modal = document.getElementById('trending-stats-modal');
        const contentDiv = document.getElementById('trending-modal-content');
        if (!modal || !contentDiv) return;

        const stats = trendDetails.stats;
        const isGoalie = trendDetails.is_goalie;
        const anomalies = trendDetails.anomalies || [];

        // Helper to format numbers
        const fmt = (val, type) => {
            if (val === null || val === undefined) return '-';
            if (type === 'toi') {
                const totalSec = Math.round(val);
                const m = Math.floor(totalSec / 60);
                const s = totalSec % 60;
                return `${m}:${s < 10 ? '0' : ''}${s}`;
            }
            if (type === 'pct') return val.toFixed(1) + '%';
            if (type === 'float') return val.toFixed(2);
            return Math.round(val * 10) / 10; // 1 decimal for others
        };

        // Define Rows
        let groups = [];
        if (isGoalie) {
            groups = [
                {
                    name: "Goalie",
                    rows: [
                        { label: "Games Started", key: "gamesstarted" },
                        { label: "Wins", key: "wins" },
                        { label: "Shutouts", key: "shutouts" },
                        { label: "Losses", key: "losses" },
                        { label: "OT Losses", key: "overtimelosses" },
                        { label: "SV%", key: "savepct", type: 'float' },
                        { label: "GAA", key: "gaa", type: 'float' }
                    ]
                },
                {
                    name: "Team",
                    rows: [
                        { label: "Shots Against", key: "shotsagainst" },
                        { label: "Goals For", key: "goalsfor" }
                    ]
                }
            ];
        } else {
            groups = [
                {
                    name: "Luck Factor",
                    rows: [
                        { label: "Missed Shots", key: "missedshots" },
                        { label: "Blocked Attempts", key: "shotattemptsblocked" },
                        { label: "Takeaways", key: "takeaways" },
                        { label: "Shooting %", key: "shootingpct", type: 'pct' }
                    ]
                },
                {
                    name: "Bangers",
                    rows: [
                        { label: "Hits", key: "hits" },
                        { label: "Blocks", key: "blockedshots" },
                        { label: "PIM", key: "penaltyminutes" }
                    ]
                },
                {
                    name: "Scoring",
                    rows: [
                        { label: "Shots", key: "shots" },
                        { label: "EV Goals", key: "evgoals" },
                        { label: "EV Assists", key: "evassists" },
                        { label: "PP Goals", key: "ppgoals" },
                        { label: "PP Assists", key: "ppassists" },
                        { label: "SH Goals", key: "shgoals" },
                        { label: "SH Assists", key: "shassists" }
                    ]
                },
                {
                    name: "Deployments",
                    rows: [
                        { label: "Time On Ice", key: "timeonice", type: 'toi' },
                        { label: "EV TOI", key: "evtimeonice", type: 'toi' },
                        { label: "OT TOI", key: "ottimeonice", type: 'toi' },
                        { label: "SH TOI", key: "shtimeonice", type: 'toi' },
                        { label: "PP TOI", key: "pptimeonice", type: 'toi' }
                    ]
                }
            ];
        }

        const columns = ["Season Avg", "Last 20", "Last 10", "Last 5", "Home", "Away"];

        // Build HTML
        let html = `
            <div class="overflow-x-auto">
                <table class="min-w-full divide-y divide-gray-700 text-xs">
                    <thead>
                        <tr class="bg-gray-700">
                            <th class="px-2 py-2 text-left font-bold text-gray-200 sticky left-0 bg-gray-700 z-10">Stat</th>
                            ${columns.map(c => `<th class="px-2 py-2 text-center font-bold text-gray-200">${c}</th>`).join('')}
                        </tr>
                    </thead>
                    <tbody class="divide-y divide-gray-700 bg-gray-800">
        `;

        groups.forEach(group => {
            html += `<tr class="bg-gray-750"><td colspan="${columns.length + 1}" class="px-2 py-1 font-bold text-blue-300 uppercase tracking-wider text-[10px] sticky left-0 bg-gray-750">${group.name}</td></tr>`;

            group.rows.forEach(row => {
                html += `<tr class="hover:bg-gray-700/50">`;
                html += `<td class="px-2 py-1 text-gray-300 font-medium sticky left-0 bg-gray-800 hover:bg-gray-700/50 z-10 border-r border-gray-700">${row.label}</td>`;

                columns.forEach(col => {
                    const colData = stats[col];
                    let valStr = '-';
                    let cellClass = "";

                    if (colData) {
                        valStr = fmt(colData[row.key], row.type);

                        // Anomaly Highlighting
                        if (col === "Last 5") {
                            const isHigh = anomalies.some(a => a.includes(row.label) && (a.includes("High") || a.includes("Spike")));
                            const isLow = anomalies.some(a => a.includes(row.label) && a.includes("Low"));

                            if (isHigh) cellClass = "bg-green-900/50 text-green-200 font-bold border border-green-700";
                            if (isLow) cellClass = "bg-red-900/50 text-red-200 font-bold border border-red-700";
                        }
                    }

                    html += `<td class="px-2 py-1 text-center text-gray-400 ${cellClass}">${valStr}</td>`;
                });
                html += `</tr>`;
            });
        });

        html += `</tbody></table></div>`;
        html += `<div class="mt-2 text-[10px] text-gray-500 italic text-right">* All values normalized to Per 5 Games avg</div>`;

        document.getElementById('trending-modal-title').textContent = `${player.player_name} - Trend Analysis`;
        contentDiv.innerHTML = html;
        modal.classList.remove('hidden');
    };

    // --- [START] AUTOMATED UPDATE LOGIC ---
    if (window.autoUpdateInfo) {
        const info = window.autoUpdateInfo;

        if (info.status === 'queued') {
            console.log("Automated update queued: " + info.job_id);

            // Directly start the stream.
            // The function above will decide whether to use the Page Box or Floating Box.
            if (typeof startLogStream === 'function') {
                startLogStream(info.job_id);
            }

        } else if (info.status === 'fresh') {
            console.log("Data is fresh. No update needed.");
        }
    }
    // --- [END] AUTOMATED UPDATE LOGIC ---

    // --- Event Delegation for Raw Data Toggle ---
    document.addEventListener('change', (e) => {
        if (e.target && e.target.id === 'global-show-raw-data') {
            const val = e.target.checked;
            localStorage.setItem('showRawData', val);
            window.dispatchEvent(new CustomEvent('rawDataToggled', { detail: { showRaw: val } }));
        }
    });

    // --- Global Cat Rank Modal Logic ---
    document.addEventListener('click', (e) => {
        const modal = document.getElementById('global-cat-rank-modal');
        if (!modal) return;
        if (e.target.closest('#global-modal-close') || e.target === modal) {
            modal.classList.add('hidden');
        }
    });

    // Global function to open modal
    window.openCatRankModal = function(playerObj, categories) {
        const modal = document.getElementById('global-cat-rank-modal');
        const modalContent = document.getElementById('global-modal-content');
        const modalTitle = document.getElementById('global-modal-title');

        if (!modal || !modalContent) return;

        const showingRawMain = localStorage.getItem('showRawData') === 'true';
        const showRanksInModal = showingRawMain; // Inverse logic

        modalTitle.textContent = `${playerObj.player_name || 'Player'} - ${showRanksInModal ? 'Category Ranks' : 'Raw Stats'}`;

        let html = `<table class="w-full text-sm text-left text-gray-300">
            <thead class="text-xs text-gray-400 uppercase bg-gray-700"><tr><th class="px-3 py-2">Category</th><th class="px-3 py-2 text-right">${showRanksInModal ? 'Rank' : 'Value'}</th></tr></thead>
            <tbody class="divide-y divide-gray-700">`;

        categories.forEach(cat => {
            let displayVal = '-';
            let style = '';

            if (showRanksInModal) {
                const rank = playerObj[cat + '_cat_rank'];
                displayVal = (rank != null) ? Math.round(rank) : '-';
                if (rank && rank <= 5) style = 'color: #4ade80; font-weight: bold;';
                else if (rank && rank >= 15) style = 'color: #f87171;';
            } else {
                const raw = playerObj[cat];
                displayVal = (raw != null && !isNaN(raw)) ? parseFloat(raw).toFixed(2).replace(/[.,]00$/, "") : (raw || '-');
            }

            html += `<tr><td class="px-3 py-2 font-medium">${cat}</td><td class="px-3 py-2 text-right" style="${style}">${displayVal}</td></tr>`;
        });

        html += `</tbody></table>`;
        modalContent.innerHTML = html;
        modal.classList.remove('hidden');
    };

    let pageData = null;

    async function handleLogout() {
        window.location.href = '/logout';
    }

    async function getTimestamp() {
        try {
            const response = await fetch('/api/db_timestamp');
            const data = await response.json();
            if (response.ok && data.timestamp) {
                timestampText.textContent = `League data exists, check League Database tab for last update`;
            } else {
                timestampText.textContent = 'League data has not been updated yet.';
            }
        } catch (error) {
            console.error('Error setting timestamp:', error);
            timestampText.textContent = 'Error loading league data status.';
        }
    }

    async function initDropdowns() {
        try {
            // 1. Fetch User's Leagues
            const leagueResp = await fetch('/api/my_leagues');
            const leagueData = await leagueResp.json();

            // 2. Fetch Matchup Data (This tells us if DB exists)
            const response = await fetch('/api/matchup_page_data');
            const data = await response.json();

            // --- [START] NEW: Manage Navigation Visibility ---
            const navButtons = document.querySelectorAll('.toggle-btn');
            const dbExists = response.ok && data.db_exists;

            navButtons.forEach(btn => {
                const page = btn.getAttribute('data-page');
                if (!dbExists) {
                    // DB Missing: Hide everything except 'league-database'
                    if (page !== 'league-database') {
                        btn.classList.add('hidden');
                    } else {
                        btn.classList.remove('hidden');
                        // Force click if we are currently on a hidden page (optional safety)
                    }
                } else {
                    // DB Exists: Show everything
                    btn.classList.remove('hidden');
                }
            });

            // HTML Structure
            dropdownContainer.innerHTML = `
                <div class="flex flex-col gap-2">
                    <div class="flex items-center gap-2 justify-end">
                        <label for="league-select" class="text-sm font-bold text-blue-400 w-24 text-right">League:</label>
                        <select id="league-select" class="bg-gray-800 border border-blue-500 text-white text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-48 p-2 font-bold">
                            ${leagueOptions}
                        </select>
                    </div>
                    <div class="flex items-center gap-2 justify-end">
                        <label for="week-select" class="text-sm font-medium text-gray-300 w-24 text-right">Fantasy Week:</label>
                        <select id="week-select" class="bg-gray-700 border border-gray-600 text-white text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-48 p-2">
                            <option selected>Loading...</option>
                        </select>
                    </div>
                </div>

                <div class="flex flex-col gap-2">
                    <div class="flex items-center gap-2 justify-end">
                        <label for="your-team-select" class="text-sm font-medium text-gray-300 w-24 text-right">Your Team:</label>
                        <select id="your-team-select" class="bg-gray-700 border border-gray-600 text-white text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-48 p-2">
                            <option selected>Choose your team</option>
                        </select>
                    </div>

                     <div class="flex items-center gap-2 justify-end">
                        <label for="stat-sourcing-select" class="text-sm font-medium text-gray-300 w-24 text-right">Stat Sourcing:</label>
                        <select id="stat-sourcing-select" class="bg-gray-700 border border-gray-600 text-white text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-48 p-2">
                            <option value="projected">Projected ROS</option>
                            <option value="todate">Season To Date</option>
                            <option value="combined">Combined</option>
                        </select>
                    </div>
                </div>

                <div class="flex items-center gap-2 bg-gray-800 py-1.5 px-4 rounded-lg border border-gray-600 shadow-sm h-min self-center">
                    <input type="checkbox" id="global-show-raw-data" class="w-4 h-4 text-blue-600 bg-gray-700 border-gray-500 rounded focus:ring-blue-500 focus:ring-2 cursor-pointer">
                    <label for="global-show-raw-data" class="text-sm font-medium text-gray-300 cursor-pointer select-none">Show Raw Data</label>
                </div>
            `;

            dropdownContainer.className = 'flex items-start gap-6 bg-gray-900 p-3 rounded-lg border border-gray-700';

            // --- EVENT LISTENER FOR LEAGUE SWITCH ---
            document.getElementById('league-select').addEventListener('change', async (e) => {
                const newLeagueId = e.target.value;
                if(confirm("Switch league? This will reload the dashboard.")) {
                    const res = await fetch('/api/switch_league', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({ league_id: newLeagueId })
                    });
                    if(res.ok) {
                        // --- FIX: Clear stored week so the new league defaults to ITS current week ---
                        localStorage.removeItem('selectedWeek');
                        // Optional: Clear session flag to ensure a "fresh start" logic runs
                        sessionStorage.removeItem('fantasySessionStarted');

                        window.location.reload();
                    } else {
                        alert("Failed to switch league.");
                    }
                }
            });

            // Handle DB not existing case
            if (!response.ok || !data.db_exists) {
                // If DB missing, disable the other dropdowns but leave League enabled so they can switch out
                ['week-select', 'your-team-select', 'stat-sourcing-select'].forEach(id => {
                    document.getElementById(id).disabled = true;
                    document.getElementById(id).innerHTML = '<option>No DB Found</option>';
                });
                return;
            }

            // Populate the standard dropdowns
            pageData = data;
            populateDropdowns();
            populateStatSourcingDropdown();

            // Restore Checkbox State
            const showRawCheckbox = document.getElementById('global-show-raw-data');
            if (showRawCheckbox) {
                showRawCheckbox.checked = localStorage.getItem('showRawData') === 'true';
            }

            // Add standard listeners
            document.getElementById('week-select').addEventListener('change', (e) => { localStorage.setItem('selectedWeek', e.target.value); });
            document.getElementById('your-team-select').addEventListener('change', (e) => { localStorage.setItem('selectedTeam', e.target.value); });

            const statSourcingSelect = document.getElementById('stat-sourcing-select');
            statSourcingSelect.addEventListener('change', (e) => {
                localStorage.setItem('selectedStatSourcing', e.target.value);
                const recalculateBtn = document.getElementById('recalculate-button');
                if (recalculateBtn) recalculateBtn.click();
            });

        } catch (error) {
            console.error('Initialization error for dropdowns:', error.message);
        }
    }

    function populateDropdowns() {
        const weekSelect = document.getElementById('week-select');
        const yourTeamSelect = document.getElementById('your-team-select');

        // Populate options (existing code)
        weekSelect.innerHTML = pageData.weeks.map(week => `<option value="${week.week_num}">Week ${week.week_num} (${week.start_date} to ${week.end_date})</option>`).join('');
        yourTeamSelect.innerHTML = pageData.teams.map(team => `<option value="${team.name}">${team.name}</option>`).join('');

        const savedTeam = localStorage.getItem('selectedTeam');
        if (savedTeam) yourTeamSelect.value = savedTeam;

        // --- NEW LOGIC START: Handle Week Selection with "New Day" Reset ---
        const currentWeek = pageData.current_week;
        const todayStr = new Date().toDateString(); // e.g. "Mon Dec 11 2025"
        const lastVisitDate = localStorage.getItem('lastVisitDate');

        // Check 1: Is this a "New Day" visit? (Fixes the Sunday -> Monday issue)
        const isNewDay = lastVisitDate !== todayStr;

        // Check 2: Is this a completely fresh tab session?
        const isNewSession = !sessionStorage.getItem('fantasySessionStarted');

        if (isNewDay || isNewSession) {
            // Force update to Current Week
            weekSelect.value = currentWeek;
            localStorage.setItem('selectedWeek', currentWeek);

            // Mark session as started and update the visit date
            sessionStorage.setItem('fantasySessionStarted', 'true');
            localStorage.setItem('lastVisitDate', todayStr);

        } else {
            // Same day, same session: Allow user to stay on a previous week (e.g. if looking at history)
            const savedWeek = localStorage.getItem('selectedWeek');
            if (savedWeek) weekSelect.value = savedWeek;
            else weekSelect.value = currentWeek;
        }
        // --- NEW LOGIC END ---
    }

    function populateStatSourcingDropdown() {
        const statSourcingSelect = document.getElementById('stat-sourcing-select');
        if (!statSourcingSelect) return;

        const savedStatSourcing = localStorage.getItem('selectedStatSourcing');
        if (savedStatSourcing) {
            statSourcingSelect.value = savedStatSourcing;
        } else {
            statSourcingSelect.value = 'projected';
            localStorage.setItem('selectedStatSourcing', 'projected');
        }
    }

    if(logoutButton) logoutButton.addEventListener('click', handleLogout);
    if(timestampText) getTimestamp();

    initDropdowns();
});

// [START] LOG STREAMING FUNCTION (Required for automated updates)
function startLogStream(buildId) {
    // 1. Try to find the standard log container (on League Database page)
    const onPageLog = document.getElementById('log-container');
    let logOutput = null;

    // Check if the element exists AND is currently part of the visible DOM
    if (onPageLog && document.body.contains(onPageLog)) {
        // CASE A: We are on the League Database page.
        // Make sure the existing box is visible and use it.
        onPageLog.classList.remove('hidden');
        logOutput = onPageLog;

        // Clear previous content if restarting
        if (!logOutput.hasAttribute('data-stream-active')) {
            logOutput.innerHTML = '';
            logOutput.setAttribute('data-stream-active', 'true');
        }
    } else {
        // CASE B: We are on another page (Home, Matchups, etc.) OR the tab hasn't loaded yet.
        // Create/Use the Floating Status Box.

        // Check if we already created it (prevent duplicates)
        const existingFloating = document.getElementById('floating-log-wrapper');

        if (!existingFloating) {
            const floatingHTML = `
            <div id="floating-log-wrapper" class="fixed bottom-6 right-6 w-80 bg-gray-800 border border-blue-500 rounded-lg shadow-2xl z-50 flex flex-col font-mono text-xs overflow-hidden animate-pulse-border">
                <div class="bg-blue-900/80 p-3 border-b border-blue-500 flex justify-between items-center">
                    <div class="flex items-center gap-2">
                        <div class="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                        <span class="text-white font-bold tracking-wide">UPDATING LEAGUE DATA...</span>
                    </div>
                </div>
                <div id="floating-log-content" class="p-3 h-32 overflow-y-auto text-gray-300 space-y-1 bg-gray-900/90">
                    <div class="text-blue-400">➤ Initializing background update...</div>
                </div>
            </div>`;
            document.body.insertAdjacentHTML('beforeend', floatingHTML);
        }

        // Target the inner content area of the floating box
        logOutput = document.getElementById('floating-log-content');
    }

    if (!logOutput) {
        console.error("Could not initialize any log output container.");
        return;
    }

    // Connect to the Flask EventSource route
    const eventSource = new EventSource(`/api/db_log_stream?build_id=${buildId}`);

    eventSource.onmessage = function(event) {
        const message = event.data;

        // Handle "Done" signal
        if (message === '__DONE__') {
            eventSource.close();

            // UI Feedback
            const doneMsg = '<div class="text-green-400 font-bold mt-2 border-t border-gray-700 pt-2">➤ Update Complete. Refreshing...</div>';
            logOutput.innerHTML += doneMsg;
            logOutput.scrollTop = logOutput.scrollHeight;

            // Refresh the CURRENT page after a short delay
            setTimeout(() => {
                window.location.reload();
            }, 1000);

        }
        // Handle Errors
        else if (message.startsWith('ERROR:')) {
            logOutput.innerHTML += `<div class="text-red-500 font-bold">➤ ${message}</div>`;
            eventSource.close();
        }
        // Handle Normal Logs
        else {
            logOutput.innerHTML += `<div>➤ ${message}</div>`;
            // Auto-scroll to bottom
            logOutput.scrollTop = logOutput.scrollHeight;
        }
    };

    eventSource.onerror = function() {
        logOutput.innerHTML += '<div class="text-red-400 pt-2">➤ Connection lost (Stream closed).</div>';
        eventSource.close();
    };
}
// [END] LOG STREAMING FUNCTION
