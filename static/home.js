document.addEventListener('DOMContentLoaded', () => {
    const logoutButton = document.getElementById('logout-button');
    const timestampText = document.getElementById('timestamp-text');
    const dropdownContainer = document.getElementById('dropdown-container');

    // --- [START] AUTOMATED UPDATE LOGIC ---
    if (window.autoUpdateInfo) {
        const info = window.autoUpdateInfo;

        if (info.status === 'queued') {
            console.log("Automated update queued: " + info.job_id);

            // 1. Find the "League Database" button
            const leagueDbBtn = document.querySelector('button[data-page="league-database"]');

            if (leagueDbBtn) {
                // 2. Programmatically click it to load the page content
                leagueDbBtn.click();

                // 3. Wait for the 'db-log-output' box to appear in the DOM
                // (We check every 100ms because the page load is async)
                const waitForLogBox = setInterval(() => {
                    const staticLogBox = document.getElementById('db-log-output');

                    if (staticLogBox) {
                        clearInterval(waitForLogBox); // Stop checking

                        // 4. Insert Initial Message
                        staticLogBox.innerHTML = `<div class="text-blue-400">➤ ${info.message}</div>`;
                        staticLogBox.innerHTML += `<div class="text-gray-400">➤ Connecting to live stream...</div>`;

                        // 5. Connect to the stream
                        if (typeof startLogStream === 'function') {
                            startLogStream(info.job_id);
                        }
                    }
                }, 100);
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
            const response = await fetch('/api/matchup_page_data');
            const data = await response.json();

            if (!response.ok || !data.db_exists) {
                dropdownContainer.innerHTML = `<button id="reload-dropdowns" class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">Create DB then press to load</button>`;
                document.getElementById('reload-dropdowns').addEventListener('click', initDropdowns);
                return;
            }

            // --- FIX: REORGANIZED LAYOUT (Stacked) ---
            dropdownContainer.innerHTML = `
                <div class="flex flex-col gap-2">
                    <div class="flex items-center gap-2 justify-end">
                        <label for="week-select" class="text-sm font-medium text-gray-300 w-24 text-right">Fantasy Week:</label>
                        <select id="week-select" class="bg-gray-700 border border-gray-600 text-white text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-48 p-2">
                            <option selected>Choose a week</option>
                        </select>
                    </div>
                    <div class="flex items-center gap-2 justify-end">
                        <label for="your-team-select" class="text-sm font-medium text-gray-300 w-24 text-right">Your Team:</label>
                        <select id="your-team-select" class="bg-gray-700 border border-gray-600 text-white text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-48 p-2">
                            <option selected>Choose your team</option>
                        </select>
                    </div>
                </div>

                <div class="flex flex-col gap-2">
                    <div class="flex items-center gap-2 justify-end">
                        <label for="stat-sourcing-select" class="text-sm font-medium text-gray-300 w-24 text-right">Stat Sourcing:</label>
                        <select id="stat-sourcing-select" class="bg-gray-700 border border-gray-600 text-white text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-48 p-2">
                            <option value="projected">Projected ROS</option>
                            <option value="todate">Season To Date</option>
                            <option value="combined">Combined</option>
                        </select>
                    </div>
                    <div class="flex justify-end">
                        <div class="flex items-center gap-2 bg-gray-800 py-1.5 px-4 rounded-lg border border-gray-600 shadow-sm w-48 justify-center">
                            <input type="checkbox" id="global-show-raw-data" class="w-4 h-4 text-blue-600 bg-gray-700 border-gray-500 rounded focus:ring-blue-500 focus:ring-2 cursor-pointer">
                            <label for="global-show-raw-data" class="text-sm font-medium text-gray-300 cursor-pointer select-none">Show Raw Data</label>
                        </div>
                    </div>
                </div>
            `;
            dropdownContainer.className = 'flex items-start gap-6'; // Ensure flex and spacing

            pageData = data;
            populateDropdowns();
            populateStatSourcingDropdown();

            // Restore Checkbox State
            const showRawCheckbox = document.getElementById('global-show-raw-data');
            if (showRawCheckbox) {
                showRawCheckbox.checked = localStorage.getItem('showRawData') === 'true';
            }

            document.getElementById('week-select').addEventListener('change', (e) => { localStorage.setItem('selectedWeek', e.target.value); });
            document.getElementById('your-team-select').addEventListener('change', (e) => { localStorage.setItem('selectedTeam', e.target.value); });

            const statSourcingSelect = document.getElementById('stat-sourcing-select');
            statSourcingSelect.addEventListener('change', (e) => {
                localStorage.setItem('selectedStatSourcing', e.target.value);
                const recalculateBtn = document.getElementById('recalculate-button');
                if (recalculateBtn) recalculateBtn.click();
                else {
                    const activeTab = document.querySelector('.toggle-btn.bg-blue-600');
                    if (activeTab) activeTab.click();
                }
            });

        } catch (error) {
            console.error('Initialization error for dropdowns:', error.message);
        }
    }

    function populateDropdowns() {
        const weekSelect = document.getElementById('week-select');
        const yourTeamSelect = document.getElementById('your-team-select');

        weekSelect.innerHTML = pageData.weeks.map(week => `<option value="${week.week_num}">Week ${week.week_num} (${week.start_date} to ${week.end_date})</option>`).join('');
        yourTeamSelect.innerHTML = pageData.teams.map(team => `<option value="${team.name}">${team.name}</option>`).join('');

        const savedTeam = localStorage.getItem('selectedTeam');
        if (savedTeam) yourTeamSelect.value = savedTeam;

        if (!sessionStorage.getItem('fantasySessionStarted')) {
            const currentWeek = pageData.current_week;
            weekSelect.value = currentWeek;
            localStorage.setItem('selectedWeek', currentWeek);
            sessionStorage.setItem('fantasySessionStarted', 'true');
        } else {
            const savedWeek = localStorage.getItem('selectedWeek');
            if (savedWeek) weekSelect.value = savedWeek;
            else weekSelect.value = pageData.current_week;
        }
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
    const logOutput = document.getElementById('db-log-output');
    if (!logOutput) return;

    // Connect to the Flask EventSource route
    const eventSource = new EventSource(`/api/db_log_stream?build_id=${buildId}`);

    eventSource.onmessage = function(event) {
        const message = event.data;

        // Handle "Done" signal
        if (message === '__DONE__') {
            eventSource.close();
            logOutput.innerHTML += '<div class="text-green-400 font-bold mt-2">➤ Update Complete.</div>';

            // Refresh the page after a short delay
            setTimeout(() => {
                window.location.reload();
            }, 1500);

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
        logOutput.innerHTML += '<div class="text-red-400">➤ Connection lost (Stream closed).</div>';
        eventSource.close();
    };
}
// [END] LOG STREAMING FUNCTION
