(async function() {
    // A short delay to ensure the page elements are in the DOM
    await new Promise(resolve => setTimeout(resolve, 0));

    // --- Page Elements ---
    const errorDiv = document.getElementById('db-error-message');
    const controlsDiv = document.getElementById('goalie-controls');

    // Dropdowns
    const weekSelect = document.getElementById('week-select');
    const yourTeamSelect = document.getElementById('your-team-select');
    const opponentSelect = document.getElementById('opponent-select');

    // Containers
    const currentStatsContainer = document.getElementById('current-stats-container');
    const simulatedStatsContainer = document.getElementById('simulated-stats-container');
    const opponentStatsContainer = document.getElementById('opponent-stats-container');
    const individualStartsContainer = document.getElementById('individual-starts-container');

    // --- Global State ---
    let pageData = null;        // All teams, weeks, matchups
    let baseStarts = [];        // The original starts fetched from the server
    let simulatedStarts = [];   // The scenarios the user has checked
    let baseTotals = {};        // "Frozen" totals for Your Team
    let opponentTotals = {};    // "Frozen" totals for Opponent
    let yourTeamName = "";      // Your team's name
    let opponentTeamName = "";  // Opponent's name

    // --- Constants ---
    const SCENARIOS = [
        { name: "Shutout",       w: 1, ga: 0, sv: 30, sa: 30, toi: 60, sho: 1 },
        { name: "1GA",           w: 1, ga: 1, sv: 29, sa: 30, toi: 60, sho: 0 },
        { name: "2GA",           w: 1, ga: 2, sv: 28, sa: 30, toi: 60, sho: 0 },
        { name: "3GA",           w: 0, ga: 3, sv: 27, sa: 30, toi: 60, sho: 0 },
        { name: "4GA",           w: 0, ga: 4, sv: 26, sa: 30, toi: 60, sho: 0 },
        { name: "5GA",           w: 0, ga: 5, sv: 25, sa: 30, toi: 60, sho: 0 },
        { name: "6GA",           w: 0, ga: 6, sv: 24, sa: 30, toi: 60, sho: 0 },
        { name: "4/GA/Pulled",   w: 0, ga: 4, sv: 11, sa: 15, toi: 20, sho: 0 }
    ];

    const COLORS = {
        win: 'bg-green-800/50',
        loss: 'bg-red-800/50',
        tie: '' // No background for a tie
    };

    async function init() {
        try {
            // Add event listener for checkbox clicks
            individualStartsContainer.addEventListener('click', handleCheckboxClick);

            await fetchPageData();
            populateDropdowns();
            setupEventListeners();
            updateOpponentDropdown();
            await fetchAndRenderStats();

            controlsDiv.classList.remove('hidden');

        } catch (error) {
            console.error('Initialization error:', error);
            errorDiv.textContent = error.message;
            errorDiv.classList.remove('hidden');
            controlsDiv.classList.add('hidden');
        }
    }

    async function fetchPageData() {
        const response = await fetch('/api/matchup_page_data');
        const data = await response.json();
        if (!response.ok || !data.db_exists) {
            throw new Error(data.error || 'Database has not been initialized.');
        }
        pageData = data;
    }

    function populateDropdowns() {
        weekSelect.innerHTML = pageData.weeks.map(week =>
            `<option value="${week.week_num}">
                Week ${week.week_num} (${week.start_date} to ${week.end_date})
            </option>`
        ).join('');

        const teamOptions = pageData.teams.map(team =>
            `<option value="${team.name}">${team.name}</option>`
        ).join('');
        yourTeamSelect.innerHTML = teamOptions;
        opponentSelect.innerHTML = teamOptions;

        const savedTeam = localStorage.getItem('selectedTeam');
        if (savedTeam) {
            yourTeamSelect.value = savedTeam;
        }

        if (!sessionStorage.getItem('fantasySessionStarted')) {
            const currentWeek = pageData.current_week;
            weekSelect.value = currentWeek;
            localStorage.setItem('selectedWeek', currentWeek);
            sessionStorage.setItem('fantasySessionStarted', 'true');
        } else {
            const savedWeek = localStorage.getItem('selectedWeek');
            weekSelect.value = savedWeek ? savedWeek : pageData.current_week;
        }
    }

    function setupEventListeners() {
        weekSelect.addEventListener('change', async () => {
            localStorage.setItem('selectedWeek', weekSelect.value);
            updateOpponentDropdown();
            await fetchAndRenderStats();
        });
        yourTeamSelect.addEventListener('change', async () => {
            localStorage.setItem('selectedTeam', yourTeamSelect.value);
            updateOpponentDropdown();
            await fetchAndRenderStats();
        });
        opponentSelect.addEventListener('change', async () => {
            await fetchAndRenderStats();
        });
    }

    function updateOpponentDropdown() {
        const selectedWeek = weekSelect.value;
        const yourTeamName = yourTeamSelect.value;

        const matchup = pageData.matchups.find(m =>
            m.week == selectedWeek && (m.team1 === yourTeamName || m.team2 === yourTeamName)
        );

        if (matchup) {
            const opponentName = matchup.team1 === yourTeamName ? matchup.team2 : matchup.team1;
            opponentSelect.value = opponentName;
        } else {
            const firstOtherTeam = pageData.teams.find(t => t.name !== yourTeamName);
            if (firstOtherTeam) {
                opponentSelect.value = firstOtherTeam.name;
            }
        }
    }

    async function fetchAndRenderStats() {
        yourTeamName = yourTeamSelect.value;
        opponentTeamName = opponentSelect.value;
        const selectedWeek = weekSelect.value;

        if (!selectedWeek || !yourTeamName || !opponentTeamName) {
            currentStatsContainer.innerHTML = '<p class="text-gray-400">Please make all selections.</p>';
            return;
        }

        currentStatsContainer.innerHTML = '<p class="text-gray-400">Loading...</p>';
        simulatedStatsContainer.innerHTML = '<p class="text-gray-400">Loading...</p>';
        opponentStatsContainer.innerHTML = '<p class="text-gray-400">Loading...</p>';
        individualStartsContainer.innerHTML = '';

        try {
            const response = await fetch('/api/goalie_planning_stats', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    week: selectedWeek,
                    your_team_name: yourTeamName,
                    opponent_team_name: opponentTeamName
                })
            });

            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Failed to fetch stats.');

            baseStarts = data.your_team_stats.individual_starts || [];
            baseTotals = calculateTotals(baseStarts);
            opponentTotals = calculateTotals(data.opponent_team_stats.individual_starts || []);
            simulatedStarts = [];

            renderAllTables();

        } catch (error) {
            console.error('Error fetching stats:', error);
            currentStatsContainer.innerHTML = `<p class="text-red-400">Error: ${error.message}</p>`;
            simulatedStatsContainer.innerHTML = '';
            opponentStatsContainer.innerHTML = '';
        }
    }

    function renderAllTables() {
        const allStarts = [...baseStarts, ...simulatedStarts];
        const simulatedTotals = calculateTotals(allStarts);

        renderAggregateStatsTable(currentStatsContainer, `Current Stats (${yourTeamName})`, baseTotals, opponentTotals);
        renderAggregateStatsTable(simulatedStatsContainer, `Simulated Stats (${yourTeamName})`, simulatedTotals, opponentTotals, true);
        renderAggregateStatsTable(opponentStatsContainer, `Opponent Stats (${opponentTeamName})`, opponentTotals, null, false);

        renderIndividualStartsTable(allStarts, simulatedTotals);
    }

    function calculateTotals(starts) {
        let totalW = 0, totalGA = 0, totalSV = 0, totalSA = 0, totalSHO = 0, totalTOI = 0;

        starts.forEach(start => {
            totalW += (start.W || start.w || 0);
            totalGA += (start.GA || start.ga || 0);
            totalSV += (start.SV || start.sv || 0);
            totalSA += (start.SA || start.sa || 0);
            totalSHO += (start.SHO || start.sho || 0);
            totalTOI += (start['TOI/G'] || start.toi || 0);
        });

        const totalGAA = totalTOI > 0 ? (totalGA * 60) / totalTOI : 0;
        const totalSVpct = totalSA > 0 ? totalSV / totalSA : 0;

        return {
            starts: starts.length,
            W: totalW,
            GA: totalGA,
            SV: totalSV,
            SA: totalSA,
            SHO: totalSHO,
            TOI: totalTOI,
            GAA: totalGAA,
            SVpct: totalSVpct
        };
    }

    function getComparisonClass(value, opponentValue, lowerIsBetter = false) {
        if (lowerIsBetter) {
            if (value < opponentValue - 1e-9) return COLORS.win;
            if (value > opponentValue + 1e-9) return COLORS.loss;
        } else {
            if (value > opponentValue + 1e-9) return COLORS.win;
            if (value < opponentValue - 1e-9) return COLORS.loss;
        }
        return COLORS.tie;
    }

    function renderAggregateStatsTable(container, title, totals, opponentTotals = null, isSimulated = false) {
        const titleClass = isSimulated ? "text-blue-300" : "text-white";
        const shadowClass = isSimulated ? "shadow-blue-500/30 shadow-lg" : "shadow";

        const leagueCats = pageData.scoring_categories.map(c => c.category);
        const hasGaa = leagueCats.includes('GAA');
        const hasSvPct = leagueCats.includes('SVpct');

        // Init colors
        let wClass = COLORS.tie, gaaClass = COLORS.tie, gaClass = COLORS.tie;
        let svPctClass = COLORS.tie, svClass = COLORS.tie, shoClass = COLORS.tie;

        if (opponentTotals) {
            wClass = getComparisonClass(totals.W, opponentTotals.W, false);
            shoClass = getComparisonClass(totals.SHO, opponentTotals.SHO, false);

            // GAA comparison
            if (hasGaa) {
                const myGaa = totals.TOI > 0 ? totals.GAA : Infinity;
                const oppGaa = opponentTotals.TOI > 0 ? opponentTotals.GAA : Infinity;
                if (myGaa === Infinity && oppGaa === Infinity) gaaClass = COLORS.tie;
                else gaaClass = getComparisonClass(myGaa, oppGaa, true);
            }

            // GA comparison (Lower is better)
            gaClass = getComparisonClass(totals.GA, opponentTotals.GA, true);

            // SV% comparison
            if (hasSvPct) {
                svPctClass = getComparisonClass(totals.SVpct, opponentTotals.SVpct, false);
            }

            // SV comparison (Higher is better)
            svClass = getComparisonClass(totals.SV, opponentTotals.SV, false);
        }

        // --- STYLES ---
        // Main stat style: "px-3 py-2 whitespace-nowrap text-sm font-medium text-gray-300"
        // Sub stat style:  "px-3 py-2 whitespace-nowrap text-sm font-normal text-gray-400 pl-6"
        // Value style (Main): "px-3 py-2 whitespace-nowrap text-sm text-right" (Color comes from row class)
        // Value style (Sub):  "px-3 py-2 whitespace-nowrap text-sm text-gray-400 text-right"

        // Helper to generate row HTML based on "is main stat" status
        const renderRow = (label, value, colorClass, isMain, subLabel = label) => {
            const labelClass = isMain
                ? "px-3 py-2 whitespace-nowrap text-sm font-medium text-gray-300"
                : "px-3 py-2 whitespace-nowrap text-sm font-normal text-gray-400 pl-6";

            const valueClass = isMain
                ? "px-3 py-2 whitespace-nowrap text-sm text-right"
                : "px-3 py-2 whitespace-nowrap text-sm text-gray-400 text-right";

            return `
                <tr class="hover:bg-gray-700/50 ${colorClass}">
                    <td class="${labelClass}">${isMain ? label : subLabel}</td>
                    <td class="${valueClass}">${value}</td>
                </tr>
            `;
        };

        let rows = "";

        // 1. Starts (Always Main)
        rows += `
            <tr class="hover:bg-gray-700/50">
                <td class="px-3 py-2 whitespace-nowrap text-sm font-medium text-gray-300">Goalie Starts</td>
                <td class="px-3 py-2 whitespace-nowrap text-sm text-gray-300 text-right font-bold">${totals.starts}</td>
            </tr>`;

        // 2. Wins (Always Main)
        rows += renderRow("Wins (W)", totals.W.toFixed(0), wClass, true);

        // 3. GAA Block
        if (hasGaa) {
            // GAA is Main, GA/TOI are Sub
            rows += renderRow("Goals Against Avg (GAA)", (totals.TOI > 0 ? totals.GAA : 0).toFixed(3), gaaClass, true);
            rows += renderRow("Goals Against (GA)", totals.GA.toFixed(0), "", false, "Goals Against (GA)");
            rows += renderRow("Time on Ice (TOI)", totals.TOI.toFixed(1), "", false, "Time on Ice (TOI)");
        } else {
            // GA is Main, TOI is Sub (informational)
            rows += renderRow("Goals Against (GA)", totals.GA.toFixed(0), gaClass, true);
            rows += renderRow("Time on Ice (TOI)", totals.TOI.toFixed(1), "", false, "Time on Ice (TOI)");
        }

        // 4. SV% Block
        if (hasSvPct) {
            // SV% is Main, SV/SA are Sub
            rows += renderRow("Save Pct (SV%)", totals.SVpct.toFixed(3), svPctClass, true);
            rows += renderRow("Saves (SV)", totals.SV.toFixed(0), "", false, "Saves (SV)");
            rows += renderRow("Shots Against (SA)", totals.SA.toFixed(0), "", false, "Shots Against (SA)");
        } else {
            // SV is Main, SA is Sub (informational)
            rows += renderRow("Saves (SV)", totals.SV.toFixed(0), svClass, true);
            rows += renderRow("Shots Against (SA)", totals.SA.toFixed(0), "", false, "Shots Against (SA)");
        }

        // 5. Shutouts (Always Main)
        rows += renderRow("Shutouts (SHO)", totals.SHO.toFixed(0), shoClass, true);

        let tableHtml = `
            <div class="bg-gray-900 rounded-lg ${shadowClass}">
                <h3 class="text-lg font-bold ${titleClass} p-3 bg-gray-800 rounded-t-lg">
                    ${title}
                </h3>
                <table class="w-full divide-y divide-gray-700">
                    <thead class="bg-gray-700/50">
                        <tr>
                            <th class="px-3 py-2 text-left text-xs font-bold text-gray-300 uppercase tracking-wider">Stat</th>
                            <th class="px-3 py-2 text-right text-xs font-bold text-gray-300 uppercase tracking-wider">Value</th>
                        </tr>
                    </thead>
                    <tbody class="bg-gray-800 divide-y divide-gray-700">
                        ${rows}
                    </tbody>
                </table>
            </div>
        `;
        container.innerHTML = tableHtml;
    }

    function renderIndividualStartsTable(allStarts, totals) {
        const leagueCats = pageData.scoring_categories.map(c => c.category);

        // Define Headers based on league settings
        let headers = ['Start #', 'Date', 'Player', 'W'];

        // GAA/GA Logic
        if (leagueCats.includes('GAA')) {
            headers.push('GA', 'GAA');
        } else {
            headers.push('GA');
        }

        // SV%/SV Logic
        if (leagueCats.includes('SVpct')) {
            headers.push('SV', 'SA', 'SV%');
        } else {
            headers.push('SV', 'SA');
        }

        headers.push('SHO');

        let tableHtml = `
            <div class="bg-gray-900 rounded-lg shadow">
                <h3 class="text-lg font-bold text-white p-3 bg-gray-800 rounded-t-lg">
                    Individual Goalie Starts
                </h3>
                <div class="overflow-x-auto">
                    <table class="w-full divide-y divide-gray-700">
                        <thead class="bg-gray-700/50">
                            <tr>
                                ${headers.map(h => `<th class="px-3 py-2 text-left text-xs font-bold text-gray-300 uppercase tracking-wider">${h}</th>`).join('')}
                            </tr>
                        </thead>
                        <tbody class="bg-gray-800 divide-y divide-gray-700">
        `;

        const renderRow = (start, index, isSim = false) => {
            const rowClass = isSim ? 'hover:bg-gray-700/50 bg-blue-900/30' : 'hover:bg-gray-700/50';
            const numCell = isSim
                ? `<td class="px-3 py-2 whitespace-nowrap text-sm text-gray-300">
                    <label class="flex items-center">
                        <input type="checkbox" class="sim-checkbox form-checkbox bg-gray-800 border-gray-600 rounded" data-sim-index="${index}" checked />
                        <span class="ml-2">Remove</span>
                    </label>
                   </td>`
                : `<td class="px-3 py-2 whitespace-nowrap text-sm font-medium text-gray-300">${index + 1}</td>`;

            const name = isSim ? start.name : start.player_name;
            const w = isSim ? start.w : start.W;
            const ga = isSim ? start.ga : start.GA;
            const sv = isSim ? start.sv : start.SV;
            const sa = isSim ? start.sa : start.SA;
            const sho = isSim ? start.sho : start.SHO;

            // Calc stats if sim, or use provided if base
            let svpct, gaa;
            if (isSim) {
                svpct = start.sa > 0 ? start.sv / start.sa : 0;
                gaa = start.toi > 0 ? (start.ga * 60) / start.toi : 0;
            } else {
                svpct = start['SV%'] || 0;
                gaa = start.GAA || 0;
            }

            let html = `<tr class="${rowClass}">
                <td class="px-3 py-2 whitespace-nowrap text-sm font-medium text-gray-300">${isSim ? (baseStarts.length + index + 1) : (index + 1)}</td>
                ${isSim ? numCell.replace(index + 1, '') : `<td class="px-3 py-2 whitespace-nowrap text-sm text-gray-300">${start.date}</td>`}
                <td class="px-3 py-2 whitespace-nowrap text-sm font-medium text-gray-300">${name}</td>
                <td class="px-3 py-2 whitespace-nowrap text-sm text-gray-300">${(w || 0).toFixed(0)}</td>
                <td class="px-3 py-2 whitespace-nowrap text-sm text-gray-300">${(ga || 0).toFixed(0)}</td>`;

            if (leagueCats.includes('GAA')) {
                html += `<td class="px-3 py-2 whitespace-nowrap text-sm text-gray-300">${gaa.toFixed(3)}</td>`;
            }

            html += `
                <td class="px-3 py-2 whitespace-nowrap text-sm text-gray-300">${(sv || 0).toFixed(0)}</td>
                <td class="px-3 py-2 whitespace-nowrap text-sm text-gray-300">${(sa || 0).toFixed(0)}</td>`;

            if (leagueCats.includes('SVpct')) {
                html += `<td class="px-3 py-2 whitespace-nowrap text-sm text-gray-300">${svpct.toFixed(3)}</td>`;
            }

            html += `<td class="px-3 py-2 whitespace-nowrap text-sm text-gray-300">${(sho || 0).toFixed(0)}</td></tr>`;
            return html;
        };

        baseStarts.forEach((start, index) => {
            tableHtml += renderRow(start, index, false);
        });

        simulatedStarts.forEach((sim, index) => {
            tableHtml += renderRow(sim, index, true);
        });

        // Totals Row
        tableHtml += `
            <tr class="bg-gray-700/50 border-t-2 border-gray-500">
                <td class="px-3 py-2 whitespace-nowrap text-sm font-bold text-white">${totals.starts}</td>
                <td class="px-3 py-2 whitespace-nowrap text-sm font-bold text-white"></td>
                <td class="px-3 py-2 whitespace-nowrap text-sm font-bold text-white">TOTALS</td>
                <td class="px-3 py-2 whitespace-nowrap text-sm font-bold text-white">${totals.W.toFixed(0)}</td>
                <td class="px-3 py-2 whitespace-nowrap text-sm font-bold text-white">${totals.GA.toFixed(0)}</td>`;

        if (leagueCats.includes('GAA')) {
            tableHtml += `<td class="px-3 py-2 whitespace-nowrap text-sm font-bold text-white">${(totals.TOI > 0 ? totals.GAA : 0).toFixed(3)}</td>`;
        }

        tableHtml += `
                <td class="px-3 py-2 whitespace-nowrap text-sm font-bold text-white">${totals.SV.toFixed(0)}</td>
                <td class="px-3 py-2 whitespace-nowrap text-sm font-bold text-white">${totals.SA.toFixed(0)}</td>`;

        if (leagueCats.includes('SVpct')) {
            tableHtml += `<td class="px-3 py-2 whitespace-nowrap text-sm font-bold text-white">${(totals.TOI > 0 ? totals.SVpct : 0).toFixed(3)}</td>`;
        }

        tableHtml += `<td class="px-3 py-2 whitespace-nowrap text-sm font-bold text-white">${totals.SHO.toFixed(0)}</td></tr>`;

        // Scenarios
        const nextStartNum = allStarts.length + 1;
        SCENARIOS.forEach(scenario => {
            const newW = totals.W + scenario.w;
            const newGA = totals.GA + scenario.ga;
            const newSV = totals.SV + scenario.sv;
            const newSA = totals.SA + scenario.sa;
            const newSHO = totals.SHO + scenario.sho;
            const newTOI = totals.TOI + scenario.toi;
            const newGAA = newTOI > 0 ? (newGA * 60) / newTOI : 0;
            const newSVpct = newSA > 0 ? newSV / newSA : 0;

            tableHtml += `
                <tr class="hover:bg-gray-700/50 text-gray-400 italic">
                    <td class="px-3 py-2 whitespace-nowrap text-sm">${nextStartNum}</td>
                    <td class="px-3 py-2 whitespace-nowrap text-sm">
                        <label class="flex items-center">
                            <input type="checkbox" class="scenario-checkbox form-checkbox bg-gray-800 border-gray-600 rounded" data-scenario-name="${scenario.name}" />
                            <span class="ml-2">Use</span>
                        </label>
                    </td>
                    <td class="px-3 py-2 whitespace-nowrap text-sm font-medium">${scenario.name}</td>
                    <td class="px-3 py-2 whitespace-nowrap text-sm">${newW.toFixed(0)}</td>
                    <td class="px-3 py-2 whitespace-nowrap text-sm">${newGA.toFixed(0)}</td>`;

            if (leagueCats.includes('GAA')) {
                tableHtml += `<td class="px-3 py-2 whitespace-nowrap text-sm">${newGAA.toFixed(3)}</td>`;
            }

            tableHtml += `
                    <td class="px-3 py-2 whitespace-nowrap text-sm">${newSV.toFixed(0)}</td>
                    <td class="px-3 py-2 whitespace-nowrap text-sm">${newSA.toFixed(0)}</td>`;

            if (leagueCats.includes('SVpct')) {
                tableHtml += `<td class="px-3 py-2 whitespace-nowrap text-sm">${newSVpct.toFixed(3)}</td>`;
            }

            tableHtml += `<td class="px-3 py-2 whitespace-nowrap text-sm">${newSHO.toFixed(0)}</td></tr>`;
        });

        tableHtml += `</tbody></table></div></div>`;
        individualStartsContainer.innerHTML = tableHtml;
    }

    function handleCheckboxClick(e) {
        const target = e.target;
        if (target.classList.contains('scenario-checkbox') && target.checked) {
            const scenarioName = target.dataset.scenarioName;
            const scenarioToAdd = SCENARIOS.find(s => s.name === scenarioName);
            if (scenarioToAdd) {
                simulatedStarts.push(scenarioToAdd);
                renderAllTables();
            }
        }
        if (target.classList.contains('sim-checkbox') && !target.checked) {
            const simIndex = parseInt(target.dataset.simIndex, 10);
            if (!isNaN(simIndex) && simIndex >= 0 && simIndex < simulatedStarts.length) {
                simulatedStarts.splice(simIndex, 1);
                renderAllTables();
            }
        }
    }

    init();
})();
