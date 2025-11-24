// This script will manage the league-database.html page
(async function() {
    await new Promise(resolve => setTimeout(resolve, 0));

    const statusText = document.getElementById('db-status-text');
    const actionButton = document.getElementById('db-action-button');
    const statUpdateContainer = document.getElementById('stat-update-container');
    const statUpdateCheckbox = document.getElementById('check-stat-updates');
    const logContainer = document.getElementById('log-container');

    if (!statusText || !actionButton || !statUpdateContainer || !statUpdateCheckbox || !logContainer) {
        console.error('Database page elements not found.');
        return;
    }

    let dbExists = false; // State to track DB status

    const updateStatus = (data) => {
        dbExists = data.db_exists; // Store state

        if (data.is_test_db) {
            statusText.innerHTML = `<strong>TEST MODE ACTIVE.</strong> All pages are reading from <span class="font-mono text-green-400">${data.league_name}</span>. <br>You can still use the button below to build or update a separate, live database.`;
            actionButton.textContent = 'Build/Update Live Database';
            // In test mode, we might act like DB exists for the purpose of the UI options
            statUpdateContainer.classList.remove('hidden');
            return;
        }

        if (data.db_exists) {
            const date = new Date(data.timestamp * 1000);
            statusText.textContent = `Your league: '${data.league_name}'s data is up to date as of: ${date.toLocaleString()}`;
            actionButton.textContent = 'Update Rosters';

            // Show the stat update option
            statUpdateContainer.classList.remove('hidden');
        } else {
            statusText.textContent = "Your league's data has not been initialized. Please initialize the database.";
            actionButton.textContent = 'Initialize Database';

            // Hide the stat update option (Initial build is always full)
            statUpdateContainer.classList.add('hidden');
            statUpdateCheckbox.checked = false;
        }
    };

    const fetchStatus = async () => {
        try {
            const response = await fetch('/api/db_status');
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Failed to fetch status');
            updateStatus(data);
        } catch (error) {
            console.error('Error fetching DB status:', error);
            statusText.textContent = `Could not determine database status. ${error.message}`;
            actionButton.textContent = 'Error';
        } finally {
            actionButton.disabled = false;
            actionButton.classList.remove('opacity-50', 'cursor-not-allowed');
        }
    };

    let eventSource = null;

    const connectToLogStream = (buildId) => {
        // Close any existing stream
        if (eventSource) {
            eventSource.close();
        }

        actionButton.textContent = 'Update in Progress...';

        eventSource = new EventSource(`/api/db_log_stream?build_id=${buildId}`);

        eventSource.onmessage = function(event) {

            // --- MODIFICATION: Handle Invalid Build ID error first ---
            if (event.data.startsWith('ERROR: Invalid build ID') || event.data.startsWith('ERROR: No build_id')) {
                logContainer.innerHTML = `
                    <p class="text-yellow-400 font-bold">The Database Update is in fact in progress, only you are unable to see the log.</p>
                    <p class="text-gray-300 mt-2">This can happen if the build was started from another device or a previous session, and that build has just completed.</p>
                    <p class="text-gray-300">Please wait a few minutes, and refresh the website to see if the update has been complete.</p>
                    <p class="text-gray-300">Please do not start an additional update, thank you.</p>
                `;
                logContainer.scrollTop = logContainer.scrollHeight; // Auto-scroll

                // Manually stop the stream and reset the button
                eventSource.close();
                eventSource = null;
                actionButton.disabled = false;
                actionButton.classList.remove('opacity-50', 'cursor-not-allowed');
                actionButton.textContent = 'Refresh Status';
                fetchStatus();
                return;
            }
            // --- END MODIFICATION ---

            // 3. Check for the sentinel message
            if (event.data === '__DONE__') {
                eventSource.close();
                eventSource = null;
                logContainer.scrollTop = logContainer.scrollHeight;
                // Re-enable button and refresh status
                actionButton.disabled = false;
                actionButton.classList.remove('opacity-50', 'cursor-not-allowed');
                actionButton.textContent = 'Update Complete';
                fetchStatus(); // Refresh main status
                return;
            }

            const p = document.createElement('p');
            p.textContent = event.data;

            if (event.data.startsWith('--- SUCCESS:')) {
                p.className = 'text-green-400 font-bold';
            } else if (event.data.startsWith('--- ERROR:') || event.data.startsWith('--- FATAL ERROR:')) {
                p.className = 'text-red-400 font-bold';
            } else if (event.data.startsWith('ERROR:')) {
                p.className = 'text-red-400';
            } else if (event.data.startsWith('---')) {
                 p.className = 'text-yellow-400';
            } else {
                p.className = 'text-gray-300';
            }
            logContainer.appendChild(p);
            logContainer.scrollTop = logContainer.scrollHeight; // Auto-scroll
        };

        eventSource.onerror = function(err) {
            console.error('EventSource failed:', err);
            const p = document.createElement('p');
            p.className = 'text-red-500';
            p.textContent = 'Connection to log stream lost. Refreshing status...';
            logContainer.appendChild(p);
            if (eventSource) eventSource.close();
            eventSource = null;
            // Refresh the main status when the stream closes
            fetchStatus();
            // Re-enable button
            actionButton.disabled = false;
            actionButton.classList.remove('opacity-50', 'cursor-not-allowed');
            actionButton.textContent = 'Stream Error';
        };
    };
    // --- END NEW FUNCTION ---

    const handleDbAction = async (event) => {
        event.preventDefault();
        actionButton.disabled = true;
        actionButton.classList.add('opacity-50', 'cursor-not-allowed');
        actionButton.textContent = 'Starting Update...';
        logContainer.innerHTML = '';

        if (eventSource) eventSource.close();

        // --- LOGIC BRANCHING ---
        let captureLineups = false;
        let rosterUpdatesOnly = false;

        if (!dbExists) {
            // Scenario 1: No DB exists -> Force Full Build
            captureLineups = true;
            rosterUpdatesOnly = false;
        } else {
            // Scenario 2: DB exists
            if (statUpdateCheckbox.checked) {
                // "Check for Stat Updated" -> Partial Stat Update
                captureLineups = false;
                rosterUpdatesOnly = false;
            } else {
                // Default -> Roster Updates Only
                captureLineups = false;
                rosterUpdatesOnly = true;
            }
        }
        // -----------------------

        try {
            const options = {
                'capture_lineups': captureLineups,
                'roster_updates_only': rosterUpdatesOnly,
                'skip_static': false,
                'skip_players': false
            };

            const response = await fetch('/api/db_action', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(options)
            });

            if (!response.ok) {
                const err = await response.json();
                if (response.status === 409 && err.build_id) {
                    logContainer.innerHTML = '<p class="text-yellow-400">A build is already in progress. Attempting to connect to the log stream...</p>';
                    connectToLogStream(err.build_id);
                } else {
                     throw new Error(err.error || `Server error: ${response.status}`);
                }
            } else {
                const data = await response.json();
                if (!data.success || !data.build_id) {
                     throw new Error('Failed to start build process. Server did not return a build_id.');
                }
                connectToLogStream(data.build_id);
            }

        } catch (error) {
            console.error('Error performing DB action:', error);
            logContainer.innerHTML = `<p class="text-red-400">Error: ${error.message}</p>`;
            actionButton.disabled = false;
            actionButton.classList.remove('opacity-50', 'cursor-not-allowed');
            actionButton.textContent = 'Update Failed';
        }
    };

    actionButton.addEventListener('click', handleDbAction);
    fetchStatus();

})();
