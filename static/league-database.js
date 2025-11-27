// This script will manage the league-database.html page
(async function() {
    await new Promise(resolve => setTimeout(resolve, 0));

    const statusText = document.getElementById('db-status-text');
    const actionButton = document.getElementById('db-action-button');
    const statUpdateContainer = document.getElementById('stat-update-container');
    const statUpdateCheckbox = document.getElementById('check-stat-updates');
    const logContainer = document.getElementById('log-container');
    const infoText = document.getElementById('db-info-text');

    if (!statusText || !actionButton || !statUpdateContainer || !statUpdateCheckbox || !logContainer || !infoText) {
        console.error('Database page elements not found.');
        return;
    }

    initPremiumFeatures();

    let dbExists = false; // State to track DB status

    const updateStatus = (data) => {
        dbExists = data.db_exists; // Store state

        if (data.db_exists) {
            const date = new Date(data.timestamp * 1000);
            statusText.textContent = `Your league: '${data.league_name}'s data is up to date as of: ${date.toLocaleString()}`;
            actionButton.textContent = 'Update Rosters';

            // Show the stat update option
            statUpdateContainer.classList.remove('hidden');
        } else {
            statusText.textContent = "Your league's data has not been initialized. Please initialize the database.";
            actionButton.textContent = 'Initialize Database';
            statUpdateContainer.classList.add('hidden');
            statUpdateCheckbox.checked = false;

            // 4. Set dynamic text for missing database
            infoText.innerHTML = `Please click Initialize Database to build the file tailored to your league. This can be a lengthy process, especially if it is later in the season as each fantasy day for each team in your league must be called.
            <br><br>
            You do not need to remain on the site for the job to run, so feel free to close the page, and simply refresh after some time (15-30minutes).`;
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
                'roster_updates_only': rosterUpdatesOnly
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

function initPremiumFeatures() {
    const container = document.getElementById('premium-actions-container');
    const actionBtn = document.getElementById('btn-premium-action');
    const exploreBtn = document.getElementById('btn-explore-benefits');
    const statusText = document.getElementById('premium-status-text');

    const giftModal = document.getElementById('modal-gift-premium');
    const benefitsModal = document.getElementById('modal-benefits');

    const claimBtn = document.getElementById('btn-claim-gift');
    const closeGiftBtn = document.getElementById('btn-close-gift');
    const closeBenefitsBtn = document.getElementById('btn-close-benefits');

    if(!container) return; // Guard clause if elements don't exist

    // 1. Fetch Status
    fetch('/api/user_status')
        .then(res => res.json())
        .then(data => {
            container.classList.remove('hidden');

            if (data.is_premium) {
                actionBtn.textContent = "Extend Premium";
                statusText.innerHTML = `🌟 Premium Status: <span class="text-green-400">Active</span> <span class="text-xs text-gray-500 block">Expires: ${data.expiration_date}</span>`;
            } else {
                actionBtn.textContent = "Go Premium";
                statusText.innerHTML = `Status: <span class="text-gray-400">Free Tier</span>`;
            }

            actionBtn.onclick = () => {
                giftModal.classList.remove('hidden');
                giftModal.classList.add('flex');
            };

            exploreBtn.onclick = () => {
                benefitsModal.classList.remove('hidden');
                benefitsModal.classList.add('flex');
            };
        })
        .catch(err => console.error("Error fetching user status:", err));

    // 2. Modal Logic
    const closeModal = (modal) => {
        modal.classList.add('hidden');
        modal.classList.remove('flex');
    };

    closeGiftBtn.onclick = () => closeModal(giftModal);
    closeBenefitsBtn.onclick = () => closeModal(benefitsModal);

    window.onclick = (event) => {
        if (event.target == giftModal) closeModal(giftModal);
        if (event.target == benefitsModal) closeModal(benefitsModal);
    };

    // 3. Claim Gift Logic
    claimBtn.onclick = async () => {
        claimBtn.textContent = "Updating...";
        claimBtn.disabled = true;
        try {
            const res = await fetch('/api/gift_premium', { method: 'POST' });
            const result = await res.json();
            if (result.success) {
                alert("Success! Your account has been upgraded to Lifetime Premium.");
                window.location.reload();
            } else {
                alert("Error: " + (result.error || "Unknown error occurred."));
                claimBtn.textContent = "Claim Lifetime Account";
                claimBtn.disabled = false;
            }
        } catch (e) {
            alert("Network error occurred.");
            claimBtn.textContent = "Claim Lifetime Account";
            claimBtn.disabled = false;
        }
    };
}
