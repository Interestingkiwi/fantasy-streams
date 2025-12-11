// This script will manage the league-database.html page
(async function() {
    await new Promise(resolve => setTimeout(resolve, 0));

    const statusText = document.getElementById('db-status-text');
    const actionButton = document.getElementById('db-action-button');
    const statUpdateContainer = document.getElementById('stat-update-container');
    const statUpdateCheckbox = document.getElementById('check-stat-updates');
    let logContainer = document.getElementById('log-container');
    const infoText = document.getElementById('db-info-text');

    if (!statusText || !actionButton || !statUpdateContainer || !statUpdateCheckbox || !logContainer || !infoText) {
        console.error('Database page elements not found.');
        return;
    }

    initPremiumFeatures();

    let dbExists = false;

    const updateStatus = (data) => {
        dbExists = data.db_exists;

        if (data.db_exists) {
            // --- CASE: DB EXISTS (Established League) ---
            const date = new Date(data.timestamp * 1000);
            statusText.textContent = `Your league: '${data.league_name}'s data is up to date as of: ${date.toLocaleString()}`;
            actionButton.textContent = 'Update Rosters';

            statUpdateContainer.classList.remove('hidden');

            // 1. Hide the Hardcoded Log Container
            logContainer.classList.add('hidden');

            // 2. Change its ID so window.startLogStream CANNOT find it.
            // This forces startLogStream to use the Floating Corner Window.
            logContainer.id = 'log-container-disabled';

        } else {
            // --- CASE: NO DB (Initialization Phase) ---
            statusText.textContent = "Your league's data has not been initialized. Please initialize the database.";
            actionButton.textContent = 'Initialize Database';
            statUpdateContainer.classList.add('hidden');
            statUpdateCheckbox.checked = false;

            infoText.innerHTML = `Please click Initialize Database to build the file tailored to your league. This can be a lengthy process, especially if it is later in the season as each fantasy day for each team in your league must be called.
            <br><br>
            You do not need to remain on the site for the job to run, so feel free to close the page, and simply refresh after some time (15-30minutes).`;

            // 1. Show the Hardcoded Log Container
            logContainer.classList.remove('hidden');

            // 2. Ensure ID is correct so window.startLogStream uses IT (large view) instead of floating window.
            logContainer.id = 'log-container';
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

    // --- REMOVED LOCAL connectToLogStream FUNCTION ENTIRELY ---
    // We now rely exclusively on window.startLogStream from home.js

    const handleDbAction = async (event) => {
        event.preventDefault();
        actionButton.disabled = true;
        actionButton.classList.add('opacity-50', 'cursor-not-allowed');
        actionButton.textContent = 'Starting Update...';

        // Clear log container if visible
        if (logContainer && logContainer.id === 'log-container') {
             logContainer.innerHTML = '';
        }

        // Logic Branching
        let captureLineups = false;
        let rosterUpdatesOnly = false;

        if (!dbExists) {
            captureLineups = true;
            rosterUpdatesOnly = false;
        } else {
            if (statUpdateCheckbox.checked) {
                captureLineups = false;
                rosterUpdatesOnly = false;
            } else {
                captureLineups = false;
                rosterUpdatesOnly = true;
            }
        }

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
                    // Use Global Streamer
                    if (window.startLogStream) window.startLogStream(err.build_id);
                } else {
                     throw new Error(err.error || `Server error: ${response.status}`);
                }
            } else {
                const data = await response.json();
                if (!data.success || !data.build_id) {
                     throw new Error('Failed to start build process. Server did not return a build_id.');
                }
                // Use Global Streamer
                if (window.startLogStream) window.startLogStream(data.build_id);
            }

        } catch (error) {
            console.error('Error performing DB action:', error);
            // Fallback display if global streamer fails or logic error
            if(logContainer.id === 'log-container') {
                logContainer.innerHTML = `<p class="text-red-400">Error: ${error.message}</p>`;
            } else {
                alert(`Error: ${error.message}`);
            }
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
