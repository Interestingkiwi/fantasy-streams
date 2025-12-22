# Fantasy Streams - H2H Fantasy Hockey Optimizer

**Fantasy Streams** is a powerful web application designed to give Head-to-Head (H2H) fantasy hockey managers a competitive edge. It connects directly to your Yahoo! Fantasy Sports league to provide advanced analytics, optimal lineup planning, and automated tools that go far beyond the standard Yahoo dashboard.

## 🚀 Key Features

### 📊 League Analytics
-   **League Database:** Automatically fetches and caches your entire league's history, rosters, and settings into a PostgreSQL database for instant analysis.
-   **Matchup Dashboard:** Compare your team against your current opponent with live stats, projected stats, and category-level win probabilities.
-   **Smart Caching:** Background workers keep data fresh without hitting Yahoo API rate limits.

### 🛠️ Manager Tools
-   **Free Agent Finder:** Advanced filtering for waiver wire players using ROS (Rest of Season) projections, L10/L5 trends, and schedule density.
-   **Goalie Planner:** Analyze upcoming goalie matchups, back-to-backs, and strength of schedule to maximize starts.
-   **Trade Helper:** Visualize how a potential trade impacts your team's category strengths and weaknesses.
-   **Schedule Insights:** Identify "Off-Day" games to maximize your roster utilization (Games Played).

### ⚡ Automation
-   **Scheduled Add/Drops:** Set up player transactions to execute automatically at a specific date and time—perfect for beating opponents to waiver wire pickups while you sleep.

---

## 🏗️ Tech Stack

-   **Backend:** Python (Flask), Gevent, Gunicorn
-   **Database:** PostgreSQL (Primary Data), Redis (Job Queues & Caching)
-   **Worker Queue:** RQ (Redis Queue) for handling background ETL jobs and automated transactions.
-   **Frontend:** HTML5, Tailwind CSS, JavaScript (Vanilla ES6)
-   **APIs:** Yahoo! Fantasy Sports API (`yfpy`, `yahoo_fantasy_api`)
-   **Deployment:** Render.com (Docker/Native)

---

## ⚙️ Setup & Installation

### Prerequisites
1.  **Yahoo Developer Account:** Create an app at [Yahoo Developer Network](https://developer.yahoo.com/apps/) with `Fantasy Sports Read/Write` permissions.
2.  **PostgreSQL:** A local or hosted Postgres database.
3.  **Redis:** A local or hosted Redis instance.

### Environment Variables
Create a `.env` file in the root directory:
```bash
# Flask
FLASK_APP=app.py
FLASK_ENV=development
FLASK_SECRET_KEY=your_secure_random_key

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/fantasy_db
REDIS_URL=redis://localhost:6379

# Yahoo API (From Yahoo Developer Console)
YAHOO_CONSUMER_KEY=your_client_id
YAHOO_CONSUMER_SECRET=your_client_secret
