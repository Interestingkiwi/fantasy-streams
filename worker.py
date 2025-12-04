import os
import redis
import threading
import scheduler  # <-- Import your scheduler script
from rq import Worker, Queue

# Get the Redis URL from the environment variable
redis_url = os.getenv('REDIS_URL', 'redis://red-d4ae0rur433s73eil750:6379') # Added default for safety
if not redis_url:
    raise RuntimeError("REDIS_URL environment variable not set.")

# --- PRIORITY QUEUE CONFIGURATION ---
# The worker will empty the 'high' queue before touching 'low'
listen = ['high', 'low', 'default']

conn = redis.from_url(redis_url)

def run_scheduler_in_background():
    """
    Wrapper to run the scheduler logic in a separate thread.
    """
    print("Starting background scheduler thread...")
    scheduler.start_scheduler()

if __name__ == '__main__':

    # --- Start the Scheduler Thread ---
    scheduler_thread = threading.Thread(target=run_scheduler_in_background, daemon=True)
    scheduler_thread.start()

    # --- Start the Main RQ Worker ---
    # We create Queue objects for each priority level
    queues = [Queue(name, connection=conn) for name in listen]

    worker = Worker(queues, connection=conn)

    print(f"RQ Worker starting, listening on queues: {listen}")
    worker.work()
