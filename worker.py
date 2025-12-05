import os
import redis
import threading
import scheduler
from rq import Worker, Queue

# Get the Redis URL
redis_url = os.getenv('REDIS_URL', 'redis://red-d4ae0rur433s73eil750:6379')
if not redis_url:
    raise RuntimeError("REDIS_URL environment variable not set.")

listen = ['high', 'low', 'default']
conn = redis.from_url(redis_url)

def run_scheduler_in_background():
    print("Starting background scheduler thread...")
    scheduler.start_scheduler()

def start_worker():
    """
    Main entry point for the worker.
    """
    # 1. Check if this instance should run the scheduler
    should_run_scheduler = os.getenv('RUN_SCHEDULER', 'False').lower() == 'true'

    if should_run_scheduler:
        print(">>> ROLE: Primary (Scheduler + Worker)")
        scheduler_thread = threading.Thread(target=run_scheduler_in_background, daemon=True)
        scheduler_thread.start()
    else:
        print(">>> ROLE: Worker Only (No Scheduler)")

    # 2. Start the RQ Worker
    try:
        queues = [Queue(name, connection=conn) for name in listen]
        worker = Worker(queues, connection=conn)

        print(f"RQ Worker starting, listening on queues: {listen}")
        worker.work()
    except Exception as e:
        print(f"Worker failed to start: {e}")

if __name__ == '__main__':
    start_worker()
