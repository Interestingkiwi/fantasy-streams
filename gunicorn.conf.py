import logging

class HealthCheckFilter(logging.Filter):
    def filter(self, record):
        # Filter out log records containing GET /healthz
        return 'GET /healthz' not in record.getMessage()

def on_starting(server):
    # Attach the filter to the Gunicorn access logger
    server.log.access_log.addFilter(HealthCheckFilter())
