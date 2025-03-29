import os
import sys
import logging
import multiprocessing

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_with_gunicorn():
    """
    Run the Flask app with Gunicorn using optimized settings
    """
    # Import these here to avoid loading them when using Flask development server
    import gunicorn.app.base
    from gunicorn.six import iteritems

    class StandaloneApplication(gunicorn.app.base.BaseApplication):
        def __init__(self, app, options=None):
            self.options = options or {}
            self.application = app
            super().__init__()

        def load_config(self):
            config = {key: value for key, value in iteritems(self.options)
                     if key in self.cfg.settings and value is not None}
            for key, value in iteritems(config):
                self.cfg.set(key.lower(), value)

        def load(self):
            return self.application

    # Set the number of workers based on available CPUs but optimize for memory
    # Using fewer workers with more threads to balance throughput and memory usage
    available_cpus = multiprocessing.cpu_count()
    workers = min(2, max(1, int(available_cpus / 4)))  # Reduced worker count to save memory
    threads = 4  # Increased threads per worker
    
    # Using sync worker class for better stability with high memory operations
    # Setting max requests to trigger worker recycling
    
    logger.info(f"Starting Gunicorn with {workers} workers and {threads} threads per worker")
    
    from app import app
    
    options = {
        'bind': '0.0.0.0:5000',
        'workers': workers,
        'worker_class': 'sync',  # Using sync worker for better memory stability
        'threads': threads,
        'timeout': 90,  # 90 seconds timeout - reduced to catch issues earlier
        'reload': True,
        'preload_app': False,  # Don't preload to reduce memory usage
        'reuse_port': True,
        'max_requests': 10,  # Recycle workers after 10 requests to prevent memory buildup
        'max_requests_jitter': 3,  # Add jitter to prevent all workers recycling at once
        'worker_tmp_dir': '/tmp',  # Use /tmp for worker temporary files
        'accesslog': '-',  # Log to stdout
        'errorlog': '-',   # Log errors to stdout
        'loglevel': 'info'
    }
    
    StandaloneApplication(app, options).run()

if __name__ == "__main__":
    run_with_gunicorn()