#!/bin/bash

echo "Starting memory-optimized Gunicorn server..."

# Set ulimit for the process to prevent runaway memory usage
# Set to 1GB memory limit for each process
ulimit -v 1048576 2>/dev/null || true

# Enable garbage collection in Python
export PYTHONUNBUFFERED=1

# Execute gunicorn with memory optimized settings
exec gunicorn \
    --bind 0.0.0.0:5000 \
    --workers 1 \
    --threads 4 \
    --worker-class sync \
    --timeout 90 \
    --max-requests 10 \
    --max-requests-jitter 3 \
    --reuse-port \
    --reload \
    main:app