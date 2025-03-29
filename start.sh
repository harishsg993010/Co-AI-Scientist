#!/bin/bash

# Kill any existing Python processes (for development only)
# pkill -f python || true

echo "Starting optimized server with memory management..."

# Set ulimit for the process to prevent runaway memory usage
# Set to 1GB memory limit for each process
ulimit -v 1048576

# Clear any cached memory to start fresh
sync && echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true

# Enable garbage collection in Python
export PYTHONUNBUFFERED=1
export PYTHONGC=1

# Use a memory-optimized command line for Python
python -X gc.debug server.py