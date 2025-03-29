import os
import gc
import logging
from dotenv import load_dotenv

# Configure memory optimization and garbage collection
def optimize_memory():
    """Configure Python's memory management for optimal performance with large language models"""
    # Enable garbage collection debugging if needed
    gc.set_debug(gc.DEBUG_STATS)
    
    # Set aggressive garbage collection thresholds
    # Lower thresholds mean more frequent collection
    gc.set_threshold(700, 10, 5)  # Default is (700, 10, 10)
    
    # Perform a full collection now
    collected = gc.collect(generation=2)
    logging.info(f"Initial garbage collection: freed {collected} objects")
    
    # Return optimization status
    return True

# Load environment variables
load_dotenv()

# Verify we have the required API key
if not os.environ.get("OPENAI_API_KEY"):
    print("WARNING: OPENAI_API_KEY environment variable is not set.")
    print("The application will attempt to run, but CrewAI agents might not function correctly.")

# Apply memory optimizations
logging.basicConfig(level=logging.INFO)
memory_optimized = optimize_memory()
logging.info(f"Memory optimization enabled: {memory_optimized}")

from app import app

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
