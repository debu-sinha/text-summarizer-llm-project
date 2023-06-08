import os
import sys
import logging

# Define the logging format
logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"

# Set up the log directory and file path
log_dir = "logs"
log_filepath = os.path.join(log_dir, "running_logs.log")

# Create the log directory if it doesn't exist
os.makedirs(log_dir, exist_ok=True)

# Configure the logging
logging.basicConfig(
    level=logging.INFO,               # Set the log level to INFO
    format=logging_str,               # Use the defined logging format
    handlers=[
        logging.FileHandler(log_filepath),      # Log to a file using the specified log file path
        logging.StreamHandler(sys.stdout)       # Log to the console (stdout)
    ]
)

# Get the logger object with the name "textSummarizerLogger"
logger = logging.getLogger("textSummarizerLogger")
