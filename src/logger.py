import sys
from loguru import logger

# Configure the logger
logger.remove() # Remove the default handler
logger.add(
    sys.stderr,
    level="INFO", # Set the default log level
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    colorize=True
)

# You can also add a file logger if you want to save logs to a file
# logger.add("logs/app.log", rotation="10 MB", level="DEBUG")
