from loguru import logger
import os

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

logger.remove()
logger.add(lambda msg: print(msg, end=""), level=LOG_LEVEL)

def get_logger():
    return logger
