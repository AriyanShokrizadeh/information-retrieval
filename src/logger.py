import datetime
import logging
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / f'{datetime.datetime.now().strftime("%Y-%m-%d")}.log'

logging.basicConfig(
    filename=LOG_FILE,
    format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


def get_logger(name):
    return logging.getLogger(name)
