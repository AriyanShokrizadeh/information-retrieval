import json
import re
from pathlib import Path

from .config_loader import AppConfig
from .logger import get_logger

logger = get_logger(__name__)

try:
    config = AppConfig()
    stopwords_path = config.stopwords_path
    stopwords = Path(stopwords_path).read_text(encoding="utf-8").splitlines()

    logger.info(
        f"Stopwords loaded successfully | source='{stopwords_path}', "
        f"count={len(stopwords)}"
    )
except FileNotFoundError:
    logger.error(f"Stopwords file not found at path: '{stopwords_path}'")
    raise FileNotFoundError


def tokenizer(text: str):
    if not isinstance(text, str):
        logger.error(
            "Tokenizer received a non-string input; returning empty token list."
        )
        return []

    text = text.lower().strip()
    tokens = re.findall(r"\w+", text)

    filtered = [
        token for token in tokens if token not in stopwords and not token.isnumeric()
    ]
    logger.debug(
        f"Tokenizer processed text | "
        f"tokens_in={len(tokens)}, "
        f" tokens_out={len(filtered)}"
    )
    return filtered


def parse_judgments(file_path: str):
    logger.info(f"Loading judgments from '{file_path}'")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Judgments file not found: '{file_path}'")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in judgments file '{file_path}': {e}")
        raise
    return data
