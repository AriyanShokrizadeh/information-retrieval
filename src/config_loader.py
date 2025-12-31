from pathlib import Path
from typing import Any, Dict

import yaml

from .logger import get_logger

logger = get_logger(__name__)


class AppConfig:
    CONFIG_FILE = "./config/config.yaml"
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance

    @staticmethod
    def _read_config(config_path: str):
        config_file = Path(config_path)

        if not config_file.exists():
            msg = f"Config file not found at {config_path}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        try:
            with config_file.open("r") as f:
                return yaml.safe_load(f)
        except yaml.YAMLError as e:
            logger.error(f"Error reading config file: {e}")
            raise

    def _load_config(self) -> None:
        config = self._read_config(self.CONFIG_FILE)

        # --- Data ingestion ---
        ingestion_cfg = config["data_ingestion"]
        self.data_root = Path(ingestion_cfg["root_dir"])

        datasets = ingestion_cfg["datasets"]
        train_cfg = datasets["train"]
        test_cfg = datasets["test"]
        val_cfg = datasets["validation"]

        # Passages
        self.train_passages_path = self.data_root / train_cfg["passages"]
        self.test_passages_path = self.data_root / test_cfg["passages"]
        self.val_passages_path = self.data_root / val_cfg["passages"]

        # Judgments
        self.test_judgments_path = self.data_root / test_cfg["judgments"]
        self.val_judgments_path = self.data_root / val_cfg["judgments"]

        # Questions
        self.test_questions_path = self.data_root / test_cfg["questions"]
        self.val_questions_path = self.data_root / val_cfg["questions"]

        # Stopwords
        self.stopwords_path = Path(ingestion_cfg["stopwords_path"])

        # --- Vocabulary ---
        vocab_cfg = config["vocabulary_settings"]
        self.vocab_size = vocab_cfg["vocab_size"]

        self.vocab_dir = Path(vocab_cfg["vocab_dir"])
        self.vocab_dir.mkdir(parents=True, exist_ok=True)

        self.tokens_path = self.vocab_dir / vocab_cfg["tokens"]

        # --- Image Retriever ---
        image_cfg = config["image_settings"]
        self.imgs_dir = Path(image_cfg["imgs_dir"])
        self.imgs_dir.mkdir(parents=True, exist_ok=True)

        self.results_plot = self.imgs_dir / image_cfg["results_plot"]

        logger.info("Configuration loaded successfully.")
