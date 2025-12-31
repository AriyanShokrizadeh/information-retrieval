import json
from collections import Counter

import pandas as pd

from .config_loader import AppConfig
from .logger import get_logger
from .utils import tokenizer

logger = get_logger(__name__)


class VocabularyBuilder:
    def __init__(self, config: AppConfig):
        self.tokens_path = config.tokens_path
        self.vocab_size = config.vocab_size

    def build(self, train_passages: pd.DataFrame):
        if train_passages is None or "passage_text" not in train_passages:
            msg = "Training passages must contain a 'passage_text' column."
            logger.error(msg)
            raise pd.errors.EmptyDataError(msg)

        logger.info("Vocabulary construction started")

        all_tokens = []
        for passage in train_passages["passage_text"]:
            all_tokens.extend(tokenizer(passage))

        counts = Counter(all_tokens)

        self._tokens = [word for word, _ in counts.most_common(self.vocab_size)]

        logger.info(
            f"Vocabulary construction completed | "
            f"unique_tokens={len(counts)}, "
            f"selected_top={len(self._tokens)}"
        )

        return self._tokens

    def save(self):
        if not self._tokens:
            msg = "Vocabulary has not been built yet. Call build() first."
            logger.error(msg)
            raise RuntimeError(msg)

        # Save tokens
        with open(self.tokens_path, "w", encoding="utf-8") as file:
            json.dump(self._tokens, file, ensure_ascii=False, indent=2)

        logger.info(f"Vocabulary artifacts saved | " f"tokens='{self.tokens_path}', ")

    @property
    def tokens(self):
        return self._tokens

    @property
    def token_mapping(self):
        return self._tokens_mapping
