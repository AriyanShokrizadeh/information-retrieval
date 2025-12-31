import json
from collections import Counter

import numpy as np

from .config_loader import AppConfig
from .logger import get_logger
from .utils import tokenizer

logger = get_logger(__name__)


class BM25Retriever:
    def __init__(self, config: AppConfig, k1: float = 1.5, b: float = 0.75):
        self.config = config
        self.k1 = k1
        self.b = b

        self.idf = {}
        self.avg_dl = 0
        self.doc_term_freqs, self.doc_lengths = [], []

    def _load_vocabulary(self):
        logger.debug(f"Loading vocabulary from {self.config.tokens_path}")

        with open(self.config.tokens_path, "r") as f:
            self.vocab = json.load(f)

        self.vocab_map = {word: idx for idx, word in enumerate(self.vocab)}
        logger.debug(f"Loaded {len(self.vocab)} vocabulary terms")

    def fit(self, passages_df):
        logger.debug("Starting BM25 training...")
        self._load_vocabulary()

        total_docs = len(passages_df)
        total_length = 0
        doc_freq_per_word = Counter()

        self.doc_lengths = []
        self.doc_term_freqs = []

        for text in passages_df["passage_text"]:
            tokens = tokenizer(text)
            doc_len = len(tokens)

            self.doc_lengths.append(doc_len)
            total_length += doc_len

            term_counts = Counter(t for t in tokens if t in self.vocab_map)
            self.doc_term_freqs.append(term_counts)

            for word in term_counts:
                doc_freq_per_word[word] += 1

        self.avg_doc_length = total_length / total_docs
        logger.debug(f"Average document length: {self.avg_doc_length:.2f}")

        logger.debug("Computing IDF values...")
        for word in self.vocab:
            n_q = doc_freq_per_word[word]
            self.idf[word] = np.log(((total_docs - n_q + 0.5) / (n_q + 0.5)) + 1)
        logger.debug("BM25 training completed successfully.")

    def _score_document(self, query_tokens, doc_index):
        score = 0.0
        doc_len = self.doc_lengths[doc_index]
        term_freqs = self.doc_term_freqs[doc_index]

        for word in query_tokens:
            if word not in self.vocab_map:
                continue
            tf = term_freqs.get(word, 0)
            idf = self.idf[word]

            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (
                1 - self.b + self.b * (doc_len / self.avg_doc_length)
            )
            score += idf * (numerator / denominator)
        return score

    def retrieve_top_k(self, query_text: str, k: int = 5):
        query_tokens = tokenizer(query_text)
        scores = [
            self._score_document(query_tokens, i)
            for i in range(len(self.doc_term_freqs))
        ]
        top_indices = np.argsort(scores)[-k:][::-1]
        return top_indices
