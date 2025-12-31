import json
from collections import Counter

import numpy as np

from .logger import get_logger
from .utils import tokenizer

logger = get_logger(__name__)


class BaseRetriever:
    def __init__(self, config, mu):
        self.config = config
        self.mu = mu

        self.doc_lengths = []
        self.doc_term_freqs = []
        self.collection_probs = {}

    def _load_vocabulary(self):
        logger.debug(f"Loading vocabulary from {self.config.tokens_path}")

        with open(self.config.tokens_path, "r") as f:
            vocab = json.load(f)

        self.vocab_map = {word: idx for idx, word in enumerate(vocab)}
        logger.debug(f"Loaded {len(vocab)} vocabulary terms")

    def fit(self, passages_df):
        self._load_vocabulary()
        total_collection_words = 0
        collection_counts = Counter()

        for text in passages_df["passage_text"]:
            tokens = tokenizer(text)
            self.doc_lengths.append(len(tokens))

            counts = Counter(t for t in tokens if t in self.vocab_map)
            self.doc_term_freqs.append(counts)

            collection_counts.update(counts)
            total_collection_words += sum(counts.values())

        for word, count in collection_counts.items():
            self.collection_probs[word] = count / total_collection_words

    def retrieve_top_k(self, query_text: str, k: int = 5):
        query_tokens = [t for t in tokenizer(query_text) if t in self.vocab_map]
        if not query_tokens:
            return np.array([])

        scores = [
            self.calculate_score(query_tokens, i) for i in range(len(self.doc_lengths))
        ]

        sorted_indices = np.argsort(scores)
        best_indices_first = sorted_indices[::-1]
        return best_indices_first[:k]

    def calculate_score(self, query_tokens, doc_idx):
        raise NotImplementedError("Child class must implement this")


class UnigramRetriever(BaseRetriever):
    def calculate_score(self, query_tokens, doc_idx):
        score = 0.0
        doc_len = self.doc_lengths[doc_idx]
        doc_counts = self.doc_term_freqs[doc_idx]

        for word in query_tokens:
            p_wc = self.collection_probs.get(word, 0)
            numerator = doc_counts[word] + (self.mu * p_wc)
            denominator = doc_len + self.mu

            score += np.log(numerator / denominator)
        return score


class BigramRetriever(BaseRetriever):
    def __init__(self, config, mu, lambda_=0.5):
        super().__init__(config, mu)
        self.lambda_ = lambda_
        self.doc_bigram_freqs = []

    def fit(self, passages_df):
        super().fit(passages_df)
        for text in passages_df["passage_text"]:
            tokens = tokenizer(text)
            pairs = []
            for i in range(len(tokens) - 1):
                w1 = tokens[i]
                w2 = tokens[i + 1]
                if w1 in self.vocab_map and w2 in self.vocab_map:
                    pairs.append((w1, w2))

            self.doc_bigram_freqs.append(Counter(pairs))

    def calculate_score(self, query_tokens, doc_idx):
        score = 0.0
        doc_len = self.doc_lengths[doc_idx]
        doc_uni = self.doc_term_freqs[doc_idx]
        doc_bi = self.doc_bigram_freqs[doc_idx]

        w0 = query_tokens[0]
        p_wc = self.collection_probs.get(w0, 0)
        p_uni_smoothed = (doc_uni[w0] + self.mu * p_wc) / (doc_len + self.mu)
        score += np.log(p_uni_smoothed) if p_uni_smoothed > 0 else -50

        for i in range(1, len(query_tokens)):
            w_prev, w_curr = query_tokens[i - 1], query_tokens[i]

            p_wc = self.collection_probs.get(w_curr, 0)
            p_uni_smoothed = (doc_uni[w_curr] + self.mu * p_wc) / (doc_len + self.mu)

            count_pair = doc_bi[(w_prev, w_curr)]
            count_prev = doc_uni[w_prev]
            p_bigram_mle = (count_pair / count_prev) if count_prev > 0 else 0

            prob = (1 - self.lambda_) * p_bigram_mle + (self.lambda_ * p_uni_smoothed)

            score += np.log(prob) if prob > 0 else -50

        return score
