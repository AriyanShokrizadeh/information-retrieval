from itertools import product

import numpy as np

from .bm25_retriever import BM25Retriever
from .language_retriever import BigramRetriever, UnigramRetriever
from .logger import get_logger
from .metrics import Evaluator

logger = get_logger(__name__)

# --- Hyperparameter Grids ---
BM25_K1_RANGE = np.linspace(1.2, 2.0, num=5).tolist()
BM25_B_RANGE = np.linspace(0.5, 1.0, num=5).tolist()

UNIGRAM_MU_RANGE = [500, 1000, 1500, 2000, 3000]
BIGRAM_LAMBDA_RANGE = np.linspace(0.1, 0.9, num=9).tolist()


def fine_tune_bm25(config, train_passage, val_passage, val_questions, val_judgments):
    logger.info("Starting BM25 Fine-Tuning")
    evaluator = Evaluator(val_judgments)
    best_map = -1
    best_params = {}

    for k1, b in product(BM25_K1_RANGE, BM25_B_RANGE):
        model = BM25Retriever(config, k1=k1, b=b)
        model.fit(train_passage)
        results = evaluator.evaluate_model(model, val_questions, val_passage)

        logger.info(f"BM25 [k1={k1:.2f}, b={b:.2f}] -> MAP: {results['MAP']:.4f}")
        if results["MAP"] > best_map:
            best_map = results["MAP"]
            best_params = {"k1": k1, "b": b}

    return best_params


def fine_tune_unigram(config, train_passage, val_passage, val_questions, val_judgments):
    logger.info("Starting Unigram Fine-Tuning")
    evaluator = Evaluator(val_judgments)
    best_map = -1
    best_mu = 1000

    for mu in UNIGRAM_MU_RANGE:
        model = UnigramRetriever(config, mu=mu)
        model.fit(train_passage)

        results = evaluator.evaluate_model(model, val_questions, val_passage)
        logger.info(f"Unigram [mu={mu}] -> MAP: {results['MAP']:.4f}")

        if results["MAP"] > best_map:
            best_map = results["MAP"]
            best_mu = mu

    return {"mu": best_mu}


def fine_tune_bigram(
    config, train_passage, val_passage, val_questions, val_judgments, best_mu
):
    logger.info(f"Starting Bigram Fine-Tuning using fixed mu={best_mu}")
    evaluator = Evaluator(val_judgments)
    best_map = -1
    best_lambda = 0.5

    for lambda_ in BIGRAM_LAMBDA_RANGE:
        model = BigramRetriever(config, mu=best_mu, lambda_=lambda_)
        model.fit(train_passage)

        results = evaluator.evaluate_model(model, val_questions, val_passage)
        logger.info(f"Bigram [lambda={lambda_:.2f}] -> MAP: {results['MAP']:.4f}")

        if results["MAP"] > best_map:
            best_map = results["MAP"]
            best_lambda = lambda_

    return {"mu": best_mu, "lambda_": best_lambda}
