import pandas as pd

from src.bm25_retriever import BM25Retriever
from src.config_loader import AppConfig
from src.fine_tuning import fine_tune_bigram, fine_tune_bm25, fine_tune_unigram
from src.language_retriever import BigramRetriever, UnigramRetriever
from src.logger import get_logger
from src.metrics import Evaluator
from src.utils import parse_judgments, plot_results
from src.vocab_builder import VocabularyBuilder

logger = get_logger(__name__)

if __name__ == "__main__":
    config = AppConfig()

    # --- Load Datasets ---
    train_passage = pd.read_json(config.train_passages_path.resolve(), orient="columns")
    val_passage = pd.read_json(config.val_passages_path, orient="columns")
    test_passage = pd.read_json(config.test_passages_path, orient="columns")

    val_questions = pd.read_json(config.val_questions_path, orient="columns")
    test_questions = pd.read_json(config.test_questions_path, orient="columns")

    val_judgments = parse_judgments(config.val_judgments_path)
    test_judgments = parse_judgments(config.test_judgments_path)

    evaluator = Evaluator(test_judgments)

    # --- Build Vocabulary ---
    vocab_builder = VocabularyBuilder(config)
    tokens = vocab_builder.build(train_passage)
    vocab_builder.save()

    # --- BM25 Retriever ---
    logger.info("--- BM25 Section ---")
    bm25_params = fine_tune_bm25(
        config, train_passage, val_passage, val_questions, val_judgments
    )

    bm25_retriever = BM25Retriever(config, **bm25_params)
    bm25_retriever.fit(test_passage)
    bm25_results = evaluator.evaluate_model(
        bm25_retriever, test_questions, test_passage
    )

    # --- Unigram Retriever ---
    logger.info("--- Unigram Section ---")
    unigram_params = fine_tune_unigram(
        config, train_passage, val_passage, val_questions, val_judgments
    )

    unigram_retriever = UnigramRetriever(config, mu=unigram_params["mu"])
    unigram_retriever.fit(test_passage)
    uni_results = evaluator.evaluate_model(
        unigram_retriever, test_questions, test_passage
    )

    # --- Bigram Retriever ---
    logger.info("--- Bigram Section ---")
    bigram_params = fine_tune_bigram(
        config,
        train_passage,
        val_passage,
        val_questions,
        val_judgments,
        best_mu=unigram_params["mu"],
    )

    bigram_retriever = BigramRetriever(config, **bigram_params)
    bigram_retriever.fit(test_passage)
    bi_results = evaluator.evaluate_model(
        bigram_retriever, test_questions, test_passage
    )

    # --- Final Comparisons ---
    print("\n" + "=" * 40)
    print("BEST PARAMETERS FOR EACH MODEL")
    print("=" * 40)
    print("BM25:", bm25_params)
    print("Unigram:", unigram_params)
    print("Bigram:", bigram_params)

    print("\n" + "=" * 40)
    print("FINAL TEST DATASET RESULTS")
    print("=" * 40)
    print(f"BM25:    {bm25_results}")
    print(f"Unigram: {uni_results}")
    print(f"Bigram:  {bi_results}")

    # --- Plot Results ---
    plot_results(bm25_results, uni_results, bi_results)
