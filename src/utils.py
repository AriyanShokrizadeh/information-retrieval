import json
import re
from pathlib import Path

import seaborn as sns
from matplotlib import pyplot as plt

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


def plot_results(bm25_results, uni_results, bi_results):
    # 1. Setup Theme
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)

    results = {"BM25": bm25_results, "Unigram": uni_results, "Bigram": bi_results}
    models = list(results.keys())

    metrics_data = [
        ("MAP", [r["MAP"] for r in results.values()]),
        ("P@5", [r["P@5"] for r in results.values()]),
        ("MRR", [r["MRR"] for r in results.values()]),
    ]

    fig, axes = plt.subplots(
        1, 3, figsize=(15, 6), constrained_layout=True, sharey=True
    )
    for ax, (metric_name, scores) in zip(axes, metrics_data):
        sns.barplot(
            x=models, y=scores, ax=ax, palette="viridis", hue=models, legend=False
        )
        ax.set_box_aspect(1)
        ax.set_title(metric_name, fontweight="bold", pad=10)
        ax.set_ylabel("Score")
        ax.set_ylim(0, 0.6)

        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", padding=3, fontsize=11)

    fig.suptitle("Retriever Performance Comparison", fontsize=18, fontweight="bold")

    plt.savefig(config.results_plot, dpi=300)
    print("Plot saved successfully.")
