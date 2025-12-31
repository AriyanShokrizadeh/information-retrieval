import numpy as np

from .logger import get_logger

logger = get_logger(__name__)


class Evaluator:
    def __init__(self, ground_truth_data):
        logger.info("Initializing Evaluator with ground truth data")
        self.ground_truth = {}

        try:
            if isinstance(ground_truth_data, dict):
                logger.debug("Ground truth provided as dictionary")
                for query_id, doc_string in ground_truth_data.items():
                    self.ground_truth[query_id] = [
                        doc_id.strip() for doc_id in doc_string.split(",")
                    ]
            else:
                logger.debug("Ground truth provided as DataFrame")
                for _, row in ground_truth_data.iterrows():
                    query_id, doc_id = row["query_id"], row["doc_id"]
                    self.ground_truth.setdefault(query_id, []).append(doc_id)

            logger.info("Ground truth loaded successfully")

        except Exception as e:
            logger.exception("Failed to initialize ground truth")
            raise e

    def calculate_p_at_5(self, retrieved_doc_ids, relevant_doc_ids):
        logger.debug("Calculating P@5")
        top_5_docs = retrieved_doc_ids[:5]
        hits = len(set(top_5_docs) & set(relevant_doc_ids))
        score = hits / 5.0
        logger.debug(f"P@5 score: {score}")
        return score

    def calculate_mrr(self, retrieved_doc_ids, relevant_doc_ids):
        logger.debug("Calculating MRR")
        for rank, doc_id in enumerate(retrieved_doc_ids):
            if doc_id in relevant_doc_ids:
                score = 1.0 / (rank + 1)
                logger.debug(f"MRR score: {score}")
                return score
        logger.debug("MRR score: 0.0 (no relevant doc found)")
        return 0.0

    def calculate_ap(self, retrieved_doc_ids, relevant_doc_ids):
        logger.debug("Calculating AP")
        total_relevant_docs = 10
        cumulative_precision = 0.0
        found_relevant = 0

        for rank, doc_id in enumerate(retrieved_doc_ids):
            if doc_id in relevant_doc_ids:
                found_relevant += 1
                precision_at_rank = found_relevant / (rank + 1)
                cumulative_precision += precision_at_rank

        score = cumulative_precision / total_relevant_docs
        logger.debug(f"AP score: {score}")
        return score

    def evaluate_model(self, model, queries_df, passages_df):
        logger.info("Starting model evaluation")
        precision_scores, mrr_scores, map_scores = [], [], []

        for _, row in queries_df.iterrows():
            query_id = str(row["query_id"])
            query_text = row["query_text"]

            if query_id not in self.ground_truth:
                continue

            try:
                top_passage_indices = model.retrieve_top_k(query_text, k=5)
                if len(top_passage_indices) == 0:
                    continue

                retrieved_doc_ids = (
                    passages_df.iloc[top_passage_indices]["doc_id"].astype(str).values
                )
            except Exception as e:
                logger.error(f"Failed retrieving docs for query_id={query_id}: {e}")
                continue

            relevant_doc_ids = self.ground_truth[query_id]

            precision_scores.append(
                self.calculate_p_at_5(retrieved_doc_ids, relevant_doc_ids)
            )
            mrr_scores.append(self.calculate_mrr(retrieved_doc_ids, relevant_doc_ids))
            map_scores.append(self.calculate_ap(retrieved_doc_ids, relevant_doc_ids))

        if not precision_scores:
            logger.error("No queries were successfully evaluated!")
            return {"P@5": 0, "MRR": 0, "MAP": 0}

        results = {
            "P@5": round(np.mean(precision_scores).item(), 4),
            "MRR": round(np.mean(mrr_scores).item(), 4),
            "MAP": round(np.mean(map_scores).item(), 4),
        }
        return results
