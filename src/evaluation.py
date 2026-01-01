"""Evaluation module for the tourism recommender system."""

import numpy as np
from typing import List, Dict, Tuple, Set, Any
from collections import defaultdict


class EvaluationModule:
    """Handles evaluation metrics for the recommender system."""
    
    def __init__(self):
        """Initialize the evaluation module."""
        pass
    
    def compute_ndcg_at_k(
        self,
        predictions: List[str],
        ground_truth: List[str],
        k: int,
        relevance_scores: Dict[str, float] = None
    ) -> float:
        """
        Compute Normalized Discounted Cumulative Gain at K (NDCG@K).
        
        NDCG@K measures the quality of ranking by comparing the predicted ranking
        to the ideal ranking based on relevance scores. It accounts for the position
        of relevant items in the ranking.
        
        Args:
            predictions: List of predicted destination IDs in ranked order
            ground_truth: List of relevant destination IDs
            k: Number of top items to consider
            relevance_scores: Optional dict mapping destination_id to relevance score.
                            If not provided, binary relevance (1 for relevant, 0 otherwise) is used.
        
        Returns:
            NDCG@K score in range [0, 1], where 1 is perfect ranking
        """
        if not predictions or not ground_truth or k <= 0:
            return 0.0
        
        # Limit predictions to top-K
        predictions_at_k = predictions[:k]
        
        # If no relevance scores provided, use binary relevance
        if relevance_scores is None:
            relevance_scores = {dest_id: 1.0 for dest_id in ground_truth}
        
        # Compute DCG (Discounted Cumulative Gain)
        dcg = 0.0
        for i, dest_id in enumerate(predictions_at_k):
            if dest_id in relevance_scores:
                relevance = relevance_scores[dest_id]
                # DCG formula: sum(rel_i / log2(i + 2))
                # i+2 because: position 0 -> log2(2), position 1 -> log2(3), etc.
                dcg += relevance / np.log2(i + 2)
        
        # Compute IDCG (Ideal DCG) - best possible ranking
        # Sort ground truth by relevance scores (descending)
        ideal_ranking = sorted(
            ground_truth,
            key=lambda x: relevance_scores.get(x, 0.0),
            reverse=True
        )[:k]
        
        idcg = 0.0
        for i, dest_id in enumerate(ideal_ranking):
            relevance = relevance_scores.get(dest_id, 0.0)
            idcg += relevance / np.log2(i + 2)
        
        # Normalize DCG by IDCG
        if idcg == 0.0:
            return 0.0
        
        ndcg = dcg / idcg
        
        # Ensure result is in [0, 1]
        return max(0.0, min(1.0, ndcg))
    
    def compute_hit_rate_at_k(
        self,
        predictions: List[str],
        ground_truth: List[str],
        k: int
    ) -> float:
        """
        Compute Hit Rate at K (HR@K).
        
        Hit Rate measures whether at least one relevant item appears in the
        top-K predictions. It's a binary metric: 1 if there's a hit, 0 otherwise.
        
        Args:
            predictions: List of predicted destination IDs in ranked order
            ground_truth: List of relevant destination IDs
            k: Number of top items to consider
        
        Returns:
            Hit rate in range [0, 1], where 1 means at least one relevant item in top-K
        """
        if not predictions or not ground_truth or k <= 0:
            return 0.0
        
        # Limit predictions to top-K
        predictions_at_k = predictions[:k]
        
        # Convert to sets for efficient intersection
        predictions_set = set(predictions_at_k)
        ground_truth_set = set(ground_truth)
        
        # Check if there's any intersection
        has_hit = len(predictions_set & ground_truth_set) > 0
        
        return 1.0 if has_hit else 0.0
    
    def compute_diversity_score(
        self,
        predictions: List[str],
        destination_types: Dict[str, str]
    ) -> float:
        """
        Compute diversity score for recommendations.
        
        Diversity measures the variety of destination types in the recommendations.
        Higher diversity means more varied types of destinations.
        
        Args:
            predictions: List of predicted destination IDs
            destination_types: Dict mapping destination_id to type (beach, cultural, etc.)
        
        Returns:
            Diversity score (non-negative), higher values indicate more diversity
        """
        if not predictions or not destination_types:
            return 0.0
        
        # Count unique types in predictions
        types_in_predictions = set()
        for dest_id in predictions:
            if dest_id in destination_types:
                types_in_predictions.add(destination_types[dest_id])
        
        # Diversity is the number of unique types
        # Normalized by the number of predictions to get a ratio
        diversity = len(types_in_predictions)
        
        # Can also compute as ratio: diversity / len(predictions)
        # But the requirement says "non-negative" without specifying normalization
        # We'll return the count of unique types
        return float(diversity)
    
    def compute_coverage_score(
        self,
        all_predictions: List[List[str]],
        catalog: Set[str]
    ) -> float:
        """
        Compute catalog coverage score.
        
        Coverage measures what fraction of the catalog is recommended across
        all users. Higher coverage means the system recommends a wider variety
        of items from the catalog.
        
        Args:
            all_predictions: List of prediction lists (one per user/query)
            catalog: Set of all available destination IDs in the catalog
        
        Returns:
            Coverage score in range [0, 1], where 1 means all catalog items recommended
        """
        if not catalog or not all_predictions:
            return 0.0
        
        # Collect all unique destinations that were recommended
        recommended_items = set()
        for predictions in all_predictions:
            recommended_items.update(predictions)
        
        # Coverage is the fraction of catalog items that were recommended
        coverage = len(recommended_items & catalog) / len(catalog)
        
        # Ensure result is in [0, 1]
        return max(0.0, min(1.0, coverage))
    
    def compute_all_metrics(
        self,
        predictions: List[str],
        ground_truth: List[str],
        k: int,
        destination_types: Dict[str, str] = None,
        relevance_scores: Dict[str, float] = None
    ) -> Dict[str, float]:
        """
        Compute all evaluation metrics for a single prediction.
        
        Args:
            predictions: List of predicted destination IDs in ranked order
            ground_truth: List of relevant destination IDs
            k: Number of top items to consider
            destination_types: Optional dict mapping destination_id to type
            relevance_scores: Optional dict mapping destination_id to relevance score
        
        Returns:
            Dictionary containing all computed metrics
        """
        metrics = {}
        
        # Compute NDCG@K
        metrics['ndcg_at_k'] = self.compute_ndcg_at_k(
            predictions, ground_truth, k, relevance_scores
        )
        
        # Compute Hit Rate@K
        metrics['hit_rate_at_k'] = self.compute_hit_rate_at_k(
            predictions, ground_truth, k
        )
        
        # Compute Diversity (if destination types provided)
        if destination_types is not None:
            metrics['diversity'] = self.compute_diversity_score(
                predictions, destination_types
            )
        
        return metrics
    
    def evaluate_batch(
        self,
        batch_predictions: List[List[str]],
        batch_ground_truth: List[List[str]],
        k: int,
        catalog: Set[str] = None,
        destination_types: Dict[str, str] = None,
        batch_relevance_scores: List[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Evaluate a batch of predictions and return aggregated metrics.
        
        Args:
            batch_predictions: List of prediction lists (one per query)
            batch_ground_truth: List of ground truth lists (one per query)
            k: Number of top items to consider
            catalog: Optional set of all available destination IDs
            destination_types: Optional dict mapping destination_id to type
            batch_relevance_scores: Optional list of relevance score dicts (one per query)
        
        Returns:
            Dictionary containing averaged metrics across the batch
        """
        if not batch_predictions or not batch_ground_truth:
            return {}
        
        if len(batch_predictions) != len(batch_ground_truth):
            raise ValueError("Batch predictions and ground truth must have same length")
        
        # Initialize metric accumulators
        ndcg_scores = []
        hit_rate_scores = []
        diversity_scores = []
        
        # Compute metrics for each query
        for i, (predictions, ground_truth) in enumerate(zip(batch_predictions, batch_ground_truth)):
            relevance_scores = None
            if batch_relevance_scores is not None and i < len(batch_relevance_scores):
                relevance_scores = batch_relevance_scores[i]
            
            # NDCG@K
            ndcg = self.compute_ndcg_at_k(predictions, ground_truth, k, relevance_scores)
            ndcg_scores.append(ndcg)
            
            # Hit Rate@K
            hit_rate = self.compute_hit_rate_at_k(predictions, ground_truth, k)
            hit_rate_scores.append(hit_rate)
            
            # Diversity
            if destination_types is not None:
                diversity = self.compute_diversity_score(predictions, destination_types)
                diversity_scores.append(diversity)
        
        # Aggregate metrics
        aggregated_metrics = {
            'ndcg_at_k': np.mean(ndcg_scores) if ndcg_scores else 0.0,
            'hit_rate_at_k': np.mean(hit_rate_scores) if hit_rate_scores else 0.0,
        }
        
        if diversity_scores:
            aggregated_metrics['diversity'] = np.mean(diversity_scores)
        
        # Compute coverage if catalog provided
        if catalog is not None:
            coverage = self.compute_coverage_score(batch_predictions, catalog)
            aggregated_metrics['coverage'] = coverage
        
        return aggregated_metrics
