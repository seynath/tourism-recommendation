"""
Comprehensive Research Evaluation Framework for Tourism Recommender System.

This module provides rigorous evaluation methods for undergraduate research including:
- Train/Test/Validation splits with cross-validation
- Baseline comparisons (Random, Popularity, Individual models)
- Standard IR metrics (Precision, Recall, F1, NDCG, MAP)
- Statistical significance testing with confidence intervals
- Ablation study for component contribution analysis
- User study framework for qualitative evaluation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from collections import defaultdict
from dataclasses import dataclass, field
from scipy import stats
from sklearn.model_selection import KFold
import random
import time
from datetime import datetime


@dataclass
class EvaluationMetrics:
    """Container for all evaluation metrics."""
    
    # Ranking metrics
    ndcg_at_k: float = 0.0
    map_at_k: float = 0.0  # Mean Average Precision
    mrr: float = 0.0  # Mean Reciprocal Rank
    
    # Classification metrics
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    f1_at_k: float = 0.0
    
    # Coverage metrics
    hit_rate_at_k: float = 0.0
    coverage: float = 0.0
    diversity: float = 0.0
    novelty: float = 0.0
    
    # Performance metrics
    latency_ms: float = 0.0
    
    # Confidence intervals (for statistical significance)
    ndcg_ci: Tuple[float, float] = (0.0, 0.0)
    precision_ci: Tuple[float, float] = (0.0, 0.0)
    recall_ci: Tuple[float, float] = (0.0, 0.0)


@dataclass
class AblationResult:
    """Results from ablation study."""
    
    component_name: str
    metrics_with: EvaluationMetrics
    metrics_without: EvaluationMetrics
    contribution_percentage: float = 0.0


@dataclass
class BaselineResult:
    """Results from baseline comparison."""
    
    baseline_name: str
    metrics: EvaluationMetrics
    improvement_over_random: float = 0.0


class ResearchEvaluator:
    """
    Comprehensive evaluation framework for research-grade analysis.
    
    Provides:
    1. Proper train/test splits with temporal awareness
    2. K-fold cross-validation
    3. Multiple baseline comparisons
    4. Statistical significance testing
    5. Ablation studies
    6. User study framework
    """
    
    def __init__(self, random_seed: int = 42):
        """Initialize evaluator with random seed for reproducibility."""
        self.random_seed = random_seed
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # Store results for reporting
        self.evaluation_history: List[Dict] = []
        self.baseline_results: Dict[str, BaselineResult] = {}
        self.ablation_results: List[AblationResult] = []
    
    # ==================== DATA SPLITTING ====================
    
    def temporal_train_test_split(
        self,
        df: pd.DataFrame,
        test_ratio: float = 0.2,
        validation_ratio: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data temporally - older reviews for training, newer for testing.
        
        This is more realistic than random splitting because it simulates
        how the system would be used in production (trained on past data,
        predicting future preferences).
        
        Args:
            df: DataFrame with 'published_date' column
            test_ratio: Fraction of data for testing
            validation_ratio: Fraction of data for validation
            
        Returns:
            Tuple of (train_df, validation_df, test_df)
        """
        # Sort by date
        df_sorted = df.sort_values('published_date', ascending=True).reset_index(drop=True)
        
        n = len(df_sorted)
        train_end = int(n * (1 - test_ratio - validation_ratio))
        val_end = int(n * (1 - test_ratio))
        
        train_df = df_sorted.iloc[:train_end].copy()
        val_df = df_sorted.iloc[train_end:val_end].copy()
        test_df = df_sorted.iloc[val_end:].copy()
        
        return train_df, val_df, test_df
    
    def user_based_split(
        self,
        df: pd.DataFrame,
        test_ratio: float = 0.2,
        min_interactions: int = 5
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data by holding out some interactions per user.
        
        For each user with enough interactions, hold out a portion
        for testing. This tests personalization quality.
        
        Args:
            df: DataFrame with user interactions
            test_ratio: Fraction of each user's interactions to hold out
            min_interactions: Minimum interactions needed to include user
            
        Returns:
            Tuple of (train_df, test_df)
        """
        train_rows = []
        test_rows = []
        
        for user_id, group in df.groupby('user_id'):
            if len(group) < min_interactions:
                # Not enough data - use all for training
                train_rows.append(group)
            else:
                # Sort by date and split
                group_sorted = group.sort_values('published_date')
                n = len(group_sorted)
                split_idx = int(n * (1 - test_ratio))
                
                train_rows.append(group_sorted.iloc[:split_idx])
                test_rows.append(group_sorted.iloc[split_idx:])
        
        train_df = pd.concat(train_rows, ignore_index=True) if train_rows else pd.DataFrame()
        test_df = pd.concat(test_rows, ignore_index=True) if test_rows else pd.DataFrame()
        
        return train_df, test_df
    
    def k_fold_cross_validation(
        self,
        df: pd.DataFrame,
        k: int = 5
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generate K-fold cross-validation splits.
        
        Args:
            df: DataFrame to split
            k: Number of folds
            
        Returns:
            List of (train_df, test_df) tuples
        """
        kf = KFold(n_splits=k, shuffle=True, random_state=self.random_seed)
        
        folds = []
        indices = np.arange(len(df))
        
        for train_idx, test_idx in kf.split(indices):
            train_df = df.iloc[train_idx].copy()
            test_df = df.iloc[test_idx].copy()
            folds.append((train_df, test_df))
        
        return folds
    
    # ==================== METRICS COMPUTATION ====================
    
    def compute_precision_at_k(
        self,
        predictions: List[str],
        ground_truth: List[str],
        k: int
    ) -> float:
        """
        Compute Precision@K.
        
        Precision@K = |relevant items in top-K| / K
        
        Args:
            predictions: Ranked list of predicted item IDs
            ground_truth: List of relevant item IDs
            k: Number of top items to consider
            
        Returns:
            Precision@K score in [0, 1]
        """
        if k <= 0 or not predictions:
            return 0.0
        
        top_k = set(predictions[:k])
        relevant = set(ground_truth)
        
        return len(top_k & relevant) / k
    
    def compute_recall_at_k(
        self,
        predictions: List[str],
        ground_truth: List[str],
        k: int
    ) -> float:
        """
        Compute Recall@K.
        
        Recall@K = |relevant items in top-K| / |all relevant items|
        
        Args:
            predictions: Ranked list of predicted item IDs
            ground_truth: List of relevant item IDs
            k: Number of top items to consider
            
        Returns:
            Recall@K score in [0, 1]
        """
        if not ground_truth or k <= 0:
            return 0.0
        
        top_k = set(predictions[:k])
        relevant = set(ground_truth)
        
        return len(top_k & relevant) / len(relevant)
    
    def compute_f1_at_k(
        self,
        predictions: List[str],
        ground_truth: List[str],
        k: int
    ) -> float:
        """
        Compute F1@K (harmonic mean of Precision@K and Recall@K).
        
        Args:
            predictions: Ranked list of predicted item IDs
            ground_truth: List of relevant item IDs
            k: Number of top items to consider
            
        Returns:
            F1@K score in [0, 1]
        """
        precision = self.compute_precision_at_k(predictions, ground_truth, k)
        recall = self.compute_recall_at_k(predictions, ground_truth, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def compute_ndcg_at_k(
        self,
        predictions: List[str],
        ground_truth: List[str],
        k: int,
        relevance_scores: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Compute Normalized Discounted Cumulative Gain at K.
        
        Args:
            predictions: Ranked list of predicted item IDs
            ground_truth: List of relevant item IDs
            k: Number of top items to consider
            relevance_scores: Optional relevance scores (default: binary)
            
        Returns:
            NDCG@K score in [0, 1]
        """
        if not predictions or not ground_truth or k <= 0:
            return 0.0
        
        # Use binary relevance if no scores provided
        if relevance_scores is None:
            relevance_scores = {item: 1.0 for item in ground_truth}
        
        # Compute DCG
        dcg = 0.0
        for i, item in enumerate(predictions[:k]):
            if item in relevance_scores:
                dcg += relevance_scores[item] / np.log2(i + 2)
        
        # Compute IDCG (ideal DCG)
        ideal_ranking = sorted(
            ground_truth,
            key=lambda x: relevance_scores.get(x, 0.0),
            reverse=True
        )[:k]
        
        idcg = 0.0
        for i, item in enumerate(ideal_ranking):
            idcg += relevance_scores.get(item, 0.0) / np.log2(i + 2)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def compute_map_at_k(
        self,
        predictions: List[str],
        ground_truth: List[str],
        k: int
    ) -> float:
        """
        Compute Mean Average Precision at K.
        
        MAP@K = (1/min(k, |relevant|)) * sum(P@i * rel(i)) for i in 1..k
        
        Args:
            predictions: Ranked list of predicted item IDs
            ground_truth: List of relevant item IDs
            k: Number of top items to consider
            
        Returns:
            MAP@K score in [0, 1]
        """
        if not predictions or not ground_truth or k <= 0:
            return 0.0
        
        relevant = set(ground_truth)
        num_relevant = min(k, len(relevant))
        
        if num_relevant == 0:
            return 0.0
        
        ap_sum = 0.0
        relevant_count = 0
        
        for i, item in enumerate(predictions[:k]):
            if item in relevant:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                ap_sum += precision_at_i
        
        return ap_sum / num_relevant
    
    def compute_mrr(
        self,
        predictions: List[str],
        ground_truth: List[str]
    ) -> float:
        """
        Compute Mean Reciprocal Rank.
        
        MRR = 1 / rank of first relevant item
        
        Args:
            predictions: Ranked list of predicted item IDs
            ground_truth: List of relevant item IDs
            
        Returns:
            MRR score in [0, 1]
        """
        if not predictions or not ground_truth:
            return 0.0
        
        relevant = set(ground_truth)
        
        for i, item in enumerate(predictions):
            if item in relevant:
                return 1.0 / (i + 1)
        
        return 0.0
    
    def compute_hit_rate_at_k(
        self,
        predictions: List[str],
        ground_truth: List[str],
        k: int
    ) -> float:
        """
        Compute Hit Rate at K (binary: 1 if any relevant item in top-K).
        
        Args:
            predictions: Ranked list of predicted item IDs
            ground_truth: List of relevant item IDs
            k: Number of top items to consider
            
        Returns:
            1.0 if hit, 0.0 otherwise
        """
        if not predictions or not ground_truth or k <= 0:
            return 0.0
        
        top_k = set(predictions[:k])
        relevant = set(ground_truth)
        
        return 1.0 if len(top_k & relevant) > 0 else 0.0
    
    def compute_diversity(
        self,
        predictions: List[str],
        item_categories: Dict[str, str]
    ) -> float:
        """
        Compute diversity as number of unique categories in recommendations.
        
        Args:
            predictions: List of recommended item IDs
            item_categories: Mapping from item ID to category
            
        Returns:
            Number of unique categories
        """
        if not predictions:
            return 0.0
        
        categories = set()
        for item in predictions:
            if item in item_categories:
                categories.add(item_categories[item])
        
        return float(len(categories))
    
    def compute_novelty(
        self,
        predictions: List[str],
        item_popularity: Dict[str, int],
        total_interactions: int
    ) -> float:
        """
        Compute novelty as average self-information of recommended items.
        
        Novelty = -log2(popularity) averaged over recommendations.
        Higher novelty means recommending less popular (more novel) items.
        
        Args:
            predictions: List of recommended item IDs
            item_popularity: Mapping from item ID to interaction count
            total_interactions: Total number of interactions in dataset
            
        Returns:
            Average novelty score
        """
        if not predictions or total_interactions == 0:
            return 0.0
        
        novelty_sum = 0.0
        count = 0
        
        for item in predictions:
            pop = item_popularity.get(item, 1)
            prob = pop / total_interactions
            if prob > 0:
                novelty_sum += -np.log2(prob)
                count += 1
        
        return novelty_sum / count if count > 0 else 0.0
    
    def compute_coverage(
        self,
        all_predictions: List[List[str]],
        catalog: set
    ) -> float:
        """
        Compute catalog coverage.
        
        Coverage = |unique recommended items| / |catalog|
        
        Args:
            all_predictions: List of recommendation lists
            catalog: Set of all available items
            
        Returns:
            Coverage score in [0, 1]
        """
        if not catalog:
            return 0.0
        
        recommended = set()
        for preds in all_predictions:
            recommended.update(preds)
        
        return len(recommended & catalog) / len(catalog)
    
    def compute_all_metrics(
        self,
        predictions: List[str],
        ground_truth: List[str],
        k: int,
        item_categories: Optional[Dict[str, str]] = None,
        relevance_scores: Optional[Dict[str, float]] = None
    ) -> EvaluationMetrics:
        """
        Compute all metrics for a single prediction.
        
        Args:
            predictions: Ranked list of predicted item IDs
            ground_truth: List of relevant item IDs
            k: Number of top items to consider
            item_categories: Optional category mapping for diversity
            relevance_scores: Optional relevance scores for NDCG
            
        Returns:
            EvaluationMetrics object with all computed metrics
        """
        metrics = EvaluationMetrics()
        
        metrics.precision_at_k = self.compute_precision_at_k(predictions, ground_truth, k)
        metrics.recall_at_k = self.compute_recall_at_k(predictions, ground_truth, k)
        metrics.f1_at_k = self.compute_f1_at_k(predictions, ground_truth, k)
        metrics.ndcg_at_k = self.compute_ndcg_at_k(predictions, ground_truth, k, relevance_scores)
        metrics.map_at_k = self.compute_map_at_k(predictions, ground_truth, k)
        metrics.mrr = self.compute_mrr(predictions, ground_truth)
        metrics.hit_rate_at_k = self.compute_hit_rate_at_k(predictions, ground_truth, k)
        
        if item_categories:
            metrics.diversity = self.compute_diversity(predictions, item_categories)
        
        return metrics

    
    # ==================== STATISTICAL SIGNIFICANCE ====================
    
    def compute_confidence_interval(
        self,
        scores: List[float],
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Compute confidence interval for a list of scores.
        
        Uses t-distribution for small samples.
        
        Args:
            scores: List of metric scores
            confidence: Confidence level (default 0.95 for 95% CI)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if len(scores) < 2:
            mean = np.mean(scores) if scores else 0.0
            return (mean, mean)
        
        n = len(scores)
        mean = np.mean(scores)
        std_err = stats.sem(scores)
        
        # t-value for confidence interval
        t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
        
        margin = t_value * std_err
        
        return (mean - margin, mean + margin)
    
    def paired_t_test(
        self,
        scores_a: List[float],
        scores_b: List[float]
    ) -> Tuple[float, float]:
        """
        Perform paired t-test between two sets of scores.
        
        Args:
            scores_a: Scores from method A
            scores_b: Scores from method B
            
        Returns:
            Tuple of (t_statistic, p_value)
        """
        if len(scores_a) != len(scores_b) or len(scores_a) < 2:
            return (0.0, 1.0)
        
        t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
        
        return (t_stat, p_value)
    
    def wilcoxon_test(
        self,
        scores_a: List[float],
        scores_b: List[float]
    ) -> Tuple[float, float]:
        """
        Perform Wilcoxon signed-rank test (non-parametric alternative to t-test).
        
        Args:
            scores_a: Scores from method A
            scores_b: Scores from method B
            
        Returns:
            Tuple of (statistic, p_value)
        """
        if len(scores_a) != len(scores_b) or len(scores_a) < 2:
            return (0.0, 1.0)
        
        try:
            stat, p_value = stats.wilcoxon(scores_a, scores_b)
            return (stat, p_value)
        except ValueError:
            # All differences are zero
            return (0.0, 1.0)
    
    def bootstrap_confidence_interval(
        self,
        scores: List[float],
        n_bootstrap: int = 1000,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Compute bootstrap confidence interval.
        
        More robust for non-normal distributions.
        
        Args:
            scores: List of metric scores
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if len(scores) < 2:
            mean = np.mean(scores) if scores else 0.0
            return (mean, mean)
        
        bootstrap_means = []
        n = len(scores)
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(scores, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))
        
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_means, alpha / 2 * 100)
        upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)
        
        return (lower, upper)
    
    # ==================== BASELINE COMPARISONS ====================
    
    def random_baseline(
        self,
        catalog: List[str],
        k: int
    ) -> List[str]:
        """
        Generate random recommendations.
        
        Args:
            catalog: List of all available items
            k: Number of recommendations
            
        Returns:
            Random list of k items
        """
        return random.sample(catalog, min(k, len(catalog)))
    
    def popularity_baseline(
        self,
        item_popularity: Dict[str, int],
        k: int
    ) -> List[str]:
        """
        Generate popularity-based recommendations.
        
        Args:
            item_popularity: Mapping from item ID to interaction count
            k: Number of recommendations
            
        Returns:
            Top-k most popular items
        """
        sorted_items = sorted(
            item_popularity.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [item for item, _ in sorted_items[:k]]
    
    def evaluate_baseline(
        self,
        baseline_name: str,
        baseline_predictions: List[List[str]],
        ground_truths: List[List[str]],
        k: int,
        item_categories: Optional[Dict[str, str]] = None
    ) -> BaselineResult:
        """
        Evaluate a baseline method.
        
        Args:
            baseline_name: Name of the baseline
            baseline_predictions: List of prediction lists
            ground_truths: List of ground truth lists
            k: Number of top items to consider
            item_categories: Optional category mapping
            
        Returns:
            BaselineResult with aggregated metrics
        """
        all_metrics = []
        
        for preds, gt in zip(baseline_predictions, ground_truths):
            metrics = self.compute_all_metrics(preds, gt, k, item_categories)
            all_metrics.append(metrics)
        
        # Aggregate metrics
        aggregated = EvaluationMetrics()
        
        if all_metrics:
            aggregated.precision_at_k = np.mean([m.precision_at_k for m in all_metrics])
            aggregated.recall_at_k = np.mean([m.recall_at_k for m in all_metrics])
            aggregated.f1_at_k = np.mean([m.f1_at_k for m in all_metrics])
            aggregated.ndcg_at_k = np.mean([m.ndcg_at_k for m in all_metrics])
            aggregated.map_at_k = np.mean([m.map_at_k for m in all_metrics])
            aggregated.mrr = np.mean([m.mrr for m in all_metrics])
            aggregated.hit_rate_at_k = np.mean([m.hit_rate_at_k for m in all_metrics])
            aggregated.diversity = np.mean([m.diversity for m in all_metrics])
            
            # Compute confidence intervals
            aggregated.ndcg_ci = self.compute_confidence_interval(
                [m.ndcg_at_k for m in all_metrics]
            )
            aggregated.precision_ci = self.compute_confidence_interval(
                [m.precision_at_k for m in all_metrics]
            )
            aggregated.recall_ci = self.compute_confidence_interval(
                [m.recall_at_k for m in all_metrics]
            )
        
        result = BaselineResult(
            baseline_name=baseline_name,
            metrics=aggregated
        )
        
        self.baseline_results[baseline_name] = result
        
        return result
    
    # ==================== ABLATION STUDY ====================
    
    def ablation_study(
        self,
        full_system_predictions: List[List[str]],
        ablated_predictions: Dict[str, List[List[str]]],
        ground_truths: List[List[str]],
        k: int,
        item_categories: Optional[Dict[str, str]] = None
    ) -> List[AblationResult]:
        """
        Perform ablation study to measure component contributions.
        
        Args:
            full_system_predictions: Predictions from full system
            ablated_predictions: Dict mapping component name to predictions without that component
            ground_truths: List of ground truth lists
            k: Number of top items to consider
            item_categories: Optional category mapping
            
        Returns:
            List of AblationResult objects
        """
        # Evaluate full system
        full_result = self.evaluate_baseline(
            "Full System",
            full_system_predictions,
            ground_truths,
            k,
            item_categories
        )
        
        results = []
        
        for component_name, ablated_preds in ablated_predictions.items():
            # Evaluate system without this component
            ablated_result = self.evaluate_baseline(
                f"Without {component_name}",
                ablated_preds,
                ground_truths,
                k,
                item_categories
            )
            
            # Calculate contribution
            full_ndcg = full_result.metrics.ndcg_at_k
            ablated_ndcg = ablated_result.metrics.ndcg_at_k
            
            if ablated_ndcg > 0:
                contribution = ((full_ndcg - ablated_ndcg) / ablated_ndcg) * 100
            else:
                contribution = 100.0 if full_ndcg > 0 else 0.0
            
            ablation_result = AblationResult(
                component_name=component_name,
                metrics_with=full_result.metrics,
                metrics_without=ablated_result.metrics,
                contribution_percentage=contribution
            )
            
            results.append(ablation_result)
        
        self.ablation_results = results
        return results
    
    # ==================== CROSS-VALIDATION EVALUATION ====================
    
    def cross_validate(
        self,
        df: pd.DataFrame,
        train_and_predict_fn: Callable,
        k_folds: int = 5,
        top_k: int = 10,
        item_categories: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Perform k-fold cross-validation.
        
        Args:
            df: Full dataset
            train_and_predict_fn: Function that takes (train_df, test_users) and returns predictions
            k_folds: Number of folds
            top_k: Number of recommendations to evaluate
            item_categories: Optional category mapping
            
        Returns:
            Dictionary with aggregated metrics and per-fold results
        """
        folds = self.k_fold_cross_validation(df, k_folds)
        
        fold_metrics = []
        
        for fold_idx, (train_df, test_df) in enumerate(folds):
            print(f"  Evaluating fold {fold_idx + 1}/{k_folds}...")
            
            # Get test users and their ground truth
            test_users = test_df['user_id'].unique()
            ground_truths = {}
            
            for user_id in test_users:
                user_test = test_df[test_df['user_id'] == user_id]
                # Ground truth: destinations with rating >= 4
                relevant = user_test[user_test['rating'] >= 4]['destination_id'].tolist()
                if relevant:
                    ground_truths[user_id] = relevant
            
            # Get predictions
            predictions = train_and_predict_fn(train_df, list(ground_truths.keys()))
            
            # Evaluate
            metrics_list = []
            for user_id, gt in ground_truths.items():
                if user_id in predictions:
                    preds = predictions[user_id]
                    metrics = self.compute_all_metrics(preds, gt, top_k, item_categories)
                    metrics_list.append(metrics)
            
            if metrics_list:
                fold_result = {
                    'fold': fold_idx + 1,
                    'ndcg': np.mean([m.ndcg_at_k for m in metrics_list]),
                    'precision': np.mean([m.precision_at_k for m in metrics_list]),
                    'recall': np.mean([m.recall_at_k for m in metrics_list]),
                    'f1': np.mean([m.f1_at_k for m in metrics_list]),
                    'hit_rate': np.mean([m.hit_rate_at_k for m in metrics_list]),
                    'map': np.mean([m.map_at_k for m in metrics_list]),
                    'mrr': np.mean([m.mrr for m in metrics_list]),
                }
                fold_metrics.append(fold_result)
        
        # Aggregate across folds
        if fold_metrics:
            aggregated = {
                'mean_ndcg': np.mean([f['ndcg'] for f in fold_metrics]),
                'std_ndcg': np.std([f['ndcg'] for f in fold_metrics]),
                'mean_precision': np.mean([f['precision'] for f in fold_metrics]),
                'std_precision': np.std([f['precision'] for f in fold_metrics]),
                'mean_recall': np.mean([f['recall'] for f in fold_metrics]),
                'std_recall': np.std([f['recall'] for f in fold_metrics]),
                'mean_f1': np.mean([f['f1'] for f in fold_metrics]),
                'std_f1': np.std([f['f1'] for f in fold_metrics]),
                'mean_hit_rate': np.mean([f['hit_rate'] for f in fold_metrics]),
                'std_hit_rate': np.std([f['hit_rate'] for f in fold_metrics]),
                'mean_map': np.mean([f['map'] for f in fold_metrics]),
                'std_map': np.std([f['map'] for f in fold_metrics]),
                'mean_mrr': np.mean([f['mrr'] for f in fold_metrics]),
                'std_mrr': np.std([f['mrr'] for f in fold_metrics]),
                'fold_results': fold_metrics,
                'ndcg_ci': self.compute_confidence_interval([f['ndcg'] for f in fold_metrics]),
                'precision_ci': self.compute_confidence_interval([f['precision'] for f in fold_metrics]),
            }
        else:
            aggregated = {}
        
        return aggregated



@dataclass
class UserStudyQuestion:
    """Question for user study."""
    
    question_id: str
    question_text: str
    question_type: str  # 'likert', 'ranking', 'open_ended', 'binary'
    options: Optional[List[str]] = None


@dataclass
class UserStudyResponse:
    """Response from user study."""
    
    participant_id: str
    question_id: str
    response: Any
    timestamp: datetime = field(default_factory=datetime.now)


class UserStudyFramework:
    """
    Framework for conducting user studies.
    
    Provides:
    - Standardized questionnaires
    - Response collection
    - Statistical analysis of responses
    """
    
    def __init__(self):
        self.questions: List[UserStudyQuestion] = []
        self.responses: List[UserStudyResponse] = []
        self._setup_default_questions()
    
    def _setup_default_questions(self):
        """Setup default user study questions for recommender evaluation."""
        
        # Relevance questions (Likert scale 1-5)
        self.questions.extend([
            UserStudyQuestion(
                question_id="rel_1",
                question_text="The recommended destinations match my travel preferences.",
                question_type="likert",
                options=["1 - Strongly Disagree", "2 - Disagree", "3 - Neutral", "4 - Agree", "5 - Strongly Agree"]
            ),
            UserStudyQuestion(
                question_id="rel_2",
                question_text="I would consider visiting the recommended destinations.",
                question_type="likert",
                options=["1 - Strongly Disagree", "2 - Disagree", "3 - Neutral", "4 - Agree", "5 - Strongly Agree"]
            ),
            UserStudyQuestion(
                question_id="rel_3",
                question_text="The recommendations are appropriate for the current weather/season.",
                question_type="likert",
                options=["1 - Strongly Disagree", "2 - Disagree", "3 - Neutral", "4 - Agree", "5 - Strongly Agree"]
            ),
        ])
        
        # Diversity questions
        self.questions.extend([
            UserStudyQuestion(
                question_id="div_1",
                question_text="The recommendations offer a good variety of destination types.",
                question_type="likert",
                options=["1 - Strongly Disagree", "2 - Disagree", "3 - Neutral", "4 - Agree", "5 - Strongly Agree"]
            ),
            UserStudyQuestion(
                question_id="div_2",
                question_text="I discovered new destinations I wasn't aware of before.",
                question_type="likert",
                options=["1 - Strongly Disagree", "2 - Disagree", "3 - Neutral", "4 - Agree", "5 - Strongly Agree"]
            ),
        ])
        
        # Usability questions
        self.questions.extend([
            UserStudyQuestion(
                question_id="use_1",
                question_text="The recommendation explanations are helpful.",
                question_type="likert",
                options=["1 - Strongly Disagree", "2 - Disagree", "3 - Neutral", "4 - Agree", "5 - Strongly Agree"]
            ),
            UserStudyQuestion(
                question_id="use_2",
                question_text="The system responds quickly enough.",
                question_type="likert",
                options=["1 - Strongly Disagree", "2 - Disagree", "3 - Neutral", "4 - Agree", "5 - Strongly Agree"]
            ),
        ])
        
        # Overall satisfaction
        self.questions.extend([
            UserStudyQuestion(
                question_id="sat_1",
                question_text="Overall, I am satisfied with the recommendations.",
                question_type="likert",
                options=["1 - Strongly Disagree", "2 - Disagree", "3 - Neutral", "4 - Agree", "5 - Strongly Agree"]
            ),
            UserStudyQuestion(
                question_id="sat_2",
                question_text="I would use this system to plan my trips to Sri Lanka.",
                question_type="likert",
                options=["1 - Strongly Disagree", "2 - Disagree", "3 - Neutral", "4 - Agree", "5 - Strongly Agree"]
            ),
        ])
        
        # Comparison question
        self.questions.append(
            UserStudyQuestion(
                question_id="comp_1",
                question_text="Rank the recommendation approaches from best to worst:",
                question_type="ranking",
                options=["Ensemble (Combined)", "Collaborative Filtering", "Content-Based", "Context-Aware"]
            )
        )
        
        # Open-ended feedback
        self.questions.append(
            UserStudyQuestion(
                question_id="open_1",
                question_text="What improvements would you suggest for the recommendation system?",
                question_type="open_ended"
            )
        )
    
    def get_questionnaire(self) -> List[Dict]:
        """Get questionnaire in dictionary format for display."""
        return [
            {
                'id': q.question_id,
                'text': q.question_text,
                'type': q.question_type,
                'options': q.options
            }
            for q in self.questions
        ]
    
    def record_response(
        self,
        participant_id: str,
        question_id: str,
        response: Any
    ):
        """Record a user study response."""
        self.responses.append(UserStudyResponse(
            participant_id=participant_id,
            question_id=question_id,
            response=response
        ))
    
    def analyze_likert_responses(self) -> Dict[str, Dict]:
        """
        Analyze Likert scale responses.
        
        Returns:
            Dictionary with statistics for each question
        """
        likert_questions = [q for q in self.questions if q.question_type == 'likert']
        
        results = {}
        
        for question in likert_questions:
            q_responses = [
                r.response for r in self.responses
                if r.question_id == question.question_id
            ]
            
            # Convert to numeric (assuming 1-5 scale)
            numeric_responses = []
            for resp in q_responses:
                if isinstance(resp, (int, float)):
                    numeric_responses.append(resp)
                elif isinstance(resp, str) and resp[0].isdigit():
                    numeric_responses.append(int(resp[0]))
            
            if numeric_responses:
                results[question.question_id] = {
                    'question': question.question_text,
                    'n': len(numeric_responses),
                    'mean': np.mean(numeric_responses),
                    'std': np.std(numeric_responses),
                    'median': np.median(numeric_responses),
                    'min': min(numeric_responses),
                    'max': max(numeric_responses),
                    'distribution': {
                        i: numeric_responses.count(i) for i in range(1, 6)
                    }
                }
        
        return results
    
    def compute_sus_score(self) -> float:
        """
        Compute System Usability Scale (SUS) score if applicable.
        
        SUS is a standard usability metric (0-100 scale).
        
        Returns:
            SUS score or -1 if not enough data
        """
        # This is a simplified version - full SUS requires 10 specific questions
        usability_responses = [
            r for r in self.responses
            if r.question_id.startswith('use_') or r.question_id.startswith('sat_')
        ]
        
        if not usability_responses:
            return -1.0
        
        # Average of usability/satisfaction scores, scaled to 0-100
        scores = []
        for resp in usability_responses:
            if isinstance(resp.response, (int, float)):
                scores.append(resp.response)
            elif isinstance(resp.response, str) and resp.response[0].isdigit():
                scores.append(int(resp.response[0]))
        
        if scores:
            # Scale from 1-5 to 0-100
            return ((np.mean(scores) - 1) / 4) * 100
        
        return -1.0
    
    def generate_report(self) -> str:
        """Generate a summary report of user study results."""
        report = []
        report.append("=" * 60)
        report.append("USER STUDY RESULTS REPORT")
        report.append("=" * 60)
        report.append(f"\nTotal Participants: {len(set(r.participant_id for r in self.responses))}")
        report.append(f"Total Responses: {len(self.responses)}")
        
        # Likert analysis
        likert_results = self.analyze_likert_responses()
        
        if likert_results:
            report.append("\n" + "-" * 40)
            report.append("LIKERT SCALE RESPONSES")
            report.append("-" * 40)
            
            for q_id, stats in likert_results.items():
                report.append(f"\n{stats['question']}")
                report.append(f"  Mean: {stats['mean']:.2f} (SD: {stats['std']:.2f})")
                report.append(f"  Median: {stats['median']:.1f}")
                report.append(f"  N: {stats['n']}")
        
        # SUS score
        sus = self.compute_sus_score()
        if sus >= 0:
            report.append("\n" + "-" * 40)
            report.append("USABILITY SCORE")
            report.append("-" * 40)
            report.append(f"  SUS Score: {sus:.1f}/100")
            
            if sus >= 80:
                report.append("  Interpretation: Excellent")
            elif sus >= 68:
                report.append("  Interpretation: Good")
            elif sus >= 50:
                report.append("  Interpretation: OK")
            else:
                report.append("  Interpretation: Needs Improvement")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)


class DatasetAnalyzer:
    """
    Analyze dataset quality and characteristics.
    
    Provides insights into:
    - Data distribution
    - Sparsity
    - User/item coverage
    - Temporal patterns
    """
    
    def __init__(self, df: pd.DataFrame):
        """Initialize with dataset."""
        self.df = df
    
    def get_basic_stats(self) -> Dict[str, Any]:
        """Get basic dataset statistics."""
        return {
            'total_reviews': len(self.df),
            'unique_users': self.df['user_id'].nunique(),
            'unique_destinations': self.df['destination_id'].nunique(),
            'avg_reviews_per_user': len(self.df) / self.df['user_id'].nunique(),
            'avg_reviews_per_destination': len(self.df) / self.df['destination_id'].nunique(),
            'rating_mean': self.df['rating'].mean(),
            'rating_std': self.df['rating'].std(),
            'rating_distribution': self.df['rating'].value_counts().to_dict(),
        }
    
    def get_sparsity(self) -> float:
        """
        Calculate matrix sparsity.
        
        Sparsity = 1 - (observed / possible)
        
        Returns:
            Sparsity ratio (0 = dense, 1 = sparse)
        """
        n_users = self.df['user_id'].nunique()
        n_items = self.df['destination_id'].nunique()
        n_observed = len(self.df)
        n_possible = n_users * n_items
        
        return 1 - (n_observed / n_possible) if n_possible > 0 else 1.0
    
    def get_user_distribution(self) -> Dict[str, Any]:
        """Analyze user activity distribution."""
        user_counts = self.df.groupby('user_id').size()
        
        return {
            'min_reviews': user_counts.min(),
            'max_reviews': user_counts.max(),
            'median_reviews': user_counts.median(),
            'users_with_1_review': (user_counts == 1).sum(),
            'users_with_5plus_reviews': (user_counts >= 5).sum(),
            'users_with_10plus_reviews': (user_counts >= 10).sum(),
            'cold_start_ratio': (user_counts < 5).sum() / len(user_counts),
        }
    
    def get_item_distribution(self) -> Dict[str, Any]:
        """Analyze item popularity distribution."""
        item_counts = self.df.groupby('destination_id').size()
        
        return {
            'min_reviews': item_counts.min(),
            'max_reviews': item_counts.max(),
            'median_reviews': item_counts.median(),
            'items_with_1_review': (item_counts == 1).sum(),
            'items_with_10plus_reviews': (item_counts >= 10).sum(),
            'items_with_50plus_reviews': (item_counts >= 50).sum(),
            'long_tail_ratio': (item_counts < item_counts.median()).sum() / len(item_counts),
        }
    
    def get_temporal_distribution(self) -> Dict[str, Any]:
        """Analyze temporal patterns in the data."""
        if 'published_date' not in self.df.columns:
            return {}
        
        df_with_dates = self.df.dropna(subset=['published_date'])
        
        if len(df_with_dates) == 0:
            return {}
        
        return {
            'date_range_start': df_with_dates['published_date'].min(),
            'date_range_end': df_with_dates['published_date'].max(),
            'reviews_with_dates': len(df_with_dates),
            'reviews_without_dates': len(self.df) - len(df_with_dates),
        }
    
    def get_location_type_distribution(self) -> Dict[str, int]:
        """Get distribution of location types."""
        if 'location_type' not in self.df.columns:
            return {}
        
        return self.df['location_type'].value_counts().to_dict()
    
    def generate_report(self) -> str:
        """Generate comprehensive dataset analysis report."""
        report = []
        report.append("=" * 60)
        report.append("DATASET ANALYSIS REPORT")
        report.append("=" * 60)
        
        # Basic stats
        basic = self.get_basic_stats()
        report.append("\n--- BASIC STATISTICS ---")
        report.append(f"Total Reviews: {basic['total_reviews']:,}")
        report.append(f"Unique Users: {basic['unique_users']:,}")
        report.append(f"Unique Destinations: {basic['unique_destinations']:,}")
        report.append(f"Avg Reviews/User: {basic['avg_reviews_per_user']:.2f}")
        report.append(f"Avg Reviews/Destination: {basic['avg_reviews_per_destination']:.2f}")
        report.append(f"Rating Mean: {basic['rating_mean']:.2f}")
        report.append(f"Rating Std: {basic['rating_std']:.2f}")
        
        # Sparsity
        sparsity = self.get_sparsity()
        report.append(f"\n--- SPARSITY ---")
        report.append(f"Matrix Sparsity: {sparsity:.4f} ({sparsity*100:.2f}%)")
        
        # User distribution
        user_dist = self.get_user_distribution()
        report.append(f"\n--- USER DISTRIBUTION ---")
        report.append(f"Min Reviews/User: {user_dist['min_reviews']}")
        report.append(f"Max Reviews/User: {user_dist['max_reviews']}")
        report.append(f"Median Reviews/User: {user_dist['median_reviews']:.1f}")
        report.append(f"Cold Start Users (<5 reviews): {user_dist['cold_start_ratio']*100:.1f}%")
        
        # Item distribution
        item_dist = self.get_item_distribution()
        report.append(f"\n--- DESTINATION DISTRIBUTION ---")
        report.append(f"Min Reviews/Destination: {item_dist['min_reviews']}")
        report.append(f"Max Reviews/Destination: {item_dist['max_reviews']}")
        report.append(f"Median Reviews/Destination: {item_dist['median_reviews']:.1f}")
        report.append(f"Long Tail Ratio: {item_dist['long_tail_ratio']*100:.1f}%")
        
        # Location types
        loc_types = self.get_location_type_distribution()
        if loc_types:
            report.append(f"\n--- LOCATION TYPES ---")
            for loc_type, count in sorted(loc_types.items(), key=lambda x: x[1], reverse=True):
                report.append(f"  {loc_type}: {count}")
        
        # Temporal
        temporal = self.get_temporal_distribution()
        if temporal:
            report.append(f"\n--- TEMPORAL COVERAGE ---")
            report.append(f"Date Range: {temporal['date_range_start']} to {temporal['date_range_end']}")
            report.append(f"Reviews with Dates: {temporal['reviews_with_dates']:,}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
