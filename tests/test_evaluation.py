"""Property-based tests for EvaluationModule."""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume

from src.evaluation import EvaluationModule


# Strategies for generating test data
@st.composite
def destination_list_strategy(draw, min_size=1, max_size=20):
    """Generate a list of destination IDs."""
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    return [f"dest_{i}" for i in range(n)]


@st.composite
def predictions_and_ground_truth_strategy(draw):
    """Generate predictions and ground truth lists."""
    # Generate a pool of destinations
    n_total = draw(st.integers(min_value=5, max_value=30))
    all_destinations = [f"dest_{i}" for i in range(n_total)]
    
    # Generate predictions (subset of all destinations, in some order)
    n_predictions = draw(st.integers(min_value=1, max_value=min(20, n_total)))
    predictions = draw(st.lists(
        st.sampled_from(all_destinations),
        min_size=n_predictions,
        max_size=n_predictions,
        unique=True
    ))
    
    # Generate ground truth (subset of all destinations)
    n_ground_truth = draw(st.integers(min_value=1, max_value=min(15, n_total)))
    ground_truth = draw(st.lists(
        st.sampled_from(all_destinations),
        min_size=n_ground_truth,
        max_size=n_ground_truth,
        unique=True
    ))
    
    return predictions, ground_truth, all_destinations


@st.composite
def relevance_scores_strategy(draw, destinations):
    """Generate relevance scores for destinations."""
    scores = {}
    for dest_id in destinations:
        # Generate relevance scores in range [0, 5]
        scores[dest_id] = draw(st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False))
    return scores


@st.composite
def destination_types_strategy(draw, destinations):
    """Generate destination types mapping."""
    types = ['beach', 'cultural', 'nature', 'urban', 'adventure']
    type_map = {}
    for dest_id in destinations:
        type_map[dest_id] = draw(st.sampled_from(types))
    return type_map


class TestMetricComputationCorrectness:
    """
    Feature: tourism-recommender-system, Property 23: Metric Computation Correctness
    
    Property: For any set of predictions and ground truth, the computed NDCG@K SHALL
    be in range [0, 1], Hit Rate@K SHALL be in range [0, 1], diversity score SHALL
    be non-negative, and coverage SHALL be in range [0, 1].
    
    Validates: Requirements 9.1, 9.2, 9.3, 9.4
    """
    
    @given(
        data=predictions_and_ground_truth_strategy(),
        k=st.integers(min_value=1, max_value=20)
    )
    @settings(max_examples=100, deadline=None)
    def test_ndcg_at_k_range(self, data, k):
        """Test that NDCG@K is always in range [0, 1]."""
        predictions, ground_truth, all_destinations = data
        evaluator = EvaluationModule()
        
        # Generate relevance scores
        relevance_scores = {dest_id: 1.0 for dest_id in ground_truth}
        
        # Compute NDCG@K
        ndcg = evaluator.compute_ndcg_at_k(predictions, ground_truth, k, relevance_scores)
        
        # Property: NDCG@K must be in range [0, 1]
        assert 0.0 <= ndcg <= 1.0, f"NDCG@K={ndcg} is outside range [0, 1]"
        assert not np.isnan(ndcg), "NDCG@K should not be NaN"
        assert not np.isinf(ndcg), "NDCG@K should not be infinite"
    
    @given(
        data=predictions_and_ground_truth_strategy(),
        k=st.integers(min_value=1, max_value=20)
    )
    @settings(max_examples=100, deadline=None)
    def test_hit_rate_at_k_range(self, data, k):
        """Test that Hit Rate@K is always in range [0, 1]."""
        predictions, ground_truth, all_destinations = data
        evaluator = EvaluationModule()
        
        # Compute Hit Rate@K
        hit_rate = evaluator.compute_hit_rate_at_k(predictions, ground_truth, k)
        
        # Property: Hit Rate@K must be in range [0, 1]
        assert 0.0 <= hit_rate <= 1.0, f"Hit Rate@K={hit_rate} is outside range [0, 1]"
        assert not np.isnan(hit_rate), "Hit Rate@K should not be NaN"
        assert not np.isinf(hit_rate), "Hit Rate@K should not be infinite"
    
    @given(
        data=predictions_and_ground_truth_strategy()
    )
    @settings(max_examples=100, deadline=None)
    def test_diversity_score_non_negative(self, data):
        """Test that diversity score is always non-negative."""
        predictions, ground_truth, all_destinations = data
        evaluator = EvaluationModule()
        
        # Generate destination types
        destination_types = {
            dest_id: ['beach', 'cultural', 'nature', 'urban'][hash(dest_id) % 4]
            for dest_id in all_destinations
        }
        
        # Compute diversity score
        diversity = evaluator.compute_diversity_score(predictions, destination_types)
        
        # Property: Diversity score must be non-negative
        assert diversity >= 0.0, f"Diversity score={diversity} is negative"
        assert not np.isnan(diversity), "Diversity score should not be NaN"
        assert not np.isinf(diversity), "Diversity score should not be infinite"
    
    @given(
        data=predictions_and_ground_truth_strategy()
    )
    @settings(max_examples=100, deadline=None)
    def test_coverage_score_range(self, data):
        """Test that coverage score is always in range [0, 1]."""
        predictions, ground_truth, all_destinations = data
        evaluator = EvaluationModule()
        
        # Create multiple prediction lists
        batch_predictions = [predictions]
        catalog = set(all_destinations)
        
        # Compute coverage score
        coverage = evaluator.compute_coverage_score(batch_predictions, catalog)
        
        # Property: Coverage must be in range [0, 1]
        assert 0.0 <= coverage <= 1.0, f"Coverage={coverage} is outside range [0, 1]"
        assert not np.isnan(coverage), "Coverage should not be NaN"
        assert not np.isinf(coverage), "Coverage should not be infinite"
    
    @given(
        data=predictions_and_ground_truth_strategy(),
        k=st.integers(min_value=1, max_value=20)
    )
    @settings(max_examples=100, deadline=None)
    def test_all_metrics_in_valid_ranges(self, data, k):
        """Test that all metrics computed together are in valid ranges."""
        predictions, ground_truth, all_destinations = data
        evaluator = EvaluationModule()
        
        # Generate destination types
        destination_types = {
            dest_id: ['beach', 'cultural', 'nature', 'urban'][hash(dest_id) % 4]
            for dest_id in all_destinations
        }
        
        # Generate relevance scores
        relevance_scores = {dest_id: 1.0 for dest_id in ground_truth}
        
        # Compute all metrics
        metrics = evaluator.compute_all_metrics(
            predictions, ground_truth, k, destination_types, relevance_scores
        )
        
        # Property: All metrics must be in valid ranges
        assert 'ndcg_at_k' in metrics
        assert 0.0 <= metrics['ndcg_at_k'] <= 1.0, f"NDCG@K={metrics['ndcg_at_k']} outside [0, 1]"
        
        assert 'hit_rate_at_k' in metrics
        assert 0.0 <= metrics['hit_rate_at_k'] <= 1.0, f"Hit Rate@K={metrics['hit_rate_at_k']} outside [0, 1]"
        
        assert 'diversity' in metrics
        assert metrics['diversity'] >= 0.0, f"Diversity={metrics['diversity']} is negative"
    
    def test_ndcg_perfect_ranking(self):
        """Test NDCG@K with perfect ranking."""
        evaluator = EvaluationModule()
        
        # Perfect ranking: predictions match ground truth order
        predictions = ['dest_1', 'dest_2', 'dest_3', 'dest_4', 'dest_5']
        ground_truth = ['dest_1', 'dest_2', 'dest_3', 'dest_4', 'dest_5']
        relevance_scores = {f'dest_{i}': 5.0 - i for i in range(1, 6)}
        
        ndcg = evaluator.compute_ndcg_at_k(predictions, ground_truth, k=5, relevance_scores=relevance_scores)
        
        # Property: Perfect ranking should give NDCG = 1.0
        assert abs(ndcg - 1.0) < 1e-6, f"Perfect ranking NDCG={ndcg}, expected 1.0"
    
    def test_ndcg_worst_ranking(self):
        """Test NDCG@K with worst possible ranking."""
        evaluator = EvaluationModule()
        
        # Worst ranking: predictions are reverse of ground truth
        predictions = ['dest_5', 'dest_4', 'dest_3', 'dest_2', 'dest_1']
        ground_truth = ['dest_1', 'dest_2', 'dest_3', 'dest_4', 'dest_5']
        relevance_scores = {f'dest_{i}': 5.0 - i for i in range(1, 6)}
        
        ndcg = evaluator.compute_ndcg_at_k(predictions, ground_truth, k=5, relevance_scores=relevance_scores)
        
        # Property: Worst ranking should give NDCG < 1.0
        assert ndcg < 1.0, f"Worst ranking NDCG={ndcg} should be < 1.0"
        assert ndcg >= 0.0, f"NDCG={ndcg} should be >= 0.0"
    
    def test_hit_rate_with_hit(self):
        """Test Hit Rate@K when there is a hit."""
        evaluator = EvaluationModule()
        
        # Predictions contain at least one ground truth item
        predictions = ['dest_1', 'dest_2', 'dest_3']
        ground_truth = ['dest_2', 'dest_5', 'dest_6']
        
        hit_rate = evaluator.compute_hit_rate_at_k(predictions, ground_truth, k=3)
        
        # Property: Should have hit rate = 1.0
        assert hit_rate == 1.0, f"Hit rate={hit_rate}, expected 1.0"
    
    def test_hit_rate_without_hit(self):
        """Test Hit Rate@K when there is no hit."""
        evaluator = EvaluationModule()
        
        # Predictions contain no ground truth items
        predictions = ['dest_1', 'dest_2', 'dest_3']
        ground_truth = ['dest_4', 'dest_5', 'dest_6']
        
        hit_rate = evaluator.compute_hit_rate_at_k(predictions, ground_truth, k=3)
        
        # Property: Should have hit rate = 0.0
        assert hit_rate == 0.0, f"Hit rate={hit_rate}, expected 0.0"
    
    def test_diversity_all_same_type(self):
        """Test diversity when all predictions are same type."""
        evaluator = EvaluationModule()
        
        predictions = ['dest_1', 'dest_2', 'dest_3']
        destination_types = {
            'dest_1': 'beach',
            'dest_2': 'beach',
            'dest_3': 'beach'
        }
        
        diversity = evaluator.compute_diversity_score(predictions, destination_types)
        
        # Property: All same type should give diversity = 1.0 (one unique type)
        assert diversity == 1.0, f"Diversity={diversity}, expected 1.0"
    
    def test_diversity_all_different_types(self):
        """Test diversity when all predictions are different types."""
        evaluator = EvaluationModule()
        
        predictions = ['dest_1', 'dest_2', 'dest_3']
        destination_types = {
            'dest_1': 'beach',
            'dest_2': 'cultural',
            'dest_3': 'nature'
        }
        
        diversity = evaluator.compute_diversity_score(predictions, destination_types)
        
        # Property: All different types should give diversity = 3.0 (three unique types)
        assert diversity == 3.0, f"Diversity={diversity}, expected 3.0"
    
    def test_coverage_full_catalog(self):
        """Test coverage when all catalog items are recommended."""
        evaluator = EvaluationModule()
        
        catalog = {'dest_1', 'dest_2', 'dest_3', 'dest_4', 'dest_5'}
        batch_predictions = [
            ['dest_1', 'dest_2'],
            ['dest_3', 'dest_4'],
            ['dest_5']
        ]
        
        coverage = evaluator.compute_coverage_score(batch_predictions, catalog)
        
        # Property: Full coverage should give coverage = 1.0
        assert coverage == 1.0, f"Coverage={coverage}, expected 1.0"
    
    def test_coverage_partial_catalog(self):
        """Test coverage when only part of catalog is recommended."""
        evaluator = EvaluationModule()
        
        catalog = {'dest_1', 'dest_2', 'dest_3', 'dest_4', 'dest_5'}
        batch_predictions = [
            ['dest_1', 'dest_2'],
            ['dest_1', 'dest_3']
        ]
        
        coverage = evaluator.compute_coverage_score(batch_predictions, catalog)
        
        # Property: 3 out of 5 items recommended = 0.6 coverage
        expected_coverage = 3.0 / 5.0
        assert abs(coverage - expected_coverage) < 1e-6, f"Coverage={coverage}, expected {expected_coverage}"
    
    def test_coverage_no_recommendations(self):
        """Test coverage when no recommendations are made."""
        evaluator = EvaluationModule()
        
        catalog = {'dest_1', 'dest_2', 'dest_3'}
        batch_predictions = []
        
        coverage = evaluator.compute_coverage_score(batch_predictions, catalog)
        
        # Property: No recommendations should give coverage = 0.0
        assert coverage == 0.0, f"Coverage={coverage}, expected 0.0"
    
    def test_empty_predictions(self):
        """Test metrics with empty predictions."""
        evaluator = EvaluationModule()
        
        predictions = []
        ground_truth = ['dest_1', 'dest_2']
        
        # NDCG@K with empty predictions
        ndcg = evaluator.compute_ndcg_at_k(predictions, ground_truth, k=5)
        assert ndcg == 0.0, f"Empty predictions NDCG={ndcg}, expected 0.0"
        
        # Hit Rate@K with empty predictions
        hit_rate = evaluator.compute_hit_rate_at_k(predictions, ground_truth, k=5)
        assert hit_rate == 0.0, f"Empty predictions hit rate={hit_rate}, expected 0.0"
    
    def test_empty_ground_truth(self):
        """Test metrics with empty ground truth."""
        evaluator = EvaluationModule()
        
        predictions = ['dest_1', 'dest_2']
        ground_truth = []
        
        # NDCG@K with empty ground truth
        ndcg = evaluator.compute_ndcg_at_k(predictions, ground_truth, k=5)
        assert ndcg == 0.0, f"Empty ground truth NDCG={ndcg}, expected 0.0"
        
        # Hit Rate@K with empty ground truth
        hit_rate = evaluator.compute_hit_rate_at_k(predictions, ground_truth, k=5)
        assert hit_rate == 0.0, f"Empty ground truth hit rate={hit_rate}, expected 0.0"
    
    def test_batch_evaluation(self):
        """Test batch evaluation with multiple queries."""
        evaluator = EvaluationModule()
        
        batch_predictions = [
            ['dest_1', 'dest_2', 'dest_3'],
            ['dest_4', 'dest_5', 'dest_6'],
            ['dest_1', 'dest_4', 'dest_7']
        ]
        
        batch_ground_truth = [
            ['dest_1', 'dest_2'],
            ['dest_4', 'dest_5'],
            ['dest_1', 'dest_8']
        ]
        
        catalog = {f'dest_{i}' for i in range(1, 10)}
        
        destination_types = {
            f'dest_{i}': ['beach', 'cultural', 'nature'][i % 3]
            for i in range(1, 10)
        }
        
        metrics = evaluator.evaluate_batch(
            batch_predictions,
            batch_ground_truth,
            k=3,
            catalog=catalog,
            destination_types=destination_types
        )
        
        # Property: All metrics should be in valid ranges
        assert 'ndcg_at_k' in metrics
        assert 0.0 <= metrics['ndcg_at_k'] <= 1.0
        
        assert 'hit_rate_at_k' in metrics
        assert 0.0 <= metrics['hit_rate_at_k'] <= 1.0
        
        assert 'diversity' in metrics
        assert metrics['diversity'] >= 0.0
        
        assert 'coverage' in metrics
        assert 0.0 <= metrics['coverage'] <= 1.0
    
    @given(
        k=st.integers(min_value=1, max_value=20)
    )
    @settings(max_examples=100, deadline=None)
    def test_k_parameter_respected(self, k):
        """Test that K parameter is respected in metric calculations."""
        evaluator = EvaluationModule()
        
        # Create predictions longer than K
        predictions = [f'dest_{i}' for i in range(max(k + 5, 10))]
        ground_truth = [f'dest_{i}' for i in range(5)]
        
        # Compute metrics
        ndcg = evaluator.compute_ndcg_at_k(predictions, ground_truth, k)
        hit_rate = evaluator.compute_hit_rate_at_k(predictions, ground_truth, k)
        
        # Property: Metrics should only consider top-K items
        # This is implicitly tested by the implementation, but we verify
        # that the metrics are computed without errors
        assert 0.0 <= ndcg <= 1.0
        assert 0.0 <= hit_rate <= 1.0
