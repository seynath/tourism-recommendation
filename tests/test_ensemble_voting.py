"""Property-based tests for EnsembleVotingSystem module."""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from collections import defaultdict

from src.ensemble_voting import EnsembleVotingSystem
from src.data_models import Context, WeatherInfo


# Strategies for generating test data
@st.composite
def predictions_strategy(draw):
    """Generate predictions from multiple models."""
    n_destinations = draw(st.integers(min_value=3, max_value=20))
    destination_ids = [f"dest_{i}" for i in range(n_destinations)]
    
    # Generate predictions for each model
    predictions = {}
    
    # Collaborative filter predictions
    cf_scores = draw(st.lists(
        st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        min_size=n_destinations,
        max_size=n_destinations
    ))
    predictions['collaborative'] = list(zip(destination_ids, cf_scores))
    predictions['collaborative'].sort(key=lambda x: x[1], reverse=True)
    
    # Content-based filter predictions
    cb_scores = draw(st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=n_destinations,
        max_size=n_destinations
    ))
    predictions['content_based'] = list(zip(destination_ids, cb_scores))
    predictions['content_based'].sort(key=lambda x: x[1], reverse=True)
    
    # Context-aware engine predictions
    ca_scores = draw(st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=n_destinations,
        max_size=n_destinations
    ))
    predictions['context_aware'] = list(zip(destination_ids, ca_scores))
    predictions['context_aware'].sort(key=lambda x: x[1], reverse=True)
    
    return predictions


@st.composite
def context_strategy(draw):
    """Generate a valid Context object."""
    weather = WeatherInfo(
        condition=draw(st.sampled_from(['sunny', 'cloudy', 'rainy', 'stormy'])),
        temperature=draw(st.floats(min_value=20.0, max_value=35.0)),
        humidity=draw(st.floats(min_value=40.0, max_value=90.0)),
        precipitation_chance=draw(st.floats(min_value=0.0, max_value=1.0))
    )
    
    context = Context(
        location=(
            draw(st.floats(min_value=5.9, max_value=9.8)),  # Sri Lanka latitude
            draw(st.floats(min_value=79.5, max_value=81.9))  # Sri Lanka longitude
        ),
        weather=weather,
        season=draw(st.sampled_from(['dry', 'monsoon', 'inter-monsoon'])),
        day_of_week=draw(st.integers(min_value=0, max_value=6)),
        is_holiday=draw(st.booleans()),
        is_peak_season=draw(st.booleans()),
        user_type=draw(st.sampled_from(['cold_start', 'regular', 'frequent']))
    )
    
    return context


class TestWeightedVotingCorrectness:
    """
    Feature: tourism-recommender-system, Property 12: Weighted Voting Correctness
    
    Property: For any set of model predictions and weights, the weighted voting result
    SHALL equal the sum of (prediction_score × weight) for each model, normalized.
    
    Validates: Requirements 5.1
    """
    
    @given(
        predictions=predictions_strategy(),
        context=context_strategy()
    )
    @settings(max_examples=100, deadline=None)
    def test_weighted_voting_correctness(self, predictions, context):
        """Test that weighted voting correctly computes weighted sum of predictions."""
        ensemble = EnsembleVotingSystem()
        
        # Get adjusted weights for this context
        adjusted_weights = ensemble.adjust_weights(context)
        
        # Perform weighted voting
        result = ensemble.weighted_voting(predictions, context)
        
        # Verify the weighted voting calculation
        # For each destination, manually calculate expected weighted score
        for dest_id, aggregated_score in result:
            # Calculate expected score
            expected_score = 0.0
            total_weight = 0.0
            
            for model_name, model_predictions in predictions.items():
                if model_name not in adjusted_weights:
                    continue
                
                weight = adjusted_weights[model_name]
                total_weight += weight
                
                # Find this destination's score in this model's predictions
                model_score = None
                for pred_dest_id, pred_score in model_predictions:
                    if pred_dest_id == dest_id:
                        model_score = pred_score
                        break
                
                if model_score is not None:
                    expected_score += weight * model_score
            
            # Normalize by total weight
            if total_weight > 0:
                expected_score /= total_weight
            
            # Property: Aggregated score should equal weighted sum (normalized)
            # Allow small floating-point tolerance
            assert abs(aggregated_score - expected_score) < 1e-6, \
                f"Destination {dest_id}: aggregated={aggregated_score}, expected={expected_score}"
    
    @given(
        predictions=predictions_strategy()
    )
    @settings(max_examples=100, deadline=None)
    def test_weighted_voting_with_default_weights(self, predictions):
        """Test weighted voting with default weights (no context adjustment)."""
        ensemble = EnsembleVotingSystem()
        
        # Create a neutral context (no adjustments)
        context = Context(
            location=(7.0, 80.0),
            weather=WeatherInfo(
                condition='sunny',
                temperature=28.0,
                humidity=60.0,
                precipitation_chance=0.0
            ),
            season='dry',
            day_of_week=3,
            is_holiday=False,
            is_peak_season=False,
            user_type='regular'
        )
        
        # Perform weighted voting
        result = ensemble.weighted_voting(predictions, context)
        
        # Verify using default weights
        default_weights = ensemble.DEFAULT_WEIGHTS
        
        for dest_id, aggregated_score in result:
            expected_score = 0.0
            total_weight = 0.0
            
            for model_name, model_predictions in predictions.items():
                if model_name not in default_weights:
                    continue
                
                weight = default_weights[model_name]
                total_weight += weight
                
                # Find this destination's score
                for pred_dest_id, pred_score in model_predictions:
                    if pred_dest_id == dest_id:
                        expected_score += weight * pred_score
                        break
            
            # Normalize
            if total_weight > 0:
                expected_score /= total_weight
            
            # Property: Should match weighted sum with default weights
            assert abs(aggregated_score - expected_score) < 1e-6, \
                f"Destination {dest_id}: aggregated={aggregated_score}, expected={expected_score}"
    
    def test_weighted_voting_simple_case(self):
        """Test weighted voting with a simple manual case."""
        ensemble = EnsembleVotingSystem()
        
        # Simple predictions: 2 destinations, 2 models
        predictions = {
            'collaborative': [('dest_1', 4.0), ('dest_2', 3.0)],
            'content_based': [('dest_1', 0.8), ('dest_2', 0.6)]
        }
        
        # Neutral context
        context = Context(
            location=(7.0, 80.0),
            weather=WeatherInfo(
                condition='sunny',
                temperature=28.0,
                humidity=60.0,
                precipitation_chance=0.0
            ),
            season='dry',
            day_of_week=3,
            is_holiday=False,
            is_peak_season=False,
            user_type='regular'
        )
        
        result = ensemble.weighted_voting(predictions, context)
        
        # Calculate expected scores manually
        # Default weights: collaborative=0.35, content_based=0.25, context_aware=0.25, neural=0.15
        # Total weight for these two models: 0.35 + 0.25 = 0.60
        
        # dest_1: (0.35 * 4.0 + 0.25 * 0.8) / 0.60 = (1.4 + 0.2) / 0.60 = 1.6 / 0.60 = 2.667
        # dest_2: (0.35 * 3.0 + 0.25 * 0.6) / 0.60 = (1.05 + 0.15) / 0.60 = 1.2 / 0.60 = 2.0
        
        expected_dest_1 = (0.35 * 4.0 + 0.25 * 0.8) / 0.60
        expected_dest_2 = (0.35 * 3.0 + 0.25 * 0.6) / 0.60
        
        # Find actual scores
        actual_scores = dict(result)
        
        assert abs(actual_scores['dest_1'] - expected_dest_1) < 1e-6
        assert abs(actual_scores['dest_2'] - expected_dest_2) < 1e-6
        
        # Verify ordering (dest_1 should rank higher)
        assert result[0][0] == 'dest_1'
        assert result[1][0] == 'dest_2'


class TestBordaCountCorrectness:
    """
    Feature: tourism-recommender-system, Property 13: Borda Count Correctness
    
    Property: For any set of ranked lists, the Borda count aggregation SHALL assign
    points equal to (n - rank) where n is the number of items, and the final ranking
    SHALL be sorted by total points descending.
    
    Validates: Requirements 5.2
    """
    
    @given(predictions=predictions_strategy())
    @settings(max_examples=100, deadline=None)
    def test_borda_count_correctness(self, predictions):
        """Test that Borda count correctly assigns points based on rank."""
        ensemble = EnsembleVotingSystem()
        
        # Perform Borda count
        result = ensemble.borda_count(predictions)
        
        # Collect all unique destinations
        all_destinations = set()
        for model_predictions in predictions.values():
            for dest_id, _ in model_predictions:
                all_destinations.add(dest_id)
        
        n = len(all_destinations)
        
        # Manually calculate expected Borda scores
        expected_scores = defaultdict(float)
        
        for model_name, model_predictions in predictions.items():
            for rank, (dest_id, _) in enumerate(model_predictions):
                # Borda count: points = n - rank - 1 (0-indexed)
                points = n - rank - 1
                expected_scores[dest_id] += points
        
        # Verify each destination's Borda score
        result_dict = dict(result)
        
        for dest_id in all_destinations:
            actual_score = result_dict.get(dest_id, 0.0)
            expected_score = expected_scores[dest_id]
            
            # Property: Borda score should equal sum of (n - rank - 1) across all models
            assert abs(actual_score - expected_score) < 1e-6, \
                f"Destination {dest_id}: actual={actual_score}, expected={expected_score}"
        
        # Property: Result should be sorted by Borda score descending
        for i in range(len(result) - 1):
            assert result[i][1] >= result[i+1][1], \
                f"Result not sorted: {result[i][1]} < {result[i+1][1]}"
    
    def test_borda_count_simple_case(self):
        """Test Borda count with a simple manual case."""
        ensemble = EnsembleVotingSystem()
        
        # Simple case: 3 destinations, 2 models
        # Model 1 ranking: dest_1 (rank 0), dest_2 (rank 1), dest_3 (rank 2)
        # Model 2 ranking: dest_2 (rank 0), dest_1 (rank 1), dest_3 (rank 2)
        predictions = {
            'model_1': [('dest_1', 5.0), ('dest_2', 4.0), ('dest_3', 3.0)],
            'model_2': [('dest_2', 5.0), ('dest_1', 4.0), ('dest_3', 3.0)]
        }
        
        result = ensemble.borda_count(predictions)
        
        # Calculate expected Borda scores
        # n = 3 destinations
        # dest_1: (3-0-1) + (3-1-1) = 2 + 1 = 3
        # dest_2: (3-1-1) + (3-0-1) = 1 + 2 = 3
        # dest_3: (3-2-1) + (3-2-1) = 0 + 0 = 0
        
        result_dict = dict(result)
        
        assert result_dict['dest_1'] == 3.0
        assert result_dict['dest_2'] == 3.0
        assert result_dict['dest_3'] == 0.0
        
        # dest_1 and dest_2 should be tied at top, dest_3 at bottom
        top_two = {result[0][0], result[1][0]}
        assert top_two == {'dest_1', 'dest_2'}
        assert result[2][0] == 'dest_3'


class TestContextBasedWeightAdjustment:
    """
    Feature: tourism-recommender-system, Property 14: Context-Based Weight Adjustment
    
    Property: For any context type (cold_start, weather_critical, peak_season),
    the ensemble weights SHALL be adjusted by the exact amounts specified:
    - cold_start: +0.2 content_based, -0.2 collaborative
    - weather_critical: +0.15 context_aware, -0.15 neural
    - peak_season: +0.1 collaborative, -0.1 content_based
    
    Validates: Requirements 5.4, 5.5, 5.6
    """
    
    def test_cold_start_weight_adjustment(self):
        """Test weight adjustment for cold start context."""
        ensemble = EnsembleVotingSystem()
        
        # Create cold start context
        context = Context(
            location=(7.0, 80.0),
            weather=WeatherInfo(
                condition='sunny',
                temperature=28.0,
                humidity=60.0,
                precipitation_chance=0.0
            ),
            season='dry',
            day_of_week=3,
            is_holiday=False,
            is_peak_season=False,
            user_type='cold_start'
        )
        
        adjusted_weights = ensemble.adjust_weights(context)
        
        # Property: cold_start should adjust collaborative by -0.2 and content_based by +0.2
        expected_collaborative = ensemble.DEFAULT_WEIGHTS['collaborative'] - 0.20
        expected_content_based = ensemble.DEFAULT_WEIGHTS['content_based'] + 0.20
        
        assert abs(adjusted_weights['collaborative'] - expected_collaborative) < 1e-6, \
            f"Collaborative weight: {adjusted_weights['collaborative']}, expected: {expected_collaborative}"
        assert abs(adjusted_weights['content_based'] - expected_content_based) < 1e-6, \
            f"Content-based weight: {adjusted_weights['content_based']}, expected: {expected_content_based}"
    
    def test_weather_critical_weight_adjustment(self):
        """Test weight adjustment for weather critical context."""
        ensemble = EnsembleVotingSystem()
        
        # Create weather critical context (rainy)
        context = Context(
            location=(7.0, 80.0),
            weather=WeatherInfo(
                condition='rainy',
                temperature=25.0,
                humidity=85.0,
                precipitation_chance=0.8
            ),
            season='dry',
            day_of_week=3,
            is_holiday=False,
            is_peak_season=False,
            user_type='regular'
        )
        
        adjusted_weights = ensemble.adjust_weights(context)
        
        # Property: weather_critical should adjust context_aware by +0.15 and neural by -0.15
        expected_context_aware = ensemble.DEFAULT_WEIGHTS['context_aware'] + 0.15
        expected_neural = ensemble.DEFAULT_WEIGHTS['neural'] - 0.15
        
        assert abs(adjusted_weights['context_aware'] - expected_context_aware) < 1e-6, \
            f"Context-aware weight: {adjusted_weights['context_aware']}, expected: {expected_context_aware}"
        assert abs(adjusted_weights['neural'] - expected_neural) < 1e-6, \
            f"Neural weight: {adjusted_weights['neural']}, expected: {expected_neural}"
    
    def test_peak_season_weight_adjustment(self):
        """Test weight adjustment for peak season context."""
        ensemble = EnsembleVotingSystem()
        
        # Create peak season context
        context = Context(
            location=(7.0, 80.0),
            weather=WeatherInfo(
                condition='sunny',
                temperature=30.0,
                humidity=65.0,
                precipitation_chance=0.1
            ),
            season='dry',
            day_of_week=6,
            is_holiday=False,
            is_peak_season=True,
            user_type='regular'
        )
        
        adjusted_weights = ensemble.adjust_weights(context)
        
        # Property: peak_season should adjust collaborative by +0.1 and content_based by -0.1
        expected_collaborative = ensemble.DEFAULT_WEIGHTS['collaborative'] + 0.10
        expected_content_based = ensemble.DEFAULT_WEIGHTS['content_based'] - 0.10
        
        assert abs(adjusted_weights['collaborative'] - expected_collaborative) < 1e-6, \
            f"Collaborative weight: {adjusted_weights['collaborative']}, expected: {expected_collaborative}"
        assert abs(adjusted_weights['content_based'] - expected_content_based) < 1e-6, \
            f"Content-based weight: {adjusted_weights['content_based']}, expected: {expected_content_based}"
    
    def test_monsoon_season_triggers_weather_critical(self):
        """Test that monsoon season triggers weather_critical adjustment."""
        ensemble = EnsembleVotingSystem()
        
        # Create monsoon context
        context = Context(
            location=(7.0, 80.0),
            weather=WeatherInfo(
                condition='cloudy',
                temperature=26.0,
                humidity=80.0,
                precipitation_chance=0.5
            ),
            season='monsoon',
            day_of_week=3,
            is_holiday=False,
            is_peak_season=False,
            user_type='regular'
        )
        
        adjusted_weights = ensemble.adjust_weights(context)
        
        # Property: monsoon season should trigger weather_critical adjustment
        expected_context_aware = ensemble.DEFAULT_WEIGHTS['context_aware'] + 0.15
        expected_neural = ensemble.DEFAULT_WEIGHTS['neural'] - 0.15
        
        assert abs(adjusted_weights['context_aware'] - expected_context_aware) < 1e-6
        assert abs(adjusted_weights['neural'] - expected_neural) < 1e-6
    
    @given(context=context_strategy())
    @settings(max_examples=100, deadline=None)
    def test_weight_adjustments_are_exact(self, context):
        """Test that weight adjustments match exact specified amounts."""
        ensemble = EnsembleVotingSystem()
        
        adjusted_weights = ensemble.adjust_weights(context)
        context_type = ensemble._get_context_type(context)
        
        # Verify adjustments match specifications
        if context_type in ensemble.CONTEXT_WEIGHT_ADJUSTMENTS:
            expected_adjustments = ensemble.CONTEXT_WEIGHT_ADJUSTMENTS[context_type]
            
            for model_name, expected_adjustment in expected_adjustments.items():
                if model_name in ensemble.DEFAULT_WEIGHTS:
                    expected_weight = ensemble.DEFAULT_WEIGHTS[model_name] + expected_adjustment
                    # Ensure non-negative
                    expected_weight = max(0.0, expected_weight)
                    
                    actual_weight = adjusted_weights[model_name]
                    
                    # Property: Adjustment should be exactly as specified
                    assert abs(actual_weight - expected_weight) < 1e-6, \
                        f"Context {context_type}, model {model_name}: actual={actual_weight}, expected={expected_weight}"


class TestTopKOutputSize:
    """
    Feature: tourism-recommender-system, Property 15: Top-K Output Size
    
    Property: For any recommendation request with parameter K, the output list
    SHALL contain exactly min(K, available_destinations) items.
    
    Validates: Requirements 5.7
    """
    
    @given(
        predictions=predictions_strategy(),
        k=st.integers(min_value=1, max_value=30)
    )
    @settings(max_examples=100, deadline=None)
    def test_top_k_output_size(self, predictions, k):
        """Test that top-K selection returns exactly min(K, available) items."""
        ensemble = EnsembleVotingSystem()
        
        # Get all unique destinations
        all_destinations = set()
        for model_predictions in predictions.values():
            for dest_id, _ in model_predictions:
                all_destinations.add(dest_id)
        
        available = len(all_destinations)
        
        # Perform Borda count (any voting method works)
        result = ensemble.borda_count(predictions)
        
        # Select top-K
        top_k_result = ensemble._select_top_k(result, k)
        
        # Property: Output size should be exactly min(K, available_destinations)
        expected_size = min(k, available)
        actual_size = len(top_k_result)
        
        assert actual_size == expected_size, \
            f"Output size: {actual_size}, expected: {expected_size} (K={k}, available={available})"
    
    def test_top_k_with_k_larger_than_available(self):
        """Test top-K when K > available destinations."""
        ensemble = EnsembleVotingSystem()
        
        # 3 destinations
        predictions = [
            ('dest_1', 5.0),
            ('dest_2', 4.0),
            ('dest_3', 3.0)
        ]
        
        # Request top-10 (more than available)
        result = ensemble._select_top_k(predictions, k=10)
        
        # Property: Should return all 3 destinations
        assert len(result) == 3
    
    def test_top_k_with_k_smaller_than_available(self):
        """Test top-K when K < available destinations."""
        ensemble = EnsembleVotingSystem()
        
        # 5 destinations
        predictions = [
            ('dest_1', 5.0),
            ('dest_2', 4.0),
            ('dest_3', 3.0),
            ('dest_4', 2.0),
            ('dest_5', 1.0)
        ]
        
        # Request top-3
        result = ensemble._select_top_k(predictions, k=3)
        
        # Property: Should return exactly 3 destinations
        assert len(result) == 3
        
        # Should be the top 3
        assert result[0][0] == 'dest_1'
        assert result[1][0] == 'dest_2'
        assert result[2][0] == 'dest_3'
    
    def test_top_k_with_k_equal_to_available(self):
        """Test top-K when K == available destinations."""
        ensemble = EnsembleVotingSystem()
        
        # 5 destinations
        predictions = [
            ('dest_1', 5.0),
            ('dest_2', 4.0),
            ('dest_3', 3.0),
            ('dest_4', 2.0),
            ('dest_5', 1.0)
        ]
        
        # Request top-5 (exactly available)
        result = ensemble._select_top_k(predictions, k=5)
        
        # Property: Should return all 5 destinations
        assert len(result) == 5
