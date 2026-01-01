"""Tests for the recommendation API."""

import pytest
from hypothesis import given, strategies as st, settings
from typing import Dict, List

from src.data_models import (
    RecommendationRequest,
    Recommendation,
    Context,
    WeatherInfo,
    UserProfile,
    LocationFeatures,
)
from src.recommender_api import RecommenderAPI
from src.ensemble_voting import EnsembleVotingSystem
from src.mobile_optimizer import MobileOptimizer


# Test data generators
@st.composite
def weather_info_strategy(draw):
    """Generate random WeatherInfo."""
    return WeatherInfo(
        condition=draw(st.sampled_from(['sunny', 'cloudy', 'rainy', 'stormy'])),
        temperature=draw(st.floats(min_value=15.0, max_value=35.0)),
        humidity=draw(st.floats(min_value=0.0, max_value=100.0)),
        precipitation_chance=draw(st.floats(min_value=0.0, max_value=1.0)),
    )


@st.composite
def context_strategy(draw):
    """Generate random Context."""
    return Context(
        location=draw(
            st.tuples(
                st.floats(min_value=5.9, max_value=9.8),  # Sri Lanka lat
                st.floats(min_value=79.5, max_value=81.9),  # Sri Lanka lon
            )
        ),
        weather=draw(weather_info_strategy()),
        season=draw(st.sampled_from(['dry', 'monsoon', 'inter-monsoon'])),
        day_of_week=draw(st.integers(min_value=0, max_value=6)),
        is_holiday=draw(st.booleans()),
        is_peak_season=draw(st.booleans()),
        user_type=draw(st.sampled_from(['cold_start', 'regular', 'frequent'])),
    )


@st.composite
def location_features_strategy(draw, dest_id=None):
    """Generate random LocationFeatures."""
    if dest_id is None:
        dest_id = draw(st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))))
    
    return LocationFeatures(
        destination_id=dest_id,
        name=draw(st.text(min_size=1, max_size=50)),
        city=draw(st.text(min_size=1, max_size=30)),
        latitude=draw(st.floats(min_value=5.9, max_value=9.8)),
        longitude=draw(st.floats(min_value=79.5, max_value=81.9)),
        location_type=draw(st.sampled_from(['beach', 'cultural', 'nature', 'urban'])),
        avg_rating=draw(st.floats(min_value=1.0, max_value=5.0)),
        review_count=draw(st.integers(min_value=0, max_value=1000)),
        price_range=draw(st.sampled_from(['budget', 'mid-range', 'luxury'])),
        attributes=draw(st.lists(st.text(min_size=1, max_size=20), min_size=0, max_size=5)),
    )


@st.composite
def recommendation_request_strategy(draw):
    """Generate random RecommendationRequest."""
    return RecommendationRequest(
        user_id=draw(st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')))),
        location=draw(
            st.tuples(
                st.floats(min_value=5.9, max_value=9.8),
                st.floats(min_value=79.5, max_value=81.9),
            )
        ),
        budget=draw(st.one_of(st.none(), st.tuples(st.floats(min_value=10.0, max_value=500.0), st.floats(min_value=500.0, max_value=2000.0)))),
        travel_style=draw(st.one_of(st.none(), st.sampled_from(['beach', 'cultural', 'nature', 'adventure']))),
        group_size=draw(st.integers(min_value=1, max_value=10)),
        max_distance_km=draw(st.one_of(st.none(), st.floats(min_value=10.0, max_value=500.0))),
    )


def create_mock_api(destinations: Dict[str, LocationFeatures] = None, user_profiles: Dict[str, UserProfile] = None):
    """Create a mock RecommenderAPI for testing."""
    ensemble = EnsembleVotingSystem(models={}, strategy='weighted')
    optimizer = MobileOptimizer()
    
    if destinations is None:
        destinations = {}
    
    if user_profiles is None:
        user_profiles = {}
    
    return RecommenderAPI(
        ensemble=ensemble,
        optimizer=optimizer,
        destinations=destinations,
        user_profiles=user_profiles,
    )


# Property 18: Recommendation Output Format
@given(
    destinations=st.lists(location_features_strategy(), min_size=1, max_size=20),
    request=recommendation_request_strategy(),
    context=context_strategy(),
)
@settings(max_examples=100, deadline=None)
def test_property_18_recommendation_output_format(destinations, request, context):
    """
    Feature: tourism-recommender-system, Property 18: Recommendation Output Format
    
    For any recommendation in the output, it SHALL contain non-null destination_id,
    name, score (in range [0, 1]), and explanation string.
    
    Validates: Requirements 7.2
    """
    # Create destination dict
    dest_dict = {dest.destination_id: dest for dest in destinations}
    
    # Create API with mock ensemble that returns predictions
    api = create_mock_api(destinations=dest_dict)
    
    # Mock the ensemble predict method to return some predictions
    def mock_predict(user_id, context, candidate_items, top_k=10):
        # Return predictions for available destinations
        return [(dest_id, 0.8) for dest_id in list(dest_dict.keys())[:min(top_k, len(dest_dict))]]
    
    api.ensemble.predict = mock_predict
    
    # Get recommendations
    recommendations = api.get_recommendations(request, context)
    
    # Verify each recommendation has required fields
    for rec in recommendations:
        # Non-null destination_id
        assert rec.destination_id is not None
        assert isinstance(rec.destination_id, str)
        assert len(rec.destination_id) > 0
        
        # Non-null name
        assert rec.name is not None
        assert isinstance(rec.name, str)
        
        # Score in range [0, 1]
        assert rec.score is not None
        assert isinstance(rec.score, (int, float))
        assert 0.0 <= rec.score <= 1.0
        
        # Non-null explanation string
        assert rec.explanation is not None
        assert isinstance(rec.explanation, str)
        assert len(rec.explanation) > 0


# Property 19: Filter Application Correctness
@given(
    destinations=st.lists(location_features_strategy(), min_size=5, max_size=20),
    budget_min=st.floats(min_value=10.0, max_value=200.0),
    budget_max=st.floats(min_value=200.0, max_value=1000.0),
    max_distance=st.floats(min_value=10.0, max_value=200.0),
    context=context_strategy(),
)
@settings(max_examples=100, deadline=None)
def test_property_19_filter_application_correctness(destinations, budget_min, budget_max, max_distance, context):
    """
    Feature: tourism-recommender-system, Property 19: Filter Application Correctness
    
    For any recommendation list with budget and distance filters applied, all remaining
    items SHALL satisfy: budget_min <= estimated_cost <= budget_max AND distance_km <= max_distance.
    
    Validates: Requirements 7.3, 7.4
    """
    # Create destination dict
    dest_dict = {dest.destination_id: dest for dest in destinations}
    
    # Create API
    api = create_mock_api(destinations=dest_dict)
    
    # Create request with filters
    request = RecommendationRequest(
        user_id='test_user',
        location=(7.0, 80.0),  # Central Sri Lanka
        budget=(budget_min, budget_max),
        max_distance_km=max_distance,
        group_size=2,
    )
    
    # Mock the ensemble predict method
    def mock_predict(user_id, context, candidate_items, top_k=10):
        return [(dest_id, 0.8) for dest_id in list(dest_dict.keys())[:min(top_k, len(dest_dict))]]
    
    api.ensemble.predict = mock_predict
    
    # Get recommendations
    recommendations = api.get_recommendations(request, context)
    
    # Verify all recommendations pass filters
    for rec in recommendations:
        # Budget filter
        if rec.estimated_cost is not None:
            assert budget_min <= rec.estimated_cost <= budget_max, \
                f"Cost {rec.estimated_cost} not in range [{budget_min}, {budget_max}]"
        
        # Distance filter
        if rec.distance_km is not None:
            assert rec.distance_km <= max_distance, \
                f"Distance {rec.distance_km} exceeds max {max_distance}"


# Property 20: Invalid User Handling
@given(
    destinations=st.lists(location_features_strategy(), min_size=1, max_size=10),
    invalid_user_id=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
    context=context_strategy(),
)
@settings(max_examples=100, deadline=None)
def test_property_20_invalid_user_handling(destinations, invalid_user_id, context):
    """
    Feature: tourism-recommender-system, Property 20: Invalid User Handling
    
    For any invalid or unknown user_id, the system SHALL treat the user as a
    cold_start user with is_cold_start = true.
    
    Validates: Requirements 7.5
    """
    # Create destination dict
    dest_dict = {dest.destination_id: dest for dest in destinations}
    
    # Create API with NO user profiles (so any user is invalid)
    api = create_mock_api(destinations=dest_dict, user_profiles={})
    
    # Create request with invalid user
    request = RecommendationRequest(
        user_id=invalid_user_id,
        location=(7.0, 80.0),
        group_size=1,
    )
    
    # Mock the ensemble predict method
    def mock_predict(user_id, context, candidate_items, top_k=10):
        return [(dest_id, 0.8) for dest_id in list(dest_dict.keys())[:min(top_k, len(dest_dict))]]
    
    api.ensemble.predict = mock_predict
    
    # Get recommendations
    recommendations = api.get_recommendations(request, context)
    
    # Verify user was created as cold start
    assert invalid_user_id in api.user_profiles
    user_profile = api.user_profiles[invalid_user_id]
    assert user_profile.is_cold_start is True
    
    # Verify context was updated
    assert context.user_type == 'cold_start'


# Property 21: Diversity in Recommendations
@given(
    num_destinations=st.integers(min_value=10, max_value=30),
    context=context_strategy(),
)
@settings(max_examples=100, deadline=None)
def test_property_21_diversity_in_recommendations(num_destinations, context):
    """
    Feature: tourism-recommender-system, Property 21: Diversity in Recommendations
    
    For any recommendation list of size >= 5, the list SHALL contain at least 2
    different destination types (e.g., beach, cultural, nature).
    
    Validates: Requirements 7.6
    """
    # Create destinations with different types
    destination_types = ['beach', 'cultural', 'nature', 'urban']
    destinations = []
    
    for i in range(num_destinations):
        dest_type = destination_types[i % len(destination_types)]
        dest = LocationFeatures(
            destination_id=f'dest_{i}',
            name=f'Destination {i}',
            city='Test City',
            latitude=7.0 + i * 0.1,
            longitude=80.0 + i * 0.1,
            location_type=dest_type,
            avg_rating=4.0,
            review_count=100,
            price_range='mid-range',
            attributes=[],
        )
        destinations.append(dest)
    
    dest_dict = {dest.destination_id: dest for dest in destinations}
    
    # Create API
    api = create_mock_api(destinations=dest_dict)
    
    # Create request
    request = RecommendationRequest(
        user_id='test_user',
        location=(7.0, 80.0),
        group_size=1,
    )
    
    # Mock the ensemble predict method to return many predictions
    def mock_predict(user_id, context, candidate_items, top_k=10):
        # Return all destinations with same score (to test diversity reranking)
        return [(dest_id, 0.8) for dest_id in list(dest_dict.keys())[:min(top_k, len(dest_dict))]]
    
    api.ensemble.predict = mock_predict
    
    # Get recommendations
    recommendations = api.get_recommendations(request, context)
    
    # If we have at least 5 recommendations, check diversity
    if len(recommendations) >= 5:
        # Get unique destination types in top 5
        top_5_types = set()
        for rec in recommendations[:5]:
            dest_type = api._get_destination_type(rec.destination_id)
            top_5_types.add(dest_type)
        
        # Should have at least 2 different types
        assert len(top_5_types) >= 2, \
            f"Expected at least 2 destination types, got {len(top_5_types)}: {top_5_types}"


# Unit tests for specific functionality
def test_apply_filters_budget():
    """Test budget filter application."""
    api = create_mock_api()
    
    recommendations = [
        Recommendation(
            destination_id='dest1',
            name='Dest 1',
            score=0.9,
            explanation='Test',
            estimated_cost=100.0,
        ),
        Recommendation(
            destination_id='dest2',
            name='Dest 2',
            score=0.8,
            explanation='Test',
            estimated_cost=500.0,
        ),
        Recommendation(
            destination_id='dest3',
            name='Dest 3',
            score=0.7,
            explanation='Test',
            estimated_cost=1500.0,
        ),
    ]
    
    request = RecommendationRequest(
        user_id='test',
        location=(7.0, 80.0),
        budget=(200.0, 1000.0),
    )
    
    filtered = api.apply_filters(recommendations, request)
    
    assert len(filtered) == 1
    assert filtered[0].destination_id == 'dest2'


def test_apply_filters_distance():
    """Test distance filter application."""
    api = create_mock_api()
    
    recommendations = [
        Recommendation(
            destination_id='dest1',
            name='Dest 1',
            score=0.9,
            explanation='Test',
            distance_km=50.0,
        ),
        Recommendation(
            destination_id='dest2',
            name='Dest 2',
            score=0.8,
            explanation='Test',
            distance_km=150.0,
        ),
        Recommendation(
            destination_id='dest3',
            name='Dest 3',
            score=0.7,
            explanation='Test',
            distance_km=250.0,
        ),
    ]
    
    request = RecommendationRequest(
        user_id='test',
        location=(7.0, 80.0),
        max_distance_km=100.0,
    )
    
    filtered = api.apply_filters(recommendations, request)
    
    assert len(filtered) == 1
    assert filtered[0].destination_id == 'dest1'


def test_diversity_reranking():
    """Test diversity-aware reranking."""
    destinations = {
        'dest1': LocationFeatures(
            destination_id='dest1',
            name='Beach 1',
            city='City',
            latitude=7.0,
            longitude=80.0,
            location_type='beach',
            avg_rating=4.5,
            review_count=100,
            price_range='mid-range',
        ),
        'dest2': LocationFeatures(
            destination_id='dest2',
            name='Beach 2',
            city='City',
            latitude=7.1,
            longitude=80.1,
            location_type='beach',
            avg_rating=4.5,
            review_count=100,
            price_range='mid-range',
        ),
        'dest3': LocationFeatures(
            destination_id='dest3',
            name='Cultural 1',
            city='City',
            latitude=7.2,
            longitude=80.2,
            location_type='cultural',
            avg_rating=4.5,
            review_count=100,
            price_range='mid-range',
        ),
    }
    
    api = create_mock_api(destinations=destinations)
    
    recommendations = [
        Recommendation(
            destination_id='dest1',
            name='Beach 1',
            score=0.9,
            explanation='Test',
        ),
        Recommendation(
            destination_id='dest2',
            name='Beach 2',
            score=0.85,
            explanation='Test',
        ),
        Recommendation(
            destination_id='dest3',
            name='Cultural 1',
            score=0.8,
            explanation='Test',
        ),
    ]
    
    reranked = api.apply_diversity_reranking(recommendations)
    
    # First should still be dest1 (highest score)
    assert reranked[0].destination_id == 'dest1'
    
    # Second should be dest3 (cultural) due to diversity, not dest2 (another beach)
    assert reranked[1].destination_id == 'dest3'
