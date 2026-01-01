"""Property-based tests for ContextAwareEngine module."""

import pytest
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, settings, assume
import time

from src.context_aware_engine import ContextAwareEngine
from src.data_models import Context, WeatherInfo


# Strategies for generating test data
@st.composite
def weather_info_strategy(draw):
    """Generate a valid WeatherInfo object."""
    condition = draw(st.sampled_from(['sunny', 'cloudy', 'rainy', 'stormy']))
    temperature = draw(st.floats(min_value=20.0, max_value=35.0, allow_nan=False, allow_infinity=False))
    humidity = draw(st.floats(min_value=40.0, max_value=100.0, allow_nan=False, allow_infinity=False))
    precipitation_chance = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    
    return WeatherInfo(
        condition=condition,
        temperature=temperature,
        humidity=humidity,
        precipitation_chance=precipitation_chance
    )


@st.composite
def context_strategy(draw):
    """Generate a valid Context object."""
    location = (
        draw(st.floats(min_value=5.9, max_value=9.8, allow_nan=False, allow_infinity=False)),
        draw(st.floats(min_value=79.5, max_value=81.9, allow_nan=False, allow_infinity=False))
    )
    weather = draw(weather_info_strategy())
    season = draw(st.sampled_from(['dry', 'monsoon', 'inter-monsoon']))
    day_of_week = draw(st.integers(min_value=0, max_value=6))
    is_holiday = draw(st.booleans())
    is_peak_season = draw(st.booleans())
    user_type = draw(st.sampled_from(['cold_start', 'regular', 'frequent']))
    
    return Context(
        location=location,
        weather=weather,
        season=season,
        day_of_week=day_of_week,
        is_holiday=is_holiday,
        is_peak_season=is_peak_season,
        user_type=user_type
    )


class TestWeatherContextScoring:
    """
    Feature: tourism-recommender-system, Property 10: Weather Context Scoring
    
    Property: For any context with rainy weather, beach/outdoor destinations
    SHALL receive lower scores than indoor/cultural destinations from the
    Context_Aware_Engine.
    
    Validates: Requirements 4.2, 4.3
    """
    
    @given(context=context_strategy())
    @settings(max_examples=100, deadline=None)
    def test_rainy_weather_deprioritizes_beach_destinations(self, context):
        """Test that beach destinations get lower scores in rainy weather."""
        # Create engine
        engine = ContextAwareEngine(max_depth=10)
        
        # Set up destination types
        destination_ids = ['beach_1', 'cultural_1', 'beach_2', 'cultural_2']
        location_types = {
            'beach_1': 'beach',
            'cultural_1': 'cultural',
            'beach_2': 'coastal',
            'cultural_2': 'temple'
        }
        engine.destination_ids = destination_ids
        engine.location_types = location_types
        engine.is_fitted = False  # Don't use tree, just rules
        
        # Force rainy weather
        rainy_context = Context(
            location=context.location,
            weather=WeatherInfo(
                condition='rainy',
                temperature=context.weather.temperature,
                humidity=context.weather.humidity,
                precipitation_chance=0.9
            ),
            season=context.season,
            day_of_week=context.day_of_week,
            is_holiday=context.is_holiday,
            is_peak_season=context.is_peak_season,
            user_type=context.user_type
        )
        
        # Get predictions
        predictions = engine.predict(rainy_context, destination_ids)
        scores = {dest_id: score for dest_id, score in predictions}
        
        # Property: Beach/coastal destinations should score lower than cultural in rain
        beach_scores = [scores['beach_1'], scores['beach_2']]
        cultural_scores = [scores['cultural_1'], scores['cultural_2']]
        
        avg_beach_score = np.mean(beach_scores)
        avg_cultural_score = np.mean(cultural_scores)
        
        # Beach destinations should have lower average score than cultural
        assert avg_beach_score < avg_cultural_score, \
            f"Beach avg score {avg_beach_score:.3f} should be < cultural avg score {avg_cultural_score:.3f} in rainy weather"
    
    @given(context=context_strategy())
    @settings(max_examples=100, deadline=None)
    def test_stormy_weather_deprioritizes_outdoor_destinations(self, context):
        """Test that outdoor destinations get lower scores in stormy weather."""
        # Create engine
        engine = ContextAwareEngine(max_depth=10)
        
        # Set up destination types
        destination_ids = ['outdoor_1', 'indoor_1', 'nature_1', 'museum_1']
        location_types = {
            'outdoor_1': 'outdoor',
            'indoor_1': 'indoor',
            'nature_1': 'nature',
            'museum_1': 'museum'
        }
        engine.destination_ids = destination_ids
        engine.location_types = location_types
        engine.is_fitted = False
        
        # Force stormy weather
        stormy_context = Context(
            location=context.location,
            weather=WeatherInfo(
                condition='stormy',
                temperature=context.weather.temperature,
                humidity=context.weather.humidity,
                precipitation_chance=1.0
            ),
            season=context.season,
            day_of_week=context.day_of_week,
            is_holiday=context.is_holiday,
            is_peak_season=context.is_peak_season,
            user_type=context.user_type
        )
        
        # Get predictions
        predictions = engine.predict(stormy_context, destination_ids)
        scores = {dest_id: score for dest_id, score in predictions}
        
        # Property: Outdoor/nature destinations should score lower than indoor/museum in storm
        outdoor_scores = [scores['outdoor_1'], scores['nature_1']]
        indoor_scores = [scores['indoor_1'], scores['museum_1']]
        
        avg_outdoor_score = np.mean(outdoor_scores)
        avg_indoor_score = np.mean(indoor_scores)
        
        # Outdoor destinations should have lower average score than indoor
        assert avg_outdoor_score < avg_indoor_score, \
            f"Outdoor avg score {avg_outdoor_score:.3f} should be < indoor avg score {avg_indoor_score:.3f} in stormy weather"
    
    @given(context=context_strategy())
    @settings(max_examples=100, deadline=None)
    def test_monsoon_season_boosts_cultural_destinations(self, context):
        """Test that cultural destinations get boosted during monsoon season."""
        # Create engine
        engine = ContextAwareEngine(max_depth=10)
        
        # Set up destination types
        destination_ids = ['beach_1', 'cultural_1', 'temple_1', 'coastal_1']
        location_types = {
            'beach_1': 'beach',
            'cultural_1': 'cultural',
            'temple_1': 'temple',
            'coastal_1': 'coastal'
        }
        engine.destination_ids = destination_ids
        engine.location_types = location_types
        engine.is_fitted = False
        
        # Force monsoon season
        monsoon_context = Context(
            location=context.location,
            weather=context.weather,
            season='monsoon',
            day_of_week=context.day_of_week,
            is_holiday=context.is_holiday,
            is_peak_season=context.is_peak_season,
            user_type=context.user_type
        )
        
        # Get predictions
        predictions = engine.predict(monsoon_context, destination_ids)
        scores = {dest_id: score for dest_id, score in predictions}
        
        # Property: Cultural/temple destinations should score higher than beach/coastal in monsoon
        cultural_scores = [scores['cultural_1'], scores['temple_1']]
        beach_scores = [scores['beach_1'], scores['coastal_1']]
        
        avg_cultural_score = np.mean(cultural_scores)
        avg_beach_score = np.mean(beach_scores)
        
        # Cultural destinations should have higher average score than beach in monsoon
        assert avg_cultural_score > avg_beach_score, \
            f"Cultural avg score {avg_cultural_score:.3f} should be > beach avg score {avg_beach_score:.3f} in monsoon season"
    
    def test_weather_scoring_specific_example(self):
        """Test weather scoring with a specific example."""
        # Create engine
        engine = ContextAwareEngine(max_depth=10)
        
        # Set up destinations
        destination_ids = ['beach_resort', 'temple_complex', 'nature_park', 'museum']
        location_types = {
            'beach_resort': 'beach',
            'temple_complex': 'cultural',
            'nature_park': 'nature',
            'museum': 'museum'
        }
        engine.destination_ids = destination_ids
        engine.location_types = location_types
        engine.is_fitted = False
        
        # Create rainy context
        rainy_context = Context(
            location=(7.0, 80.0),
            weather=WeatherInfo(
                condition='rainy',
                temperature=28.0,
                humidity=85.0,
                precipitation_chance=0.8
            ),
            season='monsoon',
            day_of_week=3,
            is_holiday=False,
            is_peak_season=False,
            user_type='regular'
        )
        
        # Get predictions
        predictions = engine.predict(rainy_context, destination_ids)
        scores = {dest_id: score for dest_id, score in predictions}
        
        # Verify beach is deprioritized
        assert scores['beach_resort'] < scores['temple_complex'], \
            "Beach should score lower than temple in rainy monsoon"
        assert scores['beach_resort'] < scores['museum'], \
            "Beach should score lower than museum in rainy monsoon"
        
        # Verify cultural/indoor are boosted
        assert scores['temple_complex'] > scores['nature_park'], \
            "Temple should score higher than nature park in rainy monsoon"


class TestHolidayContextBoost:
    """
    Feature: tourism-recommender-system, Property 11: Holiday Context Boost
    
    Property: For any context where is_holiday is true, cultural destinations
    SHALL receive a score boost from the Context_Aware_Engine.
    
    Validates: Requirements 4.6
    """
    
    @given(context=context_strategy())
    @settings(max_examples=100, deadline=None)
    def test_holiday_boosts_cultural_destinations(self, context):
        """Test that cultural destinations get boosted during holidays."""
        # Create engine
        engine = ContextAwareEngine(max_depth=10)
        
        # Set up destination types
        destination_ids = ['cultural_1', 'temple_1', 'beach_1', 'nature_1']
        location_types = {
            'cultural_1': 'cultural',
            'temple_1': 'temple',
            'beach_1': 'beach',
            'nature_1': 'nature'
        }
        engine.destination_ids = destination_ids
        engine.location_types = location_types
        engine.is_fitted = False
        
        # Create holiday context
        holiday_context = Context(
            location=context.location,
            weather=context.weather,
            season=context.season,
            day_of_week=context.day_of_week,
            is_holiday=True,
            is_peak_season=context.is_peak_season,
            user_type=context.user_type
        )
        
        # Create non-holiday context (same except is_holiday=False)
        non_holiday_context = Context(
            location=context.location,
            weather=context.weather,
            season=context.season,
            day_of_week=context.day_of_week,
            is_holiday=False,
            is_peak_season=context.is_peak_season,
            user_type=context.user_type
        )
        
        # Get predictions for both contexts
        holiday_predictions = engine.predict(holiday_context, destination_ids)
        non_holiday_predictions = engine.predict(non_holiday_context, destination_ids)
        
        holiday_scores = {dest_id: score for dest_id, score in holiday_predictions}
        non_holiday_scores = {dest_id: score for dest_id, score in non_holiday_predictions}
        
        # Property: Cultural/temple destinations should score higher during holidays
        cultural_dest_ids = ['cultural_1', 'temple_1']
        
        for dest_id in cultural_dest_ids:
            holiday_score = holiday_scores[dest_id]
            non_holiday_score = non_holiday_scores[dest_id]
            
            # Cultural destinations should get a boost during holidays
            assert holiday_score > non_holiday_score, \
                f"Cultural destination {dest_id} should score higher during holiday " \
                f"({holiday_score:.3f}) than non-holiday ({non_holiday_score:.3f})"
    
    @given(context=context_strategy())
    @settings(max_examples=100, deadline=None)
    def test_holiday_boost_magnitude(self, context):
        """Test that holiday boost is significant (at least 0.1 increase)."""
        # Create engine
        engine = ContextAwareEngine(max_depth=10)
        
        # Set up destination types
        destination_ids = ['temple_1', 'historical_1', 'festival_1']
        location_types = {
            'temple_1': 'temple',
            'historical_1': 'historical',
            'festival_1': 'festival'
        }
        engine.destination_ids = destination_ids
        engine.location_types = location_types
        engine.is_fitted = False
        
        # Create holiday context
        holiday_context = Context(
            location=context.location,
            weather=context.weather,
            season=context.season,
            day_of_week=context.day_of_week,
            is_holiday=True,
            is_peak_season=context.is_peak_season,
            user_type=context.user_type
        )
        
        # Create non-holiday context
        non_holiday_context = Context(
            location=context.location,
            weather=context.weather,
            season=context.season,
            day_of_week=context.day_of_week,
            is_holiday=False,
            is_peak_season=context.is_peak_season,
            user_type=context.user_type
        )
        
        # Get predictions
        holiday_predictions = engine.predict(holiday_context, destination_ids)
        non_holiday_predictions = engine.predict(non_holiday_context, destination_ids)
        
        holiday_scores = {dest_id: score for dest_id, score in holiday_predictions}
        non_holiday_scores = {dest_id: score for dest_id, score in non_holiday_predictions}
        
        # Property: Holiday boost should be at least 0.1 for cultural destinations
        for dest_id in destination_ids:
            boost = holiday_scores[dest_id] - non_holiday_scores[dest_id]
            
            # Boost should be positive and significant (at least 0.1)
            assert boost >= 0.1, \
                f"Holiday boost for {dest_id} is {boost:.3f}, expected >= 0.1"
    
    def test_holiday_boost_specific_example(self):
        """Test holiday boost with a specific example."""
        # Create engine
        engine = ContextAwareEngine(max_depth=10)
        
        # Set up destinations
        destination_ids = ['temple_of_tooth', 'beach_resort', 'cultural_center']
        location_types = {
            'temple_of_tooth': 'temple',
            'beach_resort': 'beach',
            'cultural_center': 'cultural'
        }
        engine.destination_ids = destination_ids
        engine.location_types = location_types
        engine.is_fitted = False
        
        # Create holiday context
        holiday_context = Context(
            location=(7.3, 80.6),
            weather=WeatherInfo(
                condition='sunny',
                temperature=30.0,
                humidity=70.0,
                precipitation_chance=0.1
            ),
            season='dry',
            day_of_week=0,
            is_holiday=True,
            is_peak_season=True,
            user_type='regular'
        )
        
        # Create non-holiday context
        non_holiday_context = Context(
            location=(7.3, 80.6),
            weather=WeatherInfo(
                condition='sunny',
                temperature=30.0,
                humidity=70.0,
                precipitation_chance=0.1
            ),
            season='dry',
            day_of_week=0,
            is_holiday=False,
            is_peak_season=True,
            user_type='regular'
        )
        
        # Get predictions
        holiday_predictions = engine.predict(holiday_context, destination_ids)
        non_holiday_predictions = engine.predict(non_holiday_context, destination_ids)
        
        holiday_scores = {dest_id: score for dest_id, score in holiday_predictions}
        non_holiday_scores = {dest_id: score for dest_id, score in non_holiday_predictions}
        
        # Verify cultural destinations get boosted
        assert holiday_scores['temple_of_tooth'] > non_holiday_scores['temple_of_tooth'], \
            "Temple should score higher during holiday"
        assert holiday_scores['cultural_center'] > non_holiday_scores['cultural_center'], \
            "Cultural center should score higher during holiday"
        
        # Verify boost is significant
        temple_boost = holiday_scores['temple_of_tooth'] - non_holiday_scores['temple_of_tooth']
        assert temple_boost >= 0.2, f"Temple boost {temple_boost:.3f} should be >= 0.2"


class TestContextAwareEngineInferenceLatency:
    """Test that Context-Aware Engine meets latency requirements."""
    
    @given(context=context_strategy())
    @settings(max_examples=100, deadline=None)
    def test_inference_latency_under_20ms(self, context):
        """Test that prediction inference completes within 20ms (Requirement 4.4)."""
        # Create engine
        engine = ContextAwareEngine(max_depth=10)
        
        # Set up destinations
        destination_ids = [f'dest_{i}' for i in range(50)]
        location_types = {
            dest_id: np.random.choice(['beach', 'cultural', 'nature', 'urban'])
            for dest_id in destination_ids
        }
        engine.destination_ids = destination_ids
        engine.location_types = location_types
        engine.is_fitted = False
        
        # Measure inference time
        start_time = time.perf_counter()
        predictions = engine.predict(context, destination_ids)
        end_time = time.perf_counter()
        
        inference_time_ms = (end_time - start_time) * 1000
        
        # Property: Inference should complete within 20ms (Requirement 4.4)
        assert inference_time_ms < 20.0, \
            f"Inference took {inference_time_ms:.2f}ms, expected < 20ms"
        
        # Verify predictions are returned
        assert len(predictions) == len(destination_ids), \
            "Should return predictions for all candidate items"
