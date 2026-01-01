"""Property-based tests for CollaborativeFilter module."""

import pytest
import numpy as np
from scipy import sparse
from hypothesis import given, strategies as st, settings, assume
import time

from src.collaborative_filter import CollaborativeFilter


# Strategies for generating test data
@st.composite
def rating_matrix_strategy(draw):
    """Generate a valid sparse rating matrix with user and destination IDs."""
    n_users = draw(st.integers(min_value=5, max_value=50))
    n_destinations = draw(st.integers(min_value=5, max_value=50))
    n_ratings = draw(st.integers(min_value=10, max_value=min(200, n_users * n_destinations)))
    
    # Generate user and destination IDs
    user_ids = [f"user_{i}" for i in range(n_users)]
    destination_ids = [f"dest_{i}" for i in range(n_destinations)]
    
    # Generate random ratings
    rows = draw(st.lists(
        st.integers(min_value=0, max_value=n_users-1),
        min_size=n_ratings,
        max_size=n_ratings
    ))
    cols = draw(st.lists(
        st.integers(min_value=0, max_value=n_destinations-1),
        min_size=n_ratings,
        max_size=n_ratings
    ))
    ratings = draw(st.lists(
        st.floats(min_value=1.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        min_size=n_ratings,
        max_size=n_ratings
    ))
    
    # Create sparse matrix
    rating_matrix = sparse.csr_matrix(
        (ratings, (rows, cols)),
        shape=(n_users, n_destinations),
        dtype=np.float32
    )
    
    return rating_matrix, user_ids, destination_ids


class TestColdStartConfidence:
    """
    Feature: tourism-recommender-system, Property 6: Cold Start Confidence
    
    Property: For any user with fewer than 5 ratings (cold start user),
    the Collaborative_Filter SHALL return a confidence score of exactly 0.
    
    Validates: Requirements 2.5
    """
    
    @given(data=rating_matrix_strategy())
    @settings(max_examples=100, deadline=None)
    def test_cold_start_confidence_is_zero(self, data):
        """Test that cold start users (< 5 ratings) get confidence score of 0."""
        rating_matrix, user_ids, destination_ids = data
        
        # Fit the model
        cf = CollaborativeFilter(n_factors=10)
        cf.fit(rating_matrix, user_ids, destination_ids)
        
        # Check confidence for each user
        for user_id in user_ids:
            confidence = cf.get_confidence(user_id)
            rating_count = cf.user_rating_counts.get(user_id, 0)
            
            # Property: Users with < 5 ratings should have confidence = 0
            if rating_count < 5:
                assert confidence == 0.0, f"User {user_id} with {rating_count} ratings has confidence {confidence}, expected 0.0"
            else:
                # Users with >= 5 ratings should have confidence > 0
                assert confidence > 0.0, f"User {user_id} with {rating_count} ratings has confidence {confidence}, expected > 0.0"
    
    @given(
        n_ratings=st.integers(min_value=0, max_value=4)
    )
    @settings(max_examples=100, deadline=None)
    def test_cold_start_user_confidence_boundary(self, n_ratings):
        """Test confidence boundary at exactly 5 ratings."""
        # Create a simple rating matrix with one user having n_ratings
        user_ids = ["test_user"]
        destination_ids = [f"dest_{i}" for i in range(10)]
        
        if n_ratings > 0:
            rows = [0] * n_ratings
            cols = list(range(n_ratings))
            ratings = [4.0] * n_ratings
            
            rating_matrix = sparse.csr_matrix(
                (ratings, (rows, cols)),
                shape=(1, 10),
                dtype=np.float32
            )
        else:
            # Empty rating matrix
            rating_matrix = sparse.csr_matrix((1, 10), dtype=np.float32)
        
        cf = CollaborativeFilter(n_factors=5)
        cf.fit(rating_matrix, user_ids, destination_ids)
        
        confidence = cf.get_confidence("test_user")
        
        # Property: < 5 ratings means confidence = 0
        assert confidence == 0.0, f"User with {n_ratings} ratings has confidence {confidence}, expected 0.0"
    
    def test_unknown_user_confidence_is_zero(self):
        """Test that unknown users get confidence score of 0."""
        # Create a simple rating matrix
        user_ids = ["user_1", "user_2"]
        destination_ids = ["dest_1", "dest_2"]
        
        rows = [0, 0, 1, 1, 1, 1, 1, 1]
        cols = [0, 1, 0, 1, 0, 1, 0, 1]
        ratings = [4.0, 5.0, 3.0, 4.0, 5.0, 4.0, 3.0, 5.0]
        
        rating_matrix = sparse.csr_matrix(
            (ratings, (rows, cols)),
            shape=(2, 2),
            dtype=np.float32
        )
        
        cf = CollaborativeFilter(n_factors=2)
        cf.fit(rating_matrix, user_ids, destination_ids)
        
        # Test unknown user
        confidence = cf.get_confidence("unknown_user")
        
        # Property: Unknown users should have confidence = 0
        assert confidence == 0.0, f"Unknown user has confidence {confidence}, expected 0.0"


class TestCollaborativeFilterInferenceLatency:
    """
    Feature: tourism-recommender-system, Property 7: Collaborative Filter Inference Latency
    
    Property: For any prediction request to the Collaborative_Filter,
    the inference time SHALL be less than 50ms.
    
    Validates: Requirements 2.3
    """
    
    @given(data=rating_matrix_strategy())
    @settings(max_examples=100, deadline=None)
    def test_inference_latency_under_50ms(self, data):
        """Test that prediction inference completes within 50ms."""
        rating_matrix, user_ids, destination_ids = data
        
        # Fit the model
        cf = CollaborativeFilter(n_factors=50)
        cf.fit(rating_matrix, user_ids, destination_ids)
        
        # Select a random user and candidate items
        user_id = user_ids[0]
        candidate_items = destination_ids[:min(10, len(destination_ids))]
        
        # Measure inference time
        start_time = time.perf_counter()
        predictions = cf.predict(user_id, candidate_items)
        end_time = time.perf_counter()
        
        inference_time_ms = (end_time - start_time) * 1000
        
        # Property: Inference should complete within 50ms
        assert inference_time_ms < 50.0, f"Inference took {inference_time_ms:.2f}ms, expected < 50ms"
        
        # Verify predictions are returned
        assert len(predictions) == len(candidate_items), "Should return predictions for all candidate items"
    
    def test_inference_latency_realistic_scenario(self):
        """Test inference latency with realistic data size."""
        # Create a realistic rating matrix (100 users, 200 destinations)
        n_users = 100
        n_destinations = 200
        n_ratings = 1000
        
        user_ids = [f"user_{i}" for i in range(n_users)]
        destination_ids = [f"dest_{i}" for i in range(n_destinations)]
        
        # Generate random ratings
        np.random.seed(42)
        rows = np.random.randint(0, n_users, n_ratings)
        cols = np.random.randint(0, n_destinations, n_ratings)
        ratings = np.random.uniform(1.0, 5.0, n_ratings)
        
        rating_matrix = sparse.csr_matrix(
            (ratings, (rows, cols)),
            shape=(n_users, n_destinations),
            dtype=np.float32
        )
        
        # Fit the model
        cf = CollaborativeFilter(n_factors=50)
        cf.fit(rating_matrix, user_ids, destination_ids)
        
        # Test inference for multiple users
        for i in range(10):
            user_id = user_ids[i]
            candidate_items = destination_ids[:10]
            
            start_time = time.perf_counter()
            predictions = cf.predict(user_id, candidate_items)
            end_time = time.perf_counter()
            
            inference_time_ms = (end_time - start_time) * 1000
            
            # Property: Inference should complete within 50ms
            assert inference_time_ms < 50.0, f"Inference for user {i} took {inference_time_ms:.2f}ms, expected < 50ms"
