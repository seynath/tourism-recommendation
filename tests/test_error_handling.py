"""Property-based tests for error handling in the tourism recommender system."""

import pytest
import pandas as pd
import numpy as np
from scipy import sparse
from hypothesis import given, strategies as st, settings
from datetime import datetime

from src.collaborative_filter import CollaborativeFilter
from src.content_based_filter import ContentBasedFilter
from src.data_processor import DataProcessor


class TestNoReviewDestinationHandling:
    """
    Feature: tourism-recommender-system, Property 25: No-Review Destination Handling
    
    Property: For any destination with zero reviews, it SHALL be excluded from
    Collaborative_Filter predictions but included in Content_Based_Filter predictions.
    
    Validates: Requirements 10.2
    """
    
    @given(
        n_destinations_with_reviews=st.integers(min_value=5, max_value=20),
        n_destinations_no_reviews=st.integers(min_value=1, max_value=10),
        n_users=st.integers(min_value=3, max_value=10)
    )
    @settings(max_examples=100, deadline=None)
    def test_no_review_destinations_excluded_from_cf(
        self, n_destinations_with_reviews, n_destinations_no_reviews, n_users
    ):
        """Test that destinations with no reviews are excluded from collaborative filtering."""
        # Create destinations with reviews
        dest_with_reviews = [f'dest_with_reviews_{i}' for i in range(n_destinations_with_reviews)]
        
        # Create destinations without reviews
        dest_no_reviews = [f'dest_no_reviews_{i}' for i in range(n_destinations_no_reviews)]
        
        # Create users
        user_ids = [f'user_{i}' for i in range(n_users)]
        
        # Build rating matrix (only for destinations with reviews)
        rows = []
        cols = []
        data = []
        
        for user_idx in range(n_users):
            # Each user rates some destinations
            n_ratings = min(3, n_destinations_with_reviews)
            for dest_idx in range(n_ratings):
                rows.append(user_idx)
                cols.append(dest_idx)
                data.append(float(np.random.uniform(1.0, 5.0)))
        
        rating_matrix = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(n_users, n_destinations_with_reviews),
            dtype=np.float32
        )
        
        # Train collaborative filter
        cf = CollaborativeFilter(n_factors=min(5, n_destinations_with_reviews - 1))
        cf.fit(rating_matrix, user_ids, dest_with_reviews)
        
        # Property: Destinations with reviews should be in CF's destination list
        for dest_id in dest_with_reviews:
            assert dest_id in cf.destination_ids, f"Destination {dest_id} with reviews not in CF"
        
        # Property: Destinations without reviews should NOT be in CF's destination list
        for dest_id in dest_no_reviews:
            assert dest_id not in cf.destination_ids, f"Destination {dest_id} without reviews incorrectly in CF"
        
        # Property: CF predictions for no-review destinations should use global mean
        all_candidates = dest_with_reviews + dest_no_reviews
        predictions = cf.predict(user_ids[0], all_candidates)
        
        # Find predictions for no-review destinations
        no_review_predictions = [
            (dest_id, score) for dest_id, score in predictions
            if dest_id in dest_no_reviews
        ]
        
        # All no-review destinations should get global mean score
        for dest_id, score in no_review_predictions:
            assert abs(score - cf.global_mean) < 0.01, \
                f"No-review destination {dest_id} got score {score}, expected global mean {cf.global_mean}"
    
    @given(
        n_destinations_with_reviews=st.integers(min_value=3, max_value=15),
        n_destinations_no_reviews=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=100, deadline=None)
    def test_no_review_destinations_included_in_cb(
        self, n_destinations_with_reviews, n_destinations_no_reviews
    ):
        """Test that destinations with no reviews are included in content-based filtering."""
        # Create all destinations (with and without reviews)
        dest_with_reviews = [f'dest_with_reviews_{i}' for i in range(n_destinations_with_reviews)]
        dest_no_reviews = [f'dest_no_reviews_{i}' for i in range(n_destinations_no_reviews)]
        all_destinations = dest_with_reviews + dest_no_reviews
        
        # Create descriptions and attributes for ALL destinations
        descriptions = [f'Description for {dest_id}' for dest_id in all_destinations]
        attributes = {
            dest_id: ['beach', 'surfing'] if 'with_reviews' in dest_id else ['cultural', 'temple']
            for dest_id in all_destinations
        }
        location_types = {
            dest_id: 'beach' if 'with_reviews' in dest_id else 'cultural'
            for dest_id in all_destinations
        }
        
        # Train content-based filter
        cb = ContentBasedFilter(max_features=50)
        cb.fit(descriptions, attributes, location_types)
        
        # Property: ALL destinations (with and without reviews) should be in CB's destination list
        for dest_id in all_destinations:
            assert dest_id in cb.destination_ids, \
                f"Destination {dest_id} not in content-based filter"
        
        # Property: Destinations without reviews should be in CB's destination list
        for dest_id in dest_no_reviews:
            assert dest_id in cb.destination_ids, \
                f"No-review destination {dest_id} not in content-based filter"
        
        # Property: CB should be able to generate predictions for no-review destinations
        user_preferences = {'preferred_types': ['cultural'], 'preferred_attributes': ['temple']}
        predictions = cb.predict(user_preferences, all_destinations)
        
        # Find predictions for no-review destinations
        no_review_predictions = [
            (dest_id, score) for dest_id, score in predictions
            if dest_id in dest_no_reviews
        ]
        
        # Should have predictions for all no-review destinations
        assert len(no_review_predictions) == n_destinations_no_reviews, \
            f"Expected {n_destinations_no_reviews} predictions for no-review destinations, got {len(no_review_predictions)}"
        
        # Scores should be valid (not just default values)
        for dest_id, score in no_review_predictions:
            assert 0.0 <= score <= 1.0, f"Invalid score {score} for no-review destination {dest_id}"
    
    @given(
        n_destinations=st.integers(min_value=5, max_value=20),
        n_users=st.integers(min_value=3, max_value=10)
    )
    @settings(max_examples=100, deadline=None)
    def test_cf_and_cb_handle_no_reviews_differently(
        self, n_destinations, n_users
    ):
        """Test that CF excludes and CB includes destinations with no reviews."""
        # Split destinations: some with reviews, some without
        n_with_reviews = max(3, n_destinations // 2)
        n_no_reviews = n_destinations - n_with_reviews
        
        dest_with_reviews = [f'dest_with_{i}' for i in range(n_with_reviews)]
        dest_no_reviews = [f'dest_no_{i}' for i in range(n_no_reviews)]
        all_destinations = dest_with_reviews + dest_no_reviews
        
        user_ids = [f'user_{i}' for i in range(n_users)]
        
        # Build rating matrix (only for destinations with reviews)
        rows = []
        cols = []
        data = []
        
        for user_idx in range(n_users):
            n_ratings = min(2, n_with_reviews)
            for dest_idx in range(n_ratings):
                rows.append(user_idx)
                cols.append(dest_idx)
                data.append(float(np.random.uniform(1.0, 5.0)))
        
        rating_matrix = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(n_users, n_with_reviews),
            dtype=np.float32
        )
        
        # Train CF
        cf = CollaborativeFilter(n_factors=min(5, n_with_reviews - 1))
        cf.fit(rating_matrix, user_ids, dest_with_reviews)
        
        # Train CB (with ALL destinations)
        descriptions = [f'Description for {dest_id}' for dest_id in all_destinations]
        attributes = {dest_id: ['beach'] for dest_id in all_destinations}
        
        cb = ContentBasedFilter(max_features=50)
        cb.fit(descriptions, attributes)
        
        # Property: CF should only have destinations with reviews
        assert len(cf.destination_ids) == n_with_reviews, \
            f"CF should have {n_with_reviews} destinations, got {len(cf.destination_ids)}"
        
        # Property: CB should have ALL destinations
        assert len(cb.destination_ids) == n_destinations, \
            f"CB should have {n_destinations} destinations, got {len(cb.destination_ids)}"
        
        # Property: No-review destinations should be in CB but not CF
        for dest_id in dest_no_reviews:
            assert dest_id not in cf.destination_ids, f"{dest_id} should not be in CF"
            assert dest_id in cb.destination_ids, f"{dest_id} should be in CB"
