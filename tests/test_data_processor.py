"""Property-based tests for DataProcessor module."""

import pytest
import pandas as pd
import numpy as np
from scipy import sparse
from hypothesis import given, strategies as st, settings
from datetime import datetime, timedelta

from src.data_processor import DataProcessor
from src.data_models import LocationFeatures, UserProfile


# Strategies for generating test data
@st.composite
def review_dataframe_strategy(draw):
    """Generate a valid review dataframe."""
    n_reviews = draw(st.integers(min_value=1, max_value=50))
    
    destination_ids = draw(st.lists(
        st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
        min_size=1,
        max_size=10
    ))
    
    user_ids = draw(st.lists(
        st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
        min_size=1,
        max_size=10
    ))
    
    location_types = ['beach', 'cultural', 'nature', 'urban', 'other']
    
    data = {
        'destination_id': draw(st.lists(st.sampled_from(destination_ids), min_size=n_reviews, max_size=n_reviews)),
        'destination_name': draw(st.lists(st.text(min_size=1, max_size=30), min_size=n_reviews, max_size=n_reviews)),
        'city': draw(st.lists(st.text(min_size=1, max_size=20), min_size=n_reviews, max_size=n_reviews)),
        'location_string': draw(st.lists(st.text(min_size=1, max_size=50), min_size=n_reviews, max_size=n_reviews)),
        'location_type': draw(st.lists(st.sampled_from(location_types), min_size=n_reviews, max_size=n_reviews)),
        'user_id': draw(st.lists(st.sampled_from(user_ids), min_size=n_reviews, max_size=n_reviews)),
        'rating': draw(st.lists(st.floats(min_value=1.0, max_value=5.0, allow_nan=False, allow_infinity=False), min_size=n_reviews, max_size=n_reviews)),
        'travel_date': [datetime.now() - timedelta(days=draw(st.integers(min_value=0, max_value=365))) for _ in range(n_reviews)],
        'published_date': [datetime.now() - timedelta(days=draw(st.integers(min_value=0, max_value=365))) for _ in range(n_reviews)],
        'title': draw(st.lists(st.text(min_size=1, max_size=50), min_size=n_reviews, max_size=n_reviews)),
        'text': draw(st.lists(st.text(min_size=1, max_size=200), min_size=n_reviews, max_size=n_reviews)),
        'latitude': draw(st.lists(st.floats(min_value=5.9, max_value=9.8), min_size=n_reviews, max_size=n_reviews)),
        'longitude': draw(st.lists(st.floats(min_value=79.5, max_value=81.9), min_size=n_reviews, max_size=n_reviews)),
    }
    
    return pd.DataFrame(data)


class TestDataExtractionCompleteness:
    """
    Feature: tourism-recommender-system, Property 1: Data Extraction Completeness
    
    Property: For any valid review CSV file, when loaded by the Data_Processor,
    the extracted location features SHALL contain all required fields
    (name, city, coordinates, location_type) with non-null values.
    
    Validates: Requirements 1.1
    """
    
    @given(df=review_dataframe_strategy())
    @settings(max_examples=100, deadline=None)
    def test_location_features_completeness(self, df):
        """Test that all location features have required non-null fields."""
        processor = DataProcessor()
        
        # Extract location features
        location_features = processor.extract_location_features(df)
        
        # Property: All location features must have non-null required fields
        for dest_id, features in location_features.items():
            assert features.destination_id is not None, f"destination_id is None for {dest_id}"
            assert features.name is not None and features.name != "", f"name is None or empty for {dest_id}"
            assert features.city is not None and features.city != "", f"city is None or empty for {dest_id}"
            assert features.latitude is not None, f"latitude is None for {dest_id}"
            assert features.longitude is not None, f"longitude is None for {dest_id}"
            assert features.location_type is not None and features.location_type != "", f"location_type is None or empty for {dest_id}"
            assert isinstance(features.avg_rating, (int, float)), f"avg_rating is not numeric for {dest_id}"
            assert isinstance(features.review_count, int), f"review_count is not int for {dest_id}"
            assert features.price_range in ['budget', 'mid-range', 'luxury'], f"Invalid price_range for {dest_id}"
            assert isinstance(features.attributes, list), f"attributes is not a list for {dest_id}"


class TestUserProfileConstruction:
    """
    Feature: tourism-recommender-system, Property 2: User Profile Construction
    
    Property: For any set of user rating data, the constructed UserProfile SHALL contain
    rating_history, preferred_types derived from ratings, and correct is_cold_start flag
    based on rating count.
    
    Validates: Requirements 1.2
    """
    
    @given(df=review_dataframe_strategy())
    @settings(max_examples=100, deadline=None)
    def test_user_profile_construction(self, df):
        """Test that user profiles are correctly constructed from rating data."""
        processor = DataProcessor()
        
        # Build user profiles
        user_profiles = processor.build_user_profiles(df)
        
        # Property: All user profiles must have correct structure
        for user_id, profile in user_profiles.items():
            assert profile.user_id is not None, f"user_id is None for {user_id}"
            assert isinstance(profile.rating_history, dict), f"rating_history is not dict for {user_id}"
            assert isinstance(profile.preferred_types, list), f"preferred_types is not list for {user_id}"
            assert isinstance(profile.avg_rating, (int, float)), f"avg_rating is not numeric for {user_id}"
            assert isinstance(profile.visit_count, int), f"visit_count is not int for {user_id}"
            assert isinstance(profile.is_cold_start, bool), f"is_cold_start is not bool for {user_id}"
            
            # Property: is_cold_start should be True if visit_count < 5
            if profile.visit_count < 5:
                assert profile.is_cold_start == True, f"is_cold_start should be True for user {user_id} with {profile.visit_count} visits"
            else:
                assert profile.is_cold_start == False, f"is_cold_start should be False for user {user_id} with {profile.visit_count} visits"
            
            # Property: rating_history should match visit_count
            assert len(profile.rating_history) == profile.visit_count, f"rating_history length mismatch for {user_id}"


class TestTFIDFEmbeddingValidity:
    """
    Feature: tourism-recommender-system, Property 3: TF-IDF Embedding Validity
    
    Property: For any non-empty text description, the generated TF-IDF embedding SHALL be
    a vector with dimensions ≤ max_features (500) and L2 norm equal to 1 (normalized).
    
    Validates: Requirements 1.3, 3.2
    """
    
    @given(
        descriptions=st.lists(
            st.text(min_size=1, max_size=200, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs'))),
            min_size=1,
            max_size=20
        ),
        max_features=st.integers(min_value=10, max_value=500)
    )
    @settings(max_examples=100, deadline=None)
    def test_tfidf_embedding_validity(self, descriptions, max_features):
        """Test that TF-IDF embeddings have correct dimensions and normalization."""
        processor = DataProcessor()
        
        # Generate embeddings
        embeddings = processor.generate_tfidf_embeddings(descriptions, max_features=max_features)
        
        # Property: Embeddings should have correct shape
        assert embeddings.shape[0] == len(descriptions), "Number of embeddings doesn't match number of descriptions"
        assert embeddings.shape[1] <= max_features, f"Embedding dimension {embeddings.shape[1]} exceeds max_features {max_features}"
        
        # Property: Each embedding should be L2 normalized (norm = 1)
        for i, embedding in enumerate(embeddings):
            norm = np.linalg.norm(embedding)
            # Allow small floating point error
            assert abs(norm - 1.0) < 1e-5 or norm == 0, f"Embedding {i} has norm {norm}, expected 1.0"


class TestRatingMatrixNormalization:
    """
    Feature: tourism-recommender-system, Property 4: Rating Matrix Normalization
    
    Property: For any rating matrix built from valid input data, all non-null values
    SHALL be in the range [1, 5] and the matrix SHALL be in sparse CSR format.
    
    Validates: Requirements 1.4
    """
    
    @given(df=review_dataframe_strategy())
    @settings(max_examples=100, deadline=None)
    def test_rating_matrix_normalization(self, df):
        """Test that rating matrix has correct format and value ranges."""
        processor = DataProcessor()
        
        # Debug: Check input ratings
        if df['rating'].max() > 5.0 or df['rating'].min() < 1.0:
            print(f"Invalid ratings in input: min={df['rating'].min()}, max={df['rating'].max()}")
            print(f"Rating values: {df['rating'].tolist()}")
        
        # Build rating matrix
        rating_matrix, user_ids, destination_ids = processor.build_rating_matrix(df)
        
        # Property: Matrix should be in CSR format
        assert isinstance(rating_matrix, sparse.csr_matrix), "Rating matrix is not in CSR format"
        
        # Property: All non-zero values should be in [1, 5] range
        non_zero_values = rating_matrix.data
        if len(non_zero_values) > 0:
            assert np.all(non_zero_values >= 1.0), f"Found rating < 1.0: {non_zero_values.min()}"
            assert np.all(non_zero_values <= 5.0), f"Found rating > 5.0: {non_zero_values.max()}, input ratings: {df['rating'].tolist()}"
        
        # Property: Matrix dimensions should match user and destination counts
        assert rating_matrix.shape[0] == len(user_ids), "Matrix rows don't match user count"
        assert rating_matrix.shape[1] == len(destination_ids), "Matrix columns don't match destination count"


class TestDeduplicationPreservesMostRecent:
    """
    Feature: tourism-recommender-system, Property 5: Deduplication Preserves Most Recent
    
    Property: For any dataset containing duplicate user-destination pairs,
    after deduplication, only the entry with the most recent timestamp SHALL remain
    for each pair.
    
    Validates: Requirements 1.5
    """
    
    @given(
        n_duplicates=st.integers(min_value=2, max_value=5),
        destination_id=st.text(min_size=1, max_size=20),
        user_id=st.text(min_size=1, max_size=20)
    )
    @settings(max_examples=100, deadline=None)
    def test_deduplication_preserves_most_recent(self, n_duplicates, destination_id, user_id):
        """Test that deduplication keeps the most recent review for each user-destination pair."""
        processor = DataProcessor()
        
        # Create dataframe with duplicate user-destination pairs
        base_date = datetime.now()
        dates = [base_date - timedelta(days=i) for i in range(n_duplicates)]
        
        data = {
            'destination_id': [destination_id] * n_duplicates,
            'destination_name': ['Test Destination'] * n_duplicates,
            'city': ['Test City'] * n_duplicates,
            'location_string': ['Test Location'] * n_duplicates,
            'location_type': ['beach'] * n_duplicates,
            'user_id': [user_id] * n_duplicates,
            'rating': [4.0] * n_duplicates,
            'travel_date': dates,
            'published_date': dates,
            'title': [f'Review {i}' for i in range(n_duplicates)],
            'text': [f'Text {i}' for i in range(n_duplicates)],
            'latitude': [6.9] * n_duplicates,
            'longitude': [80.0] * n_duplicates,
        }
        
        df = pd.DataFrame(data)
        
        # Deduplicate
        df_deduped = processor.deduplicate_reviews(df)
        
        # Property: Should have exactly 1 row after deduplication
        assert len(df_deduped) == 1, f"Expected 1 row after deduplication, got {len(df_deduped)}"
        
        # Property: The remaining row should have the most recent date
        most_recent_date = max(dates)
        assert df_deduped.iloc[0]['published_date'] == most_recent_date, "Deduplication did not preserve most recent review"



class TestInvalidRatingRejection:
    """
    Feature: tourism-recommender-system, Property 24: Invalid Rating Rejection
    
    Property: For any input rating outside the range [1, 5], the Data_Processor
    SHALL reject the entry and not include it in the rating matrix.
    
    Validates: Requirements 10.1
    """
    
    @given(
        valid_ratings=st.lists(
            st.floats(min_value=1.0, max_value=5.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=10
        ),
        invalid_ratings=st.lists(
            st.one_of(
                st.floats(min_value=-10.0, max_value=0.99, allow_nan=False, allow_infinity=False),
                st.floats(min_value=5.01, max_value=10.0, allow_nan=False, allow_infinity=False)
            ),
            min_size=1,
            max_size=10
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_invalid_ratings_rejected(self, valid_ratings, invalid_ratings):
        """Test that ratings outside [1, 5] are rejected."""
        processor = DataProcessor()
        
        # Create dataframe with mix of valid and invalid ratings
        all_ratings = valid_ratings + invalid_ratings
        n_total = len(all_ratings)
        n_valid = len(valid_ratings)
        n_invalid = len(invalid_ratings)
        
        data = {
            'destination_id': [f'dest_{i}' for i in range(n_total)],
            'destination_name': [f'Destination {i}' for i in range(n_total)],
            'city': ['Test City'] * n_total,
            'location_string': ['Test Location'] * n_total,
            'location_type': ['beach'] * n_total,
            'user_id': [f'user_{i}' for i in range(n_total)],
            'rating': all_ratings,
            'travel_date': [datetime.now()] * n_total,
            'published_date': [datetime.now()] * n_total,
            'title': [f'Review {i}' for i in range(n_total)],
            'text': [f'Text {i}' for i in range(n_total)],
            'latitude': [6.9] * n_total,
            'longitude': [80.0] * n_total,
        }
        
        df = pd.DataFrame(data)
        
        # Validate ratings
        validated_df = processor._validate_ratings(df)
        
        # Property: Only valid ratings should remain
        assert len(validated_df) == n_valid, f"Expected {n_valid} valid ratings, got {len(validated_df)}"
        
        # Property: All remaining ratings should be in [1, 5] range
        if len(validated_df) > 0:
            assert validated_df['rating'].min() >= 1.0, "Found rating < 1.0 after validation"
            assert validated_df['rating'].max() <= 5.0, "Found rating > 5.0 after validation"
        
        # Property: Invalid ratings count should be tracked
        assert processor.invalid_ratings_count >= n_invalid, "Invalid ratings not properly tracked"
    
    @given(
        ratings=st.lists(
            st.floats(min_value=1.0, max_value=5.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=20
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_all_valid_ratings_preserved(self, ratings):
        """Test that all valid ratings are preserved."""
        processor = DataProcessor()
        
        n_ratings = len(ratings)
        
        data = {
            'destination_id': [f'dest_{i}' for i in range(n_ratings)],
            'destination_name': [f'Destination {i}' for i in range(n_ratings)],
            'city': ['Test City'] * n_ratings,
            'location_string': ['Test Location'] * n_ratings,
            'location_type': ['beach'] * n_ratings,
            'user_id': [f'user_{i}' for i in range(n_ratings)],
            'rating': ratings,
            'travel_date': [datetime.now()] * n_ratings,
            'published_date': [datetime.now()] * n_ratings,
            'title': [f'Review {i}' for i in range(n_ratings)],
            'text': [f'Text {i}' for i in range(n_ratings)],
            'latitude': [6.9] * n_ratings,
            'longitude': [80.0] * n_ratings,
        }
        
        df = pd.DataFrame(data)
        
        # Validate ratings
        validated_df = processor._validate_ratings(df)
        
        # Property: All valid ratings should be preserved
        assert len(validated_df) == n_ratings, f"Expected {n_ratings} ratings, got {len(validated_df)}"
        
        # Property: No invalid ratings should be counted
        initial_count = processor.invalid_ratings_count
        processor._validate_ratings(df)
        assert processor.invalid_ratings_count == initial_count, "Valid ratings incorrectly marked as invalid"
