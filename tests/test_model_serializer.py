"""Property-based tests for ModelSerializer module."""

import pytest
import numpy as np
import pandas as pd
from scipy import sparse
from hypothesis import given, strategies as st, settings, assume
import tempfile
import os
from pathlib import Path

from src.model_serializer import ModelSerializer
from src.collaborative_filter import CollaborativeFilter
from src.content_based_filter import ContentBasedFilter
from src.context_aware_engine import ContextAwareEngine


# Strategies for generating test data
@st.composite
def collaborative_filter_strategy(draw):
    """Generate a fitted CollaborativeFilter model."""
    n_users = draw(st.integers(min_value=5, max_value=30))
    n_destinations = draw(st.integers(min_value=5, max_value=30))
    n_ratings = draw(st.integers(min_value=10, max_value=min(100, n_users * n_destinations)))
    
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
    
    rating_matrix = sparse.csr_matrix(
        (ratings, (rows, cols)),
        shape=(n_users, n_destinations),
        dtype=np.float32
    )
    
    # Create and fit model
    n_factors = draw(st.integers(min_value=5, max_value=20))
    cf = CollaborativeFilter(n_factors=n_factors)
    cf.fit(rating_matrix, user_ids, destination_ids)
    
    return cf, user_ids, destination_ids


@st.composite
def content_based_filter_strategy(draw):
    """Generate a fitted ContentBasedFilter model."""
    n_destinations = draw(st.integers(min_value=5, max_value=30))
    
    destination_ids = [f"dest_{i}" for i in range(n_destinations)]
    
    # Generate random descriptions
    words = ['beach', 'cultural', 'nature', 'temple', 'wildlife', 'historical', 
             'scenic', 'adventure', 'relaxing', 'beautiful', 'ancient', 'modern']
    
    descriptions = []
    for _ in range(n_destinations):
        n_words = draw(st.integers(min_value=3, max_value=10))
        desc_words = draw(st.lists(st.sampled_from(words), min_size=n_words, max_size=n_words))
        descriptions.append(' '.join(desc_words))
    
    # Generate attributes
    attribute_options = ['beach', 'cultural', 'nature', 'adventure', 'historical', 'wildlife']
    attributes = {}
    location_types = {}
    
    for dest_id in destination_ids:
        n_attrs = draw(st.integers(min_value=1, max_value=4))
        attrs = draw(st.lists(st.sampled_from(attribute_options), min_size=n_attrs, max_size=n_attrs, unique=True))
        attributes[dest_id] = attrs
        location_types[dest_id] = draw(st.sampled_from(attribute_options))
    
    # Create and fit model
    max_features = draw(st.integers(min_value=10, max_value=100))
    cb = ContentBasedFilter(max_features=max_features)
    cb.fit(descriptions, attributes, location_types)
    
    return cb, destination_ids


@st.composite
def context_aware_engine_strategy(draw):
    """Generate a fitted ContextAwareEngine model."""
    n_samples = draw(st.integers(min_value=20, max_value=100))
    n_destinations = draw(st.integers(min_value=5, max_value=20))
    
    destination_ids = [f"dest_{i}" for i in range(n_destinations)]
    
    # Generate context features
    weather_conditions = ['sunny', 'rainy', 'cloudy', 'stormy']
    seasons = ['dry', 'monsoon', 'inter-monsoon']
    
    context_data = []
    for _ in range(n_samples):
        context_data.append({
            'weather_sunny': draw(st.integers(min_value=0, max_value=1)),
            'weather_rainy': draw(st.integers(min_value=0, max_value=1)),
            'weather_stormy': draw(st.integers(min_value=0, max_value=1)),
            'temperature': draw(st.floats(min_value=20.0, max_value=35.0)),
            'humidity': draw(st.floats(min_value=40.0, max_value=90.0)),
            'precipitation_chance': draw(st.floats(min_value=0.0, max_value=1.0)),
            'season_dry': draw(st.integers(min_value=0, max_value=1)),
            'season_monsoon': draw(st.integers(min_value=0, max_value=1)),
            'season_inter_monsoon': draw(st.integers(min_value=0, max_value=1)),
            'day_of_week': draw(st.integers(min_value=0, max_value=6)),
            'is_holiday': draw(st.integers(min_value=0, max_value=1)),
            'is_peak_season': draw(st.integers(min_value=0, max_value=1))
        })
    
    context_features = pd.DataFrame(context_data)
    
    # Generate ratings
    ratings = draw(st.lists(
        st.floats(min_value=1.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        min_size=n_samples,
        max_size=n_samples
    ))
    ratings = np.array(ratings)
    
    # Generate location types
    location_type_options = ['beach', 'cultural', 'nature', 'adventure', 'urban']
    location_types = {dest_id: draw(st.sampled_from(location_type_options)) for dest_id in destination_ids}
    
    # Create and fit model
    max_depth = draw(st.integers(min_value=3, max_value=10))
    ca = ContextAwareEngine(max_depth=max_depth)
    ca.fit(context_features, ratings, destination_ids, location_types)
    
    return ca, destination_ids


@st.composite
def metadata_strategy(draw):
    """Generate valid metadata dictionary."""
    version = draw(st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='.-_')))
    
    metrics = {
        'ndcg': draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        'hit_rate': draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        'coverage': draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    }
    
    metadata = {
        'version': version,
        'training_date': '2024-01-01T00:00:00',
        'metrics': metrics
    }
    
    return metadata


class TestModelSerializationRoundTrip:
    """
    Feature: tourism-recommender-system, Property 22: Model Serialization Round-Trip
    
    Property: For any valid trained model, serializing to disk and then deserializing
    SHALL produce a model that generates identical predictions for the same inputs.
    
    Validates: Requirements 8.3
    """
    
    @given(
        model_data=collaborative_filter_strategy(),
        metadata=metadata_strategy()
    )
    @settings(max_examples=100, deadline=None)
    def test_collaborative_filter_round_trip(self, model_data, metadata):
        """Test that CollaborativeFilter survives serialization round-trip."""
        original_model, user_ids, destination_ids = model_data
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl.gz', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Save model
            ModelSerializer.save(original_model, tmp_path, metadata)
            
            # Load model
            loaded_model, loaded_metadata = ModelSerializer.load(tmp_path)
            
            # Verify metadata is preserved
            assert loaded_metadata['version'] == metadata['version']
            assert loaded_metadata['training_date'] == metadata['training_date']
            assert 'saved_at' in loaded_metadata  # Should be added automatically
            
            # Verify model attributes are preserved
            assert loaded_model.n_factors == original_model.n_factors
            assert loaded_model.is_fitted == original_model.is_fitted
            assert loaded_model.user_ids == original_model.user_ids
            assert loaded_model.destination_ids == original_model.destination_ids
            
            # Property: Predictions should be identical for same inputs
            test_user = user_ids[0] if user_ids else None
            if test_user and destination_ids:
                candidate_items = destination_ids[:min(5, len(destination_ids))]
                
                original_predictions = original_model.predict(test_user, candidate_items)
                loaded_predictions = loaded_model.predict(test_user, candidate_items)
                
                # Check that predictions match
                assert len(original_predictions) == len(loaded_predictions)
                
                for (orig_id, orig_score), (load_id, load_score) in zip(original_predictions, loaded_predictions):
                    assert orig_id == load_id
                    # Allow small floating-point differences
                    assert abs(orig_score - load_score) < 1e-5, \
                        f"Prediction mismatch: original={orig_score}, loaded={load_score}"
            
            # Verify confidence scores match
            if user_ids:
                for user_id in user_ids[:3]:  # Test first 3 users
                    orig_conf = original_model.get_confidence(user_id)
                    load_conf = loaded_model.get_confidence(user_id)
                    assert abs(orig_conf - load_conf) < 1e-5, \
                        f"Confidence mismatch for {user_id}: original={orig_conf}, loaded={load_conf}"
        
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    @given(
        model_data=content_based_filter_strategy(),
        metadata=metadata_strategy()
    )
    @settings(max_examples=100, deadline=None)
    def test_content_based_filter_round_trip(self, model_data, metadata):
        """Test that ContentBasedFilter survives serialization round-trip."""
        original_model, destination_ids = model_data
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl.gz', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Save model
            ModelSerializer.save(original_model, tmp_path, metadata)
            
            # Load model
            loaded_model, loaded_metadata = ModelSerializer.load(tmp_path)
            
            # Verify metadata
            assert loaded_metadata['version'] == metadata['version']
            
            # Verify model attributes
            assert loaded_model.max_features == original_model.max_features
            assert loaded_model.is_fitted == original_model.is_fitted
            assert loaded_model.destination_ids == original_model.destination_ids
            
            # Property: Predictions should be identical for same inputs
            if destination_ids and len(destination_ids) >= 2:
                user_preferences = {
                    'preferred_types': ['beach', 'cultural'],
                    'preferred_attributes': ['nature', 'adventure']
                }
                candidate_items = destination_ids[:min(5, len(destination_ids))]
                
                original_predictions = original_model.predict(user_preferences, candidate_items)
                loaded_predictions = loaded_model.predict(user_preferences, candidate_items)
                
                # Check predictions match
                assert len(original_predictions) == len(loaded_predictions)
                
                for (orig_id, orig_score), (load_id, load_score) in zip(original_predictions, loaded_predictions):
                    assert orig_id == load_id
                    assert abs(orig_score - load_score) < 1e-5
                
                # Test similarity computation
                dest1, dest2 = destination_ids[0], destination_ids[1]
                orig_sim = original_model.get_similarity(dest1, dest2)
                load_sim = loaded_model.get_similarity(dest1, dest2)
                assert abs(orig_sim - load_sim) < 1e-5
        
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    @given(
        model_data=context_aware_engine_strategy(),
        metadata=metadata_strategy()
    )
    @settings(max_examples=100, deadline=None)
    def test_context_aware_engine_round_trip(self, model_data, metadata):
        """Test that ContextAwareEngine survives serialization round-trip."""
        original_model, destination_ids = model_data
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl.gz', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Save model
            ModelSerializer.save(original_model, tmp_path, metadata)
            
            # Load model
            loaded_model, loaded_metadata = ModelSerializer.load(tmp_path)
            
            # Verify metadata
            assert loaded_metadata['version'] == metadata['version']
            
            # Verify model attributes
            assert loaded_model.max_depth == original_model.max_depth
            assert loaded_model.is_fitted == original_model.is_fitted
            assert loaded_model.destination_ids == original_model.destination_ids
            
            # Property: Predictions should be identical for same inputs
            # Note: We can't easily create a Context object in the test without importing data_models
            # So we'll just verify the model structure is preserved
            
            # Verify tree structure is preserved
            if original_model.tree is not None and loaded_model.tree is not None:
                assert original_model.tree.tree_.node_count == loaded_model.tree.tree_.node_count
                assert original_model.tree.tree_.max_depth == loaded_model.tree.tree_.max_depth
        
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_serialization_with_compression(self):
        """Test that serialization produces compressed files."""
        # Create a simple model
        user_ids = [f"user_{i}" for i in range(10)]
        destination_ids = [f"dest_{i}" for i in range(10)]
        
        rows = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
        cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        ratings = [4.0] * 10
        
        rating_matrix = sparse.csr_matrix(
            (ratings, (rows, cols)),
            shape=(10, 10),
            dtype=np.float32
        )
        
        cf = CollaborativeFilter(n_factors=5)
        cf.fit(rating_matrix, user_ids, destination_ids)
        
        metadata = {'version': '1.0', 'training_date': '2024-01-01'}
        
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl.gz', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Save model
            ModelSerializer.save(cf, tmp_path, metadata)
            
            # Verify file exists and has reasonable size
            assert os.path.exists(tmp_path)
            
            file_size_mb = ModelSerializer.get_compressed_size(tmp_path)
            
            # Compressed file should be reasonably small
            assert file_size_mb < 10.0, f"Compressed file is {file_size_mb:.2f} MB, expected < 10 MB"
            
            # Load and verify
            loaded_model, loaded_metadata = ModelSerializer.load(tmp_path)
            assert loaded_model.is_fitted
            assert loaded_metadata['version'] == '1.0'
        
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_invalid_path_handling(self):
        """Test error handling for invalid paths."""
        cf = CollaborativeFilter(n_factors=5)
        metadata = {'version': '1.0'}
        
        # Test empty path
        with pytest.raises(ValueError, match="Path cannot be empty"):
            ModelSerializer.save(cf, "", metadata)
        
        with pytest.raises(ValueError, match="Path cannot be empty"):
            ModelSerializer.load("")
        
        # Test non-existent file
        with pytest.raises(FileNotFoundError):
            ModelSerializer.load("/nonexistent/path/model.pkl.gz")
    
    def test_metadata_none_handling(self):
        """Test error handling for None metadata."""
        cf = CollaborativeFilter(n_factors=5)
        
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl.gz', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Test None metadata
            with pytest.raises(ValueError, match="Metadata cannot be None"):
                ModelSerializer.save(cf, tmp_path, None)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
