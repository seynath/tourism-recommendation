"""Property-based tests for ContentBasedFilter module."""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
import time

from src.content_based_filter import ContentBasedFilter


# Strategies for generating test data
@st.composite
def destination_data_strategy(draw):
    """Generate destination descriptions and attributes."""
    n_destinations = draw(st.integers(min_value=2, max_value=50))
    
    # Generate destination IDs
    destination_ids = [f"dest_{i}" for i in range(n_destinations)]
    
    # Generate descriptions (simple text)
    word_pool = ['beach', 'cultural', 'nature', 'temple', 'wildlife', 'scenic', 
                 'historical', 'modern', 'traditional', 'adventure', 'relaxing',
                 'beautiful', 'ancient', 'popular', 'quiet', 'bustling']
    
    descriptions = []
    for _ in range(n_destinations):
        n_words = draw(st.integers(min_value=3, max_value=10))
        words = draw(st.lists(
            st.sampled_from(word_pool),
            min_size=n_words,
            max_size=n_words
        ))
        descriptions.append(' '.join(words))
    
    # Generate attributes
    attribute_pool = ['beach', 'cultural', 'nature', 'adventure', 'historical',
                     'wildlife', 'scenic', 'temple', 'surfing', 'diving']
    
    attributes = {}
    for dest_id in destination_ids:
        n_attrs = draw(st.integers(min_value=0, max_value=5))
        dest_attrs = draw(st.lists(
            st.sampled_from(attribute_pool),
            min_size=n_attrs,
            max_size=n_attrs,
            unique=True
        ))
        attributes[dest_id] = dest_attrs
    
    return descriptions, attributes, destination_ids


class TestContentBasedSimilarityRange:
    """
    Feature: tourism-recommender-system, Property 8: Content-Based Similarity Range
    
    Property: For any pair of destinations, the computed cosine similarity
    SHALL be in the range [0, 1].
    
    Validates: Requirements 3.1
    """
    
    @given(data=destination_data_strategy())
    @settings(max_examples=100, deadline=None)
    def test_similarity_range_zero_to_one(self, data):
        """Test that all pairwise similarities are in [0, 1] range."""
        descriptions, attributes, destination_ids = data
        
        # Fit the model
        cbf = ContentBasedFilter(max_features=500)
        cbf.fit(descriptions, attributes)
        
        # Check all pairwise similarities
        for i, dest_id_1 in enumerate(destination_ids):
            for j, dest_id_2 in enumerate(destination_ids):
                similarity = cbf.get_similarity(dest_id_1, dest_id_2)
                
                # Property: Similarity must be in [0, 1] range
                assert 0.0 <= similarity <= 1.0, \
                    f"Similarity between {dest_id_1} and {dest_id_2} is {similarity}, expected in [0, 1]"
    
    @given(data=destination_data_strategy())
    @settings(max_examples=100, deadline=None)
    def test_similarity_matrix_range(self, data):
        """Test that entire similarity matrix is in [0, 1] range."""
        descriptions, attributes, destination_ids = data
        
        # Fit the model
        cbf = ContentBasedFilter(max_features=500)
        cbf.fit(descriptions, attributes)
        
        # Check entire similarity matrix
        similarity_matrix = cbf.similarity_matrix
        
        # Property: All values in similarity matrix must be in [0, 1]
        assert np.all(similarity_matrix >= 0.0), \
            f"Similarity matrix contains values < 0: min={similarity_matrix.min()}"
        assert np.all(similarity_matrix <= 1.0), \
            f"Similarity matrix contains values > 1: max={similarity_matrix.max()}"
    
    def test_self_similarity_is_one(self):
        """Test that similarity of a destination with itself is 1.0."""
        descriptions = ["beach resort with surfing", "cultural temple site", "nature reserve"]
        attributes = {
            "dest_0": ["beach", "surfing"],
            "dest_1": ["cultural", "temple"],
            "dest_2": ["nature", "wildlife"]
        }
        
        cbf = ContentBasedFilter(max_features=500)
        cbf.fit(descriptions, attributes)
        
        # Check self-similarity for each destination
        for dest_id in attributes.keys():
            similarity = cbf.get_similarity(dest_id, dest_id)
            
            # Self-similarity should be 1.0 (or very close due to floating point)
            assert abs(similarity - 1.0) < 1e-6, \
                f"Self-similarity for {dest_id} is {similarity}, expected 1.0"
    
    def test_similarity_is_symmetric(self):
        """Test that similarity(A, B) == similarity(B, A)."""
        descriptions = ["beach resort", "mountain hiking", "city tour"]
        attributes = {
            "dest_0": ["beach"],
            "dest_1": ["nature"],
            "dest_2": ["cultural"]
        }
        
        cbf = ContentBasedFilter(max_features=500)
        cbf.fit(descriptions, attributes)
        
        # Check symmetry
        dest_ids = list(attributes.keys())
        for i, dest_id_1 in enumerate(dest_ids):
            for j, dest_id_2 in enumerate(dest_ids):
                if i < j:  # Only check upper triangle
                    sim_12 = cbf.get_similarity(dest_id_1, dest_id_2)
                    sim_21 = cbf.get_similarity(dest_id_2, dest_id_1)
                    
                    # Similarity should be symmetric
                    assert abs(sim_12 - sim_21) < 1e-6, \
                        f"Similarity not symmetric: sim({dest_id_1}, {dest_id_2})={sim_12}, " \
                        f"sim({dest_id_2}, {dest_id_1})={sim_21}"


class TestPreferenceRankingCorrectness:
    """
    Feature: tourism-recommender-system, Property 9: Preference Ranking Correctness
    
    Property: For any user preference specification and destination set,
    destinations with matching attributes SHALL rank higher than those
    without matching attributes.
    
    Validates: Requirements 3.5
    """
    
    @given(
        preferred_type=st.sampled_from(['beach', 'cultural', 'nature', 'adventure']),
        n_matching=st.integers(min_value=1, max_value=10),
        n_non_matching=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=100, deadline=None)
    def test_matching_destinations_rank_higher(self, preferred_type, n_matching, n_non_matching):
        """Test that destinations matching preferences rank higher than non-matching ones."""
        # Create destinations with and without matching attributes
        destination_ids = []
        descriptions = []
        attributes = {}
        location_types = {}
        
        # Create matching destinations
        for i in range(n_matching):
            dest_id = f"match_{i}"
            destination_ids.append(dest_id)
            descriptions.append(f"{preferred_type} destination with activities")
            attributes[dest_id] = [preferred_type, 'popular']
            location_types[dest_id] = preferred_type
        
        # Create non-matching destinations
        other_types = ['beach', 'cultural', 'nature', 'adventure']
        other_types.remove(preferred_type)
        
        for i in range(n_non_matching):
            dest_id = f"nomatch_{i}"
            destination_ids.append(dest_id)
            other_type = other_types[i % len(other_types)]
            descriptions.append(f"{other_type} destination with activities")
            attributes[dest_id] = [other_type, 'popular']
            location_types[dest_id] = other_type
        
        # Fit the model
        cbf = ContentBasedFilter(max_features=500)
        cbf.fit(descriptions, attributes, location_types)
        
        # Make predictions with preference for the specific type
        user_preferences = {
            'preferred_types': [preferred_type],
            'preferred_attributes': []
        }
        
        predictions = cbf.predict(user_preferences, destination_ids)
        
        # Extract scores for matching and non-matching destinations
        matching_scores = [score for dest_id, score in predictions if dest_id.startswith('match_')]
        non_matching_scores = [score for dest_id, score in predictions if dest_id.startswith('nomatch_')]
        
        # Property: All matching destinations should have higher or equal scores than non-matching
        if matching_scores and non_matching_scores:
            min_matching_score = min(matching_scores)
            max_non_matching_score = max(non_matching_scores)
            
            assert min_matching_score >= max_non_matching_score, \
                f"Matching destinations should rank higher: min_match={min_matching_score}, " \
                f"max_nomatch={max_non_matching_score}"
    
    def test_preference_ranking_with_attributes(self):
        """Test that destinations with more matching attributes rank higher."""
        descriptions = [
            "beach resort with surfing and diving",
            "beach resort with surfing",
            "beach resort",
            "mountain hiking trail"
        ]
        
        attributes = {
            "dest_0": ["beach", "surfing", "diving"],
            "dest_1": ["beach", "surfing"],
            "dest_2": ["beach"],
            "dest_3": ["nature", "hiking"]
        }
        
        location_types = {
            "dest_0": "beach",
            "dest_1": "beach",
            "dest_2": "beach",
            "dest_3": "nature"
        }
        
        cbf = ContentBasedFilter(max_features=500)
        cbf.fit(descriptions, attributes, location_types)
        
        # Prefer beach destinations with surfing and diving
        user_preferences = {
            'preferred_types': ['beach'],
            'preferred_attributes': ['surfing', 'diving']
        }
        
        predictions = cbf.predict(user_preferences, list(attributes.keys()))
        
        # Extract scores
        scores = {dest_id: score for dest_id, score in predictions}
        
        # Property: More matching attributes should result in higher scores
        # dest_0 has all matches (beach + surfing + diving)
        # dest_1 has partial matches (beach + surfing)
        # dest_2 has minimal match (beach only)
        # dest_3 has no matches
        
        assert scores['dest_0'] >= scores['dest_1'], \
            f"dest_0 (all matches) should rank >= dest_1 (partial): {scores['dest_0']} vs {scores['dest_1']}"
        assert scores['dest_1'] >= scores['dest_2'], \
            f"dest_1 (partial) should rank >= dest_2 (minimal): {scores['dest_1']} vs {scores['dest_2']}"
        assert scores['dest_2'] >= scores['dest_3'], \
            f"dest_2 (minimal) should rank >= dest_3 (no match): {scores['dest_2']} vs {scores['dest_3']}"
    
    def test_no_preferences_gives_neutral_scores(self):
        """Test that with no preferences, all destinations get neutral scores."""
        descriptions = ["beach resort", "cultural temple", "nature reserve"]
        attributes = {
            "dest_0": ["beach"],
            "dest_1": ["cultural"],
            "dest_2": ["nature"]
        }
        
        cbf = ContentBasedFilter(max_features=500)
        cbf.fit(descriptions, attributes)
        
        # No preferences specified
        user_preferences = {
            'preferred_types': [],
            'preferred_attributes': []
        }
        
        predictions = cbf.predict(user_preferences, list(attributes.keys()))
        
        # All should have neutral score (0.5)
        for dest_id, score in predictions:
            assert score == 0.5, f"With no preferences, {dest_id} should have score 0.5, got {score}"

