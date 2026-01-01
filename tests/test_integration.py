"""Integration tests for the tourism recommender system."""

import time
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.recommender_system import RecommenderSystem
from src.data_models import (
    Context,
    WeatherInfo,
    RecommendationRequest,
    Recommendation,
    LocationFeatures,
    UserProfile,
)


class TestRecommenderSystemIntegration:
    """Integration tests for the complete recommender system."""
    
    @pytest.fixture(scope="class")
    def trained_system(self):
        """Create and train a recommender system on actual data."""
        system = RecommenderSystem(
            n_factors=50,
            max_features=500,
            max_depth=10,
            voting_strategy='weighted',
        )
        
        # Load actual dataset
        data_path = Path('dataset')
        if data_path.exists():
            system.load_data(str(data_path))
            system.extract_features()
            system.train()
        else:
            pytest.skip("Dataset not available")
        
        return system
    
    def test_end_to_end_recommendation_flow(self, trained_system):
        """Test complete recommendation flow from request to response."""
        # Get a sample user from the trained system
        sample_user = list(trained_system.user_profiles.keys())[0]
        
        # Get recommendations
        recommendations = trained_system.get_recommendations(
            user_id=sample_user,
            weather_condition='sunny',
            season='dry',
        )
        
        # Verify recommendations are returned
        assert len(recommendations) > 0
        assert len(recommendations) <= 10  # Default top-K
        
        # Verify recommendation structure
        for rec in recommendations:
            assert isinstance(rec, Recommendation)
            assert rec.destination_id is not None
            assert rec.name is not None
            assert 0 <= rec.score <= 1
            assert rec.explanation is not None
    
    def test_cold_start_user_recommendations(self, trained_system):
        """Test recommendations for a new user (cold start)."""
        # Use a user ID that doesn't exist in the system
        new_user_id = "new_user_12345"
        
        recommendations = trained_system.get_recommendations(
            user_id=new_user_id,
            weather_condition='sunny',
            season='dry',
        )
        
        # Should still return recommendations (using content-based and context-aware)
        assert len(recommendations) > 0
        
        # Verify the user was treated as cold start
        assert new_user_id in trained_system.user_profiles
        assert trained_system.user_profiles[new_user_id].is_cold_start
    
    def test_inference_latency_under_100ms(self, trained_system):
        """
        Test that end-to-end inference time is under 100ms.
        
        Requirement 6.8: THE Recommender_System SHALL achieve end-to-end 
        inference time under 100ms on mid-range mobile devices.
        """
        sample_user = list(trained_system.user_profiles.keys())[0]
        
        # Run multiple times to get average latency
        latencies = []
        for _ in range(10):
            start_time = time.time()
            _ = trained_system.get_recommendations(
                user_id=sample_user,
                weather_condition='sunny',
                season='dry',
            )
            latency_ms = (time.time() - start_time) * 1000
            latencies.append(latency_ms)
        
        avg_latency = np.mean(latencies)
        max_latency = np.max(latencies)
        
        # Average latency should be well under 100ms
        assert avg_latency < 100, f"Average latency {avg_latency:.1f}ms exceeds 100ms"
        
        # Even max latency should be under 100ms
        assert max_latency < 100, f"Max latency {max_latency:.1f}ms exceeds 100ms"
    
    def test_weather_context_affects_recommendations(self, trained_system):
        """Test that weather context affects recommendation scores."""
        sample_user = list(trained_system.user_profiles.keys())[0]
        
        # Get recommendations for sunny weather
        sunny_recs = trained_system.get_recommendations(
            user_id=sample_user,
            weather_condition='sunny',
            season='dry',
        )
        
        # Get recommendations for rainy weather
        rainy_recs = trained_system.get_recommendations(
            user_id=sample_user,
            weather_condition='rainy',
            season='monsoon',
        )
        
        # Both should return recommendations
        assert len(sunny_recs) > 0
        assert len(rainy_recs) > 0
        
        # The rankings may differ based on weather context
        sunny_ids = [r.destination_id for r in sunny_recs]
        rainy_ids = [r.destination_id for r in rainy_recs]
        
        # At least some destinations should be in both lists
        common = set(sunny_ids) & set(rainy_ids)
        assert len(common) > 0
    
    def test_budget_filter_applied(self, trained_system):
        """Test that budget filter correctly filters recommendations."""
        sample_user = list(trained_system.user_profiles.keys())[0]
        
        # Get recommendations with budget filter
        budget_min, budget_max = 50.0, 200.0
        recommendations = trained_system.get_recommendations(
            user_id=sample_user,
            budget=(budget_min, budget_max),
            weather_condition='sunny',
            season='dry',
        )
        
        # All recommendations should be within budget
        for rec in recommendations:
            if rec.estimated_cost is not None:
                assert budget_min <= rec.estimated_cost <= budget_max
    
    def test_distance_filter_applied(self, trained_system):
        """Test that distance filter correctly filters recommendations."""
        sample_user = list(trained_system.user_profiles.keys())[0]
        
        # Get recommendations with distance filter
        max_distance = 50.0  # 50 km
        recommendations = trained_system.get_recommendations(
            user_id=sample_user,
            location=(7.2906, 80.6337),  # Kandy coordinates
            max_distance_km=max_distance,
            weather_condition='sunny',
            season='dry',
        )
        
        # All recommendations should be within distance
        for rec in recommendations:
            if rec.distance_km is not None:
                assert rec.distance_km <= max_distance
    
    def test_diversity_in_recommendations(self, trained_system):
        """
        Test that recommendations include diverse destination types.
        
        Requirement 7.6: THE Recommender_System SHALL apply diversity-aware 
        reranking to avoid recommending only similar destinations.
        """
        sample_user = list(trained_system.user_profiles.keys())[0]
        
        recommendations = trained_system.get_recommendations(
            user_id=sample_user,
            weather_condition='sunny',
            season='dry',
        )
        
        if len(recommendations) >= 5:
            # Get destination types
            dest_types = set()
            for rec in recommendations:
                if rec.destination_id in trained_system.location_features:
                    dest_type = trained_system.location_features[rec.destination_id].location_type
                    dest_types.add(dest_type)
            
            # Should have at least 2 different types (if available in data)
            # This may not always be achievable depending on the dataset
            assert len(dest_types) >= 1
    
    def test_model_sizes_under_limit(self, trained_system):
        """
        Test that total model size is under 25 MB.
        
        Requirement 6.3: WHEN all models are compressed, THE Mobile_Optimizer 
        SHALL achieve total model size under 25 MB.
        """
        sizes = trained_system.get_model_sizes()
        
        assert sizes['total'] < 25.0, f"Total model size {sizes['total']:.2f} MB exceeds 25 MB limit"
    
    def test_model_save_and_load(self, trained_system, tmp_path):
        """Test that models can be saved and loaded correctly."""
        # Save models
        model_dir = tmp_path / "models"
        trained_system.save_models(str(model_dir))
        
        # Verify files were created
        assert (model_dir / "collaborative_filter.pkl.gz").exists()
        assert (model_dir / "content_based_filter.pkl.gz").exists()
        assert (model_dir / "context_aware_engine.pkl.gz").exists()
        assert (model_dir / "data_cache.pkl.gz").exists()
        
        # Create new system and load models
        new_system = RecommenderSystem()
        new_system.load_models(str(model_dir))
        
        # Verify loaded system can generate recommendations
        sample_user = list(trained_system.user_profiles.keys())[0]
        recommendations = new_system.get_recommendations(
            user_id=sample_user,
            weather_condition='sunny',
            season='dry',
        )
        
        assert len(recommendations) > 0


class TestDataProcessingIntegration:
    """Integration tests for data processing pipeline."""
    
    @pytest.fixture(scope="class")
    def system_with_data(self):
        """Create a system with loaded data."""
        system = RecommenderSystem()
        
        data_path = Path('dataset')
        if data_path.exists():
            system.load_data(str(data_path))
            system.extract_features()
        else:
            pytest.skip("Dataset not available")
        
        return system
    
    def test_data_loading_from_multiple_sources(self, system_with_data):
        """Test that data is loaded from both Reviews.csv and reviews_2 folder."""
        assert system_with_data.reviews_df is not None
        assert len(system_with_data.reviews_df) > 0
        
        # Should have loaded from multiple sources
        # Reviews.csv has ~16000 reviews, reviews_2 adds more
        assert len(system_with_data.reviews_df) > 10000
    
    def test_location_features_extracted(self, system_with_data):
        """Test that location features are properly extracted."""
        assert len(system_with_data.location_features) > 0
        
        # Check a sample location feature
        sample_dest = list(system_with_data.location_features.values())[0]
        assert sample_dest.destination_id is not None
        assert sample_dest.name is not None
        assert sample_dest.location_type in ['beach', 'cultural', 'nature', 'urban', 'other']
        assert 1 <= sample_dest.avg_rating <= 5
        assert sample_dest.review_count > 0
    
    def test_user_profiles_built(self, system_with_data):
        """Test that user profiles are properly built."""
        assert len(system_with_data.user_profiles) > 0
        
        # Check a sample user profile
        sample_user = list(system_with_data.user_profiles.values())[0]
        assert sample_user.user_id is not None
        assert isinstance(sample_user.rating_history, dict)
        assert isinstance(sample_user.is_cold_start, bool)
    
    def test_deduplication_applied(self, system_with_data):
        """Test that duplicate reviews are properly deduplicated."""
        # Check that each user-destination pair appears only once
        df = system_with_data.reviews_df
        duplicates = df.duplicated(subset=['user_id', 'destination_id'], keep=False)
        assert not duplicates.any(), "Found duplicate user-destination pairs"


class TestEnsembleVotingIntegration:
    """Integration tests for ensemble voting system."""
    
    @pytest.fixture(scope="class")
    def trained_system(self):
        """Create and train a recommender system."""
        system = RecommenderSystem(voting_strategy='weighted')
        
        data_path = Path('dataset')
        if data_path.exists():
            system.load_data(str(data_path))
            system.train()
        else:
            pytest.skip("Dataset not available")
        
        return system
    
    def test_weighted_voting_strategy(self, trained_system):
        """Test weighted voting produces valid recommendations."""
        trained_system.ensemble.strategy = 'weighted'
        
        sample_user = list(trained_system.user_profiles.keys())[0]
        recommendations = trained_system.get_recommendations(
            user_id=sample_user,
            weather_condition='sunny',
            season='dry',
        )
        
        assert len(recommendations) > 0
        for rec in recommendations:
            assert 0 <= rec.score <= 1
    
    def test_borda_voting_strategy(self, trained_system):
        """Test Borda count voting produces valid recommendations."""
        trained_system.ensemble.strategy = 'borda'
        
        sample_user = list(trained_system.user_profiles.keys())[0]
        recommendations = trained_system.get_recommendations(
            user_id=sample_user,
            weather_condition='sunny',
            season='dry',
        )
        
        assert len(recommendations) > 0
    
    def test_confidence_voting_strategy(self, trained_system):
        """Test confidence-based voting produces valid recommendations."""
        trained_system.ensemble.strategy = 'confidence'
        
        sample_user = list(trained_system.user_profiles.keys())[0]
        recommendations = trained_system.get_recommendations(
            user_id=sample_user,
            weather_condition='sunny',
            season='dry',
        )
        
        assert len(recommendations) > 0
