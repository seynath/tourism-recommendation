#!/usr/bin/env python
"""Script to load actual dataset and train all models."""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.recommender_system import RecommenderSystem
from src.logger import get_logger

logger = get_logger()


def main():
    """Load dataset and train all models."""
    print("=" * 60)
    print("Tourism Recommender System - Training Pipeline")
    print("=" * 60)
    
    # Initialize the recommender system
    print("\n1. Initializing recommender system...")
    recommender = RecommenderSystem(
        n_factors=50,
        max_features=500,
        max_depth=10,
        voting_strategy='weighted',
    )
    
    # Load data from dataset folder
    print("\n2. Loading review data...")
    start_time = time.time()
    
    try:
        reviews_df = recommender.load_data('dataset')
        load_time = time.time() - start_time
        print(f"   Loaded {len(reviews_df)} reviews in {load_time:.2f}s")
        print(f"   Unique destinations: {reviews_df['destination_id'].nunique()}")
        print(f"   Unique users: {reviews_df['user_id'].nunique()}")
    except Exception as e:
        print(f"   Error loading data: {e}")
        return 1
    
    # Extract features
    print("\n3. Extracting features...")
    start_time = time.time()
    
    try:
        location_features, user_profiles = recommender.extract_features()
        extract_time = time.time() - start_time
        print(f"   Extracted {len(location_features)} destination features in {extract_time:.2f}s")
        print(f"   Built {len(user_profiles)} user profiles")
        
        # Show sample destination types
        type_counts = {}
        for features in location_features.values():
            loc_type = features.location_type
            type_counts[loc_type] = type_counts.get(loc_type, 0) + 1
        print(f"   Destination types: {type_counts}")
    except Exception as e:
        print(f"   Error extracting features: {e}")
        return 1
    
    # Train models
    print("\n4. Training models...")
    start_time = time.time()
    
    try:
        recommender.train()
        train_time = time.time() - start_time
        print(f"   All models trained in {train_time:.2f}s")
    except Exception as e:
        print(f"   Error training models: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Check model sizes
    print("\n5. Model sizes:")
    sizes = recommender.get_model_sizes()
    for model_name, size_mb in sizes.items():
        print(f"   {model_name}: {size_mb:.2f} MB")
    
    # Verify total size is under 25 MB (Requirement 6.3)
    if sizes['total'] > 25.0:
        print(f"\n   WARNING: Total model size ({sizes['total']:.2f} MB) exceeds 25 MB limit!")
        print("   Applying compression...")
        recommender.compress_models()
        sizes = recommender.get_model_sizes()
        print(f"   After compression: {sizes['total']:.2f} MB")
    else:
        print(f"\n   Total model size ({sizes['total']:.2f} MB) is within 25 MB limit ✓")
    
    # Test recommendations
    print("\n6. Testing recommendations...")
    
    # Get a sample user
    sample_users = list(user_profiles.keys())[:3]
    
    for user_id in sample_users:
        print(f"\n   Recommendations for {user_id}:")
        start_time = time.time()
        
        try:
            recommendations = recommender.get_recommendations(
                user_id=user_id,
                weather_condition='sunny',
                season='dry',
            )
            latency_ms = (time.time() - start_time) * 1000
            
            print(f"   Latency: {latency_ms:.1f}ms")
            for i, rec in enumerate(recommendations[:5], 1):
                print(f"   {i}. {rec.name} (score: {rec.score:.3f})")
        except Exception as e:
            print(f"   Error getting recommendations: {e}")
    
    # Save models
    print("\n7. Saving models...")
    try:
        recommender.save_models('models')
        print("   Models saved to 'models/' directory")
    except Exception as e:
        print(f"   Error saving models: {e}")
        return 1
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
