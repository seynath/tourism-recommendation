#!/usr/bin/env python3
"""
Verification script to ensure all models can generate predictions independently.
This script tests each model's ability to fit and predict on sample data.
"""

import numpy as np
import pandas as pd
from scipy import sparse
from src.collaborative_filter import CollaborativeFilter
from src.content_based_filter import ContentBasedFilter
from src.context_aware_engine import ContextAwareEngine
from src.data_models import Context, WeatherInfo

def verify_collaborative_filter():
    """Verify CollaborativeFilter can fit and predict."""
    print("Testing CollaborativeFilter...")
    
    # Create sample rating matrix (5 users x 10 destinations)
    data = np.random.uniform(1, 5, size=20)
    row = np.random.randint(0, 5, size=20)
    col = np.random.randint(0, 10, size=20)
    rating_matrix = sparse.csr_matrix((data, (row, col)), shape=(5, 10))
    
    # Create user and destination IDs
    user_ids = [f"user_{i}" for i in range(5)]
    destination_ids = [f"dest_{i}" for i in range(10)]
    
    # Initialize and fit model
    cf = CollaborativeFilter(n_factors=10, n_epochs=5)
    cf.fit(rating_matrix, user_ids, destination_ids)
    
    # Generate predictions
    predictions = cf.predict(user_id="user_0", candidate_items=destination_ids)
    
    # Verify predictions
    assert len(predictions) == 10, "Should return 10 predictions"
    assert all(isinstance(p, tuple) and len(p) == 2 for p in predictions), "Each prediction should be (item_id, score)"
    
    # Test confidence scoring
    confidence = cf.get_confidence(user_id="user_0")
    assert 0 <= confidence <= 1, "Confidence should be in [0, 1]"
    
    print("✓ CollaborativeFilter verified successfully")
    return True

def verify_content_based_filter():
    """Verify ContentBasedFilter can fit and predict."""
    print("Testing ContentBasedFilter...")
    
    # Create sample destination descriptions
    descriptions = [
        "Beautiful beach with crystal clear water and white sand",
        "Ancient temple with historical significance and cultural heritage",
        "Wildlife sanctuary with elephants and diverse fauna",
        "Mountain peak with hiking trails and scenic views",
        "Coastal town with surfing and water sports",
        "Buddhist temple with ancient architecture",
        "National park with safari tours",
        "Beach resort with luxury amenities"
    ]
    
    attributes = {
        '0': ['beach', 'water', 'relaxation'],
        '1': ['cultural', 'historical', 'temple'],
        '2': ['nature', 'wildlife', 'safari'],
        '3': ['adventure', 'hiking', 'mountain'],
        '4': ['beach', 'adventure', 'surfing'],
        '5': ['cultural', 'temple', 'buddhist'],
        '6': ['nature', 'wildlife', 'safari'],
        '7': ['beach', 'luxury', 'resort']
    }
    
    # Initialize and fit model
    cb = ContentBasedFilter(max_features=50)
    cb.fit(descriptions, attributes)
    
    # Generate predictions with user preferences
    user_preferences = {'preferred_types': ['beach', 'relaxation']}
    predictions = cb.predict(user_preferences, candidate_items=list(range(8)))
    
    # Verify predictions
    assert len(predictions) == 8, "Should return 8 predictions"
    assert all(isinstance(p, tuple) and len(p) == 2 for p in predictions), "Each prediction should be (item_id, score)"
    assert all(0 <= score <= 1 for _, score in predictions), "Scores should be in [0, 1]"
    
    # Test similarity computation
    similar = cb.get_similar_destinations(destination_id='0', k=3)
    assert len(similar) <= 3, "Should return at most 3 similar destinations"
    
    print("✓ ContentBasedFilter verified successfully")
    return True

def verify_context_aware_engine():
    """Verify ContextAwareEngine can fit and predict."""
    print("Testing ContextAwareEngine...")
    
    # Create sample context features and ratings
    n_samples = 100
    context_features = pd.DataFrame({
        'weather_sunny': np.random.choice([0, 1], n_samples),
        'weather_rainy': np.random.choice([0, 1], n_samples),
        'weather_stormy': np.random.choice([0, 1], n_samples),
        'temperature': np.random.uniform(20, 35, n_samples),
        'humidity': np.random.uniform(60, 90, n_samples),
        'precipitation_chance': np.random.uniform(0, 1, n_samples),
        'season_dry': np.random.choice([0, 1], n_samples),
        'season_monsoon': np.random.choice([0, 1], n_samples),
        'season_inter_monsoon': np.random.choice([0, 1], n_samples),
        'day_of_week': np.random.randint(0, 7, n_samples),
        'is_holiday': np.random.choice([0, 1], n_samples),
        'is_peak_season': np.random.choice([0, 1], n_samples)
    })
    
    ratings = np.random.uniform(1, 5, n_samples)
    
    # Create destination IDs and location types
    destination_ids = ['beach_1', 'cultural_1', 'nature_1', 'beach_2', 'cultural_2']
    location_types = {
        'beach_1': 'beach',
        'cultural_1': 'cultural',
        'nature_1': 'nature',
        'beach_2': 'beach',
        'cultural_2': 'cultural'
    }
    
    # Initialize and fit model
    ca = ContextAwareEngine(max_depth=5)
    ca.fit(context_features, ratings, destination_ids, location_types)
    
    # Create test context
    test_context = Context(
        location=(6.9271, 79.8612),  # Colombo coordinates
        weather=WeatherInfo(
            condition='sunny',
            temperature=28.0,
            humidity=75.0,
            precipitation_chance=0.1
        ),
        season='dry',
        day_of_week=5,
        is_holiday=False,
        is_peak_season=True,
        user_type='regular'
    )
    
    # Generate predictions
    predictions = ca.predict(test_context, destination_ids)
    
    # Verify predictions
    assert len(predictions) == 5, "Should return 5 predictions"
    assert all(isinstance(p, tuple) and len(p) == 2 for p in predictions), "Each prediction should be (item_id, score)"
    
    # Test context type classification
    context_type = ca.get_context_type(test_context)
    assert isinstance(context_type, str), "Context type should be a string"
    
    print("✓ ContextAwareEngine verified successfully")
    return True

def main():
    """Run all model verifications."""
    print("=" * 60)
    print("Model Verification Script")
    print("=" * 60)
    print()
    
    results = []
    
    try:
        results.append(("CollaborativeFilter", verify_collaborative_filter()))
    except Exception as e:
        print(f"✗ CollaborativeFilter failed: {e}")
        results.append(("CollaborativeFilter", False))
    
    print()
    
    try:
        results.append(("ContentBasedFilter", verify_content_based_filter()))
    except Exception as e:
        print(f"✗ ContentBasedFilter failed: {e}")
        results.append(("ContentBasedFilter", False))
    
    print()
    
    try:
        results.append(("ContextAwareEngine", verify_context_aware_engine()))
    except Exception as e:
        print(f"✗ ContextAwareEngine failed: {e}")
        results.append(("ContextAwareEngine", False))
    
    print()
    print("=" * 60)
    print("Verification Summary")
    print("=" * 60)
    
    for model_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{model_name:25s} {status}")
    
    print()
    
    all_passed = all(success for _, success in results)
    if all_passed:
        print("✓ All models verified successfully!")
        print("✓ Each model can generate predictions independently")
        return 0
    else:
        print("✗ Some models failed verification")
        return 1

if __name__ == "__main__":
    exit(main())
