#!/usr/bin/env python3
"""
Script to verify that all models meet the size constraints.
"""

import numpy as np
import pickle
import sys
from pathlib import Path

# Import the models
from src.collaborative_filter import CollaborativeFilter
from src.content_based_filter import ContentBasedFilter
from src.context_aware_engine import ContextAwareEngine
from src.data_processor import DataProcessor


def get_model_size_mb(obj):
    """Calculate the size of a Python object in MB."""
    serialized = pickle.dumps(obj)
    size_bytes = len(serialized)
    size_mb = size_bytes / (1024 * 1024)
    return size_mb


def main():
    print("=" * 60)
    print("Model Size Verification")
    print("=" * 60)
    
    # Load and process data
    print("\n1. Loading and processing data...")
    processor = DataProcessor()
    
    # Load reviews
    reviews_df = processor.load_reviews('dataset/Reviews.csv')
    print(f"   Loaded {len(reviews_df)} reviews")
    
    # Build rating matrix
    rating_matrix, user_ids, destination_ids = processor.build_rating_matrix(reviews_df)
    print(f"   Built rating matrix: {rating_matrix.shape}")
    
    # Extract location features
    location_features = processor.extract_location_features(reviews_df)
    print(f"   Extracted {len(location_features)} location features")
    
    # Generate TF-IDF embeddings
    descriptions = [loc.name for loc in location_features.values()]
    embeddings = processor.generate_tfidf_embeddings(descriptions)
    print(f"   Generated embeddings: {embeddings.shape}")
    
    # Train models
    print("\n2. Training models...")
    
    # Collaborative Filter
    print("   Training Collaborative Filter...")
    cf = CollaborativeFilter(n_factors=50, n_epochs=20)
    cf.fit(rating_matrix, user_ids, destination_ids)
    cf_size = get_model_size_mb(cf)
    print(f"   ✓ Collaborative Filter size: {cf_size:.2f} MB")
    
    # Content-Based Filter
    print("   Training Content-Based Filter...")
    cb = ContentBasedFilter(max_features=500)
    cb.fit(descriptions, location_features)
    cb_size = get_model_size_mb(cb)
    print(f"   ✓ Content-Based Filter size: {cb_size:.2f} MB")
    
    # Context-Aware Engine
    print("   Training Context-Aware Engine...")
    ca = ContextAwareEngine(max_depth=10)
    # Create dummy context features for training
    import pandas as pd
    context_features = pd.DataFrame({
        'weather_rainy': np.random.randint(0, 2, len(reviews_df)),
        'is_holiday': np.random.randint(0, 2, len(reviews_df)),
        'is_peak_season': np.random.randint(0, 2, len(reviews_df)),
        'season_monsoon': np.random.randint(0, 2, len(reviews_df))
    })
    ratings = reviews_df['rating'].values
    location_types = {dest_id: loc.location_type for dest_id, loc in location_features.items()}
    ca.fit(context_features, ratings, destination_ids, location_types)
    ca_size = get_model_size_mb(ca)
    print(f"   ✓ Context-Aware Engine size: {ca_size:.2f} MB")
    
    # Calculate total size
    total_size = cf_size + cb_size + ca_size
    
    print("\n" + "=" * 60)
    print("Model Size Summary")
    print("=" * 60)
    print(f"Collaborative Filter:    {cf_size:>8.2f} MB (target: < 10 MB)")
    print(f"Content-Based Filter:    {cb_size:>8.2f} MB (target: < 5 MB)")
    print(f"Context-Aware Engine:    {ca_size:>8.2f} MB (target: < 3 MB)")
    print("-" * 60)
    print(f"Total Size:              {total_size:>8.2f} MB (target: < 25 MB)")
    print("=" * 60)
    
    # Check constraints
    print("\nConstraint Verification:")
    constraints_met = True
    
    if cf_size > 10:
        print(f"❌ Collaborative Filter exceeds 10 MB limit ({cf_size:.2f} MB)")
        constraints_met = False
    else:
        print(f"✓ Collaborative Filter within 10 MB limit")
    
    if cb_size > 5:
        print(f"❌ Content-Based Filter exceeds 5 MB limit ({cb_size:.2f} MB)")
        constraints_met = False
    else:
        print(f"✓ Content-Based Filter within 5 MB limit")
    
    if ca_size > 3:
        print(f"❌ Context-Aware Engine exceeds 3 MB limit ({ca_size:.2f} MB)")
        constraints_met = False
    else:
        print(f"✓ Context-Aware Engine within 3 MB limit")
    
    if total_size > 25:
        print(f"❌ Total size exceeds 25 MB limit ({total_size:.2f} MB)")
        constraints_met = False
    else:
        print(f"✓ Total size within 25 MB limit")
    
    print("\n" + "=" * 60)
    if constraints_met:
        print("✓ ALL SIZE CONSTRAINTS MET")
        print("=" * 60)
        return 0
    else:
        print("❌ SOME SIZE CONSTRAINTS NOT MET")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
