# Implementation Plan: Tourism Recommender System

## Overview

This implementation plan breaks down the lightweight ensemble-based tourism recommender system into discrete coding tasks. The system will be implemented in Python using scikit-learn, pandas, numpy, and Hypothesis for property-based testing.

## Tasks

- [x] 1. Set up project structure and dependencies
  - Create directory structure: `src/`, `tests/`, `data/`, `models/`
  - Initialize `pyproject.toml` with dependencies: pandas, numpy, scikit-learn, scipy, hypothesis, pytest
  - Create base data classes and type definitions
  - _Requirements: 1.1, 1.2_

- [x] 2. Implement Data Processing Module
  - [x] 2.1 Implement review data loader
    - Create `DataProcessor.load_reviews()` to parse CSV files
    - Handle both Reviews.csv format and reviews_2/*.csv format
    - Extract location features (name, city, coordinates, type)
    - _Requirements: 1.1_

  - [x] 2.2 Write property test for data extraction completeness
    - **Property 1: Data Extraction Completeness**
    - **Validates: Requirements 1.1**

  - [x] 2.3 Implement user profile builder
    - Create `DataProcessor.build_user_profiles()` 
    - Calculate rating history, preferred types, cold_start flag
    - _Requirements: 1.2_

  - [x] 2.4 Write property test for user profile construction
    - **Property 2: User Profile Construction**
    - **Validates: Requirements 1.2**

  - [x] 2.5 Implement TF-IDF embedding generator
    - Create `DataProcessor.generate_tfidf_embeddings()`
    - Limit to 500 features, normalize vectors
    - _Requirements: 1.3, 3.2_

  - [x] 2.6 Write property test for TF-IDF embedding validity
    - **Property 3: TF-IDF Embedding Validity**
    - **Validates: Requirements 1.3, 3.2**

  - [x] 2.7 Implement rating matrix builder
    - Create `DataProcessor.build_rating_matrix()`
    - Handle missing values, normalize to [1,5] range
    - Use sparse CSR format
    - _Requirements: 1.4_

  - [x] 2.8 Write property test for rating matrix normalization
    - **Property 4: Rating Matrix Normalization**
    - **Validates: Requirements 1.4**

  - [x] 2.9 Implement deduplication logic
    - Deduplicate by user-destination pair, keep most recent
    - _Requirements: 1.5_

  - [x] 2.10 Write property test for deduplication
    - **Property 5: Deduplication Preserves Most Recent**
    - **Validates: Requirements 1.5**

- [x] 3. Checkpoint - Data Processing Complete
  - Ensure all data processing tests pass
  - Verify data can be loaded from dataset folder

- [x] 4. Implement Collaborative Filtering Model
  - [x] 4.1 Implement CollaborativeFilter class
    - Use SVD from scipy or surprise library
    - Configurable n_factors (default 50)
    - Implement fit() and predict() methods
    - _Requirements: 2.1, 2.4_

  - [x] 4.2 Implement confidence scoring
    - Return confidence based on user rating count
    - Return 0 for cold start users (<5 ratings)
    - _Requirements: 2.5_

  - [x] 4.3 Write property test for cold start confidence
    - **Property 6: Cold Start Confidence**
    - **Validates: Requirements 2.5**

  - [x] 4.4 Write property test for inference latency
    - **Property 7: Collaborative Filter Inference Latency**
    - **Validates: Requirements 2.3**

- [x] 5. Implement Content-Based Filter
  - [x] 5.1 Implement ContentBasedFilter class
    - Use TF-IDF vectorizer with max 500 features
    - Compute cosine similarity matrix
    - Implement fit() and predict() methods
    - _Requirements: 3.1, 3.2, 3.4_

  - [x] 5.2 Write property test for similarity range
    - **Property 8: Content-Based Similarity Range**
    - **Validates: Requirements 3.1**

  - [x] 5.3 Implement preference matching
    - Rank destinations by attribute match score
    - _Requirements: 3.5_

  - [x] 5.4 Write property test for preference ranking
    - **Property 9: Preference Ranking Correctness**
    - **Validates: Requirements 3.5**

- [x] 6. Implement Context-Aware Engine
  - [x] 6.1 Implement ContextAwareEngine class
    - Use DecisionTreeClassifier with max_depth=10
    - Implement fit() and predict() methods
    - _Requirements: 4.1, 4.4, 4.5_

  - [x] 6.2 Implement weather-based scoring
    - Deprioritize beach destinations in rain
    - Boost cultural destinations in monsoon
    - _Requirements: 4.2, 4.3_

  - [x] 6.3 Write property test for weather context scoring
    - **Property 10: Weather Context Scoring**
    - **Validates: Requirements 4.2, 4.3**

  - [x] 6.4 Implement holiday boost logic
    - Boost cultural destinations during holidays
    - _Requirements: 4.6_

  - [x] 6.5 Write property test for holiday context boost
    - **Property 11: Holiday Context Boost**
    - **Validates: Requirements 4.6**

- [x] 7. Checkpoint - Individual Models Complete
  - Ensure all model tests pass
  - Verify each model can generate predictions independently

- [x] 8. Implement Ensemble Voting System
  - [x] 8.1 Implement weighted voting
    - Combine predictions using configurable weights
    - _Requirements: 5.1_

  - [x] 8.2 Write property test for weighted voting correctness
    - **Property 12: Weighted Voting Correctness**
    - **Validates: Requirements 5.1**

  - [x] 8.3 Implement Borda count voting
    - Aggregate rankings using Borda count method
    - _Requirements: 5.2_

  - [x] 8.4 Write property test for Borda count correctness
    - **Property 13: Borda Count Correctness**
    - **Validates: Requirements 5.2**

  - [x] 8.5 Implement confidence-based voting
    - Weight predictions by model confidence scores
    - _Requirements: 5.3_

  - [x] 8.6 Implement context-based weight adjustment
    - Adjust weights for cold_start, weather_critical, peak_season
    - _Requirements: 5.4, 5.5, 5.6_

  - [x] 8.7 Write property test for weight adjustment
    - **Property 14: Context-Based Weight Adjustment**
    - **Validates: Requirements 5.4, 5.5, 5.6**

  - [x] 8.8 Implement top-K selection
    - Return exactly K recommendations
    - _Requirements: 5.7_

  - [x] 8.9 Write property test for top-K output size
    - **Property 15: Top-K Output Size**
    - **Validates: Requirements 5.7**

- [x] 9. Implement Mobile Optimizer
  - [x] 9.1 Implement LRU cache
    - Cache frequently accessed destinations (max 100)
    - _Requirements: 6.4_

  - [x] 9.2 Write property test for LRU cache eviction
    - **Property 16: LRU Cache Eviction**
    - **Validates: Requirements 6.4**

  - [x] 9.3 Implement TTL cache for weather
    - 1 hour expiry for weather data
    - _Requirements: 6.5_

  - [x] 9.4 Write property test for TTL cache expiry
    - **Property 17: TTL Cache Expiry**
    - **Validates: Requirements 6.5**

  - [x] 9.5 Implement model quantization
    - Convert float32 to int8/float16
    - _Requirements: 6.1_

  - [x] 9.6 Implement model pruning
    - Remove up to 50% of low-importance weights
    - _Requirements: 6.2_

- [x] 10. Checkpoint - Ensemble and Optimization Complete
  - Ensure all ensemble tests pass
  - Verify total model size under 25 MB

- [x] 11. Implement Recommendation API
  - [x] 11.1 Implement RecommendationRequest and Recommendation dataclasses
    - Define input/output data structures
    - _Requirements: 7.1, 7.2_

  - [x] 11.2 Write property test for output format
    - **Property 18: Recommendation Output Format**
    - **Validates: Requirements 7.2**

  - [x] 11.3 Implement budget and distance filters
    - Filter recommendations by budget range and max distance
    - _Requirements: 7.3, 7.4_

  - [x] 11.4 Write property test for filter application
    - **Property 19: Filter Application Correctness**
    - **Validates: Requirements 7.3, 7.4**

  - [x] 11.5 Implement invalid user handling
    - Treat unknown user_id as cold start
    - _Requirements: 7.5_

  - [x] 11.6 Write property test for invalid user handling
    - **Property 20: Invalid User Handling**
    - **Validates: Requirements 7.5**

  - [x] 11.7 Implement diversity-aware reranking
    - Ensure diverse destination types in output
    - _Requirements: 7.6_

  - [x] 11.8 Write property test for diversity
    - **Property 21: Diversity in Recommendations**
    - **Validates: Requirements 7.6**

- [x] 12. Implement Model Serializer
  - [x] 12.1 Implement save and load methods
    - Serialize models with compression
    - Store metadata (version, date, metrics)
    - _Requirements: 8.1, 8.2, 8.4_

  - [x] 12.2 Write property test for serialization round-trip
    - **Property 22: Model Serialization Round-Trip**
    - **Validates: Requirements 8.3**

- [x] 13. Implement Evaluation Module
  - [x] 13.1 Implement NDCG@K calculation
    - _Requirements: 9.1_

  - [x] 13.2 Implement Hit Rate@K calculation
    - _Requirements: 9.2_

  - [x] 13.3 Implement diversity and coverage scores
    - _Requirements: 9.3, 9.4_

  - [x] 13.4 Write property test for metric computation
    - **Property 23: Metric Computation Correctness**
    - **Validates: Requirements 9.1, 9.2, 9.3, 9.4**

- [x] 14. Implement Error Handling
  - [x] 14.1 Implement invalid rating rejection
    - Reject ratings outside [1,5] range
    - _Requirements: 10.1_

  - [x] 14.2 Write property test for invalid rating rejection
    - **Property 24: Invalid Rating Rejection**
    - **Validates: Requirements 10.1**

  - [x] 14.3 Implement no-review destination handling
    - Exclude from CF, include in CB
    - _Requirements: 10.2_

  - [x] 14.4 Write property test for no-review handling
    - **Property 25: No-Review Destination Handling**
    - **Validates: Requirements 10.2**

  - [x] 14.5 Implement fallback mechanisms
    - Weather API fallback, popular destinations fallback
    - _Requirements: 10.3, 10.4_

  - [x] 14.6 Implement structured logging
    - Log errors with timestamps and context
    - _Requirements: 10.5_

- [x] 15. Integration and End-to-End Testing
  - [x] 15.1 Wire all components together
    - Create main RecommenderSystem class
    - Connect DataProcessor → Models → Ensemble → API
    - _Requirements: All_

  - [x] 15.2 Load and process actual dataset
    - Process Reviews.csv and reviews_2/*.csv files
    - Train all models on real data
    - _Requirements: 1.1, 1.2, 1.3, 1.4_

  - [x] 15.3 Write integration tests
    - Test end-to-end recommendation flow
    - Verify latency under 100ms
    - _Requirements: 6.8_

- [x] 16. Final Checkpoint
  - Ensure all tests pass
  - Verify model sizes meet constraints
  - Document usage examples

## Notes

- All tasks including property-based tests are required for comprehensive validation
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties
- Unit tests validate specific examples and edge cases
- Python with Hypothesis is used for property-based testing
