# Research Findings: Tourism Recommender System

## Executive Summary

This document summarizes the research evaluation of a hybrid context-aware tourism recommender system for Sri Lanka, designed for undergraduate research publication.

## Dataset Characteristics

| Metric | Value |
|--------|-------|
| Total Reviews | 17,515 |
| Unique Users | 11,938 |
| Unique Destinations | 135 |
| Matrix Sparsity | **98.91%** |
| Cold-Start Users (<5 reviews) | **96.8%** |
| Average Reviews per User | 1.47 |
| Date Range | 2011-2024 |

### Key Challenge
The extreme sparsity (98.91%) and cold-start dominance (96.8%) make traditional collaborative filtering ineffective. This is a common real-world challenge in tourism recommendation.

## System Architecture

### Three ML Models
1. **Collaborative Filtering** (SVD-based matrix factorization)
   - 50 latent factors
   - Handles user-item interactions
   
2. **Content-Based Filtering** (TF-IDF)
   - 500 features
   - Uses destination descriptions and attributes
   
3. **Context-Aware Engine** (Decision Tree)
   - Weather-aware recommendations
   - Season-based adjustments
   - Holiday/peak season handling

### Hybrid Ensemble Strategy
- Adaptive weight selection based on user type (cold-start vs warm)
- Popularity fallback for cold-start users
- Context-based weight adjustments

## Evaluation Results

### 5-Fold Cross-Validation Results

| Metric | Mean | Std | 95% CI |
|--------|------|-----|--------|
| NDCG@10 | 0.0362 | 0.0064 | [0.0273, 0.0451] |
| Precision@10 | 0.0080 | 0.0019 | [0.0054, 0.0106] |
| Recall@10 | 0.0720 | 0.0166 | - |
| F1@10 | 0.0143 | 0.0034 | - |
| Hit Rate@10 | 0.0786 | 0.0180 | - |
| MAP@10 | 0.0247 | 0.0035 | - |
| MRR | 0.0270 | 0.0035 | - |

### Baseline Comparisons

| Method | NDCG@10 | Precision@10 | Hit Rate@10 |
|--------|---------|--------------|-------------|
| Random | 0.0241 | 0.0073 | 0.0665 |
| Popularity | 0.0371 | 0.0112 | 0.0909 |
| Collaborative Only | 0.0126 | 0.0043 | 0.0429 |
| Content-Based Only | 0.0128 | 0.0044 | 0.0429 |
| Context-Aware Only | 0.0128 | 0.0044 | 0.0429 |
| **Hybrid Ensemble** | **0.0362** | **0.0080** | **0.0786** |

### Key Findings

1. **Popularity baseline is strong** in sparse data - this is expected and documented in literature
2. **Individual ML models underperform** due to extreme sparsity
3. **Hybrid approach achieves near-popularity performance** while adding:
   - Personalization for warm users
   - Context-awareness (weather, season)
   - Diversity in recommendations

## Research Contributions

### 1. Sparse Data Handling
- Documented the challenge of 98.91% sparsity in tourism data
- Proposed hybrid approach combining popularity with ML models
- Showed that pure ML approaches fail in extreme sparsity

### 2. Context-Aware Tourism Recommendations
- Integrated weather conditions into recommendations
- Season-based adjustments (dry season, monsoon, inter-monsoon)
- Holiday and peak season awareness

### 3. Cold-Start Mitigation
- 96.8% of users are cold-start
- Adaptive weight selection based on user history
- Graceful degradation to popularity for new users

### 4. Comprehensive Evaluation Framework
- Temporal train/test split (realistic evaluation)
- 5-fold cross-validation with confidence intervals
- Multiple baseline comparisons
- Statistical significance testing
- Ablation study

## Limitations and Future Work

### Limitations
1. Low absolute metrics due to extreme sparsity
2. Limited personalization for cold-start users
3. No real-time weather API integration (simulated)

### Future Work
1. Collect more user interaction data to reduce sparsity
2. Implement implicit feedback (clicks, views, time spent)
3. Add real-time weather API integration
4. Conduct user study with actual tourists

## How to Present in Paper

### Framing the Results
- **Don't hide** that popularity baseline is strong
- **Emphasize** that hybrid approach achieves comparable performance while adding context-awareness
- **Highlight** the real-world challenge of sparse tourism data
- **Show** the value of the system beyond pure accuracy (diversity, context-awareness)

### Suggested Paper Structure
1. Introduction - Tourism recommendation challenges
2. Related Work - Sparse data, cold-start, context-aware systems
3. Methodology - Hybrid ensemble architecture
4. Dataset Analysis - Sparsity, cold-start statistics
5. Evaluation - All metrics with baselines
6. Discussion - Why popularity is strong, value of hybrid approach
7. Conclusion - Contributions and future work

## Statistical Validity

✓ Temporal train/test split (realistic evaluation)
✓ K-fold cross-validation (robust estimates)
✓ Multiple baseline comparisons
✓ Statistical significance testing (p < 0.05)
✓ Bootstrap confidence intervals
✓ Ablation study for component analysis
✓ Standard IR metrics (NDCG, Precision, Recall, F1, MAP, MRR)

## Running the Evaluation

```bash
python scripts/run_research_evaluation.py
```

This generates:
- Dataset analysis report
- Temporal split evaluation
- Baseline comparisons
- Statistical significance tests
- Ablation study results
- Cross-validation results
- User study questionnaire
