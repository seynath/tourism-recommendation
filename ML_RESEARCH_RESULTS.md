# Machine Learning Research Results

## Aspect-Based Sentiment Analysis for Sri Lanka Tourism

**Date**: January 2026  
**Dataset**: Sri Lanka Tourism Reviews (TripAdvisor)

---

## 1. Dataset Overview

| Metric | Value |
|--------|-------|
| Total Reviews | 16,156 |
| Total Locations | 76 |
| Aspects Analyzed | 7 |
| ML Training Samples | 47,538 |
| Average Rating | 4.2/5 |

### Rating Distribution
| Rating | Count | Percentage |
|--------|-------|------------|
| 5 stars | 9,847 | 60.9% |
| 4 stars | 3,421 | 21.2% |
| 3 stars | 1,654 | 10.2% |
| 2 stars | 723 | 4.5% |
| 1 star | 511 | 3.2% |

---

## 2. Aspect Taxonomy

### Tourism-Specific Aspects

| Aspect | Icon | Keywords | Description |
|--------|------|----------|-------------|
| Scenery & Views | 🏞️ | 37 | Natural beauty, photography spots |
| Accessibility | 🚗 | 36 | Transport, parking, ease of access |
| Facilities | 🚻 | 34 | Toilets, shops, infrastructure |
| Safety & Crowds | 🛡️ | 36 | Security, crowd levels |
| Value for Money | 💰 | 32 | Pricing, worth |
| Experience & Activities | 🎯 | 38 | Tours, activities, enjoyment |
| Service & Staff | 👨‍💼 | 22 | Staff behavior, hospitality |

### Aspect Mention Statistics

| Aspect | Total Mentions | Avg Sentiment | Sentiment Label |
|--------|----------------|---------------|-----------------|
| Experience & Activities | 10,756 | +0.200 | Positive |
| Scenery & Views | 8,870 | +0.313 | Positive |
| Accessibility | 7,938 | +0.174 | Positive |
| Value for Money | 6,324 | +0.252 | Positive |
| Facilities | 4,702 | +0.206 | Positive |
| Safety & Crowds | 3,258 | +0.115 | Positive |
| Service & Staff | 1,998 | +0.169 | Positive |

---

## 3. Machine Learning Methodology

### 3.1 Dataset Building (Weak Supervision)

1. **Aspect Extraction**: Extract sentences containing aspect keywords
2. **Labeling Strategy**: Use review rating as weak supervision
   - Rating ≥ 4 → Positive
   - Rating = 3 → Neutral
   - Rating ≤ 2 → Negative

### 3.2 Feature Engineering

- **TF-IDF Vectorization**
  - Max features: 3,000-5,000
  - N-gram range: (1, 2) - unigrams and bigrams
  - Min document frequency: 2
  - Max document frequency: 95%

### 3.3 Models Evaluated

1. **Linear SVM** (LinearSVC)
   - Balanced class weights
   - Max iterations: 2,000

2. **Logistic Regression**
   - Balanced class weights
   - Max iterations: 1,000

3. **Random Forest**
   - 100 estimators
   - Max depth: 15
   - Balanced class weights

4. **Naive Bayes** (Multinomial)
   - Alpha: 0.1

### 3.4 Evaluation Protocol

- **Train/Test Split**: 80/20 with stratification
- **Cross-Validation**: 5-fold
- **Metrics**: Accuracy, Precision, Recall, F1-Score (weighted)

---

## 4. ML Results

### 4.1 Best Model Per Aspect

| Aspect | Best Model | Accuracy | Precision | Recall | F1 Score | CV F1 |
|--------|------------|----------|-----------|--------|----------|-------|
| Experience | Linear SVM | 78.12% | 77.89% | 78.12% | 77.67% | 73.67±1.15% |
| Scenery | Linear SVM | 77.45% | 76.98% | 77.45% | 76.80% | 70.17±6.76% |
| Facilities | Linear SVM | 74.89% | 74.56% | 74.89% | 74.12% | 69.98±1.27% |
| Accessibility | Linear SVM | 74.23% | 73.87% | 74.23% | 73.59% | 70.68±1.66% |
| Value | Linear SVM | 73.67% | 73.34% | 73.67% | 72.95% | 68.33±1.57% |
| Service | Logistic Regression | 73.12% | 72.78% | 73.12% | 72.47% | 72.21±1.92% |
| Safety | Naive Bayes | 71.89% | 71.45% | 71.89% | 71.14% | 70.13±1.44% |

### 4.2 Overall Performance Summary

| Metric | Value |
|--------|-------|
| Average Accuracy | 74.77% |
| Average F1 Score | 74.11% |
| Best Performing Aspect | Experience (77.67% F1) |
| Most Challenging Aspect | Safety (71.14% F1) |
| Total Training Samples | 47,538 |

### 4.3 Class Distribution Per Aspect

| Aspect | Positive | Neutral | Negative | Total |
|--------|----------|---------|----------|-------|
| Experience | 8,925 | 1,397 | 754 | 11,076 |
| Scenery | 7,363 | 1,076 | 614 | 9,053 |
| Accessibility | 7,172 | 1,183 | 631 | 8,986 |
| Value | 5,468 | 1,062 | 690 | 7,220 |
| Facilities | 4,204 | 762 | 447 | 5,413 |
| Safety | 2,939 | 456 | 302 | 3,697 |
| Service | 1,639 | 246 | 208 | 2,093 |

---

## 5. Hybrid Approach

### 5.1 ML + Lexicon Combination

The system uses a hybrid approach combining:
1. **ML Prediction**: Trained classifier output
2. **Lexicon Prediction**: Rule-based sentiment from word lists

### 5.2 Fusion Strategy

```
ML Weight: 0.7
Lexicon Weight: 0.3

If ML and Lexicon agree:
    Final = agreed sentiment
    Confidence = weighted_avg + 0.1 (boost)
Else:
    Final = higher weighted confidence wins
```

### 5.3 Benefits

- **Robustness**: Handles edge cases better
- **Interpretability**: Lexicon provides explainable keywords
- **Accuracy**: ML provides statistical learning

---

## 6. Key Findings

### 6.1 Model Performance Insights

1. **Linear SVM** performs best for most aspects (5/7)
2. **Logistic Regression** works well for Service aspect
3. **Naive Bayes** handles Safety aspect better (sparse data)

### 6.2 Aspect-Specific Insights

1. **Experience & Activities** has highest F1 (77.67%)
   - Most training data (11,076 samples)
   - Clear sentiment indicators

2. **Safety & Crowds** has lowest F1 (71.14%)
   - Least training data (3,697 samples)
   - Ambiguous sentiment (crowded can be positive/negative)

### 6.3 Data Quality Observations

- **Class Imbalance**: Positive reviews dominate (~75%)
- **Weak Supervision Limitation**: Rating may not reflect aspect-specific sentiment
- **Domain Specificity**: Tourism vocabulary improves accuracy

---

## 7. Practical Applications

### 7.1 Tourism App Integration

- Real-time review analysis
- Destination recommendations based on preferences
- Comparative analysis of locations

### 7.2 Business Intelligence

- Identify strengths/weaknesses per location
- Track sentiment trends over time
- Prioritize improvement areas

### 7.3 Research Contributions

1. First ABSA system for Sri Lanka tourism
2. Tourism-specific aspect taxonomy (7 aspects, 200+ keywords)
3. Comparison of ML vs Lexicon approaches
4. Practical API for app integration

---

## 8. Limitations & Future Work

### 8.1 Current Limitations

- Weak supervision may introduce label noise
- Class imbalance affects minority class detection
- English-only analysis

### 8.2 Future Improvements

- Fine-grained sentiment (very positive, slightly negative)
- Multi-language support (Sinhala, Tamil)
- Deep learning models (BERT, transformers)
- Aspect-opinion pair extraction

---

## 9. Reproducibility

### Run ML Training
```bash
python scripts/run_aspect_ml_training.py
```

### Export Results
```bash
python scripts/export_research_results.py
```

### Start Application
```bash
python app.py
```

---

## 10. References

1. Liu, B. (2012). Sentiment Analysis and Opinion Mining
2. Pontiki, M. et al. (2016). SemEval-2016 Task 5: ABSA
3. Schouten, K. & Frasincar, F. (2016). Survey on ABSA

---

*Generated: January 2026*
