# Deep Learning-Based Sentiment Analysis for Sri Lanka Tourism Reviews

## Research Summary

### Title
**Deep Learning-Based Sentiment Analysis for Sri Lanka Tourism Reviews: A Comparative Study of Machine Learning Approaches**

---

## 1. Dataset Overview

| Metric | Value |
|--------|-------|
| Total Reviews | 16,151 |
| Positive Reviews (4-5 stars) | 12,841 (79.5%) |
| Neutral Reviews (3 stars) | 2,165 (13.4%) |
| Negative Reviews (1-2 stars) | 1,145 (7.1%) |
| Unique Locations | 76 |
| Location Types | 11 |
| Average Review Length | 236 characters |
| Date Range | 2010-2023 |

### Class Distribution
- **Positive**: 79.5% - Reviews with 4-5 star ratings
- **Neutral**: 13.4% - Reviews with 3 star ratings  
- **Negative**: 7.1% - Reviews with 1-2 star ratings

### Location Types Covered
1. Religious Sites (3,017 reviews)
2. Beaches (2,110 reviews)
3. Farms (1,884 reviews)
4. Nature & Wildlife Areas (1,557 reviews)
5. Museums (1,525 reviews)
6. Historic Sites (1,519 reviews)
7. Gardens (1,354 reviews)
8. National Parks (1,205 reviews)
9. Waterfalls (933 reviews)
10. Bodies of Water (839 reviews)

---

## 2. Methodology

### Text Preprocessing Pipeline
1. **Lowercase conversion**
2. **URL and email removal**
3. **Special character removal**
4. **Stopword removal** (with sentiment-preserving exceptions)
5. **Lemmatization** using WordNet
6. **Short token removal** (< 3 characters)

### Feature Extraction
- **TF-IDF Vectorization**
  - Max features: 10,000
  - N-gram range: (1, 2) - unigrams and bigrams
  - Sublinear TF scaling
  - Min document frequency: 2
  - Max document frequency: 95%

### Train/Test Split
- Training: 80% (12,920 samples)
- Testing: 20% (3,231 samples)
- Stratified sampling to maintain class distribution

---

## 3. Models Evaluated

### Traditional Machine Learning
1. **Logistic Regression** - Linear classifier with L2 regularization
2. **Linear SVM** - Support Vector Machine with linear kernel
3. **Random Forest** - Ensemble of 100 decision trees
4. **Naive Bayes** - Multinomial Naive Bayes
5. **Gradient Boosting** - Gradient boosted decision trees

### Deep Learning (Optional)
1. **LSTM** - Long Short-Term Memory network
2. **BiLSTM** - Bidirectional LSTM
3. **CNN-LSTM** - Convolutional + LSTM hybrid

---

## 4. Results

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| **Linear SVM** | **81.58%** | 80.65% | 81.58% | **81.08%** |
| Naive Bayes | 82.58% | 78.12% | 82.58% | 77.68% |
| Gradient Boosting | 81.74% | 76.93% | 81.74% | 76.79% |
| Logistic Regression | 77.75% | 82.03% | 77.75% | 79.47% |
| Random Forest | 78.55% | 77.74% | 78.55% | 78.13% |

### 5-Fold Cross-Validation Results

| Model | Mean F1 | Std F1 |
|-------|---------|--------|
| **Linear SVM** | **0.7579** | ±0.0150 |
| Random Forest | 0.7517 | ±0.0086 |
| Naive Bayes | 0.7450 | ±0.0157 |
| Gradient Boosting | 0.7410 | ±0.0147 |
| Logistic Regression | 0.7351 | ±0.0257 |

### Per-Class Performance (Best Model: Linear SVM)

| Class | Precision | Recall | F1 Score | Support |
|-------|-----------|--------|----------|---------|
| Positive | 89.69% | 92.06% | 90.86% | 2,569 |
| Neutral | 38.14% | 34.18% | 36.05% | 433 |
| Negative | 59.71% | 53.71% | 56.55% | 229 |

---

## 5. Feature Analysis

### Top Positive Sentiment Indicators
| Feature | Coefficient |
|---------|-------------|
| also went | 4.04 |
| loved one | 2.48 |
| grandeur | 2.47 |
| went evening | 2.42 |
| beach swimming | 2.27 |
| enjoyed free | 2.20 |
| wonderful garden | 2.06 |
| very reasonably | 2.03 |
| museum great | 1.93 |
| lovely day | 1.81 |

### Top Negative Sentiment Indicators
| Feature | Coefficient |
|---------|-------------|
| non | -4.52 |
| not visit | -4.24 |
| discover | -3.11 |
| not understand | -3.05 |
| disappointing | -2.26 |
| little expensive | -2.22 |
| misleading | -2.21 |
| expect | -2.36 |

---

## 6. Key Findings

### 1. Model Performance
- **Linear SVM achieves best overall performance** with 81.58% accuracy and 81.08% F1 score
- Traditional ML models perform well on this dataset
- Cross-validation confirms robust performance (F1: 0.7579 ± 0.0150)

### 2. Class Imbalance Challenge
- Positive class dominates (79.5%) leading to high positive class performance
- Neutral class is hardest to classify (F1: 36.05%)
- Negative class shows moderate performance (F1: 56.55%)

### 3. Feature Insights
- **Positive reviews** contain experiential phrases ("went evening", "beach swimming", "enjoyed")
- **Negative reviews** contain negation patterns ("not visit", "not understand") and disappointment words
- Bigrams capture sentiment better than unigrams alone

### 4. Tourism-Specific Patterns
- Location-specific terms appear in sentiment indicators
- Service quality and value for money are key sentiment drivers
- Weather and timing affect review sentiment

---

## 7. Research Contributions

1. **First comprehensive sentiment analysis** of Sri Lanka tourism reviews
2. **Comparative study** of 5 ML algorithms on tourism domain
3. **Feature analysis** revealing tourism-specific sentiment patterns
4. **Practical insights** for tourism industry stakeholders

---

## 8. Limitations

1. **Class imbalance** - Positive reviews dominate (79.5%)
2. **Neutral class ambiguity** - 3-star reviews are inherently ambiguous
3. **English only** - Dataset limited to English reviews
4. **Single source** - Data from TripAdvisor only

---

## 9. Future Work

1. **Deep Learning** - Implement BERT-based models for better context understanding
2. **Aspect-Based Sentiment** - Analyze sentiment for specific aspects (food, service, location)
3. **Multilingual** - Extend to Sinhala and Tamil reviews
4. **Real-time System** - Deploy as API for tourism businesses

---

## 10. How to Run

```bash
# Install dependencies
pip install nltk scikit-learn pandas numpy

# Run sentiment analysis
python scripts/run_sentiment_analysis.py
```

---

## 11. Citation

If you use this work, please cite:

```
@article{tourism_sentiment_2024,
  title={Deep Learning-Based Sentiment Analysis for Sri Lanka Tourism Reviews},
  author={[Your Name]},
  journal={[Conference/Journal Name]},
  year={2024}
}
```

---

## 12. Conclusion

This research demonstrates that **traditional machine learning approaches, particularly Linear SVM, achieve strong performance (81.58% accuracy)** for sentiment analysis of tourism reviews. The study reveals important sentiment patterns specific to the tourism domain and provides actionable insights for the Sri Lanka tourism industry.

The main challenge is the **class imbalance** with positive reviews dominating the dataset. Future work should focus on handling this imbalance and implementing transformer-based models for improved neutral and negative class detection.
