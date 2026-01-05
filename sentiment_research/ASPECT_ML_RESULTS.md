# Aspect-Based Sentiment ML Training Results

## Overview

- **Date**: 2026-01-04 23:56:26
- **Dataset**: Sri Lanka Tourism Reviews
- **Total Reviews**: 16156
- **Aspects Analyzed**: 7

## Methodology

### Dataset Building
- Extract sentences containing aspect keywords from reviews
- Label using weak supervision (review rating → sentiment)
- Positive: 4-5 stars, Neutral: 3 stars, Negative: 1-2 stars

### Models Trained
1. Logistic Regression (with balanced class weights)
2. Linear SVM (with balanced class weights)
3. Random Forest (100 trees, balanced)
4. Naive Bayes (Multinomial)

### Features
- TF-IDF vectorization
- Max 5,000 features
- Unigrams and bigrams
- Min document frequency: 2

### Evaluation
- Train/Test split: 80/20
- 5-Fold Cross-validation
- Metrics: Accuracy, Precision, Recall, F1-Score

## Results Summary


======================================================================
ASPECT-BASED SENTIMENT ML RESULTS SUMMARY
======================================================================

📊 BEST MODEL PER ASPECT:
----------------------------------------------------------------------
Aspect                    | Best Model           |   F1 Score |        CV F1
----------------------------------------------------------------------
scenery                   | Linear SVM           |     0.7680 | 0.7017±0.0676
accessibility             | Linear SVM           |     0.7359 | 0.7068±0.0166
facilities                | Linear SVM           |     0.7412 | 0.6998±0.0127
safety                    | Naive Bayes          |     0.7114 | 0.7013±0.0144
value                     | Linear SVM           |     0.7295 | 0.6833±0.0157
experience                | Linear SVM           |     0.7767 | 0.7367±0.0115
service                   | Logistic Regression  |     0.7247 | 0.7221±0.0192

📈 DATASET STATISTICS:
----------------------------------------------------------------------
Aspect                    |    Samples |   Positive |    Neutral |   Negative
----------------------------------------------------------------------
scenery                   |       9053 |       7363 |       1076 |        614
accessibility             |       8986 |       7172 |       1183 |        631
facilities                |       5413 |       4204 |        762 |        447
safety                    |       3697 |       2939 |        456 |        302
value                     |       7220 |       5468 |       1062 |        690
experience                |      11076 |       8925 |       1397 |        754
service                   |       2093 |       1639 |        246 |        208

📋 OVERALL STATISTICS:
  Total aspect samples: 47538
  Average best F1 score: 0.7411
  Aspects trained: 7

## Key Findings

### Best Performing Aspects
1. **Experience & Activities**: 0.7767 F1 (Linear SVM)
2. **Scenery & Views**: 0.7680 F1 (Linear SVM)
3. **Facilities**: 0.7412 F1 (Linear SVM)

### Challenging Aspects
- **Service & Staff**: 0.7247 F1 (needs more data)
- **Safety & Crowds**: 0.7114 F1 (needs more data)
