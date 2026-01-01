======================================================================
SENTIMENT ANALYSIS RESEARCH REPORT
Deep Learning-Based Sentiment Analysis for Sri Lanka Tourism Reviews
======================================================================

1. DATASET SUMMARY
--------------------------------------------------
Total samples: 16151
Sentiment distribution: {'positive': 12841, 'neutral': 2165, 'negative': 1145}
Classes: ['negative', 'neutral', 'positive']

2. MODEL COMPARISON
--------------------------------------------------
Model                     |   Accuracy |  Precision |     Recall |         F1
---------------------------------------------------------------------------
Linear SVM                |     0.8158 |     0.8065 |     0.8158 |     0.8108
Logistic Regression       |     0.7775 |     0.8203 |     0.7775 |     0.7947
Random Forest             |     0.7855 |     0.7774 |     0.7855 |     0.7813
Naive Bayes               |     0.8258 |     0.7812 |     0.8258 |     0.7768
Gradient Boosting         |     0.8174 |     0.7693 |     0.8174 |     0.7679

3. BEST MODEL: Linear SVM
--------------------------------------------------
Accuracy: 0.8158
F1 Score: 0.8108

Classification Report:
              precision    recall  f1-score   support

    negative       0.60      0.54      0.57       229
     neutral       0.38      0.34      0.36       433
    positive       0.90      0.92      0.91      2569

    accuracy                           0.82      3231
   macro avg       0.63      0.60      0.61      3231
weighted avg       0.81      0.82      0.81      3231


4. FEATURE ANALYSIS (Logistic Regression)
--------------------------------------------------

Top Positive Sentiment Words:
  also went: 4.0374
  loved one: 2.4793
  grandeur: 2.4680
  went evening: 2.4193
  exact: 2.3580
  well one: 2.3426
  beach swimming: 2.2744
  enjoyed free: 2.1959
  wonderful garden: 2.0552
  very reasonably: 2.0282

Top Negative Sentiment Words:
  non: -4.5184
  not visit: -4.2392
  discover: -3.1132
  not understand: -3.0505
  discovering: -2.9242
  rubber tree: -2.5761
  expect: -2.3647
  beside: -2.2594
  disappointing: -2.2574
  site sri: -2.2317

======================================================================