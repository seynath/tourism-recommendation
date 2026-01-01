#!/usr/bin/env python
"""
Sentiment Analysis Evaluation Script for Tourism Reviews.

This script runs comprehensive sentiment analysis evaluation:
1. Data loading and preprocessing
2. Traditional ML models (LR, SVM, RF, NB, GB)
3. Deep Learning models (LSTM, BiLSTM, CNN-LSTM)
4. Cross-validation
5. Feature importance analysis
6. Generate research report

Run with: python scripts/run_sentiment_analysis.py
"""

import sys
import time
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.sentiment_analysis import (
    SentimentAnalysisPipeline,
    TextPreprocessor,
    SentimentDataset
)


def print_header(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def main():
    """Run complete sentiment analysis evaluation."""
    print_header("DEEP LEARNING-BASED SENTIMENT ANALYSIS")
    print("Sri Lanka Tourism Reviews")
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    # Initialize pipeline
    # Set binary_only=True for binary classification (positive/negative)
    # Set binary_only=False for 3-class classification (positive/neutral/negative)
    pipeline = SentimentAnalysisPipeline(
        include_neutral=True,  # Include neutral reviews
        binary_only=False      # 3-class classification
    )
    
    # Load data
    print_header("1. DATA LOADING")
    stats = pipeline.load_data('dataset/Reviews.csv')
    
    # Run evaluation
    print_header("2. MODEL TRAINING AND EVALUATION")
    
    # Check if TensorFlow is available
    run_dl = False  # Disabled by default for faster execution
    print("Deep learning models disabled for faster execution.")
    print("To enable, set run_deep_learning=True in run_evaluation()")
    
    results = pipeline.run_evaluation(
        test_size=0.2,
        run_deep_learning=False,  # Skip DL for faster results
        dl_epochs=5
    )
    
    # Cross-validation
    print_header("3. CROSS-VALIDATION (5-Fold)")
    cv_results = pipeline.run_cross_validation(cv=5)
    
    # Feature analysis
    print_header("4. FEATURE IMPORTANCE ANALYSIS")
    features = pipeline.get_feature_analysis('Logistic Regression')
    
    if features:
        if 'positive_sentiment_words' in features:
            print("\nTop 15 Positive Sentiment Words:")
            for word, score in features['positive_sentiment_words'][:15]:
                print(f"  {word}: {score:.4f}")
        
        if 'negative_sentiment_words' in features:
            print("\nTop 15 Negative Sentiment Words:")
            for word, score in features['negative_sentiment_words'][:15]:
                print(f"  {word}: {score:.4f}")
    
    # Generate report
    print_header("5. FINAL REPORT")
    report = pipeline.generate_report()
    print(report)
    
    # Save report
    report_path = 'SENTIMENT_ANALYSIS_RESULTS.md'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")
    
    # Summary statistics
    print_header("6. SUMMARY")
    
    # Best model
    best_model = max(results.items(), key=lambda x: x[1].f1_score)
    print(f"\nBest Model: {best_model[0]}")
    print(f"  Accuracy: {best_model[1].accuracy:.4f} ({best_model[1].accuracy*100:.2f}%)")
    print(f"  F1 Score: {best_model[1].f1_score:.4f}")
    print(f"  Precision: {best_model[1].precision:.4f}")
    print(f"  Recall: {best_model[1].recall:.4f}")
    
    # Cross-validation summary
    print("\nCross-Validation Results (F1 Score):")
    for model_name, cv_result in sorted(cv_results.items(), key=lambda x: x[1]['mean_f1'], reverse=True):
        print(f"  {model_name}: {cv_result['mean_f1']:.4f} ± {cv_result['std_f1']:.4f}")
    
    # Per-class performance (best model)
    print(f"\nPer-Class Performance ({best_model[0]}):")
    for class_name in best_model[1].f1_per_class:
        f1 = best_model[1].f1_per_class[class_name]
        prec = best_model[1].precision_per_class[class_name]
        rec = best_model[1].recall_per_class[class_name]
        print(f"  {class_name}: F1={f1:.4f}, Precision={prec:.4f}, Recall={rec:.4f}")
    
    total_time = time.time() - start_time
    print(f"\nTotal evaluation time: {total_time:.1f} seconds")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return {
        'results': results,
        'cv_results': cv_results,
        'features': features,
        'stats': stats
    }


if __name__ == '__main__':
    results = main()
