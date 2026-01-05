#!/usr/bin/env python
"""
Aspect-Based Sentiment ML Training Script.

This script trains ML models for aspect-level sentiment classification:
1. Builds datasets for each aspect (scenery, safety, facilities, etc.)
2. Trains multiple ML models (LR, SVM, RF, NB)
3. Evaluates with train/test split and cross-validation
4. Compares performance across aspects

Run with: python scripts/run_aspect_ml_training.py
"""

import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.aspect_ml_classifier import AspectMLPipeline, TOURISM_ASPECTS


def main():
    """Run aspect ML training pipeline."""
    print("=" * 70)
    print(" ASPECT-BASED SENTIMENT - ML TRAINING")
    print(" Sri Lanka Tourism Reviews")
    print("=" * 70)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize pipeline
    pipeline = AspectMLPipeline()
    
    # Run full pipeline
    results = pipeline.run_full_pipeline('dataset/Reviews.csv')
    
    # Print detailed results per aspect
    print("\n" + "=" * 70)
    print(" DETAILED RESULTS PER ASPECT")
    print("=" * 70)
    
    for aspect, aspect_results in results['results'].items():
        if not aspect_results:
            continue
            
        print(f"\n🎯 {TOURISM_ASPECTS[aspect]['icon']} {TOURISM_ASPECTS[aspect]['display_name']}")
        print("-" * 50)
        print(f"{'Model':<20} | {'Accuracy':>10} | {'F1 Score':>10} | {'CV F1':>15}")
        print("-" * 50)
        
        for model_name, result in sorted(aspect_results.items(), key=lambda x: x[1].f1_score, reverse=True):
            print(f"{model_name:<20} | {result.accuracy:>10.4f} | {result.f1_score:>10.4f} | {result.cv_f1_mean:.4f}±{result.cv_f1_std:.4f}")
    
    # Save detailed report
    report_content = f"""# Aspect-Based Sentiment ML Training Results

## Overview

- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Dataset**: Sri Lanka Tourism Reviews
- **Total Reviews**: {len(pipeline.df)}
- **Aspects Analyzed**: {len(TOURISM_ASPECTS)}

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

{results['report']}

## Key Findings

"""
    
    # Add key findings
    best_aspects = []
    for aspect, aspect_results in results['results'].items():
        if aspect_results:
            best = max(aspect_results.values(), key=lambda x: x.f1_score)
            best_aspects.append((aspect, best.model_name, best.f1_score))
    
    best_aspects.sort(key=lambda x: x[2], reverse=True)
    
    report_content += "### Best Performing Aspects\n"
    for i, (aspect, model, f1) in enumerate(best_aspects[:3], 1):
        report_content += f"{i}. **{TOURISM_ASPECTS[aspect]['display_name']}**: {f1:.4f} F1 ({model})\n"
    
    report_content += "\n### Challenging Aspects\n"
    for aspect, model, f1 in best_aspects[-2:]:
        report_content += f"- **{TOURISM_ASPECTS[aspect]['display_name']}**: {f1:.4f} F1 (needs more data)\n"
    
    # Save report
    with open('ASPECT_ML_RESULTS.md', 'w') as f:
        f.write(report_content)
    
    print("\n" + "=" * 70)
    print("✅ Training complete!")
    print(f"📄 Results saved to: ASPECT_ML_RESULTS.md")
    print("=" * 70)
    
    return results


if __name__ == '__main__':
    results = main()
