"""
Run ML Training for Aspect-Based Sentiment Analysis.

Usage: python scripts/run_ml_training.py
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.aspect_ml_classifier import AspectMLPipeline


def main():
    print("=" * 60)
    print("ASPECT-BASED SENTIMENT ML TRAINING")
    print("=" * 60)
    
    # Use dataset from parent directory
    csv_path = '../dataset/Reviews.csv'
    
    if not os.path.exists(csv_path):
        print(f"❌ Dataset not found: {csv_path}")
        print("Please ensure the Reviews.csv file is in the dataset folder.")
        return
    
    pipeline = AspectMLPipeline()
    results = pipeline.run_full_pipeline(csv_path)
    
    # Save report
    with open('ASPECT_ML_RESULTS.md', 'w') as f:
        f.write("# Aspect-Based Sentiment ML Results\n\n")
        f.write("```\n")
        f.write(results['report'])
        f.write("\n```\n")
    
    print("\n✅ Results saved to ASPECT_ML_RESULTS.md")
    return results


if __name__ == '__main__':
    main()
