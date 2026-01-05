"""
Export Research Results for Paper.

Usage: python scripts/export_results.py
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.aspect_sentiment import ABSAPipeline, TOURISM_ASPECTS
from src.aspect_ml_service import AspectMLService


def export_research_results():
    print("=" * 60)
    print("EXPORTING RESEARCH RESULTS")
    print("=" * 60)
    
    output_dir = 'research_output'
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = '../dataset/Reviews.csv'
    
    print("\n📊 Loading ABSA pipeline...")
    pipeline = ABSAPipeline()
    pipeline.load_and_analyze(csv_path)
    
    print("\n🤖 Loading ML models...")
    ml_service = AspectMLService.get_instance()
    ml_service.MODEL_PATH = 'models/aspect_ml/'
    if not ml_service.load_models():
        print("Training ML models...")
        ml_service.train_models(pipeline.df)
    
    # 1. Export Location Insights
    print("\n📍 Exporting location insights...")
    location_data = []
    for name, insight in pipeline.insights.items():
        row = {
            'location': name,
            'type': insight.location_type,
            'overall_sentiment': round(insight.overall_sentiment, 4),
            'recommendation_score': insight.recommendation_score,
            'total_reviews': insight.total_reviews,
            'strengths': '; '.join(insight.strengths),
            'weaknesses': '; '.join(insight.weaknesses)
        }
        for aspect in TOURISM_ASPECTS:
            row[f'{aspect}_score'] = round(insight.aspect_scores.get(aspect, 0), 4)
            row[f'{aspect}_count'] = insight.aspect_counts.get(aspect, 0)
        location_data.append(row)
    
    location_df = pd.DataFrame(location_data)
    location_df.to_csv(f'{output_dir}/location_insights.csv', index=False)
    print(f"  ✅ Saved {len(location_df)} locations")
    
    # 2. Export Aspect Statistics
    print("\n🎯 Exporting aspect statistics...")
    aspect_stats = pipeline.get_aspect_statistics()
    aspect_stats.to_csv(f'{output_dir}/aspect_statistics.csv', index=False)
    
    # 3. Export ML Evaluation Results
    print("\n🤖 Exporting ML evaluation results...")
    ml_results = ml_service.get_evaluation_results()
    ml_data = []
    for aspect, metrics in ml_results.items():
        ml_data.append({
            'aspect': aspect,
            'display_name': TOURISM_ASPECTS[aspect]['display_name'],
            'accuracy': round(metrics['accuracy'], 4),
            'precision': round(metrics['precision'], 4),
            'recall': round(metrics['recall'], 4),
            'f1_score': round(metrics['f1_score'], 4),
            'cv_f1_mean': round(metrics['cv_f1_mean'], 4),
            'cv_f1_std': round(metrics['cv_f1_std'], 4),
            'train_samples': metrics['train_samples'],
            'test_samples': metrics['test_samples'],
            'total_samples': metrics['total_samples']
        })
    
    ml_df = pd.DataFrame(ml_data).sort_values('f1_score', ascending=False)
    ml_df.to_csv(f'{output_dir}/ml_evaluation_results.csv', index=False)
    
    # 4. Export Complete JSON
    print("\n📚 Exporting complete research data...")
    df = pipeline.df
    complete_data = {
        'metadata': {
            'title': 'Aspect-Based Sentiment Analysis for Sri Lanka Tourism',
            'export_date': datetime.now().isoformat()
        },
        'dataset_statistics': {
            'total_reviews': len(df),
            'total_locations': len(pipeline.insights),
            'avg_rating': round(df['Rating'].mean(), 2) if 'Rating' in df.columns else None,
            'ml_total_samples': sum(m['total_samples'] for m in ml_results.values()),
            'ml_avg_f1_score': round(sum(m['f1_score'] for m in ml_results.values()) / len(ml_results), 4)
        },
        'ml_evaluation': ml_data,
        'aspect_statistics': aspect_stats.to_dict('records')
    }
    
    with open(f'{output_dir}/complete_research_data.json', 'w') as f:
        json.dump(complete_data, f, indent=2)
    
    print("\n" + "=" * 60)
    print("EXPORT COMPLETE")
    print("=" * 60)
    print(f"\n📁 Output: {output_dir}/")
    print(f"   - location_insights.csv")
    print(f"   - aspect_statistics.csv")
    print(f"   - ml_evaluation_results.csv")
    print(f"   - complete_research_data.json")
    
    return complete_data


if __name__ == '__main__':
    export_research_results()
