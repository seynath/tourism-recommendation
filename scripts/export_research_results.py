"""
Export Research Results for Paper.

Generates comprehensive CSV and JSON files for research documentation.
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.aspect_sentiment import ABSAPipeline, TOURISM_ASPECTS
from src.aspect_ml_service import AspectMLService


def export_research_results():
    """Export all research results to files."""
    print("=" * 60)
    print("EXPORTING RESEARCH RESULTS")
    print("=" * 60)
    
    # Create output directory
    output_dir = 'research_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize pipeline
    print("\n📊 Loading ABSA pipeline...")
    pipeline = ABSAPipeline()
    pipeline.load_and_analyze('dataset/Reviews.csv')
    
    # Initialize ML service
    print("\n🤖 Loading ML models...")
    ml_service = AspectMLService.get_instance()
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
    print(f"  ✅ Saved {len(location_df)} locations to location_insights.csv")
    
    # 2. Export Aspect Statistics
    print("\n🎯 Exporting aspect statistics...")
    aspect_stats = pipeline.get_aspect_statistics()
    aspect_stats.to_csv(f'{output_dir}/aspect_statistics.csv', index=False)
    print(f"  ✅ Saved aspect statistics to aspect_statistics.csv")
    
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
            'total_samples': metrics['total_samples'],
            'positive_samples': metrics['class_distribution'].get('positive', 0),
            'neutral_samples': metrics['class_distribution'].get('neutral', 0),
            'negative_samples': metrics['class_distribution'].get('negative', 0)
        })
    
    ml_df = pd.DataFrame(ml_data)
    ml_df = ml_df.sort_values('f1_score', ascending=False)
    ml_df.to_csv(f'{output_dir}/ml_evaluation_results.csv', index=False)
    print(f"  ✅ Saved ML results for {len(ml_df)} aspects to ml_evaluation_results.csv")
    
    # 4. Export Dataset Statistics
    print("\n📈 Exporting dataset statistics...")
    df = pipeline.df
    dataset_stats = {
        'total_reviews': len(df),
        'total_locations': len(pipeline.insights),
        'total_aspects': len(TOURISM_ASPECTS),
        'avg_rating': round(df['Rating'].mean(), 2) if 'Rating' in df.columns else None,
        'rating_distribution': df['Rating'].value_counts().to_dict() if 'Rating' in df.columns else {},
        'reviews_per_location_avg': round(len(df) / len(pipeline.insights), 1),
        'ml_total_samples': sum(m['total_samples'] for m in ml_results.values()),
        'ml_avg_f1_score': round(sum(m['f1_score'] for m in ml_results.values()) / len(ml_results), 4),
        'ml_avg_accuracy': round(sum(m['accuracy'] for m in ml_results.values()) / len(ml_results), 4),
        'export_date': datetime.now().isoformat()
    }
    
    with open(f'{output_dir}/dataset_statistics.json', 'w') as f:
        json.dump(dataset_stats, f, indent=2)
    print(f"  ✅ Saved dataset statistics to dataset_statistics.json")
    
    # 5. Export Complete Research Data (JSON)
    print("\n📚 Exporting complete research data...")
    complete_data = {
        'metadata': {
            'title': 'Aspect-Based Sentiment Analysis for Sri Lanka Tourism',
            'export_date': datetime.now().isoformat(),
            'dataset': 'Sri Lanka Tourism Reviews'
        },
        'dataset_statistics': dataset_stats,
        'aspect_definitions': [
            {
                'key': key,
                'display_name': config['display_name'],
                'icon': config['icon'],
                'keywords_count': len(config['keywords']),
                'sample_keywords': config['keywords'][:10]
            }
            for key, config in TOURISM_ASPECTS.items()
        ],
        'ml_evaluation': {
            'summary': {
                'total_aspects': len(ml_results),
                'avg_accuracy': dataset_stats['ml_avg_accuracy'],
                'avg_f1_score': dataset_stats['ml_avg_f1_score'],
                'total_samples': dataset_stats['ml_total_samples'],
                'best_aspect': ml_df.iloc[0]['aspect'] if len(ml_df) > 0 else None,
                'best_f1': ml_df.iloc[0]['f1_score'] if len(ml_df) > 0 else None
            },
            'aspects': ml_data
        },
        'aspect_statistics': aspect_stats.to_dict('records'),
        'top_locations': location_df.nlargest(10, 'recommendation_score').to_dict('records')
    }
    
    with open(f'{output_dir}/complete_research_data.json', 'w') as f:
        json.dump(complete_data, f, indent=2)
    print(f"  ✅ Saved complete research data to complete_research_data.json")
    
    # Print Summary
    print("\n" + "=" * 60)
    print("RESEARCH EXPORT SUMMARY")
    print("=" * 60)
    print(f"\n📁 Output Directory: {output_dir}/")
    print(f"\n📊 Files Generated:")
    print(f"   1. location_insights.csv ({len(location_df)} locations)")
    print(f"   2. aspect_statistics.csv (7 aspects)")
    print(f"   3. ml_evaluation_results.csv ({len(ml_df)} models)")
    print(f"   4. dataset_statistics.json")
    print(f"   5. complete_research_data.json")
    
    print(f"\n📈 Key Metrics:")
    print(f"   Total Reviews: {dataset_stats['total_reviews']:,}")
    print(f"   Total Locations: {dataset_stats['total_locations']}")
    print(f"   ML Training Samples: {dataset_stats['ml_total_samples']:,}")
    print(f"   ML Avg F1 Score: {dataset_stats['ml_avg_f1_score']:.2%}")
    print(f"   ML Avg Accuracy: {dataset_stats['ml_avg_accuracy']:.2%}")
    
    print("\n✅ Research export complete!")
    return complete_data


if __name__ == '__main__':
    export_research_results()
