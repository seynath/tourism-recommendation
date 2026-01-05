"""
API endpoints for Aspect-Based Sentiment Analysis.

Standalone API for sentiment research (without recommender system).
"""

from typing import Dict, List, Optional
import threading
import pandas as pd

from src.aspect_sentiment import (
    ABSAPipeline,
    LocationInsight,
    SmartRecommendation,
    TOURISM_ASPECTS
)
from src.aspect_ml_service import AspectMLService

_init_lock = threading.Lock()
_ml_lock = threading.Lock()


class ABSAService:
    """Service class for ABSA functionality."""
    
    _instance = None
    _pipeline = None
    _loaded = False
    _ml_service = None
    _ml_trained = False
    _df = None
    
    @classmethod
    def get_instance(cls) -> 'ABSAService':
        if cls._instance is None:
            cls._instance = ABSAService()
        return cls._instance
    
    def __init__(self):
        pass
    
    def initialize(self, csv_path: str = 'dataset/Reviews.csv'):
        """Initialize the ABSA pipeline with data."""
        with _init_lock:
            if not ABSAService._loaded:
                ABSAService._pipeline = ABSAPipeline()
                ABSAService._pipeline.load_and_analyze(csv_path)
                ABSAService._df = ABSAService._pipeline.df
                ABSAService._loaded = True
                
                ABSAService._ml_service = AspectMLService.get_instance()
                
                if not ABSAService._ml_service.load_models():
                    print("🔄 Training ML models...")
                    ABSAService._ml_service.train_models(ABSAService._df)
                
                ABSAService._ml_trained = ABSAService._ml_service.is_trained()
    
    def ensure_loaded(self):
        if not ABSAService._loaded:
            print("🔄 Initializing ABSA service with data...")
            self.initialize()
            print("✅ ABSA service initialized")
    
    def get_all_locations(self) -> List[Dict]:
        self.ensure_loaded()
        if ABSAService._pipeline is None or ABSAService._pipeline.insights is None:
            return []
        
        locations = []
        for name, insight in ABSAService._pipeline.insights.items():
            locations.append({
                'name': name,
                'type': insight.location_type,
                'rating': insight.recommendation_score,
                'total_reviews': insight.total_reviews,
                'overall_sentiment': round(insight.overall_sentiment, 3)
            })
        locations.sort(key=lambda x: x['rating'], reverse=True)
        return locations
    
    def get_location_insight(self, location_name: str) -> Optional[Dict]:
        self.ensure_loaded()
        insight = ABSAService._pipeline.get_location_insight(location_name)
        if insight is None:
            return None
        return self._insight_to_dict(insight)
    
    def get_location_aspects(self, location_name: str) -> Optional[Dict]:
        self.ensure_loaded()
        insight = ABSAService._pipeline.get_location_insight(location_name)
        if insight is None:
            return None
        
        aspects = []
        for aspect_key, score in insight.aspect_scores.items():
            aspects.append({
                'aspect': aspect_key,
                'display_name': TOURISM_ASPECTS[aspect_key]['display_name'],
                'icon': TOURISM_ASPECTS[aspect_key]['icon'],
                'score': round(score, 3),
                'score_normalized': round((score + 1) / 2, 3),
                'mentions': insight.aspect_counts.get(aspect_key, 0),
                'sentiment': 'positive' if score > 0.2 else ('negative' if score < -0.2 else 'neutral')
            })
        aspects.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'location': location_name,
            'aspects': aspects,
            'strengths': insight.strengths,
            'weaknesses': insight.weaknesses
        }
    
    def get_recommendations(self, preferred_aspects: List[str], avoid_aspects: List[str] = None,
                           location_type: str = None, min_reviews: int = 5, limit: int = 10) -> List[Dict]:
        self.ensure_loaded()
        recs = ABSAService._pipeline.recommend(
            preferred_aspects=preferred_aspects,
            avoid_aspects=avoid_aspects,
            location_type=location_type,
            top_n=limit
        )
        
        results = []
        for rec in recs:
            insight = ABSAService._pipeline.get_location_insight(rec.location_name)
            results.append({
                'location': rec.location_name,
                'type': insight.location_type if insight else 'Unknown',
                'match_score': round(rec.match_score, 3),
                'rating': insight.recommendation_score if insight else 0,
                'matching_aspects': rec.matching_aspects,
                'highlights': rec.highlights,
                'warnings': rec.warnings,
                'total_reviews': insight.total_reviews if insight else 0
            })
        return results
    
    def compare_locations(self, locations: List[str]) -> Dict:
        self.ensure_loaded()
        comparison_df = ABSAService._pipeline.compare(locations)
        
        comparison = {'locations': [], 'aspects': list(TOURISM_ASPECTS.keys())}
        for _, row in comparison_df.iterrows():
            loc_data = {
                'name': row['Location'],
                'type': row['Type'],
                'overall': row['Overall'],
                'reviews': row['Reviews'],
                'aspect_scores': {}
            }
            for aspect in TOURISM_ASPECTS:
                display_name = TOURISM_ASPECTS[aspect]['display_name']
                if display_name in row:
                    loc_data['aspect_scores'][aspect] = row[display_name]
            comparison['locations'].append(loc_data)
        return comparison
    
    def get_aspect_statistics(self) -> List[Dict]:
        self.ensure_loaded()
        stats_df = ABSAService._pipeline.get_aspect_statistics()
        
        stats = []
        for _, row in stats_df.iterrows():
            aspect_key = None
            for key, config in TOURISM_ASPECTS.items():
                if config['display_name'] == row['Aspect']:
                    aspect_key = key
                    break
            stats.append({
                'aspect': aspect_key,
                'display_name': row['Aspect'],
                'icon': row['Icon'],
                'total_mentions': int(row['Total Mentions']),
                'avg_sentiment': round(row['Avg Sentiment'], 3),
                'sentiment_label': row['Sentiment']
            })
        return stats
    
    def get_location_types(self) -> List[Dict]:
        self.ensure_loaded()
        type_counts = {}
        for insight in ABSAService._pipeline.insights.values():
            loc_type = insight.location_type
            type_counts[loc_type] = type_counts.get(loc_type, 0) + 1
        return [{'type': t, 'count': c} for t, c in sorted(type_counts.items(), key=lambda x: x[1], reverse=True)]
    
    def get_available_aspects(self) -> List[Dict]:
        return [{'key': key, 'display_name': config['display_name'], 'icon': config['icon']}
                for key, config in TOURISM_ASPECTS.items()]
    
    def analyze_review(self, text: str) -> Dict:
        self.ensure_loaded()
        aspects = ABSAService._pipeline.analyzer.analyze_review(text)
        return {
            'text': text,
            'aspects_found': [{
                'aspect': asp.aspect,
                'display_name': TOURISM_ASPECTS[asp.aspect]['display_name'],
                'icon': TOURISM_ASPECTS[asp.aspect]['icon'],
                'sentiment': asp.sentiment,
                'confidence': round(asp.confidence, 3),
                'keywords': asp.keywords_found,
                'snippet': asp.text_snippet
            } for asp in aspects],
            'total_aspects': len(aspects)
        }
    
    def analyze_review_ml(self, text: str) -> Dict:
        self.ensure_loaded()
        if not ABSAService._ml_trained or ABSAService._ml_service is None:
            return {'error': 'ML models not trained', 'aspects_found': []}
        results = ABSAService._ml_service.analyze_review_ml(text)
        return {'text': text, 'aspects_found': results, 'total_aspects': len(results), 'ml_enabled': True}
    
    def get_ml_evaluation_results(self) -> Dict:
        self.ensure_loaded()
        if not ABSAService._ml_trained or ABSAService._ml_service is None:
            return {'error': 'ML models not trained', 'results': {}, 'ml_enabled': False}
        
        try:
            results = ABSAService._ml_service.get_evaluation_results()
            
            if not results:
                return {'error': 'No evaluation results available', 'aspects': [], 'ml_enabled': False}
            
            formatted = []
            for aspect, metrics in results.items():
                if aspect not in TOURISM_ASPECTS:
                    continue
                
                # Convert class_distribution values to regular Python ints (numpy int64 not JSON serializable)
                class_dist = metrics.get('class_distribution', {})
                class_dist_clean = {k: int(v) for k, v in class_dist.items()}
                    
                formatted.append({
                    'aspect': aspect,
                    'display_name': TOURISM_ASPECTS[aspect]['display_name'],
                    'icon': TOURISM_ASPECTS[aspect]['icon'],
                    'accuracy': round(float(metrics.get('accuracy', 0)), 4),
                    'precision': round(float(metrics.get('precision', 0)), 4),
                    'recall': round(float(metrics.get('recall', 0)), 4),
                    'f1_score': round(float(metrics.get('f1_score', 0)), 4),
                    'cv_f1_mean': round(float(metrics.get('cv_f1_mean', 0)), 4),
                    'cv_f1_std': round(float(metrics.get('cv_f1_std', 0)), 4),
                    'train_samples': int(metrics.get('train_samples', 0)),
                    'test_samples': int(metrics.get('test_samples', 0)),
                    'total_samples': int(metrics.get('total_samples', 0)),
                    'class_distribution': class_dist_clean
                })
            
            formatted.sort(key=lambda x: x['f1_score'], reverse=True)
            
            avg_accuracy = sum(r['accuracy'] for r in formatted) / len(formatted) if formatted else 0
            avg_f1 = sum(r['f1_score'] for r in formatted) / len(formatted) if formatted else 0
            total_samples = sum(r['total_samples'] for r in formatted)
            
            return {
                'aspects': formatted,
                'summary': {
                    'total_aspects': len(formatted),
                    'avg_accuracy': round(avg_accuracy, 4),
                    'avg_f1_score': round(avg_f1, 4),
                    'total_samples': total_samples,
                    'best_aspect': formatted[0]['aspect'] if formatted else None,
                    'best_f1': formatted[0]['f1_score'] if formatted else 0
                },
                'ml_enabled': True
            }
        except Exception as e:
            import traceback
            return {'error': f'Error getting evaluation results: {str(e)}', 'traceback': traceback.format_exc(), 'ml_enabled': False}
    
    def get_research_export(self) -> Dict:
        self.ensure_loaded()
        
        try:
            insights_data = []
            for name, insight in ABSAService._pipeline.insights.items():
                insight_dict = {
                    'location': name,
                    'type': insight.location_type,
                    'overall_sentiment': insight.overall_sentiment,
                    'recommendation_score': insight.recommendation_score,
                    'total_reviews': insight.total_reviews,
                }
                # Add aspect scores and counts
                for asp in TOURISM_ASPECTS:
                    insight_dict[f'{asp}_score'] = insight.aspect_scores.get(asp, 0)
                    insight_dict[f'{asp}_count'] = insight.aspect_counts.get(asp, 0)
                insights_data.append(insight_dict)
            
            aspect_stats = self.get_aspect_statistics()
            ml_eval = self.get_ml_evaluation_results()
            
            df = ABSAService._df
            dataset_stats = {
                'total_reviews': len(df) if df is not None else 0,
                'total_locations': len(ABSAService._pipeline.insights) if ABSAService._pipeline else 0,
                'rating_distribution': df['Rating'].value_counts().to_dict() if df is not None and 'Rating' in df.columns else {},
                'avg_rating': float(df['Rating'].mean()) if df is not None and 'Rating' in df.columns else 0,
                'reviews_per_location': len(df) / len(ABSAService._pipeline.insights) if df is not None and ABSAService._pipeline and ABSAService._pipeline.insights else 0
            }
            
            return {
                'dataset_statistics': dataset_stats,
                'aspect_statistics': aspect_stats,
                'ml_evaluation': ml_eval,
                'location_insights': insights_data,
                'aspects_definition': [{'key': key, 'display_name': config['display_name'], 'icon': config['icon'],
                                       'keywords_count': len(config['keywords'])} for key, config in TOURISM_ASPECTS.items()]
            }
        except Exception as e:
            import traceback
            return {'error': f'Error exporting research data: {str(e)}', 'traceback': traceback.format_exc()}
    
    def _insight_to_dict(self, insight: LocationInsight) -> Dict:
        return {
            'location_name': insight.location_name,
            'location_type': insight.location_type,
            'overall_sentiment': round(insight.overall_sentiment, 3),
            'recommendation_score': insight.recommendation_score,
            'total_reviews': insight.total_reviews,
            'strengths': insight.strengths,
            'weaknesses': insight.weaknesses,
            'aspect_scores': {k: round(v, 3) for k, v in insight.aspect_scores.items()},
            'aspect_counts': insight.aspect_counts
        }


def register_absa_routes(app):
    """Register ABSA API routes with Flask app."""
    from flask import jsonify, request
    
    service = ABSAService.get_instance()
    
    @app.route('/api/absa/locations', methods=['GET'])
    def api_get_locations():
        try:
            return jsonify({'success': True, 'data': service.get_all_locations()})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/absa/locations/<location_name>', methods=['GET'])
    def api_get_location_insight(location_name):
        try:
            insight = service.get_location_insight(location_name)
            if insight is None:
                return jsonify({'success': False, 'error': 'Location not found'}), 404
            return jsonify({'success': True, 'data': insight})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/absa/locations/<location_name>/aspects', methods=['GET'])
    def api_get_location_aspects(location_name):
        try:
            aspects = service.get_location_aspects(location_name)
            if aspects is None:
                return jsonify({'success': False, 'error': 'Location not found'}), 404
            return jsonify({'success': True, 'data': aspects})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/absa/recommend', methods=['POST'])
    def api_get_recommendations():
        try:
            data = request.get_json()
            recs = service.get_recommendations(
                preferred_aspects=data.get('preferred_aspects', []),
                avoid_aspects=data.get('avoid_aspects', []),
                location_type=data.get('location_type'),
                limit=data.get('limit', 10)
            )
            return jsonify({'success': True, 'data': recs})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/absa/compare', methods=['POST'])
    def api_compare_locations():
        try:
            data = request.get_json()
            locations = data.get('locations', [])
            if len(locations) < 2:
                return jsonify({'success': False, 'error': 'Need at least 2 locations'}), 400
            return jsonify({'success': True, 'data': service.compare_locations(locations)})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/absa/aspects', methods=['GET'])
    def api_get_aspects():
        try:
            return jsonify({'success': True, 'data': service.get_available_aspects()})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/absa/aspects/stats', methods=['GET'])
    def api_get_aspect_stats():
        try:
            return jsonify({'success': True, 'data': service.get_aspect_statistics()})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/absa/types', methods=['GET'])
    def api_get_location_types():
        try:
            return jsonify({'success': True, 'data': service.get_location_types()})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/absa/analyze', methods=['POST'])
    def api_analyze_review():
        try:
            data = request.get_json()
            text = data.get('text', '')
            if not text:
                return jsonify({'success': False, 'error': 'No text provided'}), 400
            return jsonify({'success': True, 'data': service.analyze_review(text)})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/absa/analyze/ml', methods=['POST'])
    def api_analyze_review_ml():
        try:
            data = request.get_json()
            text = data.get('text', '')
            if not text:
                return jsonify({'success': False, 'error': 'No text provided'}), 400
            return jsonify({'success': True, 'data': service.analyze_review_ml(text)})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/absa/ml/evaluation', methods=['GET'])
    def api_get_ml_evaluation():
        try:
            return jsonify({'success': True, 'data': service.get_ml_evaluation_results()})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/absa/export/research', methods=['GET'])
    def api_export_research():
        try:
            return jsonify({'success': True, 'data': service.get_research_export()})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    print("✅ ABSA API routes registered")
