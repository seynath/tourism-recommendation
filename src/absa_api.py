"""
API endpoints for Aspect-Based Sentiment Analysis.

Provides REST API for:
1. Location insights with aspect scores
2. Smart recommendations based on preferences
3. Location comparisons
4. Aspect statistics

For integration with tourism app.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import json
import threading

from src.aspect_sentiment import (
    ABSAPipeline,
    LocationInsight,
    SmartRecommendation,
    TOURISM_ASPECTS
)

# Global lock for initialization
_init_lock = threading.Lock()


class ABSAService:
    """
    Service class for ABSA functionality.
    
    Provides methods for API integration.
    Uses class-level state for true singleton behavior.
    """
    
    _instance = None
    _pipeline = None  # Class-level pipeline
    _loaded = False   # Class-level loaded flag
    
    @classmethod
    def get_instance(cls) -> 'ABSAService':
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = ABSAService()
        return cls._instance
    
    def __init__(self):
        # Don't reset class-level state
        pass
    
    def initialize(self, csv_path: str = 'dataset/Reviews.csv'):
        """Initialize the ABSA pipeline with data."""
        with _init_lock:
            if not ABSAService._loaded:
                ABSAService._pipeline = ABSAPipeline()
                ABSAService._pipeline.load_and_analyze(csv_path)
                ABSAService._loaded = True
    
    def ensure_loaded(self):
        """Ensure data is loaded."""
        if not ABSAService._loaded:
            print("🔄 Initializing ABSA service with data...")
            self.initialize()
            print("✅ ABSA service initialized")
    
    @property
    def _current_pipeline(self):
        """Get the current pipeline."""
        return ABSAService._pipeline
    
    # =========================================================================
    # API Methods
    # =========================================================================
    
    def get_all_locations(self) -> List[Dict]:
        """
        Get list of all locations with basic info.
        
        Returns:
            List of location summaries
        """
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
        
        # Sort by rating
        locations.sort(key=lambda x: x['rating'], reverse=True)
        return locations
    
    def get_location_insight(self, location_name: str) -> Optional[Dict]:
        """
        Get detailed insight for a specific location.
        
        Args:
            location_name: Name of the location
            
        Returns:
            Location insight dictionary or None
        """
        self.ensure_loaded()
        
        insight = ABSAService._pipeline.get_location_insight(location_name)
        if insight is None:
            return None
        
        return self._insight_to_dict(insight)
    
    def get_location_aspects(self, location_name: str) -> Optional[Dict]:
        """
        Get aspect scores for a location (for charts/visualization).
        
        Args:
            location_name: Name of the location
            
        Returns:
            Aspect scores dictionary
        """
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
                'score_normalized': round((score + 1) / 2, 3),  # 0-1 scale
                'mentions': insight.aspect_counts.get(aspect_key, 0),
                'sentiment': 'positive' if score > 0.2 else ('negative' if score < -0.2 else 'neutral')
            })
        
        # Sort by score
        aspects.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'location': location_name,
            'aspects': aspects,
            'strengths': insight.strengths,
            'weaknesses': insight.weaknesses
        }
    
    def get_recommendations(
        self,
        preferred_aspects: List[str],
        avoid_aspects: List[str] = None,
        location_type: str = None,
        min_reviews: int = 5,
        limit: int = 10
    ) -> List[Dict]:
        """
        Get smart recommendations based on user preferences.
        
        Args:
            preferred_aspects: List of aspect keys user prefers
            avoid_aspects: List of aspect keys to avoid
            location_type: Filter by location type
            min_reviews: Minimum reviews required
            limit: Maximum recommendations to return
            
        Returns:
            List of recommendation dictionaries
        """
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
        """
        Compare multiple locations across all aspects.
        
        Args:
            locations: List of location names to compare
            
        Returns:
            Comparison data dictionary
        """
        self.ensure_loaded()
        
        comparison_df = ABSAService._pipeline.compare(locations)
        
        # Convert to structured format
        comparison = {
            'locations': [],
            'aspects': list(TOURISM_ASPECTS.keys())
        }
        
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
        """
        Get overall statistics for all aspects.
        
        Returns:
            List of aspect statistics
        """
        self.ensure_loaded()
        
        stats_df = ABSAService._pipeline.get_aspect_statistics()
        
        stats = []
        for _, row in stats_df.iterrows():
            # Find aspect key from display name
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
    
    def get_top_locations_by_aspect(
        self,
        aspect: str,
        limit: int = 10,
        min_mentions: int = 3
    ) -> List[Dict]:
        """
        Get top locations for a specific aspect.
        
        Args:
            aspect: Aspect key (e.g., 'scenery', 'safety')
            limit: Maximum locations to return
            min_mentions: Minimum mentions required
            
        Returns:
            List of top locations for the aspect
        """
        self.ensure_loaded()
        
        if aspect not in TOURISM_ASPECTS:
            return []
        
        top_df = ABSAService._pipeline.get_top_locations_by_aspect(
            aspect, top_n=limit, min_mentions=min_mentions
        )
        
        results = []
        for _, row in top_df.iterrows():
            results.append({
                'location': row['Location'],
                'type': row['Type'],
                'score': round(row['Score'], 3),
                'mentions': int(row['Mentions']),
                'total_reviews': int(row['Total Reviews'])
            })
        
        return results
    
    def get_location_types(self) -> List[Dict]:
        """
        Get all location types with counts.
        
        Returns:
            List of location types
        """
        self.ensure_loaded()
        
        type_counts = {}
        for insight in ABSAService._pipeline.insights.values():
            loc_type = insight.location_type
            if loc_type not in type_counts:
                type_counts[loc_type] = 0
            type_counts[loc_type] += 1
        
        return [
            {'type': t, 'count': c}
            for t, c in sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
        ]
    
    def get_available_aspects(self) -> List[Dict]:
        """
        Get list of available aspects for filtering.
        
        Returns:
            List of aspect definitions
        """
        return [
            {
                'key': key,
                'display_name': config['display_name'],
                'icon': config['icon']
            }
            for key, config in TOURISM_ASPECTS.items()
        ]
    
    def analyze_review(self, text: str) -> Dict:
        """
        Analyze a single review text for aspects and sentiment.
        
        Args:
            text: Review text to analyze
            
        Returns:
            Analysis results
        """
        self.ensure_loaded()
        
        aspects = ABSAService._pipeline.analyzer.analyze_review(text)
        
        return {
            'text': text,
            'aspects_found': [
                {
                    'aspect': asp.aspect,
                    'display_name': TOURISM_ASPECTS[asp.aspect]['display_name'],
                    'icon': TOURISM_ASPECTS[asp.aspect]['icon'],
                    'sentiment': asp.sentiment,
                    'confidence': round(asp.confidence, 3),
                    'keywords': asp.keywords_found,
                    'snippet': asp.text_snippet
                }
                for asp in aspects
            ],
            'total_aspects': len(aspects)
        }
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _insight_to_dict(self, insight: LocationInsight) -> Dict:
        """Convert LocationInsight to dictionary."""
        return {
            'location_name': insight.location_name,
            'location_type': insight.location_type,
            'overall_sentiment': round(insight.overall_sentiment, 3),
            'recommendation_score': insight.recommendation_score,
            'total_reviews': insight.total_reviews,
            'strengths': insight.strengths,
            'weaknesses': insight.weaknesses,
            'aspect_scores': {
                k: round(v, 3) for k, v in insight.aspect_scores.items()
            },
            'aspect_counts': insight.aspect_counts
        }


# ============================================================================
# Flask API Routes (for integration with app.py)
# ============================================================================

def register_absa_routes(app):
    """
    Register ABSA API routes with Flask app.
    
    Usage in app.py:
        from src.absa_api import register_absa_routes
        register_absa_routes(app)
    """
    from flask import jsonify, request
    
    service = ABSAService.get_instance()
    
    @app.route('/api/absa/locations', methods=['GET'])
    def api_get_locations():
        """Get all locations."""
        try:
            locations = service.get_all_locations()
            return jsonify({'success': True, 'data': locations})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/absa/locations/<location_name>', methods=['GET'])
    def api_get_location_insight(location_name):
        """Get insight for a specific location."""
        try:
            insight = service.get_location_insight(location_name)
            if insight is None:
                return jsonify({'success': False, 'error': 'Location not found'}), 404
            return jsonify({'success': True, 'data': insight})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/absa/locations/<location_name>/aspects', methods=['GET'])
    def api_get_location_aspects(location_name):
        """Get aspect scores for a location."""
        try:
            aspects = service.get_location_aspects(location_name)
            if aspects is None:
                return jsonify({'success': False, 'error': 'Location not found'}), 404
            return jsonify({'success': True, 'data': aspects})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/absa/recommend', methods=['POST'])
    def api_get_recommendations():
        """Get smart recommendations."""
        try:
            data = request.get_json()
            
            preferred = data.get('preferred_aspects', [])
            avoid = data.get('avoid_aspects', [])
            loc_type = data.get('location_type')
            limit = data.get('limit', 10)
            
            recs = service.get_recommendations(
                preferred_aspects=preferred,
                avoid_aspects=avoid,
                location_type=loc_type,
                limit=limit
            )
            return jsonify({'success': True, 'data': recs})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/absa/compare', methods=['POST'])
    def api_compare_locations():
        """Compare multiple locations."""
        try:
            data = request.get_json()
            locations = data.get('locations', [])
            
            if len(locations) < 2:
                return jsonify({'success': False, 'error': 'Need at least 2 locations'}), 400
            
            comparison = service.compare_locations(locations)
            return jsonify({'success': True, 'data': comparison})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/absa/aspects', methods=['GET'])
    def api_get_aspects():
        """Get available aspects."""
        try:
            aspects = service.get_available_aspects()
            return jsonify({'success': True, 'data': aspects})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/absa/aspects/stats', methods=['GET'])
    def api_get_aspect_stats():
        """Get aspect statistics."""
        try:
            stats = service.get_aspect_statistics()
            return jsonify({'success': True, 'data': stats})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/absa/aspects/<aspect>/top', methods=['GET'])
    def api_get_top_by_aspect(aspect):
        """Get top locations for an aspect."""
        try:
            limit = request.args.get('limit', 10, type=int)
            top = service.get_top_locations_by_aspect(aspect, limit=limit)
            return jsonify({'success': True, 'data': top})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/absa/types', methods=['GET'])
    def api_get_location_types():
        """Get location types."""
        try:
            types = service.get_location_types()
            return jsonify({'success': True, 'data': types})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/absa/analyze', methods=['POST'])
    def api_analyze_review():
        """Analyze a review text."""
        try:
            data = request.get_json()
            text = data.get('text', '')
            
            if not text:
                return jsonify({'success': False, 'error': 'No text provided'}), 400
            
            analysis = service.analyze_review(text)
            return jsonify({'success': True, 'data': analysis})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    print("✅ ABSA API routes registered")
