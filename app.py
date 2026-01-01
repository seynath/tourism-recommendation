"""
Flask API Backend for Tourism Recommender System
"""
import os
import time
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv

from src.recommender_system import RecommenderSystem

# Load environment variables
load_dotenv()

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)
app.secret_key = os.getenv('SECRET_KEY', 'dev-secret-key')

# Global recommender system instance
recommender = None


def get_recommender():
    """Lazy load the recommender system."""
    global recommender
    if recommender is None:
        print("Loading recommender system...")
        recommender = RecommenderSystem()
        model_path = os.getenv('MODEL_PATH', 'models/')
        recommender.load_models(model_path)
        print("Recommender system loaded successfully!")
    return recommender


@app.route('/')
def index():
    """Serve the main frontend page."""
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time()
    })


@app.route('/api/destinations', methods=['GET'])
def get_destinations():
    """Get list of all available destinations."""
    try:
        system = get_recommender()
        destinations = []
        for dest_id, features in system.location_features.items():
            destinations.append({
                'id': dest_id,
                'name': features.name,
                'city': features.city,
                'type': features.location_type,
                'latitude': features.latitude,
                'longitude': features.longitude,
                'avg_rating': features.avg_rating,
                'review_count': features.review_count,
                'price_range': features.price_range,
                'attributes': features.attributes
            })
        return jsonify({'destinations': destinations, 'count': len(destinations)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/recommend', methods=['POST'])
def get_recommendations():
    """Get personalized recommendations."""
    try:
        data = request.json
        system = get_recommender()
        
        # Parse request parameters
        user_id = data.get('user_id', 'anonymous')
        latitude = float(data.get('latitude', 7.8731))  # Default: Sri Lanka center
        longitude = float(data.get('longitude', 80.7718))
        weather_condition = data.get('weather', 'sunny')
        temperature = float(data.get('temperature', 28.0))
        season = data.get('season', 'dry')
        is_holiday = data.get('is_holiday', False)
        is_peak_season = data.get('is_peak_season', False)
        travel_style = data.get('travel_style', None)
        budget_min = data.get('budget_min', None)
        budget_max = data.get('budget_max', None)
        max_distance = data.get('max_distance', None)
        top_k = int(data.get('top_k', 10))
        voting_strategy = data.get('voting_strategy', 'weighted')
        
        # Determine user type
        user_type = 'cold_start'
        if user_id in system.user_profiles:
            profile = system.user_profiles[user_id]
            if not profile.is_cold_start:
                user_type = 'regular' if profile.visit_count < 10 else 'frequent'
        
        # Set voting strategy
        system.ensemble.strategy = voting_strategy
        
        # Get recommendations
        start_time = time.time()
        recommendations = system.get_recommendations(
            user_id=user_id,
            location=(latitude, longitude),
            budget=(float(budget_min), float(budget_max)) if budget_min and budget_max else None,
            travel_style=travel_style,
            max_distance_km=float(max_distance) if max_distance else None,
            weather_condition=weather_condition,
            season=season,
            is_holiday=is_holiday,
            is_peak_season=is_peak_season
        )
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Limit to top_k results
        recommendations = recommendations[:top_k]
        
        # Format response
        results = []
        for rec in recommendations:
            results.append({
                'destination_id': rec.destination_id,
                'name': rec.name,
                'score': round(rec.score, 4),
                'explanation': rec.explanation,
                'distance_km': round(rec.distance_km, 2) if rec.distance_km else None,
                'estimated_cost': rec.estimated_cost
            })
        
        return jsonify({
            'recommendations': results,
            'count': len(results),
            'inference_time_ms': round(inference_time, 2),
            'user_type': user_type,
            'voting_strategy': voting_strategy,
            'context': {
                'weather': weather_condition,
                'season': season,
                'is_holiday': is_holiday,
                'is_peak_season': is_peak_season
            }
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """Get information about loaded models."""
    try:
        system = get_recommender()
        
        # Get model sizes
        model_path = os.getenv('MODEL_PATH', 'models/')
        sizes = {}
        for model_name in ['collaborative_filter', 'content_based_filter', 'context_aware_engine']:
            path = os.path.join(model_path, f'{model_name}.pkl.gz')
            if os.path.exists(path):
                sizes[model_name] = round(os.path.getsize(path) / (1024 * 1024), 2)
        
        return jsonify({
            'models': {
                'collaborative_filter': {
                    'type': 'SVD Matrix Factorization',
                    'n_factors': system.collaborative_filter.n_factors,
                    'size_mb': sizes.get('collaborative_filter', 'N/A')
                },
                'content_based_filter': {
                    'type': 'TF-IDF Cosine Similarity',
                    'max_features': system.content_based_filter.max_features,
                    'size_mb': sizes.get('content_based_filter', 'N/A')
                },
                'context_aware_engine': {
                    'type': 'Decision Tree Classifier',
                    'max_depth': system.context_aware_engine.max_depth,
                    'size_mb': sizes.get('context_aware_engine', 'N/A')
                }
            },
            'total_size_mb': round(sum(sizes.values()), 2),
            'num_destinations': len(system.location_features),
            'num_users': len(system.user_profiles),
            'voting_strategies': ['weighted', 'borda', 'confidence']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/users', methods=['GET'])
def get_users():
    """Get sample user IDs for testing."""
    try:
        system = get_recommender()
        users = []
        for user_id, profile in list(system.user_profiles.items())[:20]:
            users.append({
                'user_id': user_id,
                'is_cold_start': profile.is_cold_start,
                'visit_count': profile.visit_count,
                'avg_rating': round(profile.avg_rating, 2),
                'preferred_types': profile.preferred_types[:3]
            })
        return jsonify({'users': users, 'total_users': len(system.user_profiles)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    host = os.getenv('API_HOST', '0.0.0.0')
    port = int(os.getenv('API_PORT', 5001))
    debug = os.getenv('FLASK_DEBUG', '1') == '1'
    
    print(f"Starting Tourism Recommender API on {host}:{port}")
    app.run(host=host, port=port, debug=debug)
