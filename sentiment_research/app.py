"""
Flask API for Aspect-Based Sentiment Analysis Research

Standalone application for sentiment analysis research.
Run: python app.py
Access: http://127.0.0.1:5002/
"""

import os
import threading
from flask import Flask, render_template, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from src.absa_api import register_absa_routes, ABSAService

load_dotenv()

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)
app.secret_key = os.getenv('SECRET_KEY', 'sentiment-research-key')

# Register ABSA API routes
register_absa_routes(app)

# Pre-initialize ABSA service in background
def init_absa_background():
    try:
        print("🔄 Pre-initializing ABSA service...")
        service = ABSAService.get_instance()
        service.initialize(csv_path='../dataset/Reviews.csv')
        print("✅ ABSA service ready!")
    except Exception as e:
        print(f"⚠️ ABSA init error: {e}")

threading.Thread(target=init_absa_background, daemon=True).start()


@app.route('/')
def index():
    """Serve the main ABSA frontend page."""
    return render_template('absa.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    import time
    return jsonify({'status': 'healthy', 'timestamp': time.time(), 'service': 'sentiment-research'})


if __name__ == '__main__':
    host = os.getenv('API_HOST', '0.0.0.0')
    port = int(os.getenv('API_PORT', 5002))  # Different port from main app
    debug = os.getenv('FLASK_DEBUG', '0') == '1'
    
    print("=" * 60)
    print("ASPECT-BASED SENTIMENT ANALYSIS RESEARCH")
    print("=" * 60)
    print(f"\n🚀 Starting server on http://{host}:{port}")
    print(f"📊 Dashboard: http://127.0.0.1:{port}/")
    print("\n⏳ First startup may take 2-3 minutes to train ML models...")
    
    app.run(host=host, port=port, debug=debug)
