# Tourism Recommender System

Lightweight Ensemble-Based Tourism Recommender System for Sri Lanka

## Overview

This system combines three recommendation approaches using ensemble voting:
- **Collaborative Filtering**: SVD-based matrix factorization for user-item predictions
- **Content-Based Filtering**: TF-IDF similarity for destination matching
- **Context-Aware Engine**: Decision tree rules for weather/season adjustments

Optimized for mobile deployment with total model size under 25 MB and inference under 100ms.

## Project Structure

```
.
├── src/                    # Source code
│   ├── __init__.py
│   ├── data_models.py      # Core data classes and type definitions
│   ├── data_processor.py   # Data loading and feature engineering
│   ├── collaborative_filter.py  # SVD-based collaborative filtering
│   ├── content_based_filter.py  # TF-IDF content-based filtering
│   ├── context_aware_engine.py  # Weather/season context rules
│   ├── ensemble_voting.py  # Voting strategies (weighted, Borda, confidence)
│   ├── mobile_optimizer.py # Caching and model compression
│   ├── recommender_api.py  # Main API interface
│   ├── recommender_system.py  # End-to-end system orchestration
│   ├── model_serializer.py # Model persistence
│   ├── evaluation.py       # Metrics (NDCG, Hit Rate, diversity)
│   └── logger.py           # Structured logging
├── tests/                  # Test suite (109 tests)
├── data/                   # Data storage
├── models/                 # Trained model storage
├── dataset/                # Raw dataset files
├── scripts/                # Training scripts
└── pyproject.toml          # Project configuration
```

## Installation

Install dependencies using pip:

```bash
pip install -e .
```

For development dependencies:

```bash
pip install -e ".[dev]"
```

## Quick Start

### 1. Train Models

```python
from src.recommender_system import RecommenderSystem

# Initialize and train the system
system = RecommenderSystem()
system.load_data('dataset/')
system.train()

# Save trained models
system.save_models('models/')
```

Or use the training script:

```bash
python scripts/train_models.py
```

### 2. Get Recommendations

```python
from src.recommender_system import RecommenderSystem
from src.data_models import Context, WeatherInfo

# Load pre-trained system
system = RecommenderSystem()
system.load_models('models/')

# Create context
context = Context(
    location=(7.2906, 80.6337),  # Kandy coordinates
    weather=WeatherInfo(
        condition='sunny',
        temperature=28.0,
        humidity=65.0,
        precipitation_chance=0.1
    ),
    season='dry',
    day_of_week=5,  # Saturday
    is_holiday=False,
    is_peak_season=True,
    user_type='regular'
)

# Get recommendations
recommendations = system.get_recommendations(
    user_id='user_123',
    context=context,
    top_k=10
)

for rec in recommendations:
    print(f"{rec.name}: {rec.score:.2f} - {rec.explanation}")
```

### 3. Using the API

```python
from src.recommender_api import RecommenderAPI, RecommendationRequest

# Initialize API
api = RecommenderAPI(system.ensemble, system.optimizer)

# Create request
request = RecommendationRequest(
    user_id='user_123',
    location=(7.2906, 80.6337),
    budget=(50.0, 200.0),  # USD range
    travel_style='cultural',
    group_size=2,
    max_distance_km=100.0
)

# Get filtered recommendations
recommendations = api.get_recommendations(request)
```

## Voting Strategies

The ensemble supports three voting strategies:

```python
from src.ensemble_voting import EnsembleVotingSystem

# Weighted voting (default)
ensemble = EnsembleVotingSystem(models, strategy='weighted')

# Borda count ranking
ensemble = EnsembleVotingSystem(models, strategy='borda')

# Confidence-based voting
ensemble = EnsembleVotingSystem(models, strategy='confidence')
```

### Dynamic Weight Adjustment

Weights automatically adjust based on context:
- **Cold start users**: +0.2 content-based, -0.2 collaborative
- **Weather-critical**: +0.15 context-aware
- **Peak season**: +0.1 collaborative

## Model Sizes

| Model | Size | Limit |
|-------|------|-------|
| Collaborative Filter | ~4.2 MB | < 10 MB |
| Content-Based Filter | ~0.15 MB | < 5 MB |
| Context-Aware Engine | ~0.01 MB | < 3 MB |
| **Total** | **~4.4 MB** | **< 25 MB** |

## Evaluation Metrics

```python
from src.evaluation import EvaluationModule

evaluator = EvaluationModule()

# Compute metrics
ndcg = evaluator.ndcg_at_k(predictions, ground_truth, k=10)
hit_rate = evaluator.hit_rate_at_k(predictions, ground_truth, k=10)
diversity = evaluator.diversity_score(recommendations, location_features)
coverage = evaluator.coverage_score(recommendations, all_destinations)
```

## Dependencies

- **pandas**: Data manipulation and CSV processing
- **numpy**: Numerical computations
- **scikit-learn**: Machine learning models (SVD, TF-IDF, Decision Trees)
- **scipy**: Sparse matrix operations
- **hypothesis**: Property-based testing framework
- **pytest**: Unit testing framework

## Testing

Run all tests:

```bash
pytest
```

Run with verbose output:

```bash
pytest -v
```

Run specific test file:

```bash
pytest tests/test_ensemble_voting.py -v
```

Run tests with coverage:

```bash
pytest --cov=src --cov-report=html
```

## Development

This project follows spec-driven development with property-based testing:
- Requirements: `.kiro/specs/tourism-recommender-system/requirements.md`
- Design: `.kiro/specs/tourism-recommender-system/design.md`
- Tasks: `.kiro/specs/tourism-recommender-system/tasks.md`

### Property-Based Tests

The system includes 25 correctness properties validated using Hypothesis:
- Data extraction completeness
- Rating matrix normalization
- Cold start confidence handling
- Voting correctness (weighted, Borda, confidence)
- Filter application correctness
- Model serialization round-trip

## License

MIT License

---

## Web Frontend

The system includes a web-based frontend for interactive testing and visualization.

### Running the Frontend

1. Install web dependencies:
```bash
pip install flask flask-cors python-dotenv requests
```

2. Configure environment variables (optional):
```bash
cp .env.example .env
# Edit .env to add your API keys
```

3. Start the Flask server:
```bash
python app.py
```

4. Open your browser to `http://localhost:5000`

### Frontend Features

- **Interactive Map**: Visualize destinations on a Leaflet map of Sri Lanka
- **User Selection**: Test with different user profiles (cold start, regular, frequent)
- **Context Controls**: Adjust weather, season, holidays, and peak season settings
- **Preference Filters**: Filter by travel style, budget, and distance
- **Voting Strategy**: Compare weighted, Borda count, and confidence-based voting
- **Real-time Metrics**: View inference time and model weight adjustments
- **Model Info Panel**: See model sizes and configuration

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main frontend page |
| `/api/health` | GET | Health check |
| `/api/destinations` | GET | List all destinations |
| `/api/users` | GET | List sample users |
| `/api/model-info` | GET | Model information |
| `/api/recommend` | POST | Get recommendations |

### Example API Request

```bash
curl -X POST http://localhost:5000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "anonymous",
    "latitude": 7.2906,
    "longitude": 80.6337,
    "weather": "sunny",
    "season": "dry",
    "is_holiday": false,
    "is_peak_season": true,
    "top_k": 5,
    "voting_strategy": "weighted"
  }'
```
