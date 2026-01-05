# Sri Lanka Tourism Sentiment Analysis System

**Aspect-Based Sentiment Analysis with Machine Learning for Sri Lanka Tourism**

A comprehensive machine learning system that analyzes 16,000+ tourism reviews to provide aspect-level sentiment insights and smart destination recommendations.

---

## 🎯 Research Features

### Machine Learning Components
- **Overall Sentiment Classification** - Linear SVM with 81.58% accuracy
- **Aspect-Level ML Classification** - Separate models for 7 tourism aspects (74.11% avg F1)
- **Hybrid Analysis** - Combines ML + Lexicon approaches for robust predictions

### Aspect-Based Sentiment Analysis
- **7 Tourism Aspects**: Scenery, Safety, Facilities, Value, Accessibility, Experience, Service
- **200+ Domain Keywords**: Tourism-specific vocabulary
- **76 Destinations**: Comprehensive insights for Sri Lanka locations

### Smart Features
- **Preference-Based Recommendations** - Match destinations to user preferences
- **Location Comparison** - Compare multiple destinations across all aspects
- **Real-time Analysis** - Analyze any review text instantly

---

## 📊 ML Results Summary

### Overall Sentiment Classification
| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Linear SVM | 81.58% | 81.08% |
| Logistic Regression | 80.92% | 80.45% |
| Random Forest | 78.34% | 77.89% |

### Aspect-Level ML Classification
| Aspect | Best Model | F1 Score | Samples |
|--------|------------|----------|---------|
| Experience & Activities | Linear SVM | 77.67% | 11,076 |
| Scenery & Views | Linear SVM | 76.80% | 9,053 |
| Facilities | Linear SVM | 74.12% | 5,413 |
| Accessibility | Linear SVM | 73.59% | 8,986 |
| Value for Money | Linear SVM | 72.95% | 7,220 |
| Service & Staff | Logistic Regression | 72.47% | 2,093 |
| Safety & Crowds | Naive Bayes | 71.14% | 3,697 |

**Total ML Training Samples: 47,538**

---

## 📋 Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- 4GB RAM minimum

---

## 🚀 Quick Start Guide

### Step 1: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# or: venv\Scripts\activate  # Windows
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Download NLTK Data
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt_tab')"
```

### Step 4: Run the Application
```bash
python app.py
```

### Step 5: Access the Application
- **Sentiment Analysis Dashboard**: http://127.0.0.1:5001/absa
- **Recommender System**: http://127.0.0.1:5001/

> ⚠️ **Note**: First startup takes ~2-3 minutes to train ML models. Wait for "✅ ABSA service ready!" message.

---

## 📊 Running Analysis Scripts

### Run Complete ML Training
```bash
python scripts/run_aspect_ml_training.py
```

### Export Research Results
```bash
python scripts/export_research_results.py
```

### Run Sentiment Analysis Evaluation
```bash
python scripts/run_sentiment_analysis.py
```

---

## 🔌 API Endpoints

### Core Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/absa/locations` | GET | Get all locations with ratings |
| `/api/absa/locations/<name>` | GET | Get detailed insight for a location |
| `/api/absa/locations/<name>/aspects` | GET | Get aspect scores for a location |
| `/api/absa/recommend` | POST | Get smart recommendations |
| `/api/absa/compare` | POST | Compare multiple locations |

### ML Endpoints (NEW)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/absa/analyze` | POST | Analyze review (lexicon-based) |
| `/api/absa/analyze/ml` | POST | Analyze review (ML hybrid) |
| `/api/absa/ml/evaluation` | GET | Get ML model evaluation metrics |
| `/api/absa/export/research` | GET | Export all research data |

### Example API Usage
```bash
# ML-based review analysis
curl -X POST http://127.0.0.1:5001/api/absa/analyze/ml \
  -H "Content-Type: application/json" \
  -d '{"text": "Beautiful scenery but very crowded and expensive."}'

# Get ML evaluation results
curl http://127.0.0.1:5001/api/absa/ml/evaluation

# Export research data
curl http://127.0.0.1:5001/api/absa/export/research
```

---

## 📁 Project Structure

```
tourism-recommender-system/
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── dataset/
│   └── Reviews.csv                 # Tourism reviews (16,156 reviews)
├── src/
│   ├── aspect_sentiment.py         # ABSA core implementation
│   ├── aspect_ml_service.py        # ML service (NEW)
│   ├── aspect_ml_classifier.py     # ML training pipeline
│   ├── absa_api.py                 # API service layer
│   └── sentiment_analysis.py       # Overall sentiment ML
├── scripts/
│   ├── run_aspect_ml_training.py   # Train ML models
│   ├── export_research_results.py  # Export for paper (NEW)
│   └── run_sentiment_analysis.py   # Evaluate sentiment
├── models/
│   └── aspect_ml/                  # Trained ML models (NEW)
├── research_output/                # Exported research data (NEW)
├── templates/
│   └── absa.html                   # Enhanced frontend with charts
└── data/
    ├── aspect_statistics.csv
    └── location_insights.csv
```

---

## 📈 Research Output Files

After running `python scripts/export_research_results.py`:

```
research_output/
├── location_insights.csv       # 76 locations with aspect scores
├── aspect_statistics.csv       # 7 aspects with sentiment stats
├── ml_evaluation_results.csv   # ML metrics per aspect
├── dataset_statistics.json     # Overall dataset stats
└── complete_research_data.json # All data for paper
```

---

## 🔧 Configuration

Edit `.env` file:
```env
FLASK_DEBUG=0           # Keep 0 for production
API_PORT=5001           # Change port if needed
```

---

## 📚 Research Documentation

- `ABSA_RESEARCH.md` - Detailed ABSA methodology
- `ASPECT_ML_RESULTS.md` - ML training results
- `SENTIMENT_ANALYSIS_RESEARCH.md` - Overall sentiment methodology
- `SENTIMENT_ANALYSIS_RESULTS.md` - Evaluation results

---

## 🐛 Troubleshooting

### Port Already in Use
```bash
lsof -ti:5001 | xargs kill -9
```

### NLTK Data Not Found
```bash
python -c "import nltk; nltk.download('all')"
```

### Slow First Load
First startup trains ML models (~2-3 minutes). Subsequent loads are instant as models are cached.

---

## 📄 License

This project is for academic research purposes.
