# Sri Lanka Tourism Sentiment Analysis System

**Aspect-Based Sentiment Analysis with Smart Insights for Sri Lanka Tourism**

A machine learning system that analyzes 16,000+ tourism reviews to provide aspect-level sentiment insights and smart destination recommendations.

---

## 🎯 Features

- **Aspect-Based Sentiment Analysis** - Analyzes 7 tourism aspects (scenery, safety, facilities, value, accessibility, experience, service)
- **76 Destinations** - Comprehensive insights for Sri Lanka tourist locations
- **Smart Recommendations** - Match destinations to user preferences
- **Location Comparison** - Compare multiple destinations side-by-side
- **REST API** - Production-ready API for app integration
- **Web Interface** - Interactive frontend for exploring insights

---

## 📋 Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- 4GB RAM minimum (for processing 16,000+ reviews)

---

## 🚀 Quick Start Guide

### Step 1: Clone/Download the Project

```bash
# If using git
git clone <repository-url>
cd tourism-recommender-system

# Or extract the downloaded zip file
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download NLTK Data

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### Step 5: Verify Dataset

Make sure the dataset file exists:
```bash
ls dataset/Reviews.csv
```

### Step 6: Run the Application

```bash
python app.py
```

You should see:
```
✅ ABSA API routes registered
🔄 Pre-initializing ABSA service...
Starting Tourism Recommender API on 0.0.0.0:5001
...
Generated insights for 76 locations
✅ ABSA service ready!
```

### Step 7: Access the Application

Open your browser and go to:
- **Sentiment Analysis Dashboard**: http://127.0.0.1:5001/absa
- **Recommender System**: http://127.0.0.1:5001/

> ⚠️ **Note**: First startup takes ~90 seconds to analyze all reviews. Wait for "✅ ABSA service ready!" message.

---

## 📊 Running Analysis Scripts

### Run Sentiment Analysis Evaluation
```bash
python scripts/run_sentiment_analysis.py
```

### Run Aspect-Based Analysis
```bash
python scripts/run_aspect_analysis.py
```

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/absa/locations` | GET | Get all locations with ratings |
| `/api/absa/locations/<name>` | GET | Get detailed insight for a location |
| `/api/absa/locations/<name>/aspects` | GET | Get aspect scores for a location |
| `/api/absa/recommend` | POST | Get smart recommendations |
| `/api/absa/compare` | POST | Compare multiple locations |
| `/api/absa/aspects` | GET | Get available aspects |
| `/api/absa/aspects/stats` | GET | Get aspect statistics |
| `/api/absa/analyze` | POST | Analyze a review text |

### Example API Usage

```bash
# Get all locations
curl http://127.0.0.1:5001/api/absa/locations

# Get recommendations
curl -X POST http://127.0.0.1:5001/api/absa/recommend \
  -H "Content-Type: application/json" \
  -d '{"preferred_aspects": ["scenery", "safety"], "limit": 5}'

# Compare locations
curl -X POST http://127.0.0.1:5001/api/absa/compare \
  -H "Content-Type: application/json" \
  -d '{"locations": ["Galle Fort", "Sigiriya The Ancient Rock Fortress"]}'

# Analyze a review
curl -X POST http://127.0.0.1:5001/api/absa/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Beautiful scenery but very crowded and expensive entrance fee."}'
```

---

## 📁 Project Structure

```
tourism-recommender-system/
├── app.py                      # Main Flask application
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables
├── dataset/
│   └── Reviews.csv             # Tourism reviews dataset
├── src/
│   ├── aspect_sentiment.py     # ABSA core implementation
│   ├── absa_api.py             # API service layer
│   ├── sentiment_analysis.py   # Traditional ML sentiment
│   ├── recommender_system.py   # Recommendation engine
│   └── ...
├── scripts/
│   ├── run_sentiment_analysis.py
│   └── run_aspect_analysis.py
├── templates/
│   ├── index.html              # Recommender frontend
│   └── absa.html               # ABSA frontend
├── data/                       # Generated outputs
│   ├── aspect_statistics.csv
│   └── location_insights.csv
└── models/                     # Trained models
```

---

## 🔧 Configuration

Edit `.env` file to configure:

```env
# Flask Configuration
FLASK_DEBUG=0                   # Set to 1 for development
API_PORT=5001                   # Change port if needed

# Model Configuration
MODEL_PATH=models/
DATA_PATH=dataset/
```

---

## 📈 Results Summary

| Metric | Value |
|--------|-------|
| Total Reviews Analyzed | 16,156 |
| Locations Covered | 76 |
| Aspects Tracked | 7 |
| Best ML Model | Linear SVM (81.58% accuracy) |
| Processing Time | ~90 seconds |

### Aspect Statistics

| Aspect | Mentions | Avg Sentiment |
|--------|----------|---------------|
| Experience & Activities | 10,756 | +0.200 |
| Scenery & Views | 8,870 | +0.313 |
| Accessibility | 7,938 | +0.174 |
| Value for Money | 6,324 | +0.252 |
| Facilities | 4,702 | +0.206 |
| Safety & Crowds | 3,258 | +0.115 |
| Service & Staff | 1,998 | +0.169 |

---

## 🐛 Troubleshooting

### Port Already in Use
```bash
# Kill process on port 5001
lsof -ti:5001 | xargs kill -9
```

### NLTK Data Not Found
```bash
python -c "import nltk; nltk.download('all')"
```

### Module Not Found
```bash
pip install -r requirements.txt --upgrade
```

### Slow First Load
The first startup takes ~90 seconds to analyze 16,000+ reviews. This is normal. Subsequent API calls are instant.

---

## 📚 Research Documentation

- `ABSA_RESEARCH.md` - Detailed research documentation
- `SENTIMENT_ANALYSIS_RESEARCH.md` - Sentiment analysis methodology
- `SENTIMENT_ANALYSIS_RESULTS.md` - Evaluation results

---

## 👨‍💻 Author

[Your Name]
[Your University]
[Year]

---

## 📄 License

This project is for academic research purposes.
