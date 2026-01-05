# Aspect-Based Sentiment Analysis Research

**Machine Learning-Based Sentiment Analysis for Sri Lanka Tourism Reviews**

This is a standalone research project for Aspect-Based Sentiment Analysis (ABSA) with ML classification.

---

## 📊 Research Overview

| Metric | Value |
|--------|-------|
| Total Reviews | 16,156 |
| Locations Analyzed | 76 |
| Aspects | 7 |
| ML Training Samples | 47,538 |
| Average ML F1 Score | 74.11% |

### ML Results by Aspect

| Aspect | Model | F1 Score |
|--------|-------|----------|
| Experience & Activities | Linear SVM | 77.67% |
| Scenery & Views | Linear SVM | 76.80% |
| Facilities | Linear SVM | 74.12% |
| Accessibility | Linear SVM | 73.59% |
| Value for Money | Linear SVM | 72.95% |
| Service & Staff | Logistic Regression | 72.47% |
| Safety & Crowds | Naive Bayes | 71.14% |

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd sentiment_research
pip install -r requirements.txt
```

### 2. Download NLTK Data
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt_tab')"
```

### 3. Run the Application
```bash
python app.py
```

### 4. Access Dashboard
Open: http://127.0.0.1:5002/

---

## 📁 Project Structure

```
sentiment_research/
├── app.py                      # Flask application
├── requirements.txt            # Dependencies
├── .env                        # Configuration
├── README.md                   # This file
│
├── src/                        # Source code
│   ├── aspect_sentiment.py     # Core ABSA implementation
│   ├── aspect_ml_classifier.py # ML training pipeline
│   ├── aspect_ml_service.py    # ML service for API
│   ├── absa_api.py             # API endpoints
│   └── sentiment_analysis.py   # Overall sentiment
│
├── scripts/                    # Utility scripts
│   ├── run_ml_training.py      # Train ML models
│   └── export_results.py       # Export for paper
│
├── templates/                  # Frontend
│   └── absa.html               # Dashboard with charts
│
├── models/                     # Trained models (auto-generated)
│   └── aspect_ml/
│
├── research_output/            # Exported data (auto-generated)
│
└── Documentation/
    ├── ML_RESEARCH_RESULTS.md
    ├── ABSA_RESEARCH.md
    └── ASPECT_ML_RESULTS.md
```

---

## 🔬 Running Research Scripts

### Train ML Models
```bash
python scripts/run_ml_training.py
```

### Export Results for Paper
```bash
python scripts/export_results.py
```

This generates:
- `research_output/location_insights.csv`
- `research_output/aspect_statistics.csv`
- `research_output/ml_evaluation_results.csv`
- `research_output/complete_research_data.json`

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/absa/locations` | GET | All locations |
| `/api/absa/locations/<name>/aspects` | GET | Aspect scores |
| `/api/absa/recommend` | POST | Smart recommendations |
| `/api/absa/compare` | POST | Compare locations |
| `/api/absa/analyze` | POST | Lexicon analysis |
| `/api/absa/analyze/ml` | POST | ML hybrid analysis |
| `/api/absa/ml/evaluation` | GET | ML metrics |
| `/api/absa/export/research` | GET | Export all data |

---

## 📈 Dashboard Features

1. **Explore Locations** - Browse 76 destinations with aspect scores
2. **Smart Recommendations** - Find destinations by preferences
3. **Compare Locations** - Side-by-side comparison
4. **Analyze Review** - Test with custom text (Lexicon + ML)
5. **ML Evaluation** - View model performance charts
6. **Research Export** - Download data for paper

---

## 📚 Research Documentation

- `ML_RESEARCH_RESULTS.md` - Complete ML methodology and results
- `ABSA_RESEARCH.md` - ABSA system documentation
- `ASPECT_ML_RESULTS.md` - Training output

---

## ⚠️ Notes

- First startup takes 2-3 minutes to train ML models
- Dataset (`Reviews.csv`) should be in `../dataset/` folder
- Models are cached after first training
- Port 5002 (different from main recommender app)
