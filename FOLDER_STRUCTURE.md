# Project Folder Structure

This project contains TWO separate applications:

---

## 1. 📊 Sentiment Research (NEW - Standalone)

**Location:** `sentiment_research/`

**Purpose:** Aspect-Based Sentiment Analysis with ML - For your research paper

**Run:** 
```bash
cd sentiment_research
python app.py
```

**Access:** http://127.0.0.1:5002/

### Files:
```
sentiment_research/
├── app.py                      # Main Flask app (Port 5002)
├── requirements.txt            # Dependencies
├── README.md                   # Documentation
├── .env                        # Configuration
│
├── src/                        # Source code
│   ├── aspect_sentiment.py     # Core ABSA (7 aspects, 200+ keywords)
│   ├── aspect_ml_classifier.py # ML training (Linear SVM, etc.)
│   ├── aspect_ml_service.py    # ML service for API
│   ├── absa_api.py             # REST API endpoints
│   └── sentiment_analysis.py   # Overall sentiment ML
│
├── scripts/                    # Research scripts
│   ├── run_ml_training.py      # Train ML models
│   └── export_results.py       # Export for paper
│
├── templates/absa.html         # Dashboard with charts
├── models/                     # Trained ML models
├── research_output/            # Exported CSV/JSON
│
└── Documentation/
    ├── ML_RESEARCH_RESULTS.md
    ├── ABSA_RESEARCH.md
    └── ASPECT_ML_RESULTS.md
```

---

## 2. 🗺️ Recommender System (Original)

**Location:** Root folder (`./`)

**Purpose:** Tourism destination recommender with collaborative filtering

**Run:**
```bash
python app.py
```

**Access:** http://127.0.0.1:5001/

### Files:
```
./
├── app.py                      # Main Flask app (Port 5001)
├── requirements.txt            # Dependencies
├── README.md                   # Documentation
│
├── src/                        # Source code
│   ├── recommender_system.py   # Main recommender
│   ├── collaborative_filter.py # SVD matrix factorization
│   ├── content_based_filter.py # TF-IDF similarity
│   ├── context_aware_engine.py # Context-aware recommendations
│   ├── ensemble_voting.py      # Model voting
│   └── ... (other files)
│
├── models/                     # Trained recommender models
├── dataset/Reviews.csv         # Shared dataset
└── templates/index.html        # Recommender frontend
```

---

## 📂 Shared Resources

Both applications use the same dataset:
- **Dataset:** `dataset/Reviews.csv` (16,156 reviews)

---

## 🚀 Quick Start

### For Sentiment Research (Your Paper):
```bash
cd sentiment_research
pip install -r requirements.txt
python app.py
# Open: http://127.0.0.1:5002/
```

### For Recommender System:
```bash
pip install -r requirements.txt
python app.py
# Open: http://127.0.0.1:5001/
```

---

## 📊 Key Differences

| Feature | Sentiment Research | Recommender |
|---------|-------------------|-------------|
| Port | 5002 | 5001 |
| Focus | ML Sentiment Analysis | Destination Recommendations |
| ML Models | Linear SVM per aspect | SVD, TF-IDF, Decision Tree |
| Output | Aspect scores, sentiment | Ranked destinations |
| Research | ✅ For paper | ❌ Not needed |
