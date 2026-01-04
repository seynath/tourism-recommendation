# Aspect-Based Sentiment Analysis for Sri Lanka Tourism

## Research Summary

### Title
**Aspect-Based Sentiment Analysis with Smart Insights for Sri Lanka Tourism: A Domain-Specific Approach**

---

## 1. Research Innovation

### What Makes This Unique

| Innovation | Description |
|------------|-------------|
| **Domain-Specific Taxonomy** | First tourism-specific aspect taxonomy for Sri Lanka |
| **Smart Recommendations** | Preference-based location matching using aspect scores |
| **Practical Application** | Ready-to-deploy API for tourism apps |
| **Multi-Aspect Analysis** | 7 tourism-specific aspects vs generic sentiment |

### Research Contributions

1. **Novel Aspect Taxonomy** - 7 tourism-specific aspects with 200+ keywords
2. **Hybrid Sentiment Analysis** - Lexicon + context-aware negation handling
3. **Location Intelligence** - Aggregated insights from 16,000+ reviews
4. **Smart Recommendation Engine** - Preference-based matching algorithm
5. **Practical API** - Production-ready endpoints for app integration

---

## 2. Tourism Aspect Taxonomy

### Defined Aspects

| Aspect | Icon | Keywords (Sample) | Purpose |
|--------|------|-------------------|---------|
| **Scenery & Views** | 🏞️ | view, beautiful, stunning, photography | Visual appeal |
| **Accessibility** | 🚗 | parking, transport, road, distance | Ease of access |
| **Facilities** | 🚻 | toilet, clean, food, shop, wifi | Available amenities |
| **Safety & Crowds** | 🛡️ | safe, crowded, scam, peaceful | Security concerns |
| **Value for Money** | 💰 | price, expensive, worth, free | Cost assessment |
| **Experience** | 🎯 | guide, tour, fun, recommend | Activity quality |
| **Service & Staff** | 👨‍💼 | staff, friendly, helpful, rude | Human interaction |

### Keyword Coverage
- **Total Keywords**: 200+
- **Multi-word Phrases**: 50+
- **Negation Handling**: 15 patterns

---

## 3. Methodology

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ABSA Pipeline                             │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Aspect     │───▶│  Sentiment   │───▶│   Location   │  │
│  │  Extractor   │    │  Analyzer    │    │   Insights   │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                   │                    │          │
│         ▼                   ▼                    ▼          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Keyword    │    │   Lexicon    │    │    Smart     │  │
│  │   Matching   │    │  + Negation  │    │  Recommender │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Processing Steps

1. **Aspect Extraction**
   - Sentence tokenization
   - Keyword matching with word boundaries
   - Multi-word phrase detection
   - Overlap removal (longer matches prioritized)

2. **Sentiment Analysis**
   - Lexicon-based scoring (positive/negative word counts)
   - Negation detection and handling
   - Context window analysis (±50 characters)
   - Confidence calculation

3. **Location Aggregation**
   - Per-aspect score averaging
   - Strength/weakness identification
   - Recommendation score calculation (0-5 scale)

4. **Smart Recommendations**
   - User preference matching
   - Aspect-weighted scoring
   - Penalty for avoided aspects

---

## 4. Results

### Dataset Analysis

| Metric | Value |
|--------|-------|
| Total Reviews Analyzed | 16,156 |
| Locations Covered | 76 |
| Location Types | 11 |
| Aspects Extracted | 43,846 mentions |

### Aspect Statistics

| Aspect | Total Mentions | Avg Sentiment | Overall |
|--------|----------------|---------------|---------|
| Experience & Activities | 10,756 | +0.200 | Positive |
| Scenery & Views | 8,870 | +0.313 | Positive |
| Accessibility | 7,938 | +0.174 | Positive |
| Value for Money | 6,324 | +0.252 | Positive |
| Facilities | 4,702 | +0.206 | Positive |
| Safety & Crowds | 3,258 | +0.115 | Positive |
| Service & Staff | 1,998 | +0.169 | Positive |

### Top Locations by Aspect

**Best Scenery:**
1. Ruwanwelisaya (0.63)
2. Galle Fort (0.62)
3. Gregory Lake (0.58)

**Best Safety:**
1. Martin Wickramasinghe Folk Museum (0.83)
2. Brief Garden (0.47)
3. Victoria Park (0.45)

**Best Value:**
1. Tissa Wewa (0.61)
2. St Clair's Falls (0.58)
3. Nallur Kovil (0.46)

---

## 5. Smart Recommendations

### Use Cases Demonstrated

| Scenario | Preferences | Top Recommendation |
|----------|-------------|-------------------|
| Photography Enthusiast | scenery, accessibility | Gregory Lake |
| Family with Kids | safety, facilities | Martin Wickramasinghe Museum |
| Budget Traveler | value, experience | Tissa Wewa |

### Recommendation Algorithm

```python
match_score = Σ(preferred_aspect_scores) / len(preferred_aspects)
             - Σ(avoided_aspect_penalties)
```

---

## 6. API Endpoints

### Available Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/absa/locations` | GET | All locations with ratings |
| `/api/absa/locations/<name>` | GET | Detailed location insight |
| `/api/absa/locations/<name>/aspects` | GET | Aspect scores for charts |
| `/api/absa/recommend` | POST | Smart recommendations |
| `/api/absa/compare` | POST | Compare multiple locations |
| `/api/absa/aspects` | GET | Available aspects |
| `/api/absa/aspects/stats` | GET | Aspect statistics |
| `/api/absa/aspects/<aspect>/top` | GET | Top locations by aspect |
| `/api/absa/analyze` | POST | Analyze new review text |

### Example API Usage

```javascript
// Get recommendations for photography enthusiast
fetch('/api/absa/recommend', {
  method: 'POST',
  body: JSON.stringify({
    preferred_aspects: ['scenery', 'accessibility'],
    limit: 5
  })
})

// Compare beaches
fetch('/api/absa/compare', {
  method: 'POST',
  body: JSON.stringify({
    locations: ['Arugam Bay', 'Bentota Beach', 'Hikkaduwa Beach']
  })
})
```

---

## 7. App Integration Features

### For Tourism App

1. **Location Detail Page**
   - Aspect radar chart
   - Strengths & weaknesses badges
   - Review highlights by aspect

2. **Search & Filter**
   - Filter by aspect scores
   - "Best for photography" tags
   - "Family-friendly" indicators

3. **Personalized Recommendations**
   - User preference selection
   - Matched locations
   - Explanation of why recommended

4. **Comparison Tool**
   - Side-by-side aspect comparison
   - Visual score bars
   - Decision helper

---

## 8. Research Limitations

1. **Lexicon-Based Approach** - May miss sarcasm and complex sentiment
2. **English Only** - Limited to English reviews
3. **Keyword Coverage** - Some aspects may have incomplete keywords
4. **Static Analysis** - No real-time learning from new reviews

---

## 9. Future Enhancements

1. **Deep Learning ABSA** - Use BERT for aspect extraction
2. **Multilingual Support** - Add Sinhala/Tamil analysis
3. **Real-time Updates** - Continuous learning from new reviews
4. **User Feedback Loop** - Improve recommendations from user ratings

---

## 10. How to Run

```bash
# Run ABSA analysis
python scripts/run_aspect_analysis.py

# Start API server
python app.py

# Test API
curl http://localhost:5001/api/absa/locations
```

---

## 11. Files Created

| File | Purpose |
|------|---------|
| `src/aspect_sentiment.py` | Core ABSA implementation |
| `src/absa_api.py` | API service and routes |
| `scripts/run_aspect_analysis.py` | Evaluation script |
| `data/aspect_statistics.csv` | Aspect stats output |
| `data/location_insights.csv` | Location insights output |

---

## 12. Citation

```bibtex
@article{absa_tourism_2024,
  title={Aspect-Based Sentiment Analysis with Smart Insights for Sri Lanka Tourism},
  author={[Your Name]},
  journal={[Conference/Journal Name]},
  year={2024}
}
```

---

## 13. Conclusion

This research presents a **novel Aspect-Based Sentiment Analysis system** specifically designed for Sri Lanka tourism reviews. The system:

- Extracts **7 tourism-specific aspects** from 16,000+ reviews
- Provides **actionable insights** for 76 locations
- Enables **smart recommendations** based on user preferences
- Offers **production-ready API** for tourism app integration

The combination of domain-specific aspect taxonomy, practical insights generation, and app-ready API makes this research both **academically valuable** and **practically applicable** for the tourism industry.
