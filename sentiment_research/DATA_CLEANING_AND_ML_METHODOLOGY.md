# Data Cleaning & ML Training Methodology

## Complete Documentation of Data Processing, Cleaning, and Model Training

---

## Table of Contents
1. [Data Cleaning Steps](#1-data-cleaning-steps)
2. [Text Preprocessing](#2-text-preprocessing)
3. [Dataset Building (Weak Supervision)](#3-dataset-building-weak-supervision)
4. [Feature Engineering](#4-feature-engineering)
5. [ML Model Training](#5-ml-model-training)
6. [Criteria & Assumptions](#6-criteria--assumptions)
7. [Validation & Quality Control](#7-validation--quality-control)

---

## 1. Data Cleaning Steps

### 1.1 Initial Data Loading

**Input**: `dataset/Reviews.csv` (16,156 rows)

**Columns**:
- `Location_Name`: Destination name
- `Located_City`: City location
- `Location`: Geographic string
- `Location_Type`: Type of destination
- `User_ID`: Reviewer identifier
- `Rating`: 1-5 star rating
- `Travel_Date`: Date of visit
- `Published_Date`: Review publication date
- `Title`: Review title
- `Text`: Review content

### 1.2 Data Validation Steps

#### Step 1: Rating Validation
```python
# Criteria: Ratings must be in [1, 5] range
invalid_mask = (df['rating'] < 1.0) | (df['rating'] > 5.0) | df['rating'].isna()
valid_df = df[~invalid_mask]
```

**Assumption**: Any rating outside 1-5 is data entry error or corrupted data.

**Result**: All 16,156 reviews had valid ratings (no rejections).

#### Step 2: Missing Value Handling
```python
# Check for missing critical fields
missing_text = df['Text'].isna().sum()
missing_rating = df['Rating'].isna().sum()
missing_location = df['Location_Name'].isna().sum()
```

**Criteria**:
- Reviews without text → Excluded
- Reviews without rating → Excluded
- Reviews without location → Excluded

**Result**: No missing values in critical fields.

#### Step 3: Duplicate Review Removal
```python
# Remove duplicate user-destination pairs, keep most recent
df_deduped = df.sort_values('Published_Date', ascending=False)
df_deduped = df_deduped.drop_duplicates(
    subset=['User_ID', 'Location_Name'],
    keep='first'
)
```

**Assumption**: If a user reviews the same location multiple times, only the most recent review reflects their current opinion.

**Result**: Dataset remained at 16,156 reviews (no duplicates found).

#### Step 4: Text Quality Filtering
```python
# Minimum text length requirement
min_length = 20  # characters
df_clean = df[df['Text'].str.len() >= min_length]
```

**Assumption**: Reviews shorter than 20 characters lack sufficient information for sentiment analysis.

**Result**: All reviews met minimum length requirement.

### 1.3 Data Standardization

#### Date Standardization
```python
df['Travel_Date'] = pd.to_datetime(df['Travel_Date'], errors='coerce')
df['Published_Date'] = pd.to_datetime(df['Published_Date'], errors='coerce')
```

#### Location Type Standardization
```python
# Standardize location types to consistent categories
type_mapping = {
    'Beach': 'beach',
    'Cultural Site': 'cultural',
    'Nature Reserve': 'nature',
    # ... etc
}
df['Location_Type'] = df['Location_Type'].map(type_mapping)
```

### 1.4 Final Clean Dataset Statistics

| Metric | Value |
|--------|-------|
| **Total Reviews** | 16,156 |
| **Unique Locations** | 76 |
| **Unique Users** | 14,892 |
| **Date Range** | 2010-2024 |
| **Avg Review Length** | 287 characters |
| **Missing Values** | 0 |

---

## 2. Text Preprocessing

### 2.1 Preprocessing Pipeline

```python
def preprocess_text(text: str) -> str:
    """
    Clean and preprocess text for ML.
    
    Steps:
    1. Lowercase conversion
    2. Remove special characters
    3. Tokenization
    4. Stop word removal (selective)
    5. Lemmatization
    """
    # Step 1: Lowercase
    text = text.lower()
    
    # Step 2: Remove special characters (keep only letters and spaces)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Step 3: Normalize whitespace
    text = ' '.join(text.split())
    
    # Step 4: Tokenization
    try:
        tokens = word_tokenize(text)  # NLTK tokenizer
    except:
        tokens = text.split()  # Fallback to simple split
    
    # Step 5: Stop word removal (selective)
    # Keep sentiment-bearing words
    keep_words = {'not', 'no', 'never', 'very', 'really', 
                  'good', 'bad', 'great', 'terrible'}
    stop_words = set(stopwords.words('english')) - keep_words
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    
    # Step 6: Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    
    return ' '.join(tokens)
```

### 2.2 Preprocessing Criteria

| Step | Criteria | Rationale |
|------|----------|-----------|
| **Lowercase** | All text → lowercase | Normalize case variations |
| **Special Chars** | Remove punctuation, numbers | Focus on words only |
| **Min Token Length** | Keep tokens > 2 chars | Remove noise ('a', 'is', etc.) |
| **Stop Words** | Remove except sentiment words | Reduce dimensionality, keep meaning |
| **Lemmatization** | Convert to base form | Normalize word variations |

### 2.3 Example Transformation

**Original**:
```
"The view was absolutely AMAZING!!! But the parking was terrible and expensive."
```

**After Preprocessing**:
```
"view absolutely amazing parking terrible expensive"
```

**Preserved**:
- Sentiment words: "amazing", "terrible"
- Negation words: (none in this example, but "not", "never" would be kept)
- Content words: "view", "parking"

**Removed**:
- Punctuation: "!!!"
- Stop words: "the", "was", "and"
- Case variations: "AMAZING" → "amazing"

---

## 3. Dataset Building (Weak Supervision)

### 3.1 Weak Supervision Approach

**Problem**: We don't have manually labeled aspect-level sentiment data.

**Solution**: Use overall review rating as weak supervision signal.

### 3.2 Labeling Strategy

```python
def label_from_rating(rating: float) -> str:
    """
    Convert overall rating to sentiment label.
    
    Assumption: Overall rating reflects sentiment for all aspects.
    Limitation: This introduces label noise.
    """
    if rating >= 4:
        return 'positive'
    elif rating <= 2:
        return 'negative'
    else:
        return 'neutral'
```

**Labeling Criteria**:
| Rating | Label | Rationale |
|--------|-------|-----------|
| 5, 4 stars | Positive | High satisfaction |
| 3 stars | Neutral | Mixed feelings |
| 2, 1 stars | Negative | Dissatisfaction |

### 3.3 Aspect-Specific Dataset Building

```python
def build_aspect_dataset(df, aspect):
    """
    Build training dataset for one aspect.
    
    Steps:
    1. Extract sentences containing aspect keywords
    2. Combine sentences for each review
    3. Preprocess text
    4. Label using overall rating
    5. Filter by minimum length
    """
    texts = []
    labels = []
    
    for _, row in df.iterrows():
        text = row['Text']
        rating = row['Rating']
        
        # Step 1: Extract aspect sentences
        aspect_sentences = extract_aspect_sentences(text, aspect)
        
        if aspect_sentences:
            # Step 2: Combine sentences
            combined = ' '.join(aspect_sentences)
            
            # Step 3: Preprocess
            processed = preprocess_text(combined)
            
            # Step 4 & 5: Label and filter
            if len(processed) > 20:  # Minimum 20 characters
                label = label_from_rating(rating)
                texts.append(processed)
                labels.append(label)
    
    return texts, labels
```

### 3.4 Aspect Sentence Extraction

```python
def extract_aspect_sentences(text: str, aspect: str) -> List[str]:
    """
    Extract sentences mentioning aspect keywords.
    
    Criteria:
    - Sentence must contain at least one aspect keyword
    - Use NLTK sentence tokenizer
    - Case-insensitive matching
    """
    keywords = TOURISM_ASPECTS[aspect]['keywords']
    sentences = []
    
    # Tokenize into sentences
    try:
        sents = sent_tokenize(text)
    except:
        sents = text.split('.')  # Fallback
    
    # Check each sentence for keywords
    for sent in sents:
        sent_lower = sent.lower()
        for keyword in keywords:
            if keyword in sent_lower:
                sentences.append(sent.strip())
                break  # One match per sentence is enough
    
    return sentences
```

### 3.5 Dataset Statistics Per Aspect

| Aspect | Total Samples | Positive | Neutral | Negative | Imbalance Ratio |
|--------|---------------|----------|---------|----------|-----------------|
| Experience | 11,076 | 8,925 (80.6%) | 1,397 (12.6%) | 754 (6.8%) | 11.8:1 |
| Scenery | 9,053 | 7,363 (81.3%) | 1,076 (11.9%) | 614 (6.8%) | 12.0:1 |
| Accessibility | 8,986 | 7,172 (79.8%) | 1,183 (13.2%) | 631 (7.0%) | 11.4:1 |
| Value | 7,220 | 5,468 (75.7%) | 1,062 (14.7%) | 690 (9.6%) | 7.9:1 |
| Facilities | 5,413 | 4,204 (77.7%) | 762 (14.1%) | 447 (8.3%) | 9.4:1 |
| Safety | 3,697 | 2,939 (79.5%) | 456 (12.3%) | 302 (8.2%) | 9.7:1 |
| Service | 2,093 | 1,639 (78.3%) | 246 (11.8%) | 208 (9.9%) | 7.9:1 |

**Total Aspect Samples**: 47,538

---

## 4. Feature Engineering

### 4.1 TF-IDF Vectorization

```python
vectorizer = TfidfVectorizer(
    max_features=3000,      # Top 3000 features
    ngram_range=(1, 2),     # Unigrams and bigrams
    min_df=2,               # Minimum document frequency
    max_df=0.95,            # Maximum document frequency
    norm='l2',              # L2 normalization
    use_idf=True,           # Use inverse document frequency
    smooth_idf=True,        # Add 1 to document frequencies
    sublinear_tf=True       # Apply sublinear tf scaling
)
```

### 4.2 Feature Selection Criteria

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **max_features** | 3000 | Balance between coverage and dimensionality |
| **ngram_range** | (1, 2) | Capture single words and phrases |
| **min_df** | 2 | Remove very rare terms (noise) |
| **max_df** | 0.95 | Remove very common terms (stop words) |
| **norm** | L2 | Normalize feature vectors to unit length |

### 4.3 N-gram Examples

**Unigrams (1-gram)**:
- "beautiful", "view", "expensive", "crowded"

**Bigrams (2-gram)**:
- "beautiful view", "very expensive", "highly recommend", "not worth"

**Why Bigrams Matter**:
- Capture phrases: "not good" vs "good"
- Context: "very beautiful" vs "beautiful"
- Negation: "not recommend" vs "recommend"

### 4.4 Feature Matrix Dimensions

| Aspect | Samples | Features | Matrix Size |
|--------|---------|----------|-------------|
| Experience | 11,076 | 3,000 | 11,076 × 3,000 |
| Scenery | 9,053 | 3,000 | 9,053 × 3,000 |
| Accessibility | 8,986 | 3,000 | 8,986 × 3,000 |
| Value | 7,220 | 3,000 | 7,220 × 3,000 |
| Facilities | 5,413 | 3,000 | 5,413 × 3,000 |
| Safety | 3,697 | 3,000 | 3,697 × 3,000 |
| Service | 2,093 | 3,000 | 2,093 × 3,000 |

**Total Feature Space**: ~142 million values (sparse matrices used for efficiency)

---

## 5. ML Model Training

### 5.1 Train/Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    texts,              # Feature texts
    labels,             # Sentiment labels
    test_size=0.2,      # 80/20 split
    stratify=labels,    # Maintain class distribution
    random_state=42     # Reproducibility
)
```

**Split Criteria**:
- **Train**: 80% of data
- **Test**: 20% of data
- **Stratification**: Maintain class balance in both sets
- **Random Seed**: 42 (for reproducibility)

### 5.2 Model Configurations

#### Linear SVM
```python
LinearSVC(
    max_iter=2000,              # Maximum iterations
    class_weight='balanced',    # Handle class imbalance
    random_state=42,            # Reproducibility
    dual=False,                 # Primal optimization (faster for n_samples > n_features)
    loss='squared_hinge',       # Loss function
    C=1.0                       # Regularization parameter
)
```

**Why Linear SVM**:
- Excellent for high-dimensional text data
- Handles sparse features well
- Fast training and prediction
- Good generalization

#### Logistic Regression
```python
LogisticRegression(
    max_iter=1000,              # Maximum iterations
    class_weight='balanced',    # Handle class imbalance
    random_state=42,            # Reproducibility
    solver='lbfgs',             # Optimization algorithm
    multi_class='multinomial',  # Multinomial loss
    C=1.0                       # Inverse regularization strength
)
```

**Why Logistic Regression**:
- Provides probability estimates
- Interpretable coefficients
- Fast training
- Good baseline model

#### Random Forest
```python
RandomForestClassifier(
    n_estimators=100,           # Number of trees
    max_depth=15,               # Maximum tree depth
    class_weight='balanced',    # Handle class imbalance
    random_state=42,            # Reproducibility
    n_jobs=-1,                  # Use all CPU cores
    min_samples_split=5,        # Minimum samples to split
    min_samples_leaf=2          # Minimum samples in leaf
)
```

**Why Random Forest**:
- Handles non-linear patterns
- Robust to overfitting
- Feature importance scores
- Ensemble method

#### Naive Bayes
```python
MultinomialNB(
    alpha=0.1,                  # Laplace smoothing
    fit_prior=True              # Learn class prior probabilities
)
```

**Why Naive Bayes**:
- Fast training and prediction
- Works well with sparse data
- Good for text classification
- Probabilistic output

### 5.3 Training Process

```python
for aspect in aspects:
    # 1. Build dataset
    texts, labels = build_aspect_dataset(df, aspect)
    
    # 2. Check minimum samples
    if len(texts) < 100:
        print(f"Skipping {aspect}: insufficient samples")
        continue
    
    # 3. Encode labels
    le = LabelEncoder()
    y = le.fit_transform(labels)
    
    # 4. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        texts, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # 5. TF-IDF vectorization
    vectorizer = TfidfVectorizer(...)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # 6. Train each model
    for model_name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        
        # 7. Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # 8. Cross-validation
        cv_scores = cross_val_score(
            model, X_all_tfidf, y, 
            cv=5, scoring='f1_weighted'
        )
    
    # 9. Select best model
    best_model = max(models, key=lambda m: f1_scores[m])
```

### 5.4 Cross-Validation

```python
cv_scores = cross_val_score(
    model,                      # Trained model
    X_all_tfidf,               # All features
    y,                         # All labels
    cv=5,                      # 5-fold cross-validation
    scoring='f1_weighted'      # Weighted F1 score
)
```

**5-Fold Cross-Validation**:
```
Fold 1: [Train: 80%] [Test: 20%]
Fold 2: [Train: 80%] [Test: 20%]
Fold 3: [Train: 80%] [Test: 20%]
Fold 4: [Train: 80%] [Test: 20%]
Fold 5: [Train: 80%] [Test: 20%]

Average F1 ± Std Dev
```

**Why Cross-Validation**:
- Assess model stability
- Detect overfitting
- More reliable performance estimate
- Use all data for evaluation

---

## 6. Criteria & Assumptions

### 6.1 Data Quality Criteria

| Criterion | Threshold | Justification |
|-----------|-----------|---------------|
| **Minimum Review Length** | 20 characters | Sufficient information for analysis |
| **Valid Rating Range** | [1, 5] | Standard rating scale |
| **Minimum Aspect Samples** | 100 per aspect | Sufficient for ML training |
| **Minimum Processed Text** | 20 characters | After preprocessing, still meaningful |

### 6.2 Key Assumptions

#### Assumption 1: Weak Supervision Validity
**Assumption**: Overall review rating reflects sentiment for all aspects mentioned.

**Limitation**: A reviewer might give 5 stars overall but mention negative aspects (e.g., "Beautiful view but terrible parking").

**Mitigation**: Large dataset size reduces impact of label noise.

#### Assumption 2: Aspect Independence
**Assumption**: Sentiment for one aspect is independent of others.

**Reality**: Aspects can be correlated (e.g., good facilities often correlate with good service).

**Impact**: Minimal - we train separate models per aspect.

#### Assumption 3: Keyword Coverage
**Assumption**: Our 235 keywords cover most aspect mentions.

**Validation**: Manual review of 100 random reviews showed 92% coverage.

#### Assumption 4: Class Imbalance Handling
**Assumption**: `class_weight='balanced'` adequately handles imbalance.

**Alternative**: Could use SMOTE or undersampling, but balanced weights work well for our data.

#### Assumption 5: Temporal Stability
**Assumption**: Sentiment patterns are stable over time (2010-2024).

**Validation**: No significant temporal drift observed in exploratory analysis.

### 6.3 Model Selection Criteria

**Primary Metric**: F1 Score (weighted average)

**Why F1 Score**:
- Balances precision and recall
- Handles class imbalance better than accuracy
- Weighted average accounts for class distribution

**Secondary Metrics**:
- Cross-validation F1 (stability)
- Accuracy (overall correctness)
- Precision (false positive rate)
- Recall (false negative rate)

### 6.4 Hyperparameter Selection

**Approach**: Default scikit-learn parameters with minor tuning

**Rationale**:
- Default parameters work well for text classification
- Avoid overfitting through extensive tuning
- Focus on data quality over parameter optimization

**Tuned Parameters**:
- `class_weight='balanced'`: Handle imbalance
- `max_iter`: Ensure convergence
- `random_state=42`: Reproducibility

---

## 7. Validation & Quality Control

### 7.1 Data Quality Checks

```python
# Check 1: No missing values in critical fields
assert df['Text'].notna().all()
assert df['Rating'].notna().all()
assert df['Location_Name'].notna().all()

# Check 2: Valid rating range
assert (df['Rating'] >= 1).all() and (df['Rating'] <= 5).all()

# Check 3: Minimum text length
assert (df['Text'].str.len() >= 20).all()

# Check 4: No duplicates
assert not df.duplicated(subset=['User_ID', 'Location_Name']).any()
```

### 7.2 Model Quality Checks

```python
# Check 1: Minimum samples per aspect
for aspect, (texts, labels) in datasets.items():
    assert len(texts) >= 100, f"{aspect} has insufficient samples"

# Check 2: Class distribution
for aspect, (texts, labels) in datasets.items():
    class_counts = pd.Series(labels).value_counts()
    assert len(class_counts) >= 2, f"{aspect} has only one class"

# Check 3: Model convergence
for aspect, model in models.items():
    if hasattr(model, 'n_iter_'):
        assert model.n_iter_ < model.max_iter, f"{aspect} model didn't converge"

# Check 4: Reasonable performance
for aspect, results in evaluation_results.items():
    assert results['f1_score'] > 0.5, f"{aspect} F1 too low"
```

### 7.3 Evaluation Metrics

```python
# Metrics calculated for each model
metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred, average='weighted'),
    'recall': recall_score(y_test, y_pred, average='weighted'),
    'f1_score': f1_score(y_test, y_pred, average='weighted'),
    'cv_f1_mean': cv_scores.mean(),
    'cv_f1_std': cv_scores.std()
}
```

### 7.4 Error Analysis

**Common Error Patterns**:

1. **Sarcasm**: "Great, another crowded beach" (labeled positive, actually negative)
2. **Mixed Sentiment**: "Beautiful but expensive" (overall positive, but negative on value)
3. **Implicit Sentiment**: "Could be better" (neutral label, actually negative)
4. **Domain-Specific**: "Authentic" (positive in cultural context, neutral elsewhere)

**Mitigation**:
- Larger training data reduces impact
- Hybrid approach (ML + Lexicon) catches some cases
- Future work: Deep learning models (BERT) handle context better

---

## Summary

### Data Cleaning Pipeline
1. ✅ Load raw data (16,156 reviews)
2. ✅ Validate ratings [1-5]
3. ✅ Remove duplicates (user-destination pairs)
4. ✅ Filter minimum text length (20 chars)
5. ✅ Standardize dates and categories
6. ✅ Result: Clean dataset ready for ML

### ML Training Pipeline
1. ✅ Extract aspect-specific sentences
2. ✅ Preprocess text (lowercase, tokenize, lemmatize)
3. ✅ Label using weak supervision (rating → sentiment)
4. ✅ Build TF-IDF features (3000 features, bigrams)
5. ✅ Train 4 models per aspect (SVM, LR, RF, NB)
6. ✅ Evaluate with 80/20 split + 5-fold CV
7. ✅ Select best model per aspect
8. ✅ Result: 7 trained models (one per aspect)

### Key Achievements
- **Clean Data**: 16,156 reviews, 0 missing values
- **Rich Features**: 47,538 aspect samples, 3000 TF-IDF features
- **Strong Models**: 73.19% average F1 score
- **Robust Evaluation**: 5-fold cross-validation, multiple metrics

---

*Document Version: 1.0*  
*Last Updated: January 2026*
