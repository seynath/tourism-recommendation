"""
ML-Enhanced Aspect-Based Sentiment Analysis Service.

Integrates trained ML classifiers into the live API for:
1. ML-based aspect sentiment prediction
2. Hybrid approach (ML + Lexicon)
3. Confidence-weighted predictions

Research Contribution:
- Comparison of ML vs Lexicon approaches in production
- Hybrid sentiment analysis for improved accuracy
"""

import os
import pickle
import gzip
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import re
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize

from src.aspect_sentiment import (
    TOURISM_ASPECTS, POSITIVE_WORDS, NEGATIVE_WORDS, NEGATION_WORDS,
    AspectSentiment, LocationInsight, AspectExtractor
)


@dataclass
class MLPrediction:
    """ML prediction result with confidence."""
    aspect: str
    sentiment: str
    confidence: float
    model_used: str
    keywords_found: List[str]


@dataclass
class HybridPrediction:
    """Combined ML + Lexicon prediction."""
    aspect: str
    ml_sentiment: str
    ml_confidence: float
    lexicon_sentiment: str
    lexicon_confidence: float
    final_sentiment: str
    final_confidence: float
    method_used: str  # 'ml', 'lexicon', 'hybrid'


class AspectMLService:
    """
    ML-enhanced service for aspect sentiment analysis.
    
    Features:
    - Trains ML models on startup
    - Provides ML-based predictions
    - Hybrid ML + Lexicon approach
    - Model persistence
    """
    
    _instance = None
    _models_trained = False
    _vectorizers = {}
    _models = {}
    _label_encoders = {}
    _evaluation_results = {}
    
    MODEL_PATH = 'models/aspect_ml/'
    
    def __init__(self):
        self.aspects = TOURISM_ASPECTS
        self.extractor = AspectExtractor()
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Keep sentiment words
        keep_words = {'not', 'no', 'never', 'very', 'really', 'good', 'bad', 'great', 'terrible'}
        self.stop_words -= keep_words
    
    @classmethod
    def get_instance(cls) -> 'AspectMLService':
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = AspectMLService()
        return cls._instance
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for ML."""
        if not isinstance(text, str):
            return ""
        
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = ' '.join(text.split())
        
        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()
        
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        
        return ' '.join(tokens)
    
    def extract_aspect_text(self, text: str, aspect: str) -> str:
        """Extract text related to a specific aspect."""
        if not isinstance(text, str):
            return ""
        
        keywords = self.aspects[aspect]['keywords']
        sentences = []
        
        try:
            sents = sent_tokenize(text)
        except:
            sents = text.split('.')
        
        for sent in sents:
            sent_lower = sent.lower()
            for keyword in keywords:
                if keyword in sent_lower:
                    sentences.append(sent.strip())
                    break
        
        return ' '.join(sentences)
    
    def train_models(self, df: pd.DataFrame, text_col: str = 'Text', rating_col: str = 'Rating'):
        """
        Train ML models for all aspects.
        
        Args:
            df: DataFrame with reviews
            text_col: Column with review text
            rating_col: Column with rating
        """
        print("\n" + "=" * 60)
        print("TRAINING ML MODELS FOR ASPECT SENTIMENT")
        print("=" * 60)
        
        AspectMLService._evaluation_results = {}
        
        for aspect in self.aspects:
            print(f"\n📊 Training model for: {aspect}")
            
            # Build dataset for this aspect
            texts = []
            labels = []
            
            for _, row in df.iterrows():
                text = row[text_col]
                rating = row[rating_col]
                
                aspect_text = self.extract_aspect_text(text, aspect)
                if aspect_text:
                    processed = self.preprocess_text(aspect_text)
                    if len(processed) > 20:
                        # Label based on rating
                        if rating >= 4:
                            label = 'positive'
                        elif rating <= 2:
                            label = 'negative'
                        else:
                            label = 'neutral'
                        
                        texts.append(processed)
                        labels.append(label)
            
            if len(texts) < 100:
                print(f"  ⚠️ Not enough samples ({len(texts)}), skipping...")
                continue
            
            print(f"  Samples: {len(texts)}")
            
            # Encode labels
            le = LabelEncoder()
            y = le.fit_transform(labels)
            AspectMLService._label_encoders[aspect] = le
            
            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                texts, y, test_size=0.2, stratify=y, random_state=42
            )
            
            # TF-IDF vectorization
            vectorizer = TfidfVectorizer(
                max_features=3000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
            
            X_train_tfidf = vectorizer.fit_transform(X_train)
            X_test_tfidf = vectorizer.transform(X_test)
            AspectMLService._vectorizers[aspect] = vectorizer
            
            # Train Linear SVM (best performer)
            model = LinearSVC(max_iter=2000, class_weight='balanced', random_state=42)
            model.fit(X_train_tfidf, y_train)
            AspectMLService._models[aspect] = model
            
            # Evaluate
            y_pred = model.predict(X_test_tfidf)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Cross-validation
            X_all_tfidf = vectorizer.transform(texts)
            cv_scores = cross_val_score(model, X_all_tfidf, y, cv=5, scoring='f1_weighted')
            
            AspectMLService._evaluation_results[aspect] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_f1_mean': cv_scores.mean(),
                'cv_f1_std': cv_scores.std(),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'total_samples': len(texts),
                'class_distribution': dict(zip(*np.unique(labels, return_counts=True)))
            }
            
            print(f"  ✅ Accuracy: {accuracy:.4f}, F1: {f1:.4f}, CV F1: {cv_scores.mean():.4f}±{cv_scores.std():.4f}")
        
        AspectMLService._models_trained = True
        print(f"\n✅ Trained models for {len(AspectMLService._models)} aspects")
        
        # Save models
        self.save_models()
    
    def save_models(self):
        """Save trained models to disk."""
        os.makedirs(self.MODEL_PATH, exist_ok=True)
        
        # Save vectorizers
        with gzip.open(f'{self.MODEL_PATH}vectorizers.pkl.gz', 'wb') as f:
            pickle.dump(AspectMLService._vectorizers, f)
        
        # Save models
        with gzip.open(f'{self.MODEL_PATH}models.pkl.gz', 'wb') as f:
            pickle.dump(AspectMLService._models, f)
        
        # Save label encoders
        with gzip.open(f'{self.MODEL_PATH}label_encoders.pkl.gz', 'wb') as f:
            pickle.dump(AspectMLService._label_encoders, f)
        
        # Save evaluation results
        with gzip.open(f'{self.MODEL_PATH}evaluation_results.pkl.gz', 'wb') as f:
            pickle.dump(AspectMLService._evaluation_results, f)
        
        print(f"✅ Models saved to {self.MODEL_PATH}")
    
    def load_models(self) -> bool:
        """Load trained models from disk."""
        try:
            # Check if files exist first
            vectorizers_path = f'{self.MODEL_PATH}vectorizers.pkl.gz'
            models_path = f'{self.MODEL_PATH}models.pkl.gz'
            encoders_path = f'{self.MODEL_PATH}label_encoders.pkl.gz'
            eval_path = f'{self.MODEL_PATH}evaluation_results.pkl.gz'
            
            if not all(os.path.exists(p) for p in [vectorizers_path, models_path, encoders_path, eval_path]):
                print(f"⚠️ Model files not found in {self.MODEL_PATH}")
                return False
            
            with gzip.open(vectorizers_path, 'rb') as f:
                AspectMLService._vectorizers = pickle.load(f)
            
            with gzip.open(models_path, 'rb') as f:
                AspectMLService._models = pickle.load(f)
            
            with gzip.open(encoders_path, 'rb') as f:
                AspectMLService._label_encoders = pickle.load(f)
            
            with gzip.open(eval_path, 'rb') as f:
                AspectMLService._evaluation_results = pickle.load(f)
            
            AspectMLService._models_trained = True
            print(f"✅ Loaded ML models for {len(AspectMLService._models)} aspects")
            print(f"✅ Loaded evaluation results for {len(AspectMLService._evaluation_results)} aspects")
            return True
        except Exception as e:
            import traceback
            print(f"⚠️ Could not load models: {e}")
            traceback.print_exc()
            return False
    
    def predict_ml(self, text: str, aspect: str) -> Optional[MLPrediction]:
        """
        Predict sentiment using ML model.
        
        Args:
            text: Review text
            aspect: Aspect to predict
            
        Returns:
            MLPrediction or None if model not available
        """
        if aspect not in AspectMLService._models:
            return None
        
        # Extract aspect-related text
        aspect_text = self.extract_aspect_text(text, aspect)
        if not aspect_text:
            return None
        
        # Preprocess
        processed = self.preprocess_text(aspect_text)
        if len(processed) < 10:
            return None
        
        # Vectorize
        vectorizer = AspectMLService._vectorizers[aspect]
        X = vectorizer.transform([processed])
        
        # Predict
        model = AspectMLService._models[aspect]
        pred = model.predict(X)[0]
        
        # Get label
        le = AspectMLService._label_encoders[aspect]
        label = le.inverse_transform([pred])[0]
        
        # Confidence (decision function for SVM)
        if hasattr(model, 'decision_function'):
            decision = model.decision_function(X)[0]
            # Convert to probability-like score
            confidence = 1 / (1 + np.exp(-np.abs(decision).max()))
        else:
            confidence = 0.75
        
        # Get keywords found
        aspects_found = self.extractor.extract_aspects(text)
        keywords = list(set(kw for a, kw, _, _ in aspects_found if a == aspect))
        
        return MLPrediction(
            aspect=aspect,
            sentiment=label,
            confidence=float(confidence),
            model_used='Linear SVM',
            keywords_found=keywords
        )
    
    def predict_lexicon(self, text: str, aspect: str) -> Optional[MLPrediction]:
        """Predict sentiment using lexicon-based approach."""
        aspect_text = self.extract_aspect_text(text, aspect)
        if not aspect_text:
            return None
        
        text_lower = aspect_text.lower()
        words = text_lower.split()
        
        positive_count = 0
        negative_count = 0
        
        # Check for negation
        has_negation = any(neg in text_lower for neg in NEGATION_WORDS)
        
        for word in words:
            word = re.sub(r'[^\w]', '', word)
            if word in POSITIVE_WORDS:
                positive_count += 1
            elif word in NEGATIVE_WORDS:
                negative_count += 1
        
        # Check multi-word phrases
        for phrase in POSITIVE_WORDS:
            if ' ' in phrase and phrase in text_lower:
                positive_count += 1
        for phrase in NEGATIVE_WORDS:
            if ' ' in phrase and phrase in text_lower:
                negative_count += 1
        
        # Apply negation
        if has_negation:
            positive_count, negative_count = negative_count, positive_count
        
        # Determine sentiment
        total = positive_count + negative_count
        if total == 0:
            sentiment = 'neutral'
            confidence = 0.5
        elif positive_count > negative_count:
            sentiment = 'positive'
            confidence = min(positive_count / total, 0.95)
        elif negative_count > positive_count:
            sentiment = 'negative'
            confidence = min(negative_count / total, 0.95)
        else:
            sentiment = 'neutral'
            confidence = 0.5
        
        # Get keywords
        aspects_found = self.extractor.extract_aspects(text)
        keywords = list(set(kw for a, kw, _, _ in aspects_found if a == aspect))
        
        return MLPrediction(
            aspect=aspect,
            sentiment=sentiment,
            confidence=confidence,
            model_used='Lexicon',
            keywords_found=keywords
        )
    
    def predict_hybrid(self, text: str, aspect: str) -> Optional[HybridPrediction]:
        """
        Predict sentiment using hybrid ML + Lexicon approach.
        
        Combines both methods with confidence weighting.
        """
        ml_pred = self.predict_ml(text, aspect)
        lex_pred = self.predict_lexicon(text, aspect)
        
        if ml_pred is None and lex_pred is None:
            return None
        
        # If only one is available, use it
        if ml_pred is None:
            return HybridPrediction(
                aspect=aspect,
                ml_sentiment='N/A',
                ml_confidence=0.0,
                lexicon_sentiment=lex_pred.sentiment,
                lexicon_confidence=lex_pred.confidence,
                final_sentiment=lex_pred.sentiment,
                final_confidence=lex_pred.confidence,
                method_used='lexicon'
            )
        
        if lex_pred is None:
            return HybridPrediction(
                aspect=aspect,
                ml_sentiment=ml_pred.sentiment,
                ml_confidence=ml_pred.confidence,
                lexicon_sentiment='N/A',
                lexicon_confidence=0.0,
                final_sentiment=ml_pred.sentiment,
                final_confidence=ml_pred.confidence,
                method_used='ml'
            )
        
        # Hybrid: weight by confidence
        ml_weight = 0.7  # ML gets higher weight
        lex_weight = 0.3
        
        # If they agree, boost confidence
        if ml_pred.sentiment == lex_pred.sentiment:
            final_sentiment = ml_pred.sentiment
            final_confidence = min(
                ml_pred.confidence * ml_weight + lex_pred.confidence * lex_weight + 0.1,
                0.99
            )
            method = 'hybrid_agree'
        else:
            # Use the one with higher weighted confidence
            ml_score = ml_pred.confidence * ml_weight
            lex_score = lex_pred.confidence * lex_weight
            
            if ml_score >= lex_score:
                final_sentiment = ml_pred.sentiment
                final_confidence = ml_pred.confidence
                method = 'ml'
            else:
                final_sentiment = lex_pred.sentiment
                final_confidence = lex_pred.confidence
                method = 'lexicon'
        
        return HybridPrediction(
            aspect=aspect,
            ml_sentiment=ml_pred.sentiment,
            ml_confidence=ml_pred.confidence,
            lexicon_sentiment=lex_pred.sentiment,
            lexicon_confidence=lex_pred.confidence,
            final_sentiment=final_sentiment,
            final_confidence=final_confidence,
            method_used=method
        )
    
    def analyze_review_ml(self, text: str) -> List[Dict]:
        """Analyze review using ML models."""
        results = []
        
        for aspect in self.aspects:
            pred = self.predict_hybrid(text, aspect)
            if pred:
                results.append({
                    'aspect': aspect,
                    'display_name': self.aspects[aspect]['display_name'],
                    'icon': self.aspects[aspect]['icon'],
                    'ml_sentiment': pred.ml_sentiment,
                    'ml_confidence': pred.ml_confidence,
                    'lexicon_sentiment': pred.lexicon_sentiment,
                    'lexicon_confidence': pred.lexicon_confidence,
                    'final_sentiment': pred.final_sentiment,
                    'final_confidence': pred.final_confidence,
                    'method_used': pred.method_used
                })
        
        return results
    
    def get_evaluation_results(self) -> Dict:
        """Get ML evaluation results."""
        return AspectMLService._evaluation_results
    
    def is_trained(self) -> bool:
        """Check if models are trained."""
        return AspectMLService._models_trained


# Export for API integration
ml_service = AspectMLService.get_instance()
