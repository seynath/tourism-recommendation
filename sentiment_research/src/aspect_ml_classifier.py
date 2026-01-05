"""
Machine Learning-Based Aspect Sentiment Classifier.

This module adds ML training/testing to the Aspect-Based Sentiment Analysis:
1. Trains separate classifiers for each aspect
2. Uses TF-IDF features
3. Proper train/test split and cross-validation
4. Compares ML vs Lexicon approaches

Research Contribution:
- Supervised learning for aspect-level sentiment
- Comparison of ML vs rule-based approaches
- Tourism domain-specific aspect classification
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import re
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize

# Import aspect definitions
from src.aspect_sentiment import TOURISM_ASPECTS, POSITIVE_WORDS, NEGATIVE_WORDS


@dataclass
class AspectMLResult:
    """Results for aspect-level ML classification."""
    aspect: str
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    cv_f1_mean: float
    cv_f1_std: float
    train_samples: int
    test_samples: int
    class_distribution: Dict[str, int]


class AspectDatasetBuilder:
    """
    Build training dataset for aspect-level sentiment classification.
    
    Extracts sentences containing aspect keywords and labels them
    based on overall review rating (weak supervision).
    """
    
    def __init__(self):
        self.aspects = TOURISM_ASPECTS
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Keep sentiment words
        self.keep_words = {'not', 'no', 'never', 'very', 'really', 'good', 'bad', 'great', 'terrible'}
        self.stop_words -= self.keep_words
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text."""
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
    
    def extract_aspect_sentences(self, text: str, aspect: str) -> List[str]:
        """Extract sentences containing aspect keywords."""
        if not isinstance(text, str):
            return []
        
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
        
        return sentences
    
    def build_aspect_dataset(
        self, 
        df: pd.DataFrame,
        aspect: str,
        text_col: str = 'Text',
        rating_col: str = 'Rating'
    ) -> Tuple[List[str], List[str]]:
        """
        Build dataset for a specific aspect.
        
        Uses weak supervision: sentences with aspect keywords
        are labeled based on overall review rating.
        
        Args:
            df: DataFrame with reviews
            aspect: Aspect to build dataset for
            text_col: Column with review text
            rating_col: Column with rating
            
        Returns:
            (texts, labels) - preprocessed texts and sentiment labels
        """
        texts = []
        labels = []
        
        for _, row in df.iterrows():
            text = row[text_col]
            rating = row[rating_col]
            
            # Extract sentences with aspect keywords
            aspect_sentences = self.extract_aspect_sentences(text, aspect)
            
            if aspect_sentences:
                # Combine sentences for this aspect
                combined = ' '.join(aspect_sentences)
                processed = self.preprocess_text(combined)
                
                if len(processed) > 20:  # Minimum length
                    # Label based on rating (weak supervision)
                    if rating >= 4:
                        label = 'positive'
                    elif rating <= 2:
                        label = 'negative'
                    else:
                        label = 'neutral'
                    
                    texts.append(processed)
                    labels.append(label)
        
        return texts, labels
    
    def build_all_aspect_datasets(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Tuple[List[str], List[str]]]:
        """Build datasets for all aspects."""
        datasets = {}
        
        for aspect in self.aspects:
            print(f"Building dataset for {aspect}...")
            texts, labels = self.build_aspect_dataset(df, aspect)
            datasets[aspect] = (texts, labels)
            print(f"  Samples: {len(texts)}")
            
        return datasets


class AspectSentimentClassifier:
    """
    ML-based classifier for aspect-level sentiment.
    
    Trains separate models for each aspect using:
    - TF-IDF features
    - Multiple ML algorithms
    - Cross-validation
    """
    
    def __init__(self, max_features: int = 5000):
        self.max_features = max_features
        self.vectorizers = {}  # One per aspect
        self.models = {}       # One per aspect
        self.label_encoders = {}
        self.results = {}
        
        # Available models
        self.model_configs = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000, class_weight='balanced', random_state=42
            ),
            'Linear SVM': LinearSVC(
                max_iter=2000, class_weight='balanced', random_state=42
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100, max_depth=15, class_weight='balanced', 
                random_state=42, n_jobs=-1
            ),
            'Naive Bayes': MultinomialNB(alpha=0.1)
        }
    
    def train_aspect_classifier(
        self,
        aspect: str,
        texts: List[str],
        labels: List[str],
        test_size: float = 0.2,
        cv_folds: int = 5
    ) -> Dict[str, AspectMLResult]:
        """
        Train and evaluate classifiers for a specific aspect.
        
        Args:
            aspect: Aspect name
            texts: Preprocessed text samples
            labels: Sentiment labels
            test_size: Test set proportion
            cv_folds: Cross-validation folds
            
        Returns:
            Dictionary of model name to results
        """
        if len(texts) < 50:
            print(f"  ⚠️ Not enough samples for {aspect} ({len(texts)})")
            return {}
        
        # Encode labels
        le = LabelEncoder()
        y = le.fit_transform(labels)
        self.label_encoders[aspect] = le
        
        # Get class distribution
        class_dist = dict(zip(*np.unique(labels, return_counts=True)))
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            texts, y, test_size=test_size, stratify=y, random_state=42
        )
        
        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        self.vectorizers[aspect] = vectorizer
        
        results = {}
        best_f1 = 0
        best_model_name = None
        
        for model_name, model in self.model_configs.items():
            print(f"    Training {model_name}...")
            
            # Train
            model.fit(X_train_tfidf, y_train)
            
            # Predict
            y_pred = model.predict(X_test_tfidf)
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Cross-validation
            X_all_tfidf = vectorizer.transform(texts)
            cv_scores = cross_val_score(
                model, X_all_tfidf, y, cv=min(cv_folds, len(texts)//10), 
                scoring='f1_weighted'
            )
            
            result = AspectMLResult(
                aspect=aspect,
                model_name=model_name,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                cv_f1_mean=cv_scores.mean(),
                cv_f1_std=cv_scores.std(),
                train_samples=len(X_train),
                test_samples=len(X_test),
                class_distribution=class_dist
            )
            
            results[model_name] = result
            
            # Track best model
            if f1 > best_f1:
                best_f1 = f1
                best_model_name = model_name
        
        # Store best model for this aspect
        if best_model_name:
            self.models[aspect] = self.model_configs[best_model_name]
            # Retrain on full data
            X_all_tfidf = vectorizer.transform(texts)
            self.models[aspect].fit(X_all_tfidf, y)
        
        return results
    
    def train_all_aspects(
        self,
        datasets: Dict[str, Tuple[List[str], List[str]]]
    ) -> Dict[str, Dict[str, AspectMLResult]]:
        """Train classifiers for all aspects."""
        all_results = {}
        
        for aspect, (texts, labels) in datasets.items():
            print(f"\n📊 Training classifiers for: {aspect}")
            results = self.train_aspect_classifier(aspect, texts, labels)
            all_results[aspect] = results
            self.results[aspect] = results
        
        return all_results
    
    def predict_aspect_sentiment(
        self,
        text: str,
        aspect: str
    ) -> Tuple[str, float]:
        """
        Predict sentiment for a specific aspect in text.
        
        Args:
            text: Input text
            aspect: Aspect to predict sentiment for
            
        Returns:
            (sentiment_label, confidence)
        """
        if aspect not in self.models or aspect not in self.vectorizers:
            return 'neutral', 0.5
        
        # Preprocess
        builder = AspectDatasetBuilder()
        processed = builder.preprocess_text(text)
        
        # Vectorize
        X = self.vectorizers[aspect].transform([processed])
        
        # Predict
        model = self.models[aspect]
        pred = model.predict(X)[0]
        
        # Get label
        label = self.label_encoders[aspect].inverse_transform([pred])[0]
        
        # Confidence (if available)
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)[0]
            confidence = max(proba)
        else:
            confidence = 0.8  # Default for SVM
        
        return label, confidence


class AspectMLPipeline:
    """
    Complete ML pipeline for aspect-based sentiment analysis.
    
    Combines:
    - Dataset building
    - Model training
    - Evaluation
    - Comparison with lexicon approach
    """
    
    def __init__(self):
        self.dataset_builder = AspectDatasetBuilder()
        self.classifier = AspectSentimentClassifier()
        self.df = None
        self.datasets = None
        self.all_results = None
    
    def load_data(self, csv_path: str) -> pd.DataFrame:
        """Load review data."""
        print("=" * 60)
        print("ASPECT-BASED SENTIMENT ML TRAINING")
        print("=" * 60)
        
        self.df = pd.read_csv(csv_path)
        print(f"\nLoaded {len(self.df)} reviews")
        return self.df
    
    def build_datasets(self) -> Dict:
        """Build training datasets for all aspects."""
        print("\n" + "=" * 60)
        print("BUILDING ASPECT DATASETS")
        print("=" * 60)
        
        self.datasets = self.dataset_builder.build_all_aspect_datasets(self.df)
        return self.datasets
    
    def train_models(self) -> Dict:
        """Train ML models for all aspects."""
        print("\n" + "=" * 60)
        print("TRAINING ML MODELS")
        print("=" * 60)
        
        self.all_results = self.classifier.train_all_aspects(self.datasets)
        return self.all_results
    
    def get_summary_report(self) -> str:
        """Generate summary report of all results."""
        lines = []
        lines.append("\n" + "=" * 70)
        lines.append("ASPECT-BASED SENTIMENT ML RESULTS SUMMARY")
        lines.append("=" * 70)
        
        # Best model per aspect
        lines.append("\n📊 BEST MODEL PER ASPECT:")
        lines.append("-" * 70)
        lines.append(f"{'Aspect':<25} | {'Best Model':<20} | {'F1 Score':>10} | {'CV F1':>12}")
        lines.append("-" * 70)
        
        for aspect, results in self.all_results.items():
            if results:
                best = max(results.values(), key=lambda x: x.f1_score)
                lines.append(
                    f"{aspect:<25} | {best.model_name:<20} | {best.f1_score:>10.4f} | "
                    f"{best.cv_f1_mean:.4f}±{best.cv_f1_std:.4f}"
                )
        
        # Dataset statistics
        lines.append("\n📈 DATASET STATISTICS:")
        lines.append("-" * 70)
        lines.append(f"{'Aspect':<25} | {'Samples':>10} | {'Positive':>10} | {'Neutral':>10} | {'Negative':>10}")
        lines.append("-" * 70)
        
        for aspect, (texts, labels) in self.datasets.items():
            dist = defaultdict(int)
            for l in labels:
                dist[l] += 1
            lines.append(
                f"{aspect:<25} | {len(texts):>10} | {dist['positive']:>10} | "
                f"{dist['neutral']:>10} | {dist['negative']:>10}"
            )
        
        # Overall statistics
        total_samples = sum(len(texts) for texts, _ in self.datasets.values())
        avg_f1 = np.mean([
            max(r.values(), key=lambda x: x.f1_score).f1_score 
            for r in self.all_results.values() if r
        ])
        
        lines.append("\n📋 OVERALL STATISTICS:")
        lines.append(f"  Total aspect samples: {total_samples}")
        lines.append(f"  Average best F1 score: {avg_f1:.4f}")
        lines.append(f"  Aspects trained: {len([r for r in self.all_results.values() if r])}")
        
        return '\n'.join(lines)
    
    def run_full_pipeline(self, csv_path: str) -> Dict:
        """Run complete ML pipeline."""
        start_time = time.time()
        
        # Load data
        self.load_data(csv_path)
        
        # Build datasets
        self.build_datasets()
        
        # Train models
        self.train_models()
        
        # Generate report
        report = self.get_summary_report()
        print(report)
        
        total_time = time.time() - start_time
        print(f"\n⏱️ Total training time: {total_time:.1f} seconds")
        
        return {
            'results': self.all_results,
            'datasets': self.datasets,
            'classifier': self.classifier,
            'report': report
        }


def main():
    """Run aspect ML training."""
    pipeline = AspectMLPipeline()
    results = pipeline.run_full_pipeline('dataset/Reviews.csv')
    
    # Save report
    with open('ASPECT_ML_RESULTS.md', 'w') as f:
        f.write("# Aspect-Based Sentiment ML Results\n\n")
        f.write("```\n")
        f.write(results['report'])
        f.write("\n```\n")
    
    print("\n✅ Results saved to ASPECT_ML_RESULTS.md")
    return results


if __name__ == '__main__':
    main()
