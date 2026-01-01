"""
Deep Learning-Based Sentiment Analysis for Sri Lanka Tourism Reviews.

This module implements multiple sentiment analysis approaches:
1. Traditional ML: Logistic Regression, SVM, Random Forest with TF-IDF
2. Deep Learning: LSTM, BiLSTM with word embeddings
3. Transformer: BERT-based sentiment classification

Research Focus:
- Compare traditional ML vs Deep Learning approaches
- Analyze sentiment patterns in tourism reviews
- Provide interpretable results for tourism industry insights
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import Counter
import re
import string
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import LabelEncoder

# For text preprocessing
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


@dataclass
class SentimentResult:
    """Container for sentiment analysis results."""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: np.ndarray
    classification_report: str
    training_time: float = 0.0
    
    # Per-class metrics
    precision_per_class: Dict[str, float] = None
    recall_per_class: Dict[str, float] = None
    f1_per_class: Dict[str, float] = None


class TextPreprocessor:
    """
    Text preprocessing pipeline for sentiment analysis.
    
    Steps:
    1. Lowercase conversion
    2. Remove URLs, mentions, special characters
    3. Remove stopwords
    4. Lemmatization
    """
    
    def __init__(self, remove_stopwords: bool = True, lemmatize: bool = True):
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Tourism-specific stopwords to keep (they carry sentiment)
        self.keep_words = {
            'not', 'no', 'never', 'nothing', 'nowhere', 'neither',
            'very', 'really', 'too', 'most', 'more', 'less',
            'good', 'bad', 'great', 'terrible', 'amazing', 'awful',
            'best', 'worst', 'beautiful', 'ugly', 'clean', 'dirty'
        }
        self.stop_words -= self.keep_words
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text."""
        if not isinstance(text, str):
            return ""
        
        # Lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def preprocess(self, text: str) -> str:
        """Full preprocessing pipeline."""
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize
        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words]
        
        # Lemmatize
        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        
        # Remove short tokens
        tokens = [t for t in tokens if len(t) > 2]
        
        return ' '.join(tokens)
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """Preprocess a batch of texts."""
        return [self.preprocess(text) for text in texts]


class SentimentDataset:
    """
    Dataset handler for sentiment analysis.
    
    Converts ratings to sentiment labels:
    - Positive: 4-5 stars
    - Negative: 1-2 stars
    - Neutral: 3 stars (optional, can be excluded)
    """
    
    def __init__(self, include_neutral: bool = True, binary_only: bool = False):
        """
        Initialize dataset handler.
        
        Args:
            include_neutral: Whether to include neutral (3-star) reviews
            binary_only: If True, only use positive/negative (exclude neutral)
        """
        self.include_neutral = include_neutral
        self.binary_only = binary_only
        self.preprocessor = TextPreprocessor()
        self.label_encoder = LabelEncoder()
        
    def load_and_prepare(self, csv_path: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Load dataset and prepare for sentiment analysis.
        
        Args:
            csv_path: Path to Reviews.csv
            
        Returns:
            Tuple of (prepared DataFrame, statistics dict)
        """
        # Load data
        df = pd.read_csv(csv_path)
        
        # Create sentiment labels from ratings
        def rating_to_sentiment(rating):
            if rating >= 4:
                return 'positive'
            elif rating <= 2:
                return 'negative'
            else:
                return 'neutral'
        
        df['sentiment'] = df['Rating'].apply(rating_to_sentiment)
        
        # Filter based on settings
        if self.binary_only:
            df = df[df['sentiment'] != 'neutral'].copy()
        elif not self.include_neutral:
            df = df[df['sentiment'] != 'neutral'].copy()
        
        # Preprocess text
        print("Preprocessing text...")
        df['processed_text'] = self.preprocessor.preprocess_batch(df['Text'].tolist())
        
        # Remove empty texts
        df = df[df['processed_text'].str.len() > 10].copy()
        
        # Encode labels
        df['sentiment_encoded'] = self.label_encoder.fit_transform(df['sentiment'])
        
        # Calculate statistics
        stats = {
            'total_samples': len(df),
            'sentiment_distribution': df['sentiment'].value_counts().to_dict(),
            'avg_text_length': df['processed_text'].str.len().mean(),
            'location_types': df['Location_Type'].nunique(),
            'unique_locations': df['Location_Name'].nunique(),
        }
        
        return df, stats
    
    def get_train_test_split(
        self, 
        df: pd.DataFrame, 
        test_size: float = 0.2,
        stratify: bool = True,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train and test sets.
        
        Args:
            df: Prepared DataFrame
            test_size: Fraction for test set
            stratify: Whether to stratify by sentiment
            random_state: Random seed
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        X = df['processed_text'].values
        y = df['sentiment_encoded'].values
        
        stratify_col = y if stratify else None
        
        return train_test_split(
            X, y, 
            test_size=test_size, 
            stratify=stratify_col,
            random_state=random_state
        )


class TraditionalMLSentiment:
    """
    Traditional Machine Learning approaches for sentiment analysis.
    
    Models:
    - Logistic Regression with TF-IDF
    - Support Vector Machine (LinearSVC)
    - Random Forest
    - Naive Bayes
    - Gradient Boosting
    """
    
    def __init__(self, max_features: int = 10000, ngram_range: Tuple = (1, 2)):
        """
        Initialize traditional ML sentiment analyzer.
        
        Args:
            max_features: Maximum TF-IDF features
            ngram_range: N-gram range for TF-IDF
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        
        # TF-IDF Vectorizer
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
        
        # Models - using faster configurations
        self.models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000, 
                C=1.0,
                class_weight='balanced',
                random_state=42
            ),
            'Linear SVM': LinearSVC(
                max_iter=2000,
                C=1.0,
                class_weight='balanced',
                random_state=42
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'Naive Bayes': MultinomialNB(alpha=0.1),
        }
        
        self.fitted_models = {}
        self.results = {}
    
    def fit_transform_tfidf(self, X_train: np.ndarray) -> np.ndarray:
        """Fit TF-IDF and transform training data."""
        return self.tfidf.fit_transform(X_train)
    
    def transform_tfidf(self, X: np.ndarray) -> np.ndarray:
        """Transform data using fitted TF-IDF."""
        return self.tfidf.transform(X)
    
    def train_and_evaluate(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        label_names: List[str]
    ) -> Dict[str, SentimentResult]:
        """
        Train all models and evaluate.
        
        Args:
            X_train, X_test: Text data (will be TF-IDF transformed)
            y_train, y_test: Labels
            label_names: Names of sentiment classes
            
        Returns:
            Dictionary of model name to SentimentResult
        """
        import time
        
        # Transform to TF-IDF
        print("Fitting TF-IDF vectorizer...")
        X_train_tfidf = self.fit_transform_tfidf(X_train)
        X_test_tfidf = self.transform_tfidf(X_test)
        
        print(f"TF-IDF shape: {X_train_tfidf.shape}")
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            start_time = time.time()
            
            # Train
            model.fit(X_train_tfidf, y_train)
            training_time = time.time() - start_time
            
            # Predict
            y_pred = model.predict(X_test_tfidf)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            conf_matrix = confusion_matrix(y_test, y_pred)
            class_report = classification_report(y_test, y_pred, target_names=label_names)
            
            # Per-class metrics
            precision_per_class = dict(zip(
                label_names,
                precision_score(y_test, y_pred, average=None, zero_division=0)
            ))
            recall_per_class = dict(zip(
                label_names,
                recall_score(y_test, y_pred, average=None, zero_division=0)
            ))
            f1_per_class = dict(zip(
                label_names,
                f1_score(y_test, y_pred, average=None, zero_division=0)
            ))
            
            result = SentimentResult(
                model_name=name,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                confusion_matrix=conf_matrix,
                classification_report=class_report,
                training_time=training_time,
                precision_per_class=precision_per_class,
                recall_per_class=recall_per_class,
                f1_per_class=f1_per_class
            )
            
            results[name] = result
            self.fitted_models[name] = model
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  F1 Score: {f1:.4f}")
            print(f"  Training time: {training_time:.2f}s")
        
        self.results = results
        return results
    
    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5
    ) -> Dict[str, Dict]:
        """
        Perform cross-validation for all models.
        
        Args:
            X: Text data
            y: Labels
            cv: Number of folds
            
        Returns:
            Dictionary of model name to CV results
        """
        # Transform to TF-IDF
        X_tfidf = self.fit_transform_tfidf(X)
        
        cv_results = {}
        
        for name, model in self.models.items():
            print(f"Cross-validating {name}...")
            
            scores = cross_val_score(model, X_tfidf, y, cv=cv, scoring='f1_weighted')
            
            cv_results[name] = {
                'mean_f1': scores.mean(),
                'std_f1': scores.std(),
                'scores': scores.tolist()
            }
            
            print(f"  F1: {scores.mean():.4f} ± {scores.std():.4f}")
        
        return cv_results
    
    def get_feature_importance(self, model_name: str, top_n: int = 20) -> Dict[str, List]:
        """
        Get most important features for a model.
        
        Args:
            model_name: Name of the model
            top_n: Number of top features to return
            
        Returns:
            Dictionary with positive and negative sentiment features
        """
        if model_name not in self.fitted_models:
            return {}
        
        model = self.fitted_models[model_name]
        feature_names = self.tfidf.get_feature_names_out()
        
        if hasattr(model, 'coef_'):
            # For linear models (LR, SVM)
            if len(model.coef_.shape) == 1:
                coef = model.coef_
            else:
                # Multi-class: use positive class coefficients
                coef = model.coef_[-1] if model.coef_.shape[0] > 1 else model.coef_[0]
            
            # Top positive features
            top_positive_idx = np.argsort(coef)[-top_n:][::-1]
            positive_features = [(feature_names[i], coef[i]) for i in top_positive_idx]
            
            # Top negative features
            top_negative_idx = np.argsort(coef)[:top_n]
            negative_features = [(feature_names[i], coef[i]) for i in top_negative_idx]
            
            return {
                'positive_sentiment_words': positive_features,
                'negative_sentiment_words': negative_features
            }
        
        elif hasattr(model, 'feature_importances_'):
            # For tree-based models
            importances = model.feature_importances_
            top_idx = np.argsort(importances)[-top_n:][::-1]
            top_features = [(feature_names[i], importances[i]) for i in top_idx]
            
            return {'important_features': top_features}
        
        return {}



class DeepLearningSentiment:
    """
    Deep Learning approaches for sentiment analysis.
    
    Models:
    - LSTM (Long Short-Term Memory)
    - BiLSTM (Bidirectional LSTM)
    - CNN-LSTM hybrid
    
    Note: Requires TensorFlow/Keras. Falls back gracefully if not available.
    """
    
    def __init__(
        self,
        max_words: int = 20000,
        max_len: int = 200,
        embedding_dim: int = 100
    ):
        """
        Initialize deep learning sentiment analyzer.
        
        Args:
            max_words: Maximum vocabulary size
            max_len: Maximum sequence length
            embedding_dim: Word embedding dimension
        """
        self.max_words = max_words
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        
        self.tokenizer = None
        self.models = {}
        self.history = {}
        self.results = {}
        
        # TensorFlow availability will be checked when needed
        self.tf_available = False
        self.tf = None
    
    def _check_tensorflow(self):
        """Check if TensorFlow is available (lazy loading)."""
        if self.tf is not None:
            return self.tf_available
        
        try:
            import tensorflow as tf
            self.tf_available = True
            self.tf = tf
            return True
        except ImportError:
            self.tf_available = False
            return False
    
    def prepare_sequences(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tokenize and pad sequences.
        
        Args:
            X_train, X_test: Text data
            
        Returns:
            Padded sequences for train and test
        """
        if not self._check_tensorflow():
            return None, None
        
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        
        # Fit tokenizer on training data
        self.tokenizer = Tokenizer(num_words=self.max_words, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(X_train)
        
        # Convert to sequences
        X_train_seq = self.tokenizer.texts_to_sequences(X_train)
        X_test_seq = self.tokenizer.texts_to_sequences(X_test)
        
        # Pad sequences
        X_train_pad = pad_sequences(X_train_seq, maxlen=self.max_len, padding='post', truncating='post')
        X_test_pad = pad_sequences(X_test_seq, maxlen=self.max_len, padding='post', truncating='post')
        
        return X_train_pad, X_test_pad
    
    def build_lstm_model(self, num_classes: int) -> Any:
        """Build LSTM model."""
        if not self._check_tensorflow():
            return None
        
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import (
            Embedding, LSTM, Dense, Dropout, 
            SpatialDropout1D, BatchNormalization
        )
        
        model = Sequential([
            Embedding(self.max_words, self.embedding_dim, input_length=self.max_len),
            SpatialDropout1D(0.2),
            LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True),
            LSTM(64, dropout=0.2, recurrent_dropout=0.2),
            BatchNormalization(),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_bilstm_model(self, num_classes: int) -> Any:
        """Build Bidirectional LSTM model."""
        if not self._check_tensorflow():
            return None
        
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import (
            Embedding, LSTM, Bidirectional, Dense, 
            Dropout, SpatialDropout1D, BatchNormalization
        )
        
        model = Sequential([
            Embedding(self.max_words, self.embedding_dim, input_length=self.max_len),
            SpatialDropout1D(0.2),
            Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)),
            Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)),
            BatchNormalization(),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_cnn_lstm_model(self, num_classes: int) -> Any:
        """Build CNN-LSTM hybrid model."""
        if not self._check_tensorflow():
            return None
        
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import (
            Embedding, Conv1D, MaxPooling1D, LSTM,
            Dense, Dropout, SpatialDropout1D, BatchNormalization
        )
        
        model = Sequential([
            Embedding(self.max_words, self.embedding_dim, input_length=self.max_len),
            SpatialDropout1D(0.2),
            Conv1D(128, 5, activation='relu'),
            MaxPooling1D(pool_size=2),
            LSTM(100, dropout=0.2, recurrent_dropout=0.2),
            BatchNormalization(),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_and_evaluate(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        label_names: List[str],
        epochs: int = 10,
        batch_size: int = 32,
        validation_split: float = 0.1
    ) -> Dict[str, SentimentResult]:
        """
        Train all deep learning models and evaluate.
        
        Args:
            X_train, X_test: Text data
            y_train, y_test: Labels
            label_names: Names of sentiment classes
            epochs: Training epochs
            batch_size: Batch size
            validation_split: Validation split ratio
            
        Returns:
            Dictionary of model name to SentimentResult
        """
        if not self._check_tensorflow():
            print("TensorFlow not available. Skipping deep learning models.")
            return {}
        
        import time
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        
        # Prepare sequences
        print("Preparing sequences...")
        X_train_pad, X_test_pad = self.prepare_sequences(X_train, X_test)
        
        num_classes = len(label_names)
        
        # Define models
        model_builders = {
            'LSTM': self.build_lstm_model,
            'BiLSTM': self.build_bilstm_model,
            'CNN-LSTM': self.build_cnn_lstm_model
        }
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
        ]
        
        results = {}
        
        for name, builder in model_builders.items():
            print(f"\nTraining {name}...")
            start_time = time.time()
            
            # Build model
            model = builder(num_classes)
            
            # Train
            history = model.fit(
                X_train_pad, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=1
            )
            
            training_time = time.time() - start_time
            
            # Predict
            y_pred_proba = model.predict(X_test_pad, verbose=0)
            y_pred = np.argmax(y_pred_proba, axis=1)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            conf_matrix = confusion_matrix(y_test, y_pred)
            class_report = classification_report(y_test, y_pred, target_names=label_names)
            
            # Per-class metrics
            precision_per_class = dict(zip(
                label_names,
                precision_score(y_test, y_pred, average=None, zero_division=0)
            ))
            recall_per_class = dict(zip(
                label_names,
                recall_score(y_test, y_pred, average=None, zero_division=0)
            ))
            f1_per_class = dict(zip(
                label_names,
                f1_score(y_test, y_pred, average=None, zero_division=0)
            ))
            
            result = SentimentResult(
                model_name=name,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                confusion_matrix=conf_matrix,
                classification_report=class_report,
                training_time=training_time,
                precision_per_class=precision_per_class,
                recall_per_class=recall_per_class,
                f1_per_class=f1_per_class
            )
            
            results[name] = result
            self.models[name] = model
            self.history[name] = history.history
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  F1 Score: {f1:.4f}")
            print(f"  Training time: {training_time:.2f}s")
        
        self.results = results
        return results


class SentimentAnalysisPipeline:
    """
    Complete pipeline for sentiment analysis research.
    
    Combines:
    - Data loading and preprocessing
    - Traditional ML models
    - Deep Learning models
    - Comprehensive evaluation
    - Result visualization
    """
    
    def __init__(
        self,
        include_neutral: bool = True,
        binary_only: bool = False
    ):
        """
        Initialize the pipeline.
        
        Args:
            include_neutral: Include neutral (3-star) reviews
            binary_only: Only positive/negative classification
        """
        self.dataset = SentimentDataset(
            include_neutral=include_neutral,
            binary_only=binary_only
        )
        self.traditional_ml = TraditionalMLSentiment()
        self.deep_learning = DeepLearningSentiment()
        
        self.df = None
        self.stats = None
        self.label_names = None
        self.all_results = {}
    
    def load_data(self, csv_path: str) -> Dict:
        """Load and prepare dataset."""
        print("=" * 60)
        print("LOADING AND PREPARING DATA")
        print("=" * 60)
        
        self.df, self.stats = self.dataset.load_and_prepare(csv_path)
        self.label_names = list(self.dataset.label_encoder.classes_)
        
        print(f"\nDataset Statistics:")
        print(f"  Total samples: {self.stats['total_samples']}")
        print(f"  Sentiment distribution: {self.stats['sentiment_distribution']}")
        print(f"  Average text length: {self.stats['avg_text_length']:.0f} chars")
        print(f"  Location types: {self.stats['location_types']}")
        print(f"  Unique locations: {self.stats['unique_locations']}")
        
        return self.stats
    
    def run_evaluation(
        self,
        test_size: float = 0.2,
        run_deep_learning: bool = True,
        dl_epochs: int = 10
    ) -> Dict[str, SentimentResult]:
        """
        Run complete evaluation pipeline.
        
        Args:
            test_size: Test set size
            run_deep_learning: Whether to run DL models
            dl_epochs: Epochs for DL training
            
        Returns:
            Dictionary of all results
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Split data
        X_train, X_test, y_train, y_test = self.dataset.get_train_test_split(
            self.df, test_size=test_size
        )
        
        print(f"\nTrain/Test Split:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
        
        # Traditional ML
        print("\n" + "=" * 60)
        print("TRADITIONAL MACHINE LEARNING MODELS")
        print("=" * 60)
        
        ml_results = self.traditional_ml.train_and_evaluate(
            X_train, X_test, y_train, y_test, self.label_names
        )
        self.all_results.update(ml_results)
        
        # Deep Learning
        if run_deep_learning:
            print("\n" + "=" * 60)
            print("DEEP LEARNING MODELS")
            print("=" * 60)
            
            dl_results = self.deep_learning.train_and_evaluate(
                X_train, X_test, y_train, y_test, self.label_names,
                epochs=dl_epochs
            )
            self.all_results.update(dl_results)
        
        return self.all_results
    
    def run_cross_validation(self, cv: int = 5) -> Dict:
        """Run cross-validation for traditional ML models."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("\n" + "=" * 60)
        print(f"{cv}-FOLD CROSS-VALIDATION")
        print("=" * 60)
        
        X = self.df['processed_text'].values
        y = self.df['sentiment_encoded'].values
        
        return self.traditional_ml.cross_validate(X, y, cv=cv)
    
    def get_feature_analysis(self, model_name: str = 'Logistic Regression') -> Dict:
        """Get feature importance analysis."""
        return self.traditional_ml.get_feature_importance(model_name)
    
    def generate_report(self) -> str:
        """Generate comprehensive research report."""
        report = []
        report.append("=" * 70)
        report.append("SENTIMENT ANALYSIS RESEARCH REPORT")
        report.append("Deep Learning-Based Sentiment Analysis for Sri Lanka Tourism Reviews")
        report.append("=" * 70)
        
        # Dataset info
        report.append("\n1. DATASET SUMMARY")
        report.append("-" * 50)
        if self.stats:
            report.append(f"Total samples: {self.stats['total_samples']}")
            report.append(f"Sentiment distribution: {self.stats['sentiment_distribution']}")
            report.append(f"Classes: {self.label_names}")
        
        # Results comparison
        report.append("\n2. MODEL COMPARISON")
        report.append("-" * 50)
        report.append(f"{'Model':<25} | {'Accuracy':>10} | {'Precision':>10} | {'Recall':>10} | {'F1':>10}")
        report.append("-" * 75)
        
        # Sort by F1 score
        sorted_results = sorted(
            self.all_results.items(),
            key=lambda x: x[1].f1_score,
            reverse=True
        )
        
        for name, result in sorted_results:
            report.append(
                f"{name:<25} | {result.accuracy:>10.4f} | {result.precision:>10.4f} | "
                f"{result.recall:>10.4f} | {result.f1_score:>10.4f}"
            )
        
        # Best model
        if sorted_results:
            best_name, best_result = sorted_results[0]
            report.append(f"\n3. BEST MODEL: {best_name}")
            report.append("-" * 50)
            report.append(f"Accuracy: {best_result.accuracy:.4f}")
            report.append(f"F1 Score: {best_result.f1_score:.4f}")
            report.append(f"\nClassification Report:")
            report.append(best_result.classification_report)
        
        # Feature analysis
        features = self.get_feature_analysis()
        if features:
            report.append("\n4. FEATURE ANALYSIS (Logistic Regression)")
            report.append("-" * 50)
            if 'positive_sentiment_words' in features:
                report.append("\nTop Positive Sentiment Words:")
                for word, score in features['positive_sentiment_words'][:10]:
                    report.append(f"  {word}: {score:.4f}")
            if 'negative_sentiment_words' in features:
                report.append("\nTop Negative Sentiment Words:")
                for word, score in features['negative_sentiment_words'][:10]:
                    report.append(f"  {word}: {score:.4f}")
        
        report.append("\n" + "=" * 70)
        
        return "\n".join(report)
