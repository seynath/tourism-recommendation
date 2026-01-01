"""Main RecommenderSystem class that wires all components together."""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np

from src.data_processor import DataProcessor
from src.collaborative_filter import CollaborativeFilter
from src.content_based_filter import ContentBasedFilter
from src.context_aware_engine import ContextAwareEngine
from src.ensemble_voting import EnsembleVotingSystem
from src.recommender_api import RecommenderAPI
from src.mobile_optimizer import MobileOptimizer
from src.model_serializer import ModelSerializer
from src.evaluation import EvaluationModule
from src.data_models import (
    LocationFeatures,
    UserProfile,
    Context,
    WeatherInfo,
    RecommendationRequest,
    Recommendation,
)
from src.logger import get_logger

# Get logger instance
logger = get_logger()


class RecommenderSystem:
    """
    Main class that integrates all components of the tourism recommender system.
    
    This class wires together:
    - DataProcessor: For loading and processing review data
    - CollaborativeFilter: SVD-based collaborative filtering
    - ContentBasedFilter: TF-IDF based content filtering
    - ContextAwareEngine: Context-aware rules engine
    - EnsembleVotingSystem: Combines model predictions
    - RecommenderAPI: Main API for generating recommendations
    - MobileOptimizer: Caching and optimization
    - ModelSerializer: Model persistence
    - EvaluationModule: Metrics computation
    """
    
    def __init__(
        self,
        n_factors: int = 50,
        max_features: int = 500,
        max_depth: int = 10,
        voting_strategy: str = 'weighted',
    ):
        """
        Initialize the recommender system.
        
        Args:
            n_factors: Number of latent factors for collaborative filtering
            max_features: Maximum TF-IDF features for content-based filtering
            max_depth: Maximum depth for context-aware decision tree
            voting_strategy: Ensemble voting strategy ('weighted', 'borda', 'confidence')
        """
        # Initialize components
        self.data_processor = DataProcessor()
        self.collaborative_filter = CollaborativeFilter(n_factors=n_factors)
        self.content_based_filter = ContentBasedFilter(max_features=max_features)
        self.context_aware_engine = ContextAwareEngine(max_depth=max_depth)
        self.mobile_optimizer = MobileOptimizer()
        self.model_serializer = ModelSerializer()
        self.evaluation_module = EvaluationModule()
        
        # Initialize ensemble with models
        self.ensemble = EnsembleVotingSystem(
            models={
                'collaborative': self.collaborative_filter,
                'content_based': self.content_based_filter,
                'context_aware': self.context_aware_engine,
            },
            strategy=voting_strategy,
        )
        
        # Initialize API (will be fully configured after training)
        self.api: Optional[RecommenderAPI] = None
        
        # Data storage
        self.reviews_df: Optional[pd.DataFrame] = None
        self.location_features: Dict[str, LocationFeatures] = {}
        self.user_profiles: Dict[str, UserProfile] = {}
        self.rating_matrix = None
        self.user_ids: List[str] = []
        self.destination_ids: List[str] = []
        
        # Training state
        self.is_trained = False
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load and process review data from files.
        
        Handles both Reviews.csv and reviews_2/*.csv formats.
        
        Args:
            data_path: Path to data directory or specific CSV file
            
        Returns:
            Combined DataFrame with all reviews
        """
        path = Path(data_path)
        all_reviews = []
        
        if path.is_file():
            # Single file
            df = self.data_processor.load_reviews(str(path))
            all_reviews.append(df)
            logger.logger.info(f"Loaded {len(df)} reviews from {path}")
        elif path.is_dir():
            # Directory - load all CSV files
            # First check for Reviews.csv
            reviews_csv = path / 'Reviews.csv'
            if reviews_csv.exists():
                df = self.data_processor.load_reviews(str(reviews_csv))
                all_reviews.append(df)
                logger.logger.info(f"Loaded {len(df)} reviews from Reviews.csv")
            
            # Then check for reviews_2 subdirectory
            reviews_2_dir = path / 'reviews_2'
            if reviews_2_dir.exists() and reviews_2_dir.is_dir():
                for csv_file in reviews_2_dir.glob('*.csv'):
                    try:
                        df = self.data_processor.load_reviews(str(csv_file))
                        all_reviews.append(df)
                        logger.logger.info(f"Loaded {len(df)} reviews from {csv_file.name}")
                    except Exception as e:
                        logger.logger.warning(f"Failed to load {csv_file}: {e}")
        else:
            raise FileNotFoundError(f"Data path not found: {data_path}")
        
        if not all_reviews:
            raise ValueError("No review data found")
        
        # Combine all reviews
        self.reviews_df = pd.concat(all_reviews, ignore_index=True)
        
        # Deduplicate
        self.reviews_df = self.data_processor.deduplicate_reviews(self.reviews_df)
        
        logger.logger.info(f"Total reviews after deduplication: {len(self.reviews_df)}")
        
        return self.reviews_df
    
    def extract_features(self) -> Tuple[Dict[str, LocationFeatures], Dict[str, UserProfile]]:
        """
        Extract location features and user profiles from loaded data.
        
        Returns:
            Tuple of (location_features dict, user_profiles dict)
        """
        if self.reviews_df is None:
            raise RuntimeError("Data must be loaded before extracting features")
        
        # Extract location features
        self.location_features = self.data_processor.extract_location_features(self.reviews_df)
        logger.logger.info(f"Extracted features for {len(self.location_features)} destinations")
        
        # Build user profiles
        self.user_profiles = self.data_processor.build_user_profiles(self.reviews_df)
        logger.logger.info(f"Built profiles for {len(self.user_profiles)} users")
        
        return self.location_features, self.user_profiles
    
    def train(self) -> None:
        """
        Train all models on the loaded data.
        
        This method:
        1. Builds the rating matrix for collaborative filtering
        2. Generates TF-IDF embeddings for content-based filtering
        3. Trains the context-aware decision tree
        4. Initializes the API with trained models
        """
        if self.reviews_df is None:
            raise RuntimeError("Data must be loaded before training")
        
        if not self.location_features:
            self.extract_features()
        
        start_time = time.time()
        
        # 1. Build rating matrix and train collaborative filter
        logger.logger.info("Training collaborative filter...")
        self.rating_matrix, self.user_ids, self.destination_ids = \
            self.data_processor.build_rating_matrix(self.reviews_df)
        
        self.collaborative_filter.fit(
            self.rating_matrix,
            self.user_ids,
            self.destination_ids,
        )
        cf_time = time.time() - start_time
        logger.logger.info(f"Collaborative filter trained in {cf_time:.2f}s")
        
        # 2. Generate TF-IDF embeddings and train content-based filter
        logger.logger.info("Training content-based filter...")
        cb_start = time.time()
        
        # Prepare descriptions and attributes for content-based filter
        descriptions = []
        attributes = {}
        location_types = {}
        
        for dest_id, features in self.location_features.items():
            # Create description from available text
            desc_parts = [features.name, features.city, features.location_type]
            desc_parts.extend(features.attributes)
            description = ' '.join(desc_parts)
            descriptions.append(description)
            attributes[dest_id] = features.attributes
            location_types[dest_id] = features.location_type
        
        self.content_based_filter.fit(
            descriptions=descriptions,
            attributes=attributes,
            location_types=location_types,
        )
        cb_time = time.time() - cb_start
        logger.logger.info(f"Content-based filter trained in {cb_time:.2f}s")
        
        # 3. Train context-aware engine
        logger.logger.info("Training context-aware engine...")
        ca_start = time.time()
        
        # Create context features from reviews
        context_features = self._create_context_features()
        ratings = self.reviews_df['rating'].values
        
        if len(context_features) > 0:
            self.context_aware_engine.fit(
                context_features=context_features,
                ratings=ratings,
                destination_ids=list(self.location_features.keys()),
                location_types=location_types,
            )
        ca_time = time.time() - ca_start
        logger.logger.info(f"Context-aware engine trained in {ca_time:.2f}s")
        
        # 4. Initialize API with trained models
        self.api = RecommenderAPI(
            ensemble=self.ensemble,
            optimizer=self.mobile_optimizer,
            destinations=self.location_features,
            user_profiles=self.user_profiles,
        )
        
        self.is_trained = True
        total_time = time.time() - start_time
        logger.logger.info(f"All models trained in {total_time:.2f}s")
    
    def _create_context_features(self) -> pd.DataFrame:
        """
        Create context features from review data for training context-aware engine.
        
        Returns:
            DataFrame with context features
        """
        if self.reviews_df is None or len(self.reviews_df) == 0:
            return pd.DataFrame()
        
        # Extract temporal features from travel_date
        features = pd.DataFrame()
        
        # Day of week (0-6)
        if 'travel_date' in self.reviews_df.columns:
            features['day_of_week'] = self.reviews_df['travel_date'].dt.dayofweek.fillna(0)
            
            # Month for season inference
            month = self.reviews_df['travel_date'].dt.month.fillna(6)
            
            # Season features (Sri Lanka seasons)
            # Dry season: December-March (12, 1, 2, 3)
            # Monsoon: May-September (5, 6, 7, 8, 9)
            # Inter-monsoon: April, October, November (4, 10, 11)
            features['is_dry_season'] = month.isin([12, 1, 2, 3]).astype(float)
            features['is_monsoon'] = month.isin([5, 6, 7, 8, 9]).astype(float)
            features['is_inter_monsoon'] = month.isin([4, 10, 11]).astype(float)
        else:
            features['day_of_week'] = 0.0
            features['is_dry_season'] = 0.0
            features['is_monsoon'] = 0.0
            features['is_inter_monsoon'] = 0.0
        
        # Default weather features (we don't have actual weather data in reviews)
        features['is_sunny'] = 0.5
        features['is_rainy'] = 0.2
        features['is_stormy'] = 0.0
        features['temperature'] = 28.0
        features['humidity'] = 70.0
        features['precipitation_chance'] = 0.3
        
        # Holiday feature (simplified - assume weekends are holidays)
        features['is_holiday'] = (features['day_of_week'] >= 5).astype(float)
        
        # Peak season (December-March is peak tourist season)
        features['is_peak_season'] = features['is_dry_season']
        
        return features
    
    def get_recommendations(
        self,
        user_id: str,
        location: Tuple[float, float] = (7.8731, 80.7718),  # Default: Sri Lanka center
        budget: Optional[Tuple[float, float]] = None,
        travel_style: Optional[str] = None,
        group_size: int = 1,
        max_distance_km: Optional[float] = None,
        weather_condition: str = 'sunny',
        season: str = 'dry',
        is_holiday: bool = False,
        is_peak_season: bool = False,
    ) -> List[Recommendation]:
        """
        Get personalized recommendations for a user.
        
        Args:
            user_id: User identifier
            location: User's current location (lat, lon)
            budget: Budget range (min, max)
            travel_style: Preferred travel style
            group_size: Number of travelers
            max_distance_km: Maximum distance from location
            weather_condition: Current weather condition
            season: Current season
            is_holiday: Whether it's a holiday
            is_peak_season: Whether it's peak tourist season
            
        Returns:
            List of Recommendation objects
        """
        if not self.is_trained or self.api is None:
            raise RuntimeError("System must be trained before generating recommendations")
        
        # Determine user type
        user_type = 'cold_start'
        if user_id in self.user_profiles:
            profile = self.user_profiles[user_id]
            if not profile.is_cold_start:
                user_type = 'regular' if profile.visit_count < 10 else 'frequent'
        
        # Create request
        request = RecommendationRequest(
            user_id=user_id,
            location=location,
            budget=budget,
            travel_style=travel_style,
            group_size=group_size,
            max_distance_km=max_distance_km,
        )
        
        # Create context
        context = Context(
            location=location,
            weather=WeatherInfo(
                condition=weather_condition,
                temperature=28.0,
                humidity=70.0,
                precipitation_chance=0.2 if weather_condition == 'sunny' else 0.7,
            ),
            season=season,
            day_of_week=0,
            is_holiday=is_holiday,
            is_peak_season=is_peak_season,
            user_type=user_type,
        )
        
        # Get recommendations
        start_time = time.time()
        recommendations = self.api.get_recommendations(request, context)
        latency_ms = (time.time() - start_time) * 1000
        
        logger.log_response(
            request_id=f"{user_id}_{time.time()}",
            recommendation_count=len(recommendations),
            total_latency_ms=latency_ms,
        )
        
        return recommendations
    
    def evaluate(
        self,
        test_users: List[str],
        ground_truth: Dict[str, List[str]],
        k: int = 10,
    ) -> Dict[str, float]:
        """
        Evaluate the recommender system on test data.
        
        Args:
            test_users: List of user IDs to evaluate
            ground_truth: Dict mapping user_id to list of relevant destination IDs
            k: Number of recommendations to evaluate
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_trained:
            raise RuntimeError("System must be trained before evaluation")
        
        batch_predictions = []
        batch_ground_truth = []
        
        for user_id in test_users:
            if user_id not in ground_truth:
                continue
            
            # Get recommendations
            recommendations = self.get_recommendations(user_id)
            predictions = [rec.destination_id for rec in recommendations[:k]]
            
            batch_predictions.append(predictions)
            batch_ground_truth.append(ground_truth[user_id])
        
        # Get destination types for diversity calculation
        destination_types = {
            dest_id: features.location_type
            for dest_id, features in self.location_features.items()
        }
        
        # Compute metrics
        metrics = self.evaluation_module.evaluate_batch(
            batch_predictions=batch_predictions,
            batch_ground_truth=batch_ground_truth,
            k=k,
            catalog=set(self.location_features.keys()),
            destination_types=destination_types,
        )
        
        return metrics
    
    def save_models(self, output_dir: str) -> None:
        """
        Save all trained models to disk.
        
        Args:
            output_dir: Directory to save models
        """
        if not self.is_trained:
            raise RuntimeError("System must be trained before saving")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save collaborative filter
        self.model_serializer.save(
            self.collaborative_filter,
            str(output_path / 'collaborative_filter.pkl.gz'),
            metadata={
                'model_type': 'collaborative_filter',
                'n_factors': self.collaborative_filter.n_factors,
                'n_users': len(self.user_ids),
                'n_destinations': len(self.destination_ids),
            }
        )
        
        # Save content-based filter
        self.model_serializer.save(
            self.content_based_filter,
            str(output_path / 'content_based_filter.pkl.gz'),
            metadata={
                'model_type': 'content_based_filter',
                'max_features': self.content_based_filter.max_features,
                'n_destinations': len(self.content_based_filter.destination_ids),
            }
        )
        
        # Save context-aware engine
        self.model_serializer.save(
            self.context_aware_engine,
            str(output_path / 'context_aware_engine.pkl.gz'),
            metadata={
                'model_type': 'context_aware_engine',
                'max_depth': self.context_aware_engine.max_depth,
            }
        )
        
        # Save location features and user profiles
        self.model_serializer.save(
            {
                'location_features': self.location_features,
                'user_profiles': self.user_profiles,
            },
            str(output_path / 'data_cache.pkl.gz'),
            metadata={
                'n_locations': len(self.location_features),
                'n_users': len(self.user_profiles),
            }
        )
        
        logger.logger.info(f"Models saved to {output_dir}")
    
    def load_models(self, model_dir: str) -> None:
        """
        Load trained models from disk.
        
        Args:
            model_dir: Directory containing saved models
        """
        model_path = Path(model_dir)
        
        # Load collaborative filter
        cf_path = model_path / 'collaborative_filter.pkl.gz'
        if cf_path.exists():
            self.collaborative_filter, _ = self.model_serializer.load(str(cf_path))
        
        # Load content-based filter
        cb_path = model_path / 'content_based_filter.pkl.gz'
        if cb_path.exists():
            self.content_based_filter, _ = self.model_serializer.load(str(cb_path))
        
        # Load context-aware engine
        ca_path = model_path / 'context_aware_engine.pkl.gz'
        if ca_path.exists():
            self.context_aware_engine, _ = self.model_serializer.load(str(ca_path))
        
        # Load data cache
        cache_path = model_path / 'data_cache.pkl.gz'
        if cache_path.exists():
            data_cache, _ = self.model_serializer.load(str(cache_path))
            self.location_features = data_cache['location_features']
            self.user_profiles = data_cache['user_profiles']
        
        # Re-initialize ensemble with loaded models
        self.ensemble = EnsembleVotingSystem(
            models={
                'collaborative': self.collaborative_filter,
                'content_based': self.content_based_filter,
                'context_aware': self.context_aware_engine,
            },
            strategy=self.ensemble.strategy,
        )
        
        # Initialize API
        self.api = RecommenderAPI(
            ensemble=self.ensemble,
            optimizer=self.mobile_optimizer,
            destinations=self.location_features,
            user_profiles=self.user_profiles,
        )
        
        self.is_trained = True
        logger.logger.info(f"Models loaded from {model_dir}")
    
    def get_model_sizes(self) -> Dict[str, float]:
        """
        Get sizes of all models in MB.
        
        Returns:
            Dictionary mapping model name to size in MB
        """
        sizes = {}
        
        if self.collaborative_filter.is_fitted:
            sizes['collaborative_filter'] = self.collaborative_filter.get_model_size_mb()
        
        if self.content_based_filter.is_fitted:
            sizes['content_based_filter'] = self.content_based_filter.get_model_size_mb()
        
        if self.context_aware_engine.is_fitted:
            sizes['context_aware_engine'] = self.context_aware_engine.get_model_size_mb()
        
        sizes['total'] = sum(sizes.values())
        
        return sizes
    
    def compress_models(self) -> None:
        """
        Apply compression to all models for mobile deployment.
        """
        if not self.is_trained:
            raise RuntimeError("System must be trained before compression")
        
        self.collaborative_filter.compress()
        self.content_based_filter.compress()
        self.context_aware_engine.compress()
        
        logger.logger.info("Models compressed for mobile deployment")
