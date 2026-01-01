"""Context-aware recommendation engine using decision trees."""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from sklearn.tree import DecisionTreeClassifier
from src.data_models import Context, LocationFeatures, WeatherInfo
from src.logger import get_logger

# Get logger instance
logger = get_logger()


class ContextAwareEngine:
    """Rule-based context processing using decision trees."""
    
    def __init__(self, max_depth: int = 10):
        """
        Initialize context-aware engine.
        
        Args:
            max_depth: Maximum depth of decision tree (default 10)
        """
        self.max_depth = max_depth
        self.tree = None
        self.destination_ids = []
        self.location_types: Dict[str, str] = {}
        self.is_fitted = False
        self.cached_weather: Optional[Dict[str, WeatherInfo]] = {}  # Cache for weather data
        self.default_weather = WeatherInfo(
            condition='sunny',
            temperature=28.0,
            humidity=70.0,
            precipitation_chance=0.2
        )
    
    def fit(self, context_features: pd.DataFrame, ratings: np.ndarray, 
            destination_ids: List[str], location_types: Dict[str, str]) -> None:
        """
        Train decision tree on context-rating relationships.
        
        Args:
            context_features: DataFrame with context features (weather, season, etc.)
            ratings: Array of ratings corresponding to contexts
            destination_ids: List of destination IDs
            location_types: Dictionary mapping destination_id to location type
        """
        if len(context_features) == 0:
            raise ValueError("Context features cannot be empty")
        
        if len(context_features) != len(ratings):
            raise ValueError("Number of context features must match number of ratings")
        
        self.destination_ids = destination_ids
        self.location_types = location_types
        
        # Initialize decision tree classifier
        self.tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            random_state=42,
            min_samples_split=5,
            min_samples_leaf=2
        )
        
        # Convert ratings to binary classes (good/bad) for classification
        # Ratings >= 4 are considered "good" (class 1), < 4 are "bad" (class 0)
        rating_classes = (ratings >= 4.0).astype(int)
        
        # Fit the tree
        self.tree.fit(context_features, rating_classes)
        
        self.is_fitted = True
    
    def predict(self, context: Context, 
                candidate_items: List[str]) -> List[Tuple[str, float]]:
        """
        Score destinations based on current context.
        
        This method applies context-aware scoring rules:
        - Weather-based scoring (deprioritize beach in rain, boost cultural in monsoon)
        - Holiday boost for cultural destinations
        - Season-based adjustments
        
        Requirement 10.3: If weather data is unavailable, uses cached weather data
        or defaults to season-based rules.
        
        Args:
            context: Current context information
            candidate_items: List of destination IDs to score
            
        Returns:
            List of (destination_id, score) tuples, sorted by score descending
        """
        # Apply weather fallback if needed (Requirement 10.3)
        context = self._apply_weather_fallback(context)
        
        predictions = []
        
        for dest_id in candidate_items:
            # Start with base score
            score = 0.5
            
            # Get destination type
            dest_type = self.location_types.get(dest_id, 'unknown')
            
            # Apply weather-based scoring (Requirements 4.2, 4.3)
            score = self._apply_weather_scoring(score, dest_type, context)
            
            # Apply holiday boost (Requirement 4.6)
            score = self._apply_holiday_boost(score, dest_type, context)
            
            # Apply season-based adjustments
            score = self._apply_season_adjustments(score, dest_type, context)
            
            # If tree is fitted, use it for additional scoring
            if self.is_fitted and self.tree is not None:
                context_features = self._extract_context_features(context)
                try:
                    # Get probability of "good" rating
                    tree_proba = self.tree.predict_proba([context_features])[0]
                    if len(tree_proba) > 1:
                        tree_score = tree_proba[1]  # Probability of class 1 (good rating)
                        # Blend tree score with rule-based score (70% rules, 30% tree)
                        score = 0.7 * score + 0.3 * tree_score
                except Exception:
                    # If tree prediction fails, use rule-based score only
                    pass
            
            # Normalize score to [0, 1] range
            score = min(1.0, max(0.0, score))
            
            predictions.append((dest_id, score))
        
        # Sort by score descending
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions
    
    def _apply_weather_fallback(self, context: Context) -> Context:
        """
        Apply weather fallback mechanism.
        
        Requirement 10.3: IF weather API is unavailable, THEN THE Context_Aware_Engine
        SHALL use cached weather data or default to season-based rules.
        
        Args:
            context: Current context
            
        Returns:
            Context with valid weather data
        """
        # Check if weather data is valid
        if context.weather is None or not hasattr(context.weather, 'condition'):
            # Use structured logging (Requirement 10.5)
            logger.log_fallback(
                request_id='context_aware',
                fallback_type='weather',
                reason='Weather data unavailable',
                context={'location': context.location, 'season': context.season}
            )
            
            # Try to get cached weather for this location
            location_key = f"{context.location[0]:.2f},{context.location[1]:.2f}"
            
            if location_key in self.cached_weather:
                logger.logger.info(f"Using cached weather data for {location_key}")
                context.weather = self.cached_weather[location_key]
            else:
                # Use season-based default weather
                logger.logger.info(f"Using season-based default weather for season: {context.season}")
                context.weather = self._get_season_based_weather(context.season)
                
                # Cache this default for future use
                self.cached_weather[location_key] = context.weather
        else:
            # Cache valid weather data
            location_key = f"{context.location[0]:.2f},{context.location[1]:.2f}"
            self.cached_weather[location_key] = context.weather
        
        return context
    
    def _get_season_based_weather(self, season: str) -> WeatherInfo:
        """
        Get default weather based on season.
        
        Args:
            season: Season string (dry, monsoon, inter-monsoon)
            
        Returns:
            WeatherInfo with season-appropriate defaults
        """
        if season == 'monsoon':
            return WeatherInfo(
                condition='rainy',
                temperature=26.0,
                humidity=85.0,
                precipitation_chance=0.7
            )
        elif season == 'dry':
            return WeatherInfo(
                condition='sunny',
                temperature=30.0,
                humidity=60.0,
                precipitation_chance=0.1
            )
        else:  # inter-monsoon
            return WeatherInfo(
                condition='cloudy',
                temperature=28.0,
                humidity=75.0,
                precipitation_chance=0.4
            )
    
    def _apply_weather_scoring(self, base_score: float, dest_type: str, context: Context) -> float:
        """
        Apply weather-based scoring adjustments.
        
        Requirement 4.2: Deprioritize beach destinations in rain
        Requirement 4.3: Boost cultural destinations in monsoon
        
        Args:
            base_score: Starting score
            dest_type: Destination type (beach, cultural, nature, etc.)
            context: Current context
            
        Returns:
            Adjusted score
        """
        score = base_score
        weather_condition = context.weather.condition.lower()
        
        # Deprioritize outdoor/beach destinations in rain (Requirement 4.2)
        if weather_condition in ['rainy', 'stormy']:
            if dest_type in ['beach', 'coastal', 'nature', 'outdoor']:
                score -= 0.3  # Significant penalty for outdoor activities in rain
        
        # Boost indoor/cultural destinations in monsoon season (Requirement 4.3)
        if context.season == 'monsoon':
            if dest_type in ['cultural', 'museum', 'temple', 'indoor', 'urban']:
                score += 0.2  # Boost cultural/indoor destinations
            elif dest_type in ['beach', 'coastal']:
                score -= 0.15  # Slight penalty for beach during monsoon
        
        return score
    
    def _apply_holiday_boost(self, base_score: float, dest_type: str, context: Context) -> float:
        """
        Apply holiday-based scoring boost.
        
        Requirement 4.6: Boost cultural destinations during holidays
        
        Args:
            base_score: Starting score
            dest_type: Destination type
            context: Current context
            
        Returns:
            Adjusted score
        """
        score = base_score
        
        # Boost cultural destinations during holidays (Requirement 4.6)
        if context.is_holiday:
            if dest_type in ['cultural', 'temple', 'historical', 'festival', 'museum']:
                score += 0.25  # Significant boost for cultural sites during holidays
        
        return score
    
    def _apply_season_adjustments(self, base_score: float, dest_type: str, context: Context) -> float:
        """
        Apply season-based scoring adjustments.
        
        Args:
            base_score: Starting score
            dest_type: Destination type
            context: Current context
            
        Returns:
            Adjusted score
        """
        score = base_score
        
        # Boost beach destinations during dry season
        if context.season == 'dry':
            if dest_type in ['beach', 'coastal', 'water_sports']:
                score += 0.15
        
        # Boost nature/wildlife during inter-monsoon
        if context.season == 'inter-monsoon':
            if dest_type in ['nature', 'wildlife', 'safari', 'hiking']:
                score += 0.1
        
        return score
    
    def _extract_context_features(self, context: Context) -> List[float]:
        """
        Extract numerical features from context for tree prediction.
        
        Args:
            context: Context object
            
        Returns:
            List of numerical features
        """
        features = []
        
        # Weather features
        features.append(1.0 if context.weather.condition == 'sunny' else 0.0)
        features.append(1.0 if context.weather.condition == 'rainy' else 0.0)
        features.append(1.0 if context.weather.condition == 'stormy' else 0.0)
        features.append(context.weather.temperature)
        features.append(context.weather.humidity)
        features.append(context.weather.precipitation_chance)
        
        # Season features
        features.append(1.0 if context.season == 'dry' else 0.0)
        features.append(1.0 if context.season == 'monsoon' else 0.0)
        features.append(1.0 if context.season == 'inter-monsoon' else 0.0)
        
        # Temporal features
        features.append(float(context.day_of_week))
        features.append(1.0 if context.is_holiday else 0.0)
        features.append(1.0 if context.is_peak_season else 0.0)
        
        return features
    
    def get_context_type(self, context: Context) -> str:
        """
        Classify context for weight adjustment.
        
        Args:
            context: Current context
            
        Returns:
            Context type string (e.g., 'cold_start', 'weather_critical', 'peak_season')
        """
        context_types = []
        
        # Check for cold start
        if context.user_type == 'cold_start':
            context_types.append('cold_start')
        
        # Check for weather-critical conditions
        weather_condition = context.weather.condition.lower()
        if weather_condition in ['rainy', 'stormy'] or context.season == 'monsoon':
            context_types.append('weather_critical')
        
        # Check for peak season
        if context.is_peak_season:
            context_types.append('peak_season')
        
        # Return primary context type (prioritize in order)
        if 'weather_critical' in context_types:
            return 'weather_critical'
        elif 'cold_start' in context_types:
            return 'cold_start'
        elif 'peak_season' in context_types:
            return 'peak_season'
        else:
            return 'normal'
    
    def compress(self, target_size_mb: float = 3.0) -> None:
        """
        Apply compression for mobile deployment.
        
        Args:
            target_size_mb: Target model size in MB (default 3.0)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before compression")
        
        # Decision trees are already quite compact
        # Additional compression could involve pruning the tree further
        # For now, the max_depth constraint handles size
        pass
    
    def get_model_size_mb(self) -> float:
        """
        Calculate current model size in MB.
        
        Returns:
            Model size in megabytes
        """
        if not self.is_fitted or self.tree is None:
            return 0.0
        
        # Estimate tree size based on number of nodes
        # Each node stores split info, thresholds, etc.
        n_nodes = self.tree.tree_.node_count
        bytes_per_node = 100  # Approximate
        
        size_bytes = n_nodes * bytes_per_node
        
        # Add overhead for other attributes
        size_bytes += len(self.destination_ids) * 50
        size_bytes += len(self.location_types) * 60
        
        return size_bytes / (1024 * 1024)  # Convert to MB
