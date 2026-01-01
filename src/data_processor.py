"""Data processing module for tourism recommender system."""

import ast
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data_models import LocationFeatures, UserProfile
from src.logger import get_logger

# Get logger instance
logger = get_logger()


class DataProcessor:
    """Handles raw data transformation and feature engineering."""
    
    def __init__(self):
        self.tfidf_vectorizer = None
        self.location_features_cache: Dict[str, LocationFeatures] = {}
        self.user_profiles_cache: Dict[str, UserProfile] = {}
        self.invalid_ratings_count = 0  # Track rejected ratings
    
    def load_reviews(self, file_path: str) -> pd.DataFrame:
        """
        Load and parse review CSV files.
        
        Handles both Reviews.csv format and reviews_2/*.csv format.
        Validates and rejects ratings outside [1, 5] range.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            DataFrame with standardized columns (invalid ratings removed)
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        df = pd.read_csv(file_path)
        
        # Detect format based on columns
        if 'Location_Name' in df.columns:
            # Reviews.csv format
            parsed_df = self._parse_reviews_format(df)
        elif 'placeInfo' in df.columns:
            # reviews_2/*.csv format
            parsed_df = self._parse_reviews2_format(df)
        else:
            raise ValueError(f"Unknown CSV format in {file_path}")
        
        # Validate and filter ratings (Requirement 10.1)
        parsed_df = self._validate_ratings(parsed_df)
        
        return parsed_df
    
    def _validate_ratings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate ratings and reject entries outside [1, 5] range.
        
        Requirement 10.1: IF input data contains invalid ratings (outside 1-5 range),
        THEN THE Data_Processor SHALL reject the invalid entries and log warnings.
        
        Args:
            df: DataFrame with rating column
            
        Returns:
            DataFrame with only valid ratings
        """
        if 'rating' not in df.columns:
            return df
        
        # Count invalid ratings before filtering
        invalid_mask = (df['rating'] < 1.0) | (df['rating'] > 5.0) | df['rating'].isna()
        n_invalid = invalid_mask.sum()
        
        if n_invalid > 0:
            self.invalid_ratings_count += n_invalid
            
            # Use structured logging (Requirement 10.5)
            logger.log_validation_error(
                request_id='data_processing',
                validation_type='rating',
                invalid_count=n_invalid,
                details={
                    'total_ratings': len(df),
                    'invalid_ratings': n_invalid,
                    'sample_invalid_values': df[invalid_mask]['rating'].head(5).tolist()
                }
            )
        
        # Filter to keep only valid ratings
        valid_df = df[~invalid_mask].copy()
        
        return valid_df
    
    def _parse_reviews_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse Reviews.csv format."""
        standardized = pd.DataFrame({
            'destination_id': df['Location_Name'].astype(str),
            'destination_name': df['Location_Name'],
            'city': df['Located_City'],
            'location_string': df['Location'],
            'location_type': df['Location_Type'],
            'user_id': df['User_ID'],
            'rating': df['Rating'],
            'travel_date': pd.to_datetime(df['Travel_Date'], errors='coerce'),
            'published_date': pd.to_datetime(df['Published_Date'], errors='coerce'),
            'title': df['Title'],
            'text': df['Text'],
            'latitude': np.nan,  # Not available in this format
            'longitude': np.nan,
        })
        
        return standardized
    
    def _parse_reviews2_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse reviews_2/*.csv format with placeInfo."""
        # Extract placeInfo dictionary
        place_info_list = []
        
        for idx, row in df.iterrows():
            try:
                if isinstance(row['placeInfo'], str):
                    place_info = ast.literal_eval(row['placeInfo'])
                elif isinstance(row['placeInfo'], dict):
                    place_info = row['placeInfo']
                else:
                    # Handle NaN or other non-dict/non-string values
                    place_info = {}
                place_info_list.append(place_info)
            except (ValueError, SyntaxError):
                place_info_list.append({})
        
        # Extract fields from placeInfo
        destination_ids = [p.get('id', '') if isinstance(p, dict) else '' for p in place_info_list]
        destination_names = [p.get('name', '') if isinstance(p, dict) else '' for p in place_info_list]
        cities = [p.get('addressObj', {}).get('city', '') if isinstance(p, dict) and isinstance(p.get('addressObj'), dict) else '' 
                  for p in place_info_list]
        location_strings = [p.get('locationString', '') if isinstance(p, dict) else '' for p in place_info_list]
        latitudes = [p.get('latitude', np.nan) if isinstance(p, dict) else np.nan for p in place_info_list]
        longitudes = [p.get('longitude', np.nan) if isinstance(p, dict) else np.nan for p in place_info_list]
        
        # Infer location type from file name or destination name
        location_type = self._infer_location_type(destination_names[0] if destination_names else '')
        
        standardized = pd.DataFrame({
            'destination_id': destination_ids,
            'destination_name': destination_names,
            'city': cities,
            'location_string': location_strings,
            'location_type': location_type,
            'user_id': df['id'].astype(str),  # Using review id as user_id proxy
            'rating': df['rating'],
            'travel_date': pd.to_datetime(df['travelDate'], errors='coerce'),
            'published_date': pd.to_datetime(df['publishedDate'], errors='coerce'),
            'title': df['title'],
            'text': df['text'],
            'latitude': latitudes,
            'longitude': longitudes,
        })
        
        return standardized
    
    def _infer_location_type(self, name: str) -> str:
        """Infer location type from destination name."""
        name_lower = name.lower()
        
        if any(word in name_lower for word in ['beach', 'bay', 'surf', 'coastal']):
            return 'beach'
        elif any(word in name_lower for word in ['temple', 'cultural', 'heritage', 'museum', 'palace']):
            return 'cultural'
        elif any(word in name_lower for word in ['park', 'safari', 'wildlife', 'nature', 'reserve']):
            return 'nature'
        elif any(word in name_lower for word in ['city', 'urban', 'town']):
            return 'urban'
        else:
            return 'other'
    
    def extract_location_features(self, df: pd.DataFrame) -> Dict[str, LocationFeatures]:
        """
        Extract location metadata from review dataframe.
        
        Args:
            df: Standardized review dataframe
            
        Returns:
            Dictionary mapping destination_id to LocationFeatures
        """
        location_features = {}
        
        # Group by destination
        grouped = df.groupby('destination_id')
        
        for dest_id, group in grouped:
            # Get most common values
            name = group['destination_name'].mode()[0] if not group['destination_name'].mode().empty else str(dest_id)
            city = group['city'].mode()[0] if not group['city'].mode().empty else 'Unknown'
            location_type = group['location_type'].mode()[0] if not group['location_type'].mode().empty else 'other'
            
            # Get coordinates (use first non-null value)
            latitude = group['latitude'].dropna().iloc[0] if not group['latitude'].dropna().empty else 0.0
            longitude = group['longitude'].dropna().iloc[0] if not group['longitude'].dropna().empty else 0.0
            
            # Calculate statistics
            avg_rating = group['rating'].mean()
            review_count = len(group)
            
            # Infer price range from rating distribution
            price_range = self._infer_price_range(avg_rating, review_count)
            
            # Extract attributes from text
            attributes = self._extract_attributes(group['text'].tolist(), location_type)
            
            location_features[str(dest_id)] = LocationFeatures(
                destination_id=str(dest_id),
                name=name,
                city=city,
                latitude=float(latitude),
                longitude=float(longitude),
                location_type=location_type,
                avg_rating=float(avg_rating),
                review_count=int(review_count),
                price_range=price_range,
                attributes=attributes
            )
        
        self.location_features_cache = location_features
        return location_features
    
    def _infer_price_range(self, avg_rating: float, review_count: int) -> str:
        """Infer price range from rating and popularity."""
        if avg_rating >= 4.5 and review_count > 100:
            return 'luxury'
        elif avg_rating >= 4.0:
            return 'mid-range'
        else:
            return 'budget'
    
    def _extract_attributes(self, texts: List[str], location_type: str) -> List[str]:
        """Extract attributes from review texts."""
        attributes = set()
        
        # Common attribute keywords
        attribute_keywords = {
            'beach': ['surfing', 'swimming', 'snorkeling', 'diving', 'sunset'],
            'cultural': ['historical', 'temple', 'heritage', 'ancient', 'traditional'],
            'nature': ['wildlife', 'safari', 'hiking', 'scenic', 'birds'],
            'urban': ['shopping', 'dining', 'nightlife', 'modern', 'entertainment'],
        }
        
        keywords = attribute_keywords.get(location_type, [])
        
        # Combine all texts
        combined_text = ' '.join([str(t).lower() for t in texts if pd.notna(t)])
        
        for keyword in keywords:
            if keyword in combined_text:
                attributes.add(keyword)
        
        return list(attributes)[:5]  # Limit to 5 attributes
    
    def build_user_profiles(self, df: pd.DataFrame) -> Dict[str, UserProfile]:
        """
        Create user profiles from rating history.
        
        Note: If a user has multiple reviews for the same destination,
        only the most recent one is kept in rating_history.
        
        Args:
            df: Standardized review dataframe
            
        Returns:
            Dictionary mapping user_id to UserProfile
        """
        # First deduplicate to ensure one review per user-destination pair
        df_deduped = self.deduplicate_reviews(df)
        
        user_profiles = {}
        
        # Group by user
        grouped = df_deduped.groupby('user_id')
        
        for user_id, group in grouped:
            # Build rating history
            rating_history = {}
            for _, row in group.iterrows():
                dest_id = str(row['destination_id'])
                rating = float(row['rating'])
                rating_history[dest_id] = rating
            
            # Calculate statistics
            avg_rating = group['rating'].mean()
            visit_count = len(group)
            is_cold_start = visit_count < 5
            
            # Determine preferred types
            if 'location_type' in group.columns:
                type_counts = group['location_type'].value_counts()
                preferred_types = type_counts.head(3).index.tolist()
            else:
                preferred_types = []
            
            user_profiles[str(user_id)] = UserProfile(
                user_id=str(user_id),
                rating_history=rating_history,
                preferred_types=preferred_types,
                avg_rating=float(avg_rating),
                visit_count=int(visit_count),
                is_cold_start=is_cold_start
            )
        
        self.user_profiles_cache = user_profiles
        return user_profiles
    
    def generate_tfidf_embeddings(self, descriptions: List[str], max_features: int = 500) -> np.ndarray:
        """
        Generate TF-IDF embeddings for destination descriptions.
        
        Args:
            descriptions: List of text descriptions
            max_features: Maximum number of features (default 500)
            
        Returns:
            Normalized TF-IDF matrix (n_docs x max_features)
        """
        if not descriptions:
            return np.array([])
        
        # Initialize vectorizer if not already done
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words='english',
                lowercase=True,
                norm='l2',  # L2 normalization
                min_df=1,  # Allow terms that appear in at least 1 document
                token_pattern=r'(?u)\b\w+\b'  # Match single characters too
            )
        
        try:
            # Fit and transform
            embeddings = self.tfidf_vectorizer.fit_transform(descriptions)
        except ValueError as e:
            # Handle empty vocabulary case (all stop words or empty strings)
            if "empty vocabulary" in str(e):
                # Return zero vectors with shape (n_docs, 1)
                return np.zeros((len(descriptions), 1))
            raise
        
        # Convert to dense array and ensure L2 normalization
        embeddings_dense = embeddings.toarray()
        
        # Normalize each row to unit length
        norms = np.linalg.norm(embeddings_dense, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        embeddings_normalized = embeddings_dense / norms
        
        return embeddings_normalized
    
    def build_rating_matrix(self, df: pd.DataFrame) -> Tuple[sparse.csr_matrix, List[str], List[str]]:
        """
        Construct user-item rating matrix with normalization.
        
        Note: Automatically deduplicates reviews to ensure one rating per user-destination pair.
        
        Args:
            df: Standardized review dataframe
            
        Returns:
            Tuple of (sparse CSR matrix, user_ids list, destination_ids list)
        """
        # Deduplicate first to avoid summing duplicate ratings
        df_deduped = self.deduplicate_reviews(df)
        
        # Get unique users and destinations
        user_ids = sorted(df_deduped['user_id'].unique())
        destination_ids = sorted(df_deduped['destination_id'].unique())
        
        # Create mappings
        user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
        dest_to_idx = {did: idx for idx, did in enumerate(destination_ids)}
        
        # Build sparse matrix
        rows = []
        cols = []
        data = []
        
        for _, row in df_deduped.iterrows():
            user_idx = user_to_idx[row['user_id']]
            dest_idx = dest_to_idx[row['destination_id']]
            rating = row['rating']
            
            # Ensure rating is in [1, 5] range
            if 1 <= rating <= 5:
                rows.append(user_idx)
                cols.append(dest_idx)
                data.append(float(rating))
        
        # Create sparse matrix
        rating_matrix = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(len(user_ids), len(destination_ids)),
            dtype=np.float32
        )
        
        return rating_matrix, user_ids, destination_ids
    
    def deduplicate_reviews(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Deduplicate reviews by user-destination pair, keeping most recent.
        
        Args:
            df: Review dataframe
            
        Returns:
            Deduplicated dataframe
        """
        # Ensure published_date is timezone-naive for consistent sorting
        if 'published_date' in df.columns:
            # Convert to timezone-naive if timezone-aware
            if df['published_date'].dtype == 'datetime64[ns, UTC]' or \
               (hasattr(df['published_date'].dtype, 'tz') and df['published_date'].dtype.tz is not None):
                df = df.copy()
                df['published_date'] = df['published_date'].dt.tz_localize(None)
            elif df['published_date'].apply(lambda x: hasattr(x, 'tzinfo') and x.tzinfo is not None if pd.notna(x) else False).any():
                df = df.copy()
                df['published_date'] = pd.to_datetime(df['published_date'], utc=True).dt.tz_localize(None)
        
        # Sort by published_date descending (most recent first)
        df_sorted = df.sort_values('published_date', ascending=False, na_position='last')
        
        # Drop duplicates, keeping first (most recent)
        df_deduped = df_sorted.drop_duplicates(
            subset=['user_id', 'destination_id'],
            keep='first'
        )
        
        return df_deduped.reset_index(drop=True)
