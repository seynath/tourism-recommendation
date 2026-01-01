"""Content-based filtering model using TF-IDF for tourism recommendations."""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedFilter:
    """TF-IDF based content filtering with pre-computed similarities."""
    
    def __init__(self, max_features: int = 500):
        """
        Initialize content-based filter.
        
        Args:
            max_features: Maximum number of TF-IDF features (default 500)
        """
        self.max_features = max_features
        self.vectorizer = None
        self.embeddings = None
        self.similarity_matrix = None
        self.destination_ids = []
        self.destination_attributes: Dict[str, List[str]] = {}
        self.location_types: Dict[str, str] = {}
        self.is_fitted = False
    
    def fit(self, descriptions: List[str], attributes: Dict[str, List[str]], 
            location_types: Optional[Dict[str, str]] = None) -> None:
        """
        Build TF-IDF embeddings and similarity matrix.
        
        Requirement 10.2: Content-based filter includes ALL destinations,
        even those with no reviews, since it relies on destination attributes
        rather than user ratings.
        
        Args:
            descriptions: List of destination descriptions (one per destination)
            attributes: Dictionary mapping destination_id to list of attributes
            location_types: Optional dictionary mapping destination_id to location type
                          (beach, cultural, nature, adventure, urban, etc.)
        """
        if not descriptions:
            raise ValueError("Descriptions list cannot be empty")
        
        if len(descriptions) != len(attributes):
            raise ValueError("Number of descriptions must match number of attribute entries")
        
        # Store destination IDs and attributes
        self.destination_ids = list(attributes.keys())
        self.destination_attributes = attributes
        self.location_types = location_types or {}
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words='english',
            lowercase=True,
            norm='l2',  # L2 normalization
            min_df=1,
            token_pattern=r'(?u)\b\w+\b'
        )
        
        try:
            # Fit and transform descriptions
            self.embeddings = self.vectorizer.fit_transform(descriptions)
        except ValueError as e:
            # Handle empty vocabulary case
            if "empty vocabulary" in str(e):
                # Create zero embeddings
                self.embeddings = np.zeros((len(descriptions), 1))
            else:
                raise
        
        # Convert to dense array if sparse
        if hasattr(self.embeddings, 'toarray'):
            self.embeddings = self.embeddings.toarray()
        
        # Ensure L2 normalization
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        self.embeddings = self.embeddings / norms
        
        # Pre-compute similarity matrix
        self.similarity_matrix = cosine_similarity(self.embeddings)
        
        # Clip to [0, 1] range to handle floating-point precision issues
        self.similarity_matrix = np.clip(self.similarity_matrix, 0.0, 1.0)
        
        self.is_fitted = True
    
    def predict(self, user_preferences: Dict[str, Any], 
                candidate_items: List[str]) -> List[Tuple[str, float]]:
        """
        Rank destinations by preference match.
        
        This method implements preference matching by scoring destinations based on
        how well they match the user's stated preferences for location types and attributes.
        Destinations with better matches receive higher scores.
        
        Args:
            user_preferences: Dictionary containing user preferences
                - 'preferred_types': List of preferred location types (beach, cultural, nature, adventure)
                - 'preferred_attributes': List of preferred attributes (optional)
            candidate_items: List of destination IDs to score
            
        Returns:
            List of (destination_id, score) tuples, sorted by score descending
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
        
        predictions = []
        
        # Extract preferences
        preferred_types = user_preferences.get('preferred_types', [])
        preferred_attributes = user_preferences.get('preferred_attributes', [])
        
        for dest_id in candidate_items:
            if dest_id not in self.destination_ids:
                # Unknown destination - assign neutral score
                predictions.append((dest_id, 0.5))
                continue
            
            # Calculate attribute match score
            dest_attributes = self.destination_attributes.get(dest_id, [])
            dest_type = self.location_types.get(dest_id, '')
            
            # Score based on attribute and type overlap
            score = 0.0
            
            # Check preferred types match (primary factor)
            if preferred_types:
                # Check if destination type matches any preferred type
                type_match = dest_type in preferred_types
                
                # Also check if type appears in attributes
                type_in_attrs = any(ptype in dest_attributes for ptype in preferred_types)
                
                if type_match or type_in_attrs:
                    score += 0.5  # 50% weight for type match
            
            # Check preferred attributes match (secondary factor)
            if preferred_attributes:
                matching_attrs = set(dest_attributes) & set(preferred_attributes)
                if len(preferred_attributes) > 0:
                    attr_score = len(matching_attrs) / len(preferred_attributes)
                    score += attr_score * 0.5  # 50% weight for attributes
            
            # If no preferences specified, use neutral score
            if not preferred_types and not preferred_attributes:
                score = 0.5
            
            # Normalize score to [0, 1] range
            score = min(1.0, max(0.0, score))
            
            predictions.append((dest_id, score))
        
        # Sort by score descending (higher scores = better matches)
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions
    
    def get_similar_destinations(self, destination_id: str, k: int = 10) -> List[str]:
        """
        Find k most similar destinations.
        
        Args:
            destination_id: Reference destination ID
            k: Number of similar destinations to return
            
        Returns:
            List of destination IDs sorted by similarity (most similar first)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before finding similar destinations")
        
        if destination_id not in self.destination_ids:
            return []
        
        # Get destination index
        dest_idx = self.destination_ids.index(destination_id)
        
        # Get similarity scores for this destination
        similarities = self.similarity_matrix[dest_idx]
        
        # Get indices of top-k most similar (excluding self)
        # Sort in descending order
        similar_indices = np.argsort(similarities)[::-1]
        
        # Filter out self and take top k
        similar_destinations = []
        for idx in similar_indices:
            if idx != dest_idx:
                similar_destinations.append(self.destination_ids[idx])
                if len(similar_destinations) >= k:
                    break
        
        return similar_destinations
    
    def get_similarity(self, dest_id_1: str, dest_id_2: str) -> float:
        """
        Get cosine similarity between two destinations.
        
        Args:
            dest_id_1: First destination ID
            dest_id_2: Second destination ID
            
        Returns:
            Cosine similarity score in range [0, 1]
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before computing similarity")
        
        if dest_id_1 not in self.destination_ids or dest_id_2 not in self.destination_ids:
            return 0.0
        
        idx_1 = self.destination_ids.index(dest_id_1)
        idx_2 = self.destination_ids.index(dest_id_2)
        
        similarity = self.similarity_matrix[idx_1, idx_2]
        
        # Ensure in [0, 1] range (cosine similarity can be negative, but with L2 norm it should be [0, 1])
        return float(np.clip(similarity, 0.0, 1.0))
    
    def compress(self, target_size_mb: float = 5.0) -> None:
        """
        Apply compression for mobile deployment.
        
        Args:
            target_size_mb: Target model size in MB (default 5.0)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before compression")
        
        # Convert embeddings to float16 for quantization
        if self.embeddings is not None:
            self.embeddings = self.embeddings.astype(np.float16)
        
        # Convert similarity matrix to float16
        if self.similarity_matrix is not None:
            self.similarity_matrix = self.similarity_matrix.astype(np.float16)
    
    def get_model_size_mb(self) -> float:
        """
        Calculate current model size in MB.
        
        Returns:
            Model size in megabytes
        """
        if not self.is_fitted:
            return 0.0
        
        size_bytes = 0
        
        if self.embeddings is not None:
            size_bytes += self.embeddings.nbytes
        
        if self.similarity_matrix is not None:
            size_bytes += self.similarity_matrix.nbytes
        
        # Add overhead for other attributes
        size_bytes += len(self.destination_ids) * 50  # Approximate string size
        
        for dest_id, attrs in self.destination_attributes.items():
            size_bytes += len(dest_id) * 2
            size_bytes += sum(len(attr) * 2 for attr in attrs)
        
        return size_bytes / (1024 * 1024)  # Convert to MB
