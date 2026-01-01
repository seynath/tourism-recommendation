"""Collaborative filtering model using SVD for tourism recommendations."""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds
from typing import List, Tuple, Dict, Optional


class CollaborativeFilter:
    """SVD-based collaborative filtering with mobile optimization."""
    
    def __init__(self, n_factors: int = 50, n_epochs: int = 20):
        """
        Initialize collaborative filter.
        
        Args:
            n_factors: Number of latent factors for SVD (default 50)
            n_epochs: Number of training epochs (not used for SVD, kept for interface compatibility)
        """
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.user_factors = None
        self.item_factors = None
        self.global_mean = 0.0
        self.user_ids = []
        self.destination_ids = []
        self.user_rating_counts: Dict[str, int] = {}
        self.is_fitted = False
    
    def fit(self, rating_matrix: sparse.csr_matrix, user_ids: List[str], destination_ids: List[str]) -> None:
        """
        Train SVD model on rating matrix.
        
        Only destinations with reviews are included in the collaborative filter.
        Destinations with no reviews are excluded (Requirement 10.2).
        
        Args:
            rating_matrix: Sparse user-item rating matrix (users x destinations)
            user_ids: List of user IDs corresponding to matrix rows
            destination_ids: List of destination IDs corresponding to matrix columns
        """
        if rating_matrix.shape[0] == 0 or rating_matrix.shape[1] == 0:
            raise ValueError("Rating matrix cannot be empty")
        
        self.user_ids = user_ids
        self.destination_ids = destination_ids
        
        # Calculate global mean from non-zero entries
        non_zero_ratings = rating_matrix.data
        if len(non_zero_ratings) > 0:
            self.global_mean = np.mean(non_zero_ratings)
        else:
            self.global_mean = 3.0  # Default middle rating
        
        # Count ratings per user for confidence scoring
        self.user_rating_counts = {}
        for i, user_id in enumerate(user_ids):
            row = rating_matrix.getrow(i)
            self.user_rating_counts[user_id] = row.nnz
        
        # Perform SVD decomposition
        # Use min of (n_factors, min(matrix dimensions) - 1) to avoid errors
        k = min(self.n_factors, min(rating_matrix.shape) - 1)
        
        if k < 1:
            # Matrix too small for SVD, use simple mean-based approach
            self.user_factors = np.zeros((rating_matrix.shape[0], 1))
            self.item_factors = np.zeros((rating_matrix.shape[1], 1))
            self.is_fitted = True
            return
        
        try:
            # Perform truncated SVD
            U, sigma, Vt = svds(rating_matrix.astype(np.float64), k=k)
            
            # Store factors
            # U: users x k, Vt: k x items
            # Multiply U by sigma to get user factors
            self.user_factors = U * sigma
            self.item_factors = Vt.T  # Transpose to get items x k
            
            self.is_fitted = True
            
        except Exception as e:
            # Fallback to simple mean-based approach if SVD fails
            print(f"SVD failed: {e}. Using fallback approach.")
            self.user_factors = np.zeros((rating_matrix.shape[0], 1))
            self.item_factors = np.zeros((rating_matrix.shape[1], 1))
            self.is_fitted = True
    
    def predict(self, user_id: str, candidate_items: List[str]) -> List[Tuple[str, float]]:
        """
        Generate predictions for user-item pairs.
        
        Requirement 10.2: Destinations with no reviews are excluded from collaborative
        filtering predictions (they won't be in self.destination_ids).
        
        Args:
            user_id: User ID to generate predictions for
            candidate_items: List of destination IDs to score
            
        Returns:
            List of (destination_id, predicted_score) tuples, sorted by score descending
            Note: Destinations not in training data (no reviews) will receive global mean score
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
        
        # Check if user exists in training data
        if user_id not in self.user_ids:
            # Cold start user - return global mean for all items
            predictions = [(item_id, self.global_mean) for item_id in candidate_items]
            return sorted(predictions, key=lambda x: x[1], reverse=True)
        
        # Get user index
        user_idx = self.user_ids.index(user_id)
        user_vector = self.user_factors[user_idx]
        
        # Generate predictions for candidate items
        predictions = []
        for item_id in candidate_items:
            if item_id in self.destination_ids:
                # Destination has reviews - use collaborative filtering
                item_idx = self.destination_ids.index(item_id)
                item_vector = self.item_factors[item_idx]
                
                # Predicted rating = dot product of user and item factors
                predicted_score = np.dot(user_vector, item_vector)
                
                # Clip to valid rating range [1, 5]
                predicted_score = np.clip(predicted_score, 1.0, 5.0)
                
                predictions.append((item_id, float(predicted_score)))
            else:
                # Destination has no reviews - excluded from CF (Requirement 10.2)
                # Return global mean as fallback
                predictions.append((item_id, self.global_mean))
        
        # Sort by predicted score descending
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions
    
    def get_confidence(self, user_id: str) -> float:
        """
        Return confidence score based on user's rating history.
        
        Args:
            user_id: User ID to get confidence for
            
        Returns:
            Confidence score between 0 and 1
            - Returns 0 for cold start users (< 5 ratings)
            - Returns scaled confidence for other users
        """
        if user_id not in self.user_rating_counts:
            # Unknown user - cold start
            return 0.0
        
        rating_count = self.user_rating_counts[user_id]
        
        # Cold start threshold: < 5 ratings
        if rating_count < 5:
            return 0.0
        
        # Scale confidence based on rating count
        # Use sigmoid-like function: confidence = min(1.0, rating_count / 20)
        # This gives:
        # - 5 ratings: 0.25 confidence
        # - 10 ratings: 0.5 confidence
        # - 20+ ratings: 1.0 confidence
        confidence = min(1.0, rating_count / 20.0)
        
        return confidence
    
    def compress(self, target_size_mb: float = 10.0) -> None:
        """
        Apply quantization and pruning for mobile deployment.
        
        Args:
            target_size_mb: Target model size in MB (default 10.0)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before compression")
        
        # Convert to float16 for quantization
        if self.user_factors is not None:
            self.user_factors = self.user_factors.astype(np.float16)
        
        if self.item_factors is not None:
            self.item_factors = self.item_factors.astype(np.float16)
        
        # TODO: Implement pruning if needed to meet target size
        # For now, quantization should be sufficient
    
    def get_model_size_mb(self) -> float:
        """
        Calculate current model size in MB.
        
        Returns:
            Model size in megabytes
        """
        if not self.is_fitted:
            return 0.0
        
        size_bytes = 0
        
        if self.user_factors is not None:
            size_bytes += self.user_factors.nbytes
        
        if self.item_factors is not None:
            size_bytes += self.item_factors.nbytes
        
        # Add overhead for other attributes
        size_bytes += len(self.user_ids) * 50  # Approximate string size
        size_bytes += len(self.destination_ids) * 50
        size_bytes += len(self.user_rating_counts) * 12  # Dict overhead
        
        return size_bytes / (1024 * 1024)  # Convert to MB
