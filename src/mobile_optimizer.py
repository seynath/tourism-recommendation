"""Mobile optimization module for tourism recommender system."""

import time
import numpy as np
from collections import OrderedDict
from typing import Any, Optional, Dict, List
from dataclasses import dataclass


class LRUCache:
    """
    Least Recently Used (LRU) cache implementation.
    
    Evicts the least recently used item when capacity is exceeded.
    """
    
    def __init__(self, maxsize: int = 100):
        """
        Initialize LRU cache.
        
        Args:
            maxsize: Maximum number of items to cache
        """
        self.maxsize = maxsize
        self.cache: OrderedDict = OrderedDict()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        if key not in self.cache:
            return None
        
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key: str, value: Any) -> None:
        """
        Put item in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        if key in self.cache:
            # Update existing key and move to end
            self.cache.move_to_end(key)
        else:
            # Add new key
            if len(self.cache) >= self.maxsize:
                # Evict least recently used (first item)
                self.cache.popitem(last=False)
        
        self.cache[key] = value
    
    def clear(self) -> None:
        """Clear all items from cache."""
        self.cache.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)


@dataclass
class CachedItem:
    """Item stored in TTL cache with expiry time."""
    value: Any
    expiry_time: float


class TTLCache:
    """
    Time-To-Live (TTL) cache implementation.
    
    Items expire after a specified time period.
    """
    
    def __init__(self, maxsize: int = 20, ttl: int = 3600):
        """
        Initialize TTL cache.
        
        Args:
            maxsize: Maximum number of items to cache
            ttl: Time-to-live in seconds (default 1 hour)
        """
        self.maxsize = maxsize
        self.ttl = ttl
        self.cache: Dict[str, CachedItem] = {}
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get item from cache if not expired.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found or expired
        """
        if key not in self.cache:
            return None
        
        item = self.cache[key]
        current_time = time.time()
        
        # Check if expired
        if current_time >= item.expiry_time:
            # Remove expired item
            del self.cache[key]
            return None
        
        return item.value
    
    def put(self, key: str, value: Any) -> None:
        """
        Put item in cache with TTL.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        # Clean up expired items if at capacity
        if len(self.cache) >= self.maxsize:
            self._cleanup_expired()
            
            # If still at capacity, remove oldest item
            if len(self.cache) >= self.maxsize:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
        
        # Add new item with expiry time
        expiry_time = time.time() + self.ttl
        self.cache[key] = CachedItem(value=value, expiry_time=expiry_time)
    
    def _cleanup_expired(self) -> None:
        """Remove all expired items from cache."""
        current_time = time.time()
        expired_keys = [
            key for key, item in self.cache.items()
            if current_time >= item.expiry_time
        ]
        for key in expired_keys:
            del self.cache[key]
    
    def clear(self) -> None:
        """Clear all items from cache."""
        self.cache.clear()
    
    def size(self) -> int:
        """Get current cache size (including expired items)."""
        return len(self.cache)


class MobileOptimizer:
    """Handles model compression and caching for mobile deployment."""
    
    def __init__(self):
        self.destination_cache = LRUCache(maxsize=100)
        self.weather_cache = TTLCache(maxsize=20, ttl=3600)
        self.user_cache = LRUCache(maxsize=50)
    
    def quantize_model(self, model: Any, bit_width: int = 8) -> Any:
        """
        Convert model to lower precision format.
        
        Supports quantization of numpy arrays and model parameters from float32 to int8 or float16.
        
        Args:
            model: Model to quantize (can be numpy array, dict of arrays, or object with weights)
            bit_width: Target bit width (8 or 16)
            
        Returns:
            Quantized model
        """
        if bit_width not in [8, 16]:
            raise ValueError(f"bit_width must be 8 or 16, got {bit_width}")
        
        # Handle numpy arrays
        if isinstance(model, np.ndarray):
            return self._quantize_array(model, bit_width)
        
        # Handle dictionaries of arrays (common for model parameters)
        elif isinstance(model, dict):
            quantized = {}
            for key, value in model.items():
                if isinstance(value, np.ndarray):
                    quantized[key] = self._quantize_array(value, bit_width)
                else:
                    quantized[key] = value
            return quantized
        
        # Handle objects with 'weights' or 'parameters' attributes
        elif hasattr(model, 'user_factors') and hasattr(model, 'item_factors'):
            # Collaborative filter model
            quantized_model = type(model).__new__(type(model))
            quantized_model.__dict__.update(model.__dict__)
            if model.user_factors is not None:
                quantized_model.user_factors = self._quantize_array(model.user_factors, bit_width)
            if model.item_factors is not None:
                quantized_model.item_factors = self._quantize_array(model.item_factors, bit_width)
            return quantized_model
        
        # Handle objects with embeddings
        elif hasattr(model, 'embeddings'):
            quantized_model = type(model).__new__(type(model))
            quantized_model.__dict__.update(model.__dict__)
            if model.embeddings is not None:
                quantized_model.embeddings = self._quantize_array(model.embeddings, bit_width)
            return quantized_model
        
        else:
            # Return as-is if we don't know how to quantize
            return model
    
    def _quantize_array(self, array: np.ndarray, bit_width: int) -> np.ndarray:
        """
        Quantize a numpy array to lower precision.
        
        Args:
            array: Array to quantize
            bit_width: Target bit width (8 or 16)
            
        Returns:
            Quantized array
        """
        if bit_width == 16:
            # Convert to float16
            return array.astype(np.float16)
        
        elif bit_width == 8:
            # Convert to int8 with scaling
            # Find min and max values
            min_val = array.min()
            max_val = array.max()
            
            # Avoid division by zero
            if max_val == min_val:
                return np.zeros_like(array, dtype=np.int8)
            
            # Scale to [-127, 127] range for int8
            scale = 127.0 / max(abs(min_val), abs(max_val))
            quantized = np.round(array * scale).astype(np.int8)
            
            return quantized
        
        return array
    
    def prune_model(self, model: Any, sparsity: float = 0.5) -> Any:
        """
        Remove low-importance weights from model.
        
        Prunes weights by setting the smallest magnitude weights to zero.
        
        Args:
            model: Model to prune (can be numpy array, dict of arrays, or object with weights)
            sparsity: Fraction of weights to remove (0.0 to 1.0)
            
        Returns:
            Pruned model
        """
        if not 0.0 <= sparsity <= 1.0:
            raise ValueError(f"sparsity must be between 0.0 and 1.0, got {sparsity}")
        
        # Handle numpy arrays
        if isinstance(model, np.ndarray):
            return self._prune_array(model, sparsity)
        
        # Handle dictionaries of arrays
        elif isinstance(model, dict):
            pruned = {}
            for key, value in model.items():
                if isinstance(value, np.ndarray):
                    pruned[key] = self._prune_array(value, sparsity)
                else:
                    pruned[key] = value
            return pruned
        
        # Handle objects with 'user_factors' and 'item_factors' (collaborative filter)
        elif hasattr(model, 'user_factors') and hasattr(model, 'item_factors'):
            pruned_model = type(model).__new__(type(model))
            pruned_model.__dict__.update(model.__dict__)
            if model.user_factors is not None:
                pruned_model.user_factors = self._prune_array(model.user_factors, sparsity)
            if model.item_factors is not None:
                pruned_model.item_factors = self._prune_array(model.item_factors, sparsity)
            return pruned_model
        
        # Handle objects with embeddings
        elif hasattr(model, 'embeddings'):
            pruned_model = type(model).__new__(type(model))
            pruned_model.__dict__.update(model.__dict__)
            if model.embeddings is not None:
                pruned_model.embeddings = self._prune_array(model.embeddings, sparsity)
            return pruned_model
        
        else:
            # Return as-is if we don't know how to prune
            return model
    
    def _prune_array(self, array: np.ndarray, sparsity: float) -> np.ndarray:
        """
        Prune a numpy array by setting smallest magnitude values to zero.
        
        Args:
            array: Array to prune
            sparsity: Fraction of weights to remove
            
        Returns:
            Pruned array (sparse representation)
        """
        if sparsity == 0.0:
            return array
        
        # Create a copy to avoid modifying original
        pruned = array.copy()
        
        # Flatten array to find threshold
        flat = np.abs(pruned.flatten())
        
        # Find threshold value (percentile corresponding to sparsity)
        threshold = np.percentile(flat, sparsity * 100)
        
        # Set values below threshold to zero
        pruned[np.abs(pruned) < threshold] = 0
        
        return pruned
    
    def compress_all_models(self, models: Dict[str, Any]) -> Dict[str, bytes]:
        """
        Apply compression pipeline to all models.
        
        Args:
            models: Dictionary of models to compress
            
        Returns:
            Dictionary of compressed models
        """
        # TODO: Implement full compression pipeline
        raise NotImplementedError("Model compression pipeline not yet implemented")
    
    def get_cached_recommendations(self, cache_key: str) -> Optional[List[Any]]:
        """
        Retrieve cached recommendations if available.
        
        Args:
            cache_key: Cache key for recommendations
            
        Returns:
            Cached recommendations or None
        """
        return self.destination_cache.get(cache_key)
    
    def cache_recommendations(self, cache_key: str, recommendations: List[Any]) -> None:
        """
        Cache recommendations.
        
        Args:
            cache_key: Cache key
            recommendations: Recommendations to cache
        """
        self.destination_cache.put(cache_key, recommendations)
    
    def get_cached_weather(self, location_key: str) -> Optional[Any]:
        """
        Retrieve cached weather data if available and not expired.
        
        Args:
            location_key: Location identifier
            
        Returns:
            Cached weather data or None
        """
        return self.weather_cache.get(location_key)
    
    def cache_weather(self, location_key: str, weather_data: Any) -> None:
        """
        Cache weather data with TTL.
        
        Args:
            location_key: Location identifier
            weather_data: Weather data to cache
        """
        self.weather_cache.put(location_key, weather_data)
    
    def should_use_server(self, network_available: bool, 
                          on_device_latency: float) -> bool:
        """
        Decide whether to call server-side models.
        
        Args:
            network_available: Whether network is available
            on_device_latency: Expected on-device latency in ms
            
        Returns:
            True if should use server, False otherwise
        """
        if not network_available:
            return False
        
        # Use server if on-device latency is too high
        if on_device_latency > 100:
            return True
        
        return False
