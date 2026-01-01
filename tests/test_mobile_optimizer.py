"""Property-based tests for MobileOptimizer module."""

import pytest
import time
import numpy as np
from hypothesis import given, strategies as st, settings, assume

from src.mobile_optimizer import LRUCache, TTLCache, MobileOptimizer


class TestLRUCacheEviction:
    """
    Feature: tourism-recommender-system, Property 16: LRU Cache Eviction
    
    Property: For any sequence of cache accesses exceeding 100 items,
    the least recently used item SHALL be evicted when a new item is added.
    
    Validates: Requirements 6.4
    """
    
    @given(
        maxsize=st.integers(min_value=1, max_value=10),
        n_items=st.integers(min_value=2, max_value=20)
    )
    @settings(max_examples=30, deadline=None)
    def test_lru_cache_eviction(self, maxsize, n_items):
        """Test that LRU cache evicts least recently used items when capacity is exceeded."""
        assume(n_items > maxsize)  # Only test when we exceed capacity
        
        cache = LRUCache(maxsize=maxsize)
        
        # Add items exceeding capacity
        for i in range(n_items):
            cache.put(f"key_{i}", f"value_{i}")
        
        # Property: Cache size should not exceed maxsize
        assert cache.size() <= maxsize, f"Cache size {cache.size()} exceeds maxsize {maxsize}"
        
        # Property: The oldest items (first maxsize items) should have been evicted
        # Only the last maxsize items should remain
        expected_remaining_start = n_items - maxsize
        
        for i in range(expected_remaining_start):
            # These should have been evicted
            assert cache.get(f"key_{i}") is None, f"key_{i} should have been evicted"
        
        for i in range(expected_remaining_start, n_items):
            # These should still be in cache
            assert cache.get(f"key_{i}") == f"value_{i}", f"key_{i} should still be in cache"
    
    @given(
        maxsize=st.integers(min_value=2, max_value=8),
        access_pattern=st.lists(
            st.integers(min_value=0, max_value=5),
            min_size=5,
            max_size=15
        )
    )
    @settings(max_examples=30, deadline=None)
    def test_lru_cache_access_order(self, maxsize, access_pattern):
        """Test that accessing items updates their recency and prevents eviction."""
        cache = LRUCache(maxsize=maxsize)
        
        # Initialize cache with maxsize items
        for i in range(maxsize):
            cache.put(f"key_{i}", f"value_{i}")
        
        # Access items according to pattern
        for idx in access_pattern:
            if idx < maxsize:
                cache.get(f"key_{idx}")
        
        # Add one more item to trigger eviction
        cache.put("new_key", "new_value")
        
        # Property: Cache size should still be maxsize
        assert cache.size() == maxsize, f"Cache size should be {maxsize}, got {cache.size()}"
        
        # Property: The new key should be in cache
        assert cache.get("new_key") == "new_value", "New key should be in cache"
        
        # Property: At least one old key should have been evicted
        remaining_count = sum(1 for i in range(maxsize) if cache.get(f"key_{i}") is not None)
        assert remaining_count == maxsize - 1, f"Expected {maxsize - 1} old keys, got {remaining_count}"


class TestTTLCacheExpiry:
    """
    Feature: tourism-recommender-system, Property 17: TTL Cache Expiry
    
    Property: For any cached weather data, accessing it after 1 hour SHALL return a cache miss.
    
    Validates: Requirements 6.5
    """
    
    @given(
        ttl=st.integers(min_value=1, max_value=3),
        wait_time=st.integers(min_value=1, max_value=5)
    )
    @settings(max_examples=20, deadline=None)
    def test_ttl_cache_expiry(self, ttl, wait_time):
        """Test that TTL cache expires items after the specified time."""
        cache = TTLCache(maxsize=10, ttl=ttl)
        
        # Add item to cache
        cache.put("test_key", "test_value")
        
        # Property: Item should be retrievable immediately
        assert cache.get("test_key") == "test_value", "Item should be in cache immediately"
        
        # Wait for specified time
        time.sleep(wait_time)
        
        # Property: Item should be expired if wait_time >= ttl
        result = cache.get("test_key")
        if wait_time >= ttl:
            assert result is None, f"Item should be expired after {wait_time}s with TTL {ttl}s"
        else:
            assert result == "test_value", f"Item should not be expired after {wait_time}s with TTL {ttl}s"
    
    @given(
        n_items=st.integers(min_value=1, max_value=5)
    )
    @settings(max_examples=10, deadline=None)
    def test_ttl_cache_expiry_one_hour(self, n_items):
        """Test that weather cache with 1 hour TTL expires correctly."""
        # Use very short TTL for testing (1 second instead of 1 hour)
        cache = TTLCache(maxsize=20, ttl=1)
        
        # Add items
        for i in range(n_items):
            cache.put(f"location_{i}", {"temp": 25.0 + i, "condition": "sunny"})
        
        # Property: All items should be retrievable immediately
        for i in range(n_items):
            result = cache.get(f"location_{i}")
            assert result is not None, f"location_{i} should be in cache immediately"
            assert result["temp"] == 25.0 + i, f"location_{i} should have correct data"
        
        # Wait for TTL to expire
        time.sleep(1.1)
        
        # Property: All items should be expired
        for i in range(n_items):
            result = cache.get(f"location_{i}")
            assert result is None, f"location_{i} should be expired after TTL"
    
    @given(
        maxsize=st.integers(min_value=2, max_value=8),
        n_items=st.integers(min_value=3, max_value=12)
    )
    @settings(max_examples=20, deadline=None)
    def test_ttl_cache_capacity(self, maxsize, n_items):
        """Test that TTL cache respects maxsize limit."""
        assume(n_items > maxsize)
        
        cache = TTLCache(maxsize=maxsize, ttl=3600)
        
        # Add items exceeding capacity
        for i in range(n_items):
            cache.put(f"key_{i}", f"value_{i}")
        
        # Property: Cache size should not exceed maxsize
        assert cache.size() <= maxsize, f"Cache size {cache.size()} exceeds maxsize {maxsize}"


class TestMobileOptimizerIntegration:
    """Integration tests for MobileOptimizer."""
    
    def test_mobile_optimizer_initialization(self):
        """Test that MobileOptimizer initializes with correct cache configurations."""
        optimizer = MobileOptimizer()
        
        # Check that caches are initialized
        assert optimizer.destination_cache is not None
        assert optimizer.weather_cache is not None
        assert optimizer.user_cache is not None
        
        # Check cache sizes
        assert optimizer.destination_cache.maxsize == 100
        assert optimizer.weather_cache.maxsize == 20
        assert optimizer.weather_cache.ttl == 3600
        assert optimizer.user_cache.maxsize == 50
    
    def test_recommendation_caching(self):
        """Test caching and retrieval of recommendations."""
        optimizer = MobileOptimizer()
        
        recommendations = [
            {"id": "1", "name": "Destination 1", "score": 0.9},
            {"id": "2", "name": "Destination 2", "score": 0.8},
        ]
        
        # Cache recommendations
        optimizer.cache_recommendations("user_123", recommendations)
        
        # Retrieve from cache
        cached = optimizer.get_cached_recommendations("user_123")
        assert cached == recommendations
        
        # Non-existent key should return None
        assert optimizer.get_cached_recommendations("user_999") is None
    
    def test_weather_caching(self):
        """Test caching and retrieval of weather data."""
        optimizer = MobileOptimizer()
        
        weather_data = {
            "condition": "sunny",
            "temperature": 28.5,
            "humidity": 65.0,
            "precipitation_chance": 0.1
        }
        
        # Cache weather data
        optimizer.cache_weather("location_kandy", weather_data)
        
        # Retrieve from cache
        cached = optimizer.get_cached_weather("location_kandy")
        assert cached == weather_data
        
        # Non-existent key should return None
        assert optimizer.get_cached_weather("location_unknown") is None
    
    def test_should_use_server_logic(self):
        """Test server usage decision logic."""
        optimizer = MobileOptimizer()
        
        # No network -> use on-device
        assert optimizer.should_use_server(False, 50) == False
        
        # Network available, low latency -> use on-device
        assert optimizer.should_use_server(True, 50) == False
        
        # Network available, high latency -> use server
        assert optimizer.should_use_server(True, 150) == True



class TestModelQuantization:
    """Unit tests for model quantization."""
    
    def test_quantize_array_to_float16(self):
        """Test quantization of numpy array to float16."""
        optimizer = MobileOptimizer()
        
        # Create test array
        array = np.array([[1.5, 2.7, 3.9], [4.2, 5.1, 6.8]], dtype=np.float32)
        
        # Quantize to float16
        quantized = optimizer.quantize_model(array, bit_width=16)
        
        # Check dtype
        assert quantized.dtype == np.float16
        
        # Check shape preserved
        assert quantized.shape == array.shape
    
    def test_quantize_array_to_int8(self):
        """Test quantization of numpy array to int8."""
        optimizer = MobileOptimizer()
        
        # Create test array
        array = np.array([[1.5, 2.7, 3.9], [4.2, 5.1, 6.8]], dtype=np.float32)
        
        # Quantize to int8
        quantized = optimizer.quantize_model(array, bit_width=8)
        
        # Check dtype
        assert quantized.dtype == np.int8
        
        # Check shape preserved
        assert quantized.shape == array.shape
        
        # Check values are in int8 range
        assert np.all(quantized >= -128)
        assert np.all(quantized <= 127)
    
    def test_quantize_dict_of_arrays(self):
        """Test quantization of dictionary containing arrays."""
        optimizer = MobileOptimizer()
        
        # Create test dict
        model = {
            'weights': np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            'bias': np.array([0.5, 1.5], dtype=np.float32),
            'metadata': 'some_string'
        }
        
        # Quantize
        quantized = optimizer.quantize_model(model, bit_width=16)
        
        # Check arrays are quantized
        assert quantized['weights'].dtype == np.float16
        assert quantized['bias'].dtype == np.float16
        
        # Check non-array values preserved
        assert quantized['metadata'] == 'some_string'
    
    def test_quantize_invalid_bit_width(self):
        """Test that invalid bit width raises error."""
        optimizer = MobileOptimizer()
        array = np.array([1.0, 2.0, 3.0])
        
        with pytest.raises(ValueError, match="bit_width must be 8 or 16"):
            optimizer.quantize_model(array, bit_width=32)


class TestModelPruning:
    """Unit tests for model pruning."""
    
    def test_prune_array_50_percent(self):
        """Test pruning 50% of array weights."""
        optimizer = MobileOptimizer()
        
        # Create test array with known values
        array = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
        
        # Prune 50%
        pruned = optimizer.prune_model(array, sparsity=0.5)
        
        # Check that approximately 50% are zero
        zero_count = np.sum(pruned == 0)
        total_count = pruned.size
        sparsity_ratio = zero_count / total_count
        
        # Allow some tolerance
        assert 0.4 <= sparsity_ratio <= 0.6
    
    def test_prune_array_preserves_large_values(self):
        """Test that pruning preserves largest magnitude values."""
        optimizer = MobileOptimizer()
        
        # Create array where largest values are obvious
        array = np.array([[10.0, 1.0, 0.5, 0.1]], dtype=np.float32)
        
        # Prune 50%
        pruned = optimizer.prune_model(array, sparsity=0.5)
        
        # Largest values should be preserved
        assert pruned[0, 0] == 10.0  # Largest value preserved
        assert pruned[0, 1] != 0  # Second largest likely preserved
    
    def test_prune_dict_of_arrays(self):
        """Test pruning dictionary containing arrays."""
        optimizer = MobileOptimizer()
        
        # Create test dict
        model = {
            'weights': np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32),
            'bias': np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
            'metadata': 'some_string'
        }
        
        # Prune
        pruned = optimizer.prune_model(model, sparsity=0.5)
        
        # Check arrays are pruned
        assert np.any(pruned['weights'] == 0)
        assert np.any(pruned['bias'] == 0)
        
        # Check non-array values preserved
        assert pruned['metadata'] == 'some_string'
    
    def test_prune_zero_sparsity(self):
        """Test that zero sparsity returns unchanged array."""
        optimizer = MobileOptimizer()
        
        array = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        pruned = optimizer.prune_model(array, sparsity=0.0)
        
        # Should be unchanged
        np.testing.assert_array_equal(pruned, array)
    
    def test_prune_invalid_sparsity(self):
        """Test that invalid sparsity raises error."""
        optimizer = MobileOptimizer()
        array = np.array([1.0, 2.0, 3.0])
        
        with pytest.raises(ValueError, match="sparsity must be between 0.0 and 1.0"):
            optimizer.prune_model(array, sparsity=1.5)
