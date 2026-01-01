"""Structured logging for the tourism recommender system."""

import logging
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class RecommenderLogger:
    """
    Structured logging for debugging and monitoring.
    
    Requirement 10.5: WHEN errors occur, THE Recommender_System SHALL log error
    details with timestamps and context for debugging.
    """
    
    def __init__(self, name: str = "tourism_recommender", level: int = logging.INFO):
        """
        Initialize the logger.
        
        Args:
            name: Logger name
            level: Logging level (default INFO)
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Create console handler if not already configured
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(level)
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            
            self.logger.addHandler(handler)
    
    def log_request(self, request_id: str, request: Dict[str, Any]) -> None:
        """
        Log incoming request with timestamp.
        
        Args:
            request_id: Unique request identifier
            request: Request data dictionary
        """
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'request_id': request_id,
            'event': 'request_received',
            'user_id': request.get('user_id'),
            'location': request.get('location'),
            'budget': request.get('budget'),
            'travel_style': request.get('travel_style'),
            'group_size': request.get('group_size'),
        }
        
        self.logger.info(f"Request: {json.dumps(log_data, cls=NumpyEncoder)}")
    
    def log_model_prediction(
        self,
        request_id: str,
        model_name: str,
        latency_ms: float,
        prediction_count: int
    ) -> None:
        """
        Log individual model performance.
        
        Args:
            request_id: Unique request identifier
            model_name: Name of the model (collaborative, content_based, context_aware)
            latency_ms: Prediction latency in milliseconds
            prediction_count: Number of predictions generated
        """
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'request_id': request_id,
            'event': 'model_prediction',
            'model_name': model_name,
            'latency_ms': latency_ms,
            'prediction_count': prediction_count,
        }
        
        self.logger.info(f"Model prediction: {json.dumps(log_data, cls=NumpyEncoder)}")
    
    def log_error(
        self,
        request_id: str,
        error_type: str,
        error_message: str,
        context: Dict[str, Any]
    ) -> None:
        """
        Log errors with full context for debugging.
        
        Requirement 10.5: Log error details with timestamps and context.
        
        Args:
            request_id: Unique request identifier
            error_type: Type/category of error
            error_message: Detailed error message
            context: Additional context information
        """
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'request_id': request_id,
            'event': 'error',
            'error_type': error_type,
            'error_message': error_message,
            'context': context,
        }
        
        self.logger.error(f"Error: {json.dumps(log_data, cls=NumpyEncoder)}")
    
    def log_cache_stats(self, cache_name: str, hits: int, misses: int) -> None:
        """
        Log cache performance metrics.
        
        Args:
            cache_name: Name of the cache (destination, weather, user)
            hits: Number of cache hits
            misses: Number of cache misses
        """
        hit_rate = hits / (hits + misses) if (hits + misses) > 0 else 0.0
        
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'event': 'cache_stats',
            'cache_name': cache_name,
            'hits': hits,
            'misses': misses,
            'hit_rate': hit_rate,
        }
        
        self.logger.info(f"Cache stats: {json.dumps(log_data, cls=NumpyEncoder)}")
    
    def log_fallback(
        self,
        request_id: str,
        fallback_type: str,
        reason: str,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log when fallback mechanisms are triggered.
        
        Args:
            request_id: Unique request identifier
            fallback_type: Type of fallback (weather, popular_destinations, etc.)
            reason: Reason for fallback
            context: Additional context information
        """
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'request_id': request_id,
            'event': 'fallback_triggered',
            'fallback_type': fallback_type,
            'reason': reason,
            'context': context or {},
        }
        
        self.logger.warning(f"Fallback: {json.dumps(log_data, cls=NumpyEncoder)}")
    
    def log_validation_error(
        self,
        request_id: str,
        validation_type: str,
        invalid_count: int,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log data validation errors.
        
        Args:
            request_id: Unique request identifier
            validation_type: Type of validation (rating, user, destination)
            invalid_count: Number of invalid entries
            details: Additional details about validation errors
        """
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'request_id': request_id,
            'event': 'validation_error',
            'validation_type': validation_type,
            'invalid_count': invalid_count,
            'details': details or {},
        }
        
        self.logger.warning(f"Validation error: {json.dumps(log_data, cls=NumpyEncoder)}")
    
    def log_response(
        self,
        request_id: str,
        recommendation_count: int,
        total_latency_ms: float
    ) -> None:
        """
        Log response generation.
        
        Args:
            request_id: Unique request identifier
            recommendation_count: Number of recommendations returned
            total_latency_ms: Total request latency in milliseconds
        """
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'request_id': request_id,
            'event': 'response_sent',
            'recommendation_count': recommendation_count,
            'total_latency_ms': total_latency_ms,
        }
        
        self.logger.info(f"Response: {json.dumps(log_data, cls=NumpyEncoder)}")


# Global logger instance
_global_logger: Optional[RecommenderLogger] = None


def get_logger(name: str = "tourism_recommender") -> RecommenderLogger:
    """
    Get or create global logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        RecommenderLogger instance
    """
    global _global_logger
    
    if _global_logger is None:
        _global_logger = RecommenderLogger(name)
    
    return _global_logger
