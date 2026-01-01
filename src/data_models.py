"""Core data models and type definitions for the tourism recommender system."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any


@dataclass
class LocationFeatures:
    """Features extracted from destination data."""
    
    destination_id: str
    name: str
    city: str
    latitude: float
    longitude: float
    location_type: str  # beach, cultural, nature, urban
    avg_rating: float
    review_count: int
    price_range: str  # budget, mid-range, luxury
    attributes: List[str] = field(default_factory=list)  # surfing, historical, wildlife, etc.


@dataclass
class UserProfile:
    """User preference profile."""
    
    user_id: str
    rating_history: Dict[str, float] = field(default_factory=dict)  # destination_id -> rating
    preferred_types: List[str] = field(default_factory=list)
    avg_rating: float = 0.0
    visit_count: int = 0
    is_cold_start: bool = True


@dataclass
class WeatherInfo:
    """Weather data for context-aware recommendations."""
    
    condition: str  # sunny, cloudy, rainy, stormy
    temperature: float
    humidity: float
    precipitation_chance: float


@dataclass
class Context:
    """Current contextual information."""
    
    location: Tuple[float, float]  # (latitude, longitude)
    weather: WeatherInfo
    season: str  # dry, monsoon, inter-monsoon
    day_of_week: int
    is_holiday: bool
    is_peak_season: bool
    user_type: str  # cold_start, regular, frequent


@dataclass
class RecommendationRequest:
    """Request structure for recommendation API."""
    
    user_id: str
    location: Tuple[float, float]  # (latitude, longitude)
    budget: Optional[Tuple[float, float]] = None  # (min, max)
    travel_style: Optional[str] = None  # beach, cultural, nature, adventure
    group_size: int = 1
    max_distance_km: Optional[float] = None


@dataclass
class Recommendation:
    """Recommendation output structure."""
    
    destination_id: str
    name: str
    score: float
    explanation: str
    distance_km: Optional[float] = None
    estimated_cost: Optional[float] = None
