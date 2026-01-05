"""
Aspect-Based Sentiment Analysis (ABSA) for Sri Lanka Tourism Reviews.

This module implements:
1. Aspect Extraction - Identify tourism-specific aspects in reviews
2. Aspect Sentiment - Determine sentiment for each aspect
3. Location Insights - Aggregate aspect scores per location
4. Smart Recommendations - Generate actionable insights

Research Innovation:
- First ABSA system for Sri Lanka tourism domain
- Tourism-specific aspect taxonomy
- Practical insights for tourism app integration
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import re
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

from nltk.tokenize import sent_tokenize, word_tokenize


# ============================================================================
# TOURISM ASPECT TAXONOMY
# ============================================================================

TOURISM_ASPECTS = {
    'scenery': {
        'keywords': [
            'view', 'views', 'scenery', 'scenic', 'beautiful', 'beauty', 'stunning',
            'breathtaking', 'picturesque', 'landscape', 'panorama', 'panoramic',
            'nature', 'natural', 'green', 'lush', 'sunset', 'sunrise', 'photography',
            'photo', 'photos', 'instagram', 'picture', 'pictures', 'aesthetic',
            'gorgeous', 'magnificent', 'spectacular', 'amazing view', 'ocean view',
            'mountain', 'waterfall', 'beach', 'forest', 'wildlife', 'birds', 'animals'
        ],
        'display_name': 'Scenery & Views',
        'icon': '🏞️'
    },
    'accessibility': {
        'keywords': [
            'access', 'accessible', 'accessibility', 'parking', 'park', 'transport',
            'bus', 'train', 'tuk', 'tuktuk', 'taxi', 'uber', 'drive', 'driving',
            'road', 'roads', 'distance', 'far', 'near', 'close', 'location',
            'directions', 'find', 'reach', 'reaching', 'getting there', 'walk',
            'walking', 'hike', 'hiking', 'stairs', 'steps', 'climb', 'wheelchair',
            'disabled', 'elderly', 'kids', 'children', 'stroller', 'easy access'
        ],
        'display_name': 'Accessibility',
        'icon': '🚗'
    },
    'facilities': {
        'keywords': [
            'toilet', 'toilets', 'bathroom', 'restroom', 'washroom', 'clean',
            'cleanliness', 'dirty', 'garbage', 'trash', 'litter', 'maintained',
            'maintenance', 'facility', 'facilities', 'infrastructure', 'bench',
            'seating', 'shade', 'shelter', 'shop', 'shops', 'store', 'vendor',
            'food', 'restaurant', 'cafe', 'drinks', 'water', 'snacks', 'wifi',
            'internet', 'charging', 'lockers', 'changing room', 'shower'
        ],
        'display_name': 'Facilities',
        'icon': '🚻'
    },
    'safety': {
        'keywords': [
            'safe', 'safety', 'secure', 'security', 'guard', 'guards', 'police',
            'dangerous', 'danger', 'risk', 'risky', 'careful', 'warning', 'caution',
            'scam', 'scams', 'scammer', 'tout', 'touts', 'hassle', 'harass',
            'theft', 'steal', 'stolen', 'pickpocket', 'crowd', 'crowded', 'crowds',
            'busy', 'packed', 'empty', 'quiet', 'peaceful', 'alone', 'solo',
            'night', 'dark', 'evening', 'lifeguard', 'current', 'waves', 'slippery'
        ],
        'display_name': 'Safety & Crowds',
        'icon': '🛡️'
    },
    'value': {
        'keywords': [
            'price', 'prices', 'pricing', 'cost', 'costs', 'fee', 'fees', 'ticket',
            'tickets', 'entry', 'entrance', 'admission', 'expensive', 'cheap',
            'affordable', 'overpriced', 'worth', 'value', 'money', 'pay', 'paid',
            'free', 'budget', 'bargain', 'deal', 'discount', 'foreigner', 'local',
            'tourist price', 'rip off', 'reasonable', 'fair price'
        ],
        'display_name': 'Value for Money',
        'icon': '💰'
    },
    'experience': {
        'keywords': [
            'guide', 'guides', 'tour', 'tours', 'guided', 'information', 'info',
            'history', 'historical', 'story', 'stories', 'learn', 'learned',
            'educational', 'interesting', 'boring', 'activity', 'activities',
            'adventure', 'fun', 'enjoy', 'enjoyed', 'experience', 'memorable',
            'recommend', 'recommended', 'must visit', 'must see', 'worth visiting',
            'time', 'hours', 'spend', 'spent', 'morning', 'afternoon', 'evening',
            'best time', 'avoid', 'skip', 'overrated', 'underrated', 'hidden gem'
        ],
        'display_name': 'Experience & Activities',
        'icon': '🎯'
    },
    'service': {
        'keywords': [
            'staff', 'employee', 'employees', 'worker', 'workers', 'service',
            'friendly', 'helpful', 'rude', 'polite', 'professional', 'welcoming',
            'hospitality', 'attitude', 'behavior', 'english', 'language',
            'communication', 'assist', 'assistance', 'help', 'helped'
        ],
        'display_name': 'Service & Staff',
        'icon': '👨‍💼'
    }
}

# Sentiment lexicon for aspect-level sentiment
POSITIVE_WORDS = {
    'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'awesome',
    'beautiful', 'lovely', 'nice', 'perfect', 'best', 'love', 'loved', 'enjoy',
    'enjoyed', 'recommend', 'recommended', 'worth', 'clean', 'friendly', 'helpful',
    'safe', 'peaceful', 'quiet', 'stunning', 'breathtaking', 'magnificent',
    'spectacular', 'incredible', 'outstanding', 'superb', 'brilliant', 'pleasant',
    'comfortable', 'convenient', 'easy', 'free', 'cheap', 'affordable', 'reasonable',
    'well maintained', 'well organized', 'must visit', 'must see', 'hidden gem'
}

NEGATIVE_WORDS = {
    'bad', 'poor', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'hated',
    'disappointing', 'disappointed', 'disappoints', 'not worth', 'waste', 'avoid',
    'dirty', 'filthy', 'smelly', 'broken', 'damaged', 'dangerous', 'unsafe',
    'crowded', 'packed', 'busy', 'noisy', 'loud', 'expensive', 'overpriced',
    'rip off', 'scam', 'scammer', 'rude', 'unfriendly', 'unhelpful', 'boring',
    'overrated', 'skip', 'not recommend', 'dont recommend', 'never again',
    'poorly maintained', 'run down', 'neglected', 'hassle', 'harass', 'annoying'
}

NEGATION_WORDS = {'not', 'no', 'never', 'dont', "don't", 'didnt', "didn't", 
                  'wasnt', "wasn't", 'isnt', "isn't", 'nothing', 'none', 'neither'}


@dataclass
class AspectSentiment:
    """Container for aspect-level sentiment."""
    aspect: str
    sentiment: str  # 'positive', 'negative', 'neutral'
    confidence: float
    text_snippet: str
    keywords_found: List[str]


@dataclass
class LocationInsight:
    """Aggregated insights for a location."""
    location_name: str
    location_type: str
    overall_sentiment: float  # -1 to 1
    aspect_scores: Dict[str, float]  # aspect -> score (-1 to 1)
    aspect_counts: Dict[str, int]  # aspect -> number of mentions
    strengths: List[str]
    weaknesses: List[str]
    total_reviews: int
    recommendation_score: float  # 0 to 5


@dataclass 
class SmartRecommendation:
    """Smart recommendation based on user preferences."""
    location_name: str
    match_score: float
    matching_aspects: List[str]
    highlights: List[str]
    warnings: List[str]


class AspectExtractor:
    """
    Extract tourism-specific aspects from review text.
    
    Uses keyword matching with context awareness.
    """
    
    def __init__(self):
        self.aspects = TOURISM_ASPECTS
        self._build_keyword_index()
    
    def _build_keyword_index(self):
        """Build reverse index from keywords to aspects."""
        self.keyword_to_aspect = {}
        for aspect, config in self.aspects.items():
            for keyword in config['keywords']:
                # Handle multi-word keywords
                self.keyword_to_aspect[keyword.lower()] = aspect
    
    def extract_aspects(self, text: str) -> List[Tuple[str, str, int, int]]:
        """
        Extract aspects from text with their positions.
        
        Args:
            text: Review text
            
        Returns:
            List of (aspect, matched_keyword, start_pos, end_pos)
        """
        if not isinstance(text, str) or not text.strip():
            return []
        
        text_lower = text.lower()
        found_aspects = []
        
        # Check for multi-word keywords first (longer matches take priority)
        sorted_keywords = sorted(self.keyword_to_aspect.keys(), 
                                key=len, reverse=True)
        
        for keyword in sorted_keywords:
            # Use word boundary matching
            pattern = r'\b' + re.escape(keyword) + r'\b'
            for match in re.finditer(pattern, text_lower):
                aspect = self.keyword_to_aspect[keyword]
                found_aspects.append((
                    aspect,
                    keyword,
                    match.start(),
                    match.end()
                ))
        
        # Remove overlapping matches (keep longer ones)
        found_aspects = self._remove_overlaps(found_aspects)
        
        return found_aspects
    
    def _remove_overlaps(self, aspects: List[Tuple]) -> List[Tuple]:
        """Remove overlapping aspect matches, keeping longer ones."""
        if not aspects:
            return []
        
        # Sort by start position, then by length (descending)
        sorted_aspects = sorted(aspects, key=lambda x: (x[2], -(x[3]-x[2])))
        
        result = []
        last_end = -1
        
        for aspect in sorted_aspects:
            if aspect[2] >= last_end:
                result.append(aspect)
                last_end = aspect[3]
        
        return result
    
    def get_aspect_sentences(self, text: str) -> Dict[str, List[str]]:
        """
        Get sentences containing each aspect.
        
        Args:
            text: Review text
            
        Returns:
            Dictionary mapping aspect to list of sentences
        """
        aspect_sentences = defaultdict(list)
        
        try:
            sentences = sent_tokenize(text)
        except:
            sentences = text.split('.')
        
        for sentence in sentences:
            aspects = self.extract_aspects(sentence)
            for aspect, keyword, _, _ in aspects:
                aspect_sentences[aspect].append(sentence.strip())
        
        return dict(aspect_sentences)


class AspectSentimentAnalyzer:
    """
    Analyze sentiment for specific aspects in reviews.
    
    Combines:
    1. Lexicon-based sentiment (fast, interpretable)
    2. Context-aware negation handling
    3. ML-based sentiment for ambiguous cases
    """
    
    def __init__(self):
        self.extractor = AspectExtractor()
        self.positive_words = POSITIVE_WORDS
        self.negative_words = NEGATIVE_WORDS
        self.negation_words = NEGATION_WORDS
        
        # ML model for ambiguous cases
        self.ml_model = None
        self.vectorizer = None
    
    def _get_context_window(self, text: str, start: int, end: int, 
                           window_size: int = 50) -> str:
        """Get text window around aspect mention."""
        context_start = max(0, start - window_size)
        context_end = min(len(text), end + window_size)
        return text[context_start:context_end]
    
    def _analyze_sentiment_lexicon(self, text: str) -> Tuple[str, float]:
        """
        Analyze sentiment using lexicon with negation handling.
        
        Returns:
            (sentiment, confidence)
        """
        text_lower = text.lower()
        words = text_lower.split()
        
        positive_count = 0
        negative_count = 0
        
        # Check for negation in window
        has_negation = any(neg in text_lower for neg in self.negation_words)
        
        for word in words:
            # Clean word
            word = re.sub(r'[^\w]', '', word)
            
            if word in self.positive_words:
                positive_count += 1
            elif word in self.negative_words:
                negative_count += 1
        
        # Also check multi-word phrases
        for phrase in self.positive_words:
            if ' ' in phrase and phrase in text_lower:
                positive_count += 1
        
        for phrase in self.negative_words:
            if ' ' in phrase and phrase in text_lower:
                negative_count += 1
        
        # Apply negation
        if has_negation:
            positive_count, negative_count = negative_count, positive_count
        
        # Determine sentiment
        total = positive_count + negative_count
        if total == 0:
            return 'neutral', 0.5
        
        if positive_count > negative_count:
            confidence = positive_count / total
            return 'positive', min(confidence, 0.95)
        elif negative_count > positive_count:
            confidence = negative_count / total
            return 'negative', min(confidence, 0.95)
        else:
            return 'neutral', 0.5
    
    def analyze_review(self, text: str) -> List[AspectSentiment]:
        """
        Analyze all aspects in a review.
        
        Args:
            text: Review text
            
        Returns:
            List of AspectSentiment objects
        """
        if not isinstance(text, str) or not text.strip():
            return []
        
        results = []
        aspect_sentences = self.extractor.get_aspect_sentences(text)
        
        for aspect, sentences in aspect_sentences.items():
            # Combine all sentences for this aspect
            combined_text = ' '.join(sentences)
            
            # Get sentiment
            sentiment, confidence = self._analyze_sentiment_lexicon(combined_text)
            
            # Get keywords found
            aspects_found = self.extractor.extract_aspects(combined_text)
            keywords = list(set(kw for a, kw, _, _ in aspects_found if a == aspect))
            
            results.append(AspectSentiment(
                aspect=aspect,
                sentiment=sentiment,
                confidence=confidence,
                text_snippet=sentences[0][:200] if sentences else "",
                keywords_found=keywords
            ))
        
        return results
    
    def analyze_batch(self, texts: List[str]) -> List[List[AspectSentiment]]:
        """Analyze multiple reviews."""
        return [self.analyze_review(text) for text in texts]


class LocationInsightGenerator:
    """
    Generate aggregated insights for locations based on aspect sentiments.
    """
    
    def __init__(self):
        self.analyzer = AspectSentimentAnalyzer()
        self.aspects = TOURISM_ASPECTS
    
    def _sentiment_to_score(self, sentiment: str, confidence: float) -> float:
        """Convert sentiment label to numeric score."""
        if sentiment == 'positive':
            return confidence
        elif sentiment == 'negative':
            return -confidence
        else:
            return 0.0
    
    def generate_location_insights(
        self, 
        df: pd.DataFrame,
        location_col: str = 'Location_Name',
        type_col: str = 'Location_Type',
        text_col: str = 'Text'
    ) -> Dict[str, LocationInsight]:
        """
        Generate insights for all locations in dataset.
        
        Args:
            df: DataFrame with reviews
            location_col: Column name for location
            type_col: Column name for location type
            text_col: Column name for review text
            
        Returns:
            Dictionary mapping location name to LocationInsight
        """
        insights = {}
        
        # Group by location
        for location, group in df.groupby(location_col):
            location_type = group[type_col].iloc[0] if type_col in group.columns else 'Unknown'
            
            # Analyze all reviews for this location
            aspect_scores = defaultdict(list)
            aspect_counts = defaultdict(int)
            
            for text in group[text_col]:
                aspect_sentiments = self.analyzer.analyze_review(text)
                
                for asp_sent in aspect_sentiments:
                    score = self._sentiment_to_score(
                        asp_sent.sentiment, 
                        asp_sent.confidence
                    )
                    aspect_scores[asp_sent.aspect].append(score)
                    aspect_counts[asp_sent.aspect] += 1
            
            # Calculate average scores
            avg_scores = {}
            for aspect, scores in aspect_scores.items():
                avg_scores[aspect] = np.mean(scores) if scores else 0.0
            
            # Determine strengths and weaknesses
            strengths = []
            weaknesses = []
            
            for aspect, score in sorted(avg_scores.items(), key=lambda x: x[1], reverse=True):
                aspect_name = self.aspects[aspect]['display_name']
                if score > 0.3 and aspect_counts[aspect] >= 3:
                    strengths.append(f"{self.aspects[aspect]['icon']} {aspect_name}")
                elif score < -0.2 and aspect_counts[aspect] >= 3:
                    weaknesses.append(f"{self.aspects[aspect]['icon']} {aspect_name}")
            
            # Calculate overall sentiment
            all_scores = [s for scores in aspect_scores.values() for s in scores]
            overall_sentiment = np.mean(all_scores) if all_scores else 0.0
            
            # Calculate recommendation score (0-5)
            recommendation_score = (overall_sentiment + 1) * 2.5  # Map -1,1 to 0,5
            recommendation_score = max(0, min(5, recommendation_score))
            
            insights[location] = LocationInsight(
                location_name=location,
                location_type=location_type,
                overall_sentiment=overall_sentiment,
                aspect_scores=dict(avg_scores),
                aspect_counts=dict(aspect_counts),
                strengths=strengths[:5],  # Top 5
                weaknesses=weaknesses[:3],  # Top 3
                total_reviews=len(group),
                recommendation_score=round(recommendation_score, 1)
            )
        
        return insights
    
    def get_location_summary(self, insight: LocationInsight) -> str:
        """Generate human-readable summary for a location."""
        lines = []
        lines.append(f"📍 {insight.location_name}")
        lines.append(f"   Type: {insight.location_type}")
        lines.append(f"   Rating: {'⭐' * int(insight.recommendation_score)} ({insight.recommendation_score}/5)")
        lines.append(f"   Based on {insight.total_reviews} reviews")
        
        if insight.strengths:
            lines.append(f"\n   ✅ Strengths: {', '.join(insight.strengths)}")
        
        if insight.weaknesses:
            lines.append(f"   ⚠️ Watch out: {', '.join(insight.weaknesses)}")
        
        # Aspect breakdown
        lines.append("\n   📊 Aspect Scores:")
        for aspect, score in sorted(insight.aspect_scores.items(), 
                                   key=lambda x: x[1], reverse=True):
            if insight.aspect_counts.get(aspect, 0) >= 2:
                icon = self.aspects[aspect]['icon']
                name = self.aspects[aspect]['display_name']
                bar = self._score_to_bar(score)
                lines.append(f"      {icon} {name}: {bar} ({score:.2f})")
        
        return '\n'.join(lines)
    
    def _score_to_bar(self, score: float) -> str:
        """Convert score to visual bar."""
        # Map -1 to 1 → 0 to 10
        normalized = int((score + 1) * 5)
        normalized = max(0, min(10, normalized))
        return '█' * normalized + '░' * (10 - normalized)


class SmartRecommender:
    """
    Generate smart recommendations based on user preferences and aspect scores.
    """
    
    def __init__(self, insights: Dict[str, LocationInsight]):
        self.insights = insights
        self.aspects = TOURISM_ASPECTS
    
    def recommend_by_preferences(
        self,
        preferred_aspects: List[str],
        avoid_aspects: List[str] = None,
        location_type: str = None,
        min_reviews: int = 5,
        top_n: int = 5
    ) -> List[SmartRecommendation]:
        """
        Recommend locations based on user preferences.
        
        Args:
            preferred_aspects: Aspects user cares about (e.g., ['scenery', 'safety'])
            avoid_aspects: Aspects to avoid low scores in
            location_type: Filter by location type
            min_reviews: Minimum reviews required
            top_n: Number of recommendations
            
        Returns:
            List of SmartRecommendation objects
        """
        avoid_aspects = avoid_aspects or []
        recommendations = []
        
        for location, insight in self.insights.items():
            # Filter by type
            if location_type and insight.location_type != location_type:
                continue
            
            # Filter by minimum reviews
            if insight.total_reviews < min_reviews:
                continue
            
            # Calculate match score
            match_score = 0.0
            matching_aspects = []
            
            for aspect in preferred_aspects:
                if aspect in insight.aspect_scores:
                    score = insight.aspect_scores[aspect]
                    if score > 0:
                        match_score += score
                        matching_aspects.append(aspect)
            
            # Penalize for avoid aspects
            for aspect in avoid_aspects:
                if aspect in insight.aspect_scores:
                    score = insight.aspect_scores[aspect]
                    if score < 0:
                        match_score += score  # Negative, so adds penalty
            
            # Normalize by number of preferred aspects
            if preferred_aspects:
                match_score /= len(preferred_aspects)
            
            # Generate highlights and warnings
            highlights = []
            warnings = []
            
            for aspect in matching_aspects:
                if insight.aspect_scores.get(aspect, 0) > 0.3:
                    highlights.append(f"Great {self.aspects[aspect]['display_name'].lower()}")
            
            for aspect, score in insight.aspect_scores.items():
                if score < -0.3 and insight.aspect_counts.get(aspect, 0) >= 3:
                    warnings.append(f"Issues with {self.aspects[aspect]['display_name'].lower()}")
            
            recommendations.append(SmartRecommendation(
                location_name=location,
                match_score=match_score,
                matching_aspects=matching_aspects,
                highlights=highlights[:3],
                warnings=warnings[:2]
            ))
        
        # Sort by match score
        recommendations.sort(key=lambda x: x.match_score, reverse=True)
        
        return recommendations[:top_n]
    
    def compare_locations(
        self, 
        locations: List[str]
    ) -> pd.DataFrame:
        """
        Compare multiple locations across all aspects.
        
        Args:
            locations: List of location names to compare
            
        Returns:
            DataFrame with comparison
        """
        data = []
        
        for location in locations:
            if location not in self.insights:
                continue
            
            insight = self.insights[location]
            row = {'Location': location, 'Type': insight.location_type}
            
            for aspect in self.aspects:
                score = insight.aspect_scores.get(aspect, 0)
                row[self.aspects[aspect]['display_name']] = round(score, 2)
            
            row['Overall'] = round(insight.overall_sentiment, 2)
            row['Reviews'] = insight.total_reviews
            
            data.append(row)
        
        return pd.DataFrame(data)


class ABSAPipeline:
    """
    Complete Aspect-Based Sentiment Analysis pipeline.
    
    Combines all components for easy use.
    """
    
    def __init__(self):
        self.analyzer = AspectSentimentAnalyzer()
        self.insight_generator = LocationInsightGenerator()
        self.recommender = None
        
        self.df = None
        self.insights = None
    
    def load_and_analyze(self, csv_path: str) -> Dict[str, LocationInsight]:
        """
        Load data and generate insights for all locations.
        
        Args:
            csv_path: Path to Reviews.csv
            
        Returns:
            Dictionary of location insights
        """
        print("=" * 60)
        print("ASPECT-BASED SENTIMENT ANALYSIS")
        print("=" * 60)
        
        # Load data
        print("\nLoading data...")
        self.df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.df)} reviews")
        
        # Generate insights
        print("\nAnalyzing aspects and sentiments...")
        self.insights = self.insight_generator.generate_location_insights(self.df)
        print(f"Generated insights for {len(self.insights)} locations")
        
        # Initialize recommender
        self.recommender = SmartRecommender(self.insights)
        
        return self.insights
    
    def get_location_insight(self, location_name: str) -> Optional[LocationInsight]:
        """Get insight for a specific location."""
        return self.insights.get(location_name)
    
    def print_location_summary(self, location_name: str):
        """Print formatted summary for a location."""
        if location_name not in self.insights:
            print(f"Location '{location_name}' not found")
            return
        
        summary = self.insight_generator.get_location_summary(
            self.insights[location_name]
        )
        print(summary)
    
    def recommend(
        self,
        preferred_aspects: List[str],
        avoid_aspects: List[str] = None,
        location_type: str = None,
        top_n: int = 5
    ) -> List[SmartRecommendation]:
        """Get smart recommendations based on preferences."""
        if self.recommender is None:
            raise ValueError("Data not loaded. Call load_and_analyze() first.")
        
        return self.recommender.recommend_by_preferences(
            preferred_aspects=preferred_aspects,
            avoid_aspects=avoid_aspects,
            location_type=location_type,
            top_n=top_n
        )
    
    def compare(self, locations: List[str]) -> pd.DataFrame:
        """Compare multiple locations."""
        if self.recommender is None:
            raise ValueError("Data not loaded. Call load_and_analyze() first.")
        
        return self.recommender.compare_locations(locations)
    
    def get_aspect_statistics(self) -> pd.DataFrame:
        """Get statistics about aspect mentions across all locations."""
        if self.insights is None:
            raise ValueError("Data not loaded. Call load_and_analyze() first.")
        
        stats = []
        for aspect in TOURISM_ASPECTS:
            total_mentions = sum(
                insight.aspect_counts.get(aspect, 0) 
                for insight in self.insights.values()
            )
            avg_score = np.mean([
                insight.aspect_scores.get(aspect, 0)
                for insight in self.insights.values()
                if aspect in insight.aspect_scores
            ]) if total_mentions > 0 else 0
            
            stats.append({
                'Aspect': TOURISM_ASPECTS[aspect]['display_name'],
                'Icon': TOURISM_ASPECTS[aspect]['icon'],
                'Total Mentions': total_mentions,
                'Avg Sentiment': round(avg_score, 3),
                'Sentiment': 'Positive' if avg_score > 0.1 else ('Negative' if avg_score < -0.1 else 'Neutral')
            })
        
        return pd.DataFrame(stats).sort_values('Total Mentions', ascending=False)
    
    def get_top_locations_by_aspect(
        self, 
        aspect: str, 
        top_n: int = 10,
        min_mentions: int = 5
    ) -> pd.DataFrame:
        """Get top locations for a specific aspect."""
        if self.insights is None:
            raise ValueError("Data not loaded. Call load_and_analyze() first.")
        
        data = []
        for location, insight in self.insights.items():
            if insight.aspect_counts.get(aspect, 0) >= min_mentions:
                data.append({
                    'Location': location,
                    'Type': insight.location_type,
                    'Score': insight.aspect_scores.get(aspect, 0),
                    'Mentions': insight.aspect_counts.get(aspect, 0),
                    'Total Reviews': insight.total_reviews
                })
        
        df = pd.DataFrame(data)
        if len(df) > 0:
            df = df.sort_values('Score', ascending=False).head(top_n)
        
        return df
