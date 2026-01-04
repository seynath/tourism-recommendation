#!/usr/bin/env python
"""
Aspect-Based Sentiment Analysis Script for Tourism Reviews.

This script runs comprehensive ABSA evaluation:
1. Load and analyze reviews by aspect
2. Generate location insights
3. Show aspect statistics
4. Demonstrate smart recommendations
5. Compare locations

Run with: python scripts/run_aspect_analysis.py
"""

import sys
import time
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from src.aspect_sentiment import (
    ABSAPipeline,
    TOURISM_ASPECTS,
    AspectSentimentAnalyzer
)


def print_header(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def main():
    """Run complete ABSA evaluation."""
    print_header("ASPECT-BASED SENTIMENT ANALYSIS")
    print("Sri Lanka Tourism Reviews")
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    # Initialize pipeline
    pipeline = ABSAPipeline()
    
    # Load and analyze
    print_header("1. LOADING AND ANALYZING DATA")
    insights = pipeline.load_and_analyze('dataset/Reviews.csv')
    
    # Aspect statistics
    print_header("2. ASPECT STATISTICS")
    aspect_stats = pipeline.get_aspect_statistics()
    print("\nAspect Mention Statistics:")
    print(aspect_stats.to_string(index=False))
    
    # Top locations by aspect
    print_header("3. TOP LOCATIONS BY ASPECT")
    
    for aspect in ['scenery', 'safety', 'value']:
        print(f"\n🏆 Top 5 Locations for {TOURISM_ASPECTS[aspect]['display_name']}:")
        top_df = pipeline.get_top_locations_by_aspect(aspect, top_n=5, min_mentions=3)
        if len(top_df) > 0:
            for _, row in top_df.iterrows():
                score_bar = '█' * int((row['Score'] + 1) * 5) + '░' * (10 - int((row['Score'] + 1) * 5))
                print(f"   {row['Location'][:30]:<30} {score_bar} ({row['Score']:.2f})")
        else:
            print("   Not enough data")
    
    # Sample location insights
    print_header("4. SAMPLE LOCATION INSIGHTS")
    
    # Get top 3 locations by review count
    top_locations = sorted(
        insights.items(), 
        key=lambda x: x[1].total_reviews, 
        reverse=True
    )[:3]
    
    for location, insight in top_locations:
        print("\n" + "-" * 50)
        pipeline.print_location_summary(location)
    
    # Smart recommendations demo
    print_header("5. SMART RECOMMENDATIONS DEMO")
    
    # Scenario 1: Photography enthusiast
    print("\n📸 Scenario: Photography enthusiast looking for scenic spots")
    print("   Preferences: scenery, accessibility")
    recs = pipeline.recommend(
        preferred_aspects=['scenery', 'accessibility'],
        top_n=5
    )
    print("\n   Recommendations:")
    for i, rec in enumerate(recs, 1):
        print(f"   {i}. {rec.location_name} (Match: {rec.match_score:.2f})")
        if rec.highlights:
            print(f"      ✅ {', '.join(rec.highlights)}")
        if rec.warnings:
            print(f"      ⚠️ {', '.join(rec.warnings)}")
    
    # Scenario 2: Family with kids
    print("\n👨‍👩‍👧‍👦 Scenario: Family with kids prioritizing safety")
    print("   Preferences: safety, facilities")
    print("   Avoid: crowds")
    recs = pipeline.recommend(
        preferred_aspects=['safety', 'facilities'],
        avoid_aspects=['safety'],  # Avoid places with safety concerns
        top_n=5
    )
    print("\n   Recommendations:")
    for i, rec in enumerate(recs, 1):
        print(f"   {i}. {rec.location_name} (Match: {rec.match_score:.2f})")
        if rec.highlights:
            print(f"      ✅ {', '.join(rec.highlights)}")
    
    # Scenario 3: Budget traveler
    print("\n💰 Scenario: Budget traveler looking for value")
    print("   Preferences: value, experience")
    recs = pipeline.recommend(
        preferred_aspects=['value', 'experience'],
        top_n=5
    )
    print("\n   Recommendations:")
    for i, rec in enumerate(recs, 1):
        print(f"   {i}. {rec.location_name} (Match: {rec.match_score:.2f})")
        if rec.highlights:
            print(f"      ✅ {', '.join(rec.highlights)}")
    
    # Location comparison
    print_header("6. LOCATION COMPARISON")
    
    # Compare top beaches (if available)
    beach_locations = [
        loc for loc, insight in insights.items() 
        if 'beach' in insight.location_type.lower()
    ][:3]
    
    if beach_locations:
        print(f"\n🏖️ Comparing Beaches: {', '.join(beach_locations)}")
        comparison = pipeline.compare(beach_locations)
        print(comparison.to_string(index=False))
    
    # Summary statistics
    print_header("7. SUMMARY STATISTICS")
    
    # Overall stats
    total_locations = len(insights)
    avg_reviews = np.mean([i.total_reviews for i in insights.values()])
    avg_score = np.mean([i.recommendation_score for i in insights.values()])
    
    print(f"\n📊 Dataset Overview:")
    print(f"   Total Locations Analyzed: {total_locations}")
    print(f"   Average Reviews per Location: {avg_reviews:.1f}")
    print(f"   Average Recommendation Score: {avg_score:.1f}/5")
    
    # Best and worst by aspect
    print(f"\n🏆 Best Performing Aspects (by avg sentiment):")
    for _, row in aspect_stats.head(3).iterrows():
        print(f"   {row['Icon']} {row['Aspect']}: {row['Avg Sentiment']:.3f}")
    
    print(f"\n⚠️ Aspects Needing Improvement:")
    for _, row in aspect_stats.tail(2).iterrows():
        if row['Avg Sentiment'] < 0.1:
            print(f"   {row['Icon']} {row['Aspect']}: {row['Avg Sentiment']:.3f}")
    
    # Save results
    print_header("8. SAVING RESULTS")
    
    # Save aspect statistics
    aspect_stats.to_csv('data/aspect_statistics.csv', index=False)
    print("   ✅ Saved aspect_statistics.csv")
    
    # Save location insights summary
    insight_data = []
    for location, insight in insights.items():
        row = {
            'Location': location,
            'Type': insight.location_type,
            'Reviews': insight.total_reviews,
            'Overall_Score': insight.overall_sentiment,
            'Recommendation': insight.recommendation_score,
            'Strengths': ', '.join(insight.strengths),
            'Weaknesses': ', '.join(insight.weaknesses)
        }
        for aspect in TOURISM_ASPECTS:
            row[f'{aspect}_score'] = insight.aspect_scores.get(aspect, None)
            row[f'{aspect}_count'] = insight.aspect_counts.get(aspect, 0)
        insight_data.append(row)
    
    insights_df = pd.DataFrame(insight_data)
    insights_df.to_csv('data/location_insights.csv', index=False)
    print("   ✅ Saved location_insights.csv")
    
    total_time = time.time() - start_time
    print(f"\n⏱️ Total analysis time: {total_time:.1f} seconds")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return {
        'pipeline': pipeline,
        'insights': insights,
        'aspect_stats': aspect_stats
    }


if __name__ == '__main__':
    results = main()
