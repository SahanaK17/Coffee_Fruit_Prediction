"""
Explainable Yield Decision Support Layer
Infers possible root health issues and generates yield improvement recommendations
based on environmental and soil conditions.
"""

from typing import Dict, List
import src.root_health_explainer as root_health_explainer

def analyze_yield_factors(
    plant_yield_kg: float,
    soil_data: dict,
    weather_data: dict,
    visual_features: dict
) -> List[str]:
    """
    Identifies possible stress factors of low yield based on soil, weather, and visual metrics.
    Uses the new root_health_explainer engine.
    """
    return root_health_explainer.identify_possible_stress_factors(soil_data, weather_data, visual_features)


def infer_root_health(soil_data: dict, weather_data: dict, visual_features: dict = None) -> List[str]:
    """
    Infers potential root-zone and crop health issues.
    Combines both soil/weather insights and visual physiological insights.
    """
    insights = root_health_explainer.infer_root_health_issues(soil_data, weather_data)
    
    if visual_features:
        visual_insights = root_health_explainer.generate_crop_health_insights(visual_features, is_same_location=False)
        insights.extend(visual_insights)
        
    return insights


def generate_recommendations(causes: List[str], insights: List[str]) -> List[str]:
    """
    Generates actionable recommendations based on stress factors and insights.
    Always guarantees at least 2 recommendations.
    """
    recommendations = []
    
    causes_str = " ".join(causes).lower()
    insights_str = " ".join(insights).lower()
    
    if "acidic" in causes_str or "acidity" in insights_str or "acidic" in insights_str:
        recommendations.append("Apply agricultural lime or wood ash to balance soil pH")
        
    if "alkaline" in causes_str or "lockout" in insights_str or "alkaline" in insights_str:
        recommendations.append("Use organic compost or sulfur to lower pH gradually")
        
    if "fertility" in causes_str or "fertility" in insights_str or "organic" in causes_str:
        recommendations.append("Apply organic compost and targeted NPK fertilizers to boost fertility")
        
    if "retention" in causes_str or "dehydration" in insights_str or "sand" in causes_str:
        recommendations.append("Apply heavy organic mulch to improve root-zone moisture retention")
        
    if "waterlogging" in causes_str or "aeration" in insights_str or "clay" in causes_str:
        recommendations.append("Improve root-zone aeration and field drainage")
        
    if "moisture stress" in causes_str or "heat stress" in causes_str or "imbalance" in causes_str:
        recommendations.append("Increase irrigation frequency and maintain shade-tree canopy")
        
    if "density" in causes_str or "formation" in causes_str or "physiological" in causes_str:
        recommendations.append("Implement corrective pruning to encourage new productive branches")
        
    if "delayed" in causes_str or "ripening" in causes_str or "maturation" in insights_str:
        recommendations.append("Monitor nutrient supplementation, especially potassium, to support fruit ripening")
        
    # Guarantee at least 2 recommendations
    if len(recommendations) == 0:
        recommendations.append("Increase organic compost application to support general plant vigor")
        recommendations.append("Improve irrigation consistency to prevent moisture-related stress")
    elif len(recommendations) == 1:
        if "compost" not in recommendations[0].lower():
            recommendations.append("Increase organic compost application to support general plant vigor")
        else:
            recommendations.append("Improve irrigation consistency to prevent moisture-related stress")
            
    # Limit to top 4 recommendations to keep UI clean
    return recommendations[:4]
