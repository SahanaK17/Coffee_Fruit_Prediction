"""
Root Health & Crop Explainability Engine
Generates scientifically realistic agricultural insights for plant-level yield prediction.
Guarantees at least TWO meaningful agricultural reasons/insights per function.
"""

from typing import List

def infer_root_health_issues(soil_data: dict, weather_data: dict) -> List[str]:
    """
    Infers possible root-zone issues based on soil and weather data.
    Always returns at least 2 insights.
    """
    insights = []
    
    ph = soil_data.get("ph", 6.5)
    soc = soil_data.get("soc", 25.0)
    sand = soil_data.get("sand", 40.0)
    clay = soil_data.get("clay", 30.0)
    rain_30d = weather_data.get("rainfall_last_30_days", 100.0)
    
    if ph < 5.5:
        insights.append("Possible nutrient absorption inefficiency due to acidic root-zone conditions")
    elif ph > 7.5:
        insights.append("Likely micronutrient lockout limiting root absorption efficiency")
        
    if soc < 20.0:
        insights.append("Reduced root-zone fertility linked to weak organic nutrient availability")
        insights.append("Possible reduction in beneficial soil microbial support")
        
    if sand > 60.0:
        insights.append("Potential root dehydration risk due to low water retention in sandy soil")
        
    if clay > 60.0:
        insights.append("Possible root aeration stress from high clay density")
        
    if rain_30d < 50.0:
        insights.append("Likely root moisture stress caused by insufficient recent rainfall")
        
    # Guarantee at least 2 insights
    if len(insights) == 0:
        insights.append("Possible minor limitations in root nutrient uptake efficiency")
        insights.append("Potential sub-optimal root-zone microbial activity affecting vigor")
    elif len(insights) == 1:
        insights.append("Possible sub-optimal organic nutrient availability in the root zone")
        
    return insights


def generate_crop_health_insights(visual_features: dict, is_same_location: bool = False) -> List[str]:
    """
    Infers physiological stress based on visual cues (fruit density, ripeness).
    Particularly useful when soil/weather are shared (Same Location mode).
    Always returns at least 2 insights.
    """
    insights = []
    
    fruit_count = visual_features.get("fruit_count", 50)
    ripeness_ratio = visual_features.get("ripeness_ratio", 0.5)
    
    if fruit_count < 40:
        insights.append("Possible internal physiological stress reducing reproductive efficiency")
        insights.append("Potential localized nutrient uptake limitation affecting fruit density")
        
    if ripeness_ratio < 0.4:
        insights.append("Likely delayed maturation indicating possible physiological stress")
        insights.append("Possible potassium assimilation inefficiency slowing fruit development")
        
    # Same location specific inferences
    if is_same_location:
        if fruit_count < 50:
            insights.append("Reduced plant vigor compared to neighboring plant sharing identical soil")
        if visual_features.get("consistency_score", 1.0) < 0.6:
            insights.append("Inconsistent fruit clustering suggesting uneven physiological resource allocation")
            
    # Guarantee at least 2 insights
    if len(insights) == 0:
        insights.append("Possible minor physiological stress affecting overall plant vigor")
        insights.append("Potential micro-variations in nutrient absorption efficiency")
    elif len(insights) == 1:
        insights.append("Possible localized micro-climate stress affecting fruit development")
        
    return insights


def identify_possible_stress_factors(soil_data: dict, weather_data: dict, visual_features: dict) -> List[str]:
    """
    Combines environmental and visual data to output possible stress factors.
    Always returns at least 2 factors.
    """
    factors = []
    
    # Soil stress
    ph = soil_data.get("ph", 6.5)
    if ph < 5.5:
        factors.append("Soil acidity stress")
    elif ph > 7.5:
        factors.append("Soil alkalinity stress")
        
    soc = soil_data.get("soc", 25.0)
    if soc < 20.0:
        factors.append("Weak organic nutrient availability")
        
    sand = soil_data.get("sand", 40.0)
    if sand > 60.0:
        factors.append("Imbalance in soil texture composition (high sand)")
        
    # Weather stress
    temp = weather_data.get("temperature", 25.0)
    if temp > 30.0:
        factors.append("Physiological heat stress leading to increased water demand")
        
    rain_30d = weather_data.get("rainfall_last_30_days", 100.0)
    if rain_30d < 50.0:
        factors.append("Moisture imbalance affecting fruit development support")
        
    # Visual stress
    if visual_features.get("fruit_count", 50) < 40:
        factors.append("Potential physiological stress limiting fruit formation")
        
    if visual_features.get("ripeness_ratio", 0.5) < 0.4:
        factors.append("Possible nutrient or metabolic stress delaying ripening")
        
    # Guarantee at least 2 factors
    if len(factors) == 0:
        factors.append("Minor physiological stress affecting growth rate")
        factors.append("Possible transient moisture or nutrient micro-imbalance")
    elif len(factors) == 1:
        factors.append("Potential sub-clinical root-zone stress factor")
        
    return factors
