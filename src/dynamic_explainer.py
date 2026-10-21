import random
from typing import Dict, List, Any

# =====================================================================
# TEMPLATES FOR DYNAMIC NATURAL LANGUAGE GENERATION
# =====================================================================

STRESS_NLG = {
    "moisture_nutrient_stress": [
        "Combined moisture deficit and weak organic fertility are severely limiting root absorption.",
        "Simultaneous low rainfall and poor soil carbon are creating a restrictive root-zone environment.",
        "Moisture imbalance coupled with suboptimal soil organics is leading to dual root-zone stress."
    ],
    "acidic_nutrient_lockout": [
        "Acidic soil conditions are actively causing micronutrient lockout, stunting plant development.",
        "Low pH is severely restricting root nutrient uptake efficiency.",
        "Soil acidity is triggering a systemic nutrient absorption bottleneck in the root zone."
    ],
    "alkaline_nutrient_lockout": [
        "Alkaline soil conditions are likely binding crucial trace minerals, preventing root uptake.",
        "High pH is causing a potential iron and zinc lockout at the root level.",
        "Suboptimal alkaline conditions are reducing overall nutrient transport efficiency."
    ],
    "root_dehydration": [
        "High sand content combined with low rainfall is causing rapid root dehydration.",
        "Excessive drainage from sandy soil under dry conditions is leading to severe moisture stress.",
        "Poor water retention capacity is creating a highly stressful, dehydrated root environment."
    ],
    "waterlogging_risk": [
        "Heavy clay composition is restricting root aeration, leading to possible suffocation.",
        "High clay density is creating a localized waterlogging risk, reducing root vigor.",
        "Poor soil drainage is likely causing a hypoxic (low oxygen) root-zone environment."
    ],
    "physiological_heat_stress": [
        "High temperatures are triggering physiological heat stress, dramatically increasing water demand.",
        "Elevated ambient heat is forcing the plant into a defensive survival state rather than fruit production.",
        "Heat-induced metabolic stress is actively slowing down reproductive development."
    ],
    "weak_reproductive_vigor": [
        "A noticeable lack of fruit density indicates a systemic breakdown in reproductive efficiency.",
        "Low fruit formation suggests an underlying physiological or localized nutrient limitation.",
        "Reduced floral initiation or fruit set is severely capping the plant's overall yield potential."
    ],
    "maturity_delay": [
        "A high proportion of unripe fruits indicates significant delayed maturation.",
        "Fruit ripening is heavily delayed, pointing to a potential potassium assimilation inefficiency.",
        "Metabolic processes required for fruit ripening appear to be severely stalled."
    ],
    "inconsistent_maturity": [
        "Fruits are developing at drastically different rates, suggesting uneven resource allocation.",
        "Highly inconsistent ripening points to fluctuating internal moisture or nutrient flow.",
        "Uneven maturity distribution indicates localized physiological stress across different branches."
    ],
    "localized_vigor_stress": [
        "Despite sharing identical environmental conditions, this plant shows significantly weaker vegetative vigor.",
        "Internal physiological limitations are causing this plant to underperform compared to its immediate neighbor.",
        "Micro-variations in the immediate root-zone or internal genetics are capping this plant's potential."
    ],
    "general_suboptimal_growth": [
        "Minor environmental fluctuations are causing transient, sub-clinical stress.",
        "A combination of minor factors is slightly depressing optimal growth rates.",
        "No single severe stress is present, but overall vigor is mildly limited."
    ],
    "economic_loss": [
        "Despite current biological conditions, production costs are exceeding potential revenue.",
        "Projected yield volume is currently insufficient to cover estimated maintenance costs.",
        "The plant's harvest value is falling below the economic break-even threshold."
    ],
    "yield_optimization": [
        "Biological conditions are favorable, allowing for minor optimizations to maximize profit.",
        "Current growth supports a profitable yield; focus can shift to long-term plant health.",
        "Economic viability is secure, offering an opportunity to refine nutrient efficiency."
    ]
}

REC_TEMPLATES = {
    "moisture_nutrient_stress": [
        {"action": "Apply organic compost and increase watering", "reason": "Compost improves nutrient availability while retaining much-needed moisture."},
        {"action": "Integrate heavy mulching and organic fertilizers", "reason": "Mulch locks in existing moisture, and organic matter slowly releases nutrients."}
    ],
    "acidic_nutrient_lockout": [
        {"action": "Apply agricultural lime or wood ash", "reason": "These cost-effective amendments will naturally raise the pH and unlock bound nutrients."},
        {"action": "Gradually incorporate calcium-rich organic matter", "reason": "Raises pH over time while improving overall soil structure."}
    ],
    "alkaline_nutrient_lockout": [
        {"action": "Add sulfur powder or pine needle compost", "reason": "Naturally acidifies the soil to free up trapped micronutrients."},
        {"action": "Use well-rotted cow dung manure", "reason": "Slowly lowers pH while providing a robust nutrient boost."}
    ],
    "root_dehydration": [
        {"action": "Implement deep, consistent watering and heavy mulching", "reason": "Counteracts rapid drainage in sandy soils by physically trapping moisture."},
        {"action": "Add biochar or organic compost to the root zone", "reason": "Physically alters sandy soil structure to dramatically improve water retention."}
    ],
    "waterlogging_risk": [
        {"action": "Loosen topsoil and improve field drainage trenches", "reason": "Breaks up heavy clay to immediately restore oxygen flow to the root zone."},
        {"action": "Incorporate coarse sand and compost", "reason": "Improves long-term soil aeration and prevents future waterlogging."}
    ],
    "physiological_heat_stress": [
        {"action": "Plant temporary shade crops or erect shade nets", "reason": "Directly lowers ambient temperature around the plant, reducing water loss."},
        {"action": "Increase early-morning irrigation frequency", "reason": "Ensures the plant is fully hydrated before peak midday heat arrives."}
    ],
    "weak_reproductive_vigor": [
        {"action": "Perform corrective structural pruning", "reason": "Redirects the plant's internal energy toward producing new, fruit-bearing branches."},
        {"action": "Apply targeted NPK fertilizers during flowering", "reason": "Provides the specific chemical building blocks needed for strong fruit set."}
    ],
    "maturity_delay": [
        {"action": "Apply wood ash or potassium-rich banana peel compost", "reason": "Potassium is the primary driver for fruit sizing and even ripening."},
        {"action": "Delay harvest and monitor potassium levels", "reason": "Gives the plant the necessary time and resources to finish the maturation cycle."}
    ],
    "inconsistent_maturity": [
        {"action": "Standardize watering schedules", "reason": "Prevents the 'boom and bust' water cycles that cause uneven fruit development."},
        {"action": "Selectively harvest only fully ripe fruits", "reason": "Reduces competition for resources, allowing remaining unripe fruits to finish developing."}
    ],
    "localized_vigor_stress": [
        {"action": "Conduct a localized root inspection and apply targeted compost", "reason": "Addresses potential hidden root damage or highly localized nutrient pockets."},
        {"action": "Prune competing neighboring plants", "reason": "Ensures this specific plant receives adequate sunlight and airflow."}
    ],
    "general_suboptimal_growth": [
        {"action": "Maintain basic agronomic hygiene", "reason": "Removing weeds and dry leaves allows the plant to focus energy entirely on growth."},
        {"action": "Apply a balanced, slow-release organic fertilizer", "reason": "Provides a steady baseline of nutrition to gently improve overall vigor."}
    ],
    "economic_loss": [
        {"action": "Review maintenance expenditures", "reason": "Reducing per-plant costs is necessary to achieve profitability."},
        {"action": "Implement high-efficiency fertilization", "reason": "Maximizes biological yield without proportionally increasing input costs."}
    ],
    "yield_optimization": [
        {"action": "Maintain current agronomic practices", "reason": "Existing practices are yielding a profitable harvest."},
        {"action": "Invest in soil organic capital", "reason": "Using profits to build soil carbon ensures long-term yield stability."}
    ]
}


# =====================================================================
# DYNAMIC ANALYSIS ENGINE
# =====================================================================

def evaluate_factors(soil: dict, weather: dict, visual: dict, is_same_location: bool, yield_delta_pct: float = 0.0, profit_status: str = "PROFIT") -> List[dict]:
    """
    Evaluates combined conditions and scores multiple stress factors.
    Returns a list of dicts: {"id": str, "weight": float, "name": str, "severity": str}
    """
    ph = soil.get("ph", 6.5)
    soc = soil.get("soc", 25.0)
    sand = soil.get("sand", 40.0)
    clay = soil.get("clay", 30.0)
    
    temp = weather.get("temperature", 25.0)
    rain_30d = weather.get("rainfall_last_30_days", 100.0)
    
    fruit_count = visual.get("fruit_count", 50)
    ripeness_ratio = visual.get("ripeness_ratio", 0.5)
    consistency_score = visual.get("consistency_score", 1.0)
    
    stresses = []
    
    # 1. Moisture & Nutrient Stress
    if soc < 20.0 and rain_30d < 60.0:
        stresses.append({"id": "moisture_nutrient_stress", "weight": 9.0, "name": "Moisture & Nutrient Deficit", "severity": "high"})
    elif soc < 20.0:
        stresses.append({"id": "moisture_nutrient_stress", "weight": 6.0, "name": "Low Organic Fertility", "severity": "medium"})
        
    # 2. Acidic Lockout
    if ph < 5.5:
        w = 8.5 if ph < 5.0 else 6.5
        sev = "high" if w > 7.0 else "medium"
        stresses.append({"id": "acidic_nutrient_lockout", "weight": w, "name": "Acidic Nutrient Lockout", "severity": sev})
        
    # 3. Alkaline Lockout
    if ph > 7.5:
        w = 8.0 if ph > 8.0 else 6.0
        sev = "high" if w > 7.0 else "medium"
        stresses.append({"id": "alkaline_nutrient_lockout", "weight": w, "name": "Alkaline Nutrient Lockout", "severity": sev})
        
    # 4. Root Dehydration
    if sand > 60.0 and rain_30d < 50.0:
        stresses.append({"id": "root_dehydration", "weight": 8.5, "name": "Severe Root Dehydration", "severity": "high"})
    elif sand > 65.0:
        stresses.append({"id": "root_dehydration", "weight": 5.5, "name": "Poor Water Retention", "severity": "medium"})
        
    # 5. Waterlogging Risk
    if clay > 60.0:
        stresses.append({"id": "waterlogging_risk", "weight": 7.0, "name": "Root Aeration Stress", "severity": "medium"})
        
    # 6. Heat Stress
    if temp > 30.0 and rain_30d < 60.0:
        stresses.append({"id": "physiological_heat_stress", "weight": 8.0, "name": "Compounded Heat Stress", "severity": "high"})
    elif temp > 32.0:
        stresses.append({"id": "physiological_heat_stress", "weight": 6.5, "name": "Physiological Heat Stress", "severity": "medium"})
        
    # 7. Weak Reproductive Vigor
    if fruit_count < 30:
        stresses.append({"id": "weak_reproductive_vigor", "weight": 9.5, "name": "Critical Reproductive Failure", "severity": "high"})
    elif fruit_count < 50:
        stresses.append({"id": "weak_reproductive_vigor", "weight": 6.0, "name": "Suboptimal Fruit Density", "severity": "medium"})
        
    # 8. Maturity Delay
    if ripeness_ratio < 0.3:
        stresses.append({"id": "maturity_delay", "weight": 7.5, "name": "Severe Maturation Delay", "severity": "medium"})
    elif ripeness_ratio < 0.5:
        stresses.append({"id": "maturity_delay", "weight": 5.0, "name": "Delayed Ripening", "severity": "low"})
        
    # 9. Inconsistent Maturity
    if consistency_score < 0.5:
        stresses.append({"id": "inconsistent_maturity", "weight": 7.0, "name": "Highly Uneven Maturation", "severity": "medium"})
        
    # 10. Same-Location Specifics
    if is_same_location and yield_delta_pct > 15.0:
        stresses.append({"id": "localized_vigor_stress", "weight": 9.0, "name": "Localized Plant Vigor Deficit", "severity": "high"})
        
    # 11. Economic Factors
    if profit_status == "LOSS":
        stresses.append({"id": "economic_loss", "weight": 10.0, "name": "Economic Viability Stress", "severity": "high"})
    elif profit_status == "PROFIT":
        stresses.append({"id": "yield_optimization", "weight": 2.0, "name": "Yield Optimization Potential", "severity": "low"})

    # Fallback if no major stress detected
    if not stresses:
        stresses.append({"id": "general_suboptimal_growth", "weight": 3.0, "name": "Suboptimal Growth Profile", "severity": "low"})
        
    return stresses


def generate_dynamic_insights(soil: dict, weather: dict, visual: dict, is_same_location: bool, yield_delta_pct: float = 0.0, profit_status: str = "PROFIT") -> Dict[str, Any]:
    """
    Main entrypoint for the dynamic explainability engine.
    Returns a complex dict with primary/secondary stresses, insights, and recommendations.
    """
    # 1. Score and rank factors
    factors = evaluate_factors(soil, weather, visual, is_same_location, yield_delta_pct, profit_status)
    factors.sort(key=lambda x: x["weight"], reverse=True)
    
    # Ensure at least two factors by injecting a generic one if needed
    if len(factors) == 1:
        if factors[0]["id"] != "general_suboptimal_growth":
            factors.append({"id": "general_suboptimal_growth", "weight": 2.0, "name": "Minor Environmental Fluctuations", "severity": "low"})
        else:
            factors.append({"id": "weak_reproductive_vigor", "weight": 1.0, "name": "Slight Vigor Limitation", "severity": "low"})
            
    primary_factor = factors[0]
    secondary_factor = factors[1]
    
    # 2. Generate natural language descriptions
    primary_desc = random.choice(STRESS_NLG[primary_factor["id"]])
    secondary_desc = random.choice(STRESS_NLG[secondary_factor["id"]])
    
    # 3. Generate structured lists for Root and Crop Health Insights
    root_insights = []
    crop_insights = []
    
    # Partition all detected factors into Root vs Crop based on ID
    root_ids = ["moisture_nutrient_stress", "acidic_nutrient_lockout", "alkaline_nutrient_lockout", "root_dehydration", "waterlogging_risk", "general_suboptimal_growth", "yield_optimization"]
    crop_ids = ["physiological_heat_stress", "weak_reproductive_vigor", "maturity_delay", "inconsistent_maturity", "localized_vigor_stress", "economic_loss"]
    
    for f in factors:
        desc = random.choice(STRESS_NLG[f["id"]])
        if f["id"] in root_ids and desc not in root_insights:
            root_insights.append(desc)
        elif f["id"] in crop_ids and desc not in crop_insights:
            crop_insights.append(desc)
            
    # Guarantee at least 2 in each list
    if len(root_insights) < 2:
        extra_root = [desc for desc in STRESS_NLG["general_suboptimal_growth"] if desc not in root_insights]
        root_insights.extend(extra_root[:2 - len(root_insights)])
        
    if len(crop_insights) < 2:
        extra_crop = [desc for desc in STRESS_NLG["weak_reproductive_vigor"] if desc not in crop_insights]
        crop_insights.extend(extra_crop[:2 - len(crop_insights)])
        
    # 4. Generate dynamic recommendations
    recs = []
    seen_actions = set()
    
    # Pull recommendations from top 3 factors
    for f in factors[:3]:
        # Pick 1 random recommendation per factor
        rec_list = REC_TEMPLATES[f["id"]]
        selected_rec = random.choice(rec_list)
        if selected_rec["action"] not in seen_actions:
            # Map severity to priority
            priority = "high" if f["severity"] == "high" else ("medium" if f["severity"] == "medium" else "low")
            recs.append({
                "action": selected_rec["action"],
                "reason": selected_rec["reason"],
                "priority": priority
            })
            seen_actions.add(selected_rec["action"])
            
    # Ensure at least 2 recommendations
    if len(recs) < 2:
        fallback = REC_TEMPLATES["general_suboptimal_growth"]
        for fb in fallback:
            if fb["action"] not in seen_actions:
                recs.append({"action": fb["action"], "reason": fb["reason"], "priority": "low"})
                if len(recs) >= 2:
                    break

    return {
        "primary_stress": {
            "name": primary_factor["name"],
            "severity": primary_factor["severity"],
            "description": primary_desc
        },
        "secondary_stress": {
            "name": secondary_factor["name"],
            "severity": secondary_factor["severity"],
            "description": secondary_desc
        },
        "root_insights": root_insights[:4],
        "crop_insights": crop_insights[:4],
        "recommendations": recs[:4]
    }
