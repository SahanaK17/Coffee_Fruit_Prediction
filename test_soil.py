import sys
sys.path.append('.')
from src.soil_service import get_soil_data
import requests

lat, lon = 12.9189, 74.8525
print("Fetching soil data for", lat, lon)
try:
    params = [
        ("lat", lat),
        ("lon", lon),
        ("depth", "0-5cm"),
        ("value", "mean"),
    ]
    for prop in ["phh2o", "clay", "sand", "silt", "soc"]:
        params.append(("property", prop))
        
    url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
    response = requests.get(url, params=params, timeout=15)
    print("Status code:", response.status_code)
    try:
        print("Response:", response.json())
    except:
        print("Raw text:", response.text)
except Exception as e:
    print("Error:", e)
