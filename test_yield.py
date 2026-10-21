import requests

url = "http://localhost:8000/predict/yield"
files = {
    'file': ('test.jpg', b'dummy_content', 'image/jpeg')
}
data = {
    'lat': 20.0,
    'lon': 75.0,
    'price_per_kg': 200.0,
    'estimated_plant_maintenance_cost': 100.0,
    'soil_type': 'loamy',
    'water_retention': 'medium',
    'drainage': 'moderate'
}
try:
    response = requests.post(url, files=files, data=data)
    print("Status:", response.status_code)
    print("JSON:", response.json())
except Exception as e:
    print("Error:", e)
