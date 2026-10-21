import requests
import os

def test_yield_api():
    url = "http://localhost:8000/predict/yield"
    # Use a sample image from the project if available, or just check the endpoint existence
    image_path = "test_download.png" # Existing image in root
    
    if not os.path.exists(image_path):
        print(f"Test image {image_path} not found. Skipping API call.")
        return

    print(f"Testing Yield API with {image_path}...")
    try:
        with open(image_path, "rb") as f:
            files = {"file": ("test.png", f, "image/png")}
            response = requests.post(url, files=files)
            
        if response.status_code == 200:
            data = response.json()
            print("Response Success!")
            print(f"Fruit Count: {data.get('fruit_count')}")
            print(f"Avg Size: {data.get('avg_fruit_size')}")
            print(f"Variance: {data.get('fruit_variance')}")
            print(f"Canopy Coverage: {data.get('canopy_coverage')}")
            print(f"Confidence: {data.get('confidence')}")
            print(f"Model: {data.get('model_architecture')}")
            print(f"Predicted Yield: {data.get('predicted_yield')} kg")
            print("Feature Importances:", data.get('feature_importances'))
        else:
            print(f"Response Failed: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Note: Server must be running for this to work
    print("This script assumes the FastAPI server is running at http://localhost:8000")
    test_yield_api()
