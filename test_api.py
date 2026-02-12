"""
Quick test script to verify the QA endpoint is working
"""
import requests

API_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("Testing /health endpoint...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}\n")

def test_qa_check(image_path):
    """Test QA check endpoint"""
    print(f"Testing /qa/check with image: {image_path}")
    
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{API_URL}/qa/check", files=files)
    
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}\n")

if __name__ == "__main__":
    # Test health
    test_health()
    
    # Test QA with a sample image (update path to your test image)
    # Uncomment and update path when you have a test image
    # test_qa_check("path/to/coffee_fruit_image.jpg")
    
    print("✅ Health check passed!")
    print("📝 To test QA endpoint, uncomment the test_qa_check line and provide an image path")
