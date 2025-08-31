import requests
import json
import time

def test_debug_api():
    """Test the debug version of the API"""
    print("🧪 Testing Debug API...")
    
    # Update this with your actual Modal URL
    base_url = "https://vuhaiandp2017--multimodel-image-gen-fastapi-app.modal.run"
    
    # Test 1: Health check
    print("\n1️⃣ Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=30)
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {str(e)}")
        return False
    
    # Test 2: Root endpoint
    print("\n2️⃣ Testing root endpoint...")
    try:
        response = requests.get(f"{base_url}/", timeout=30)
        if response.status_code == 200:
            print("✅ Root endpoint working")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Root endpoint error: {str(e)}")
    
    # Test 3: Models endpoint
    print("\n3️⃣ Testing models endpoint...")
    try:
        response = requests.get(f"{base_url}/models", timeout=30)
        if response.status_code == 200:
            print("✅ Models endpoint working")
            print(f"Available models: {json.dumps(response.json(), indent=2)}")
        else:
            print(f"❌ Models endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Models endpoint error: {str(e)}")
    
    # Test 4: Simple image generation
    print("\n4️⃣ Testing image generation...")
    try:
        data = {
            "prompt": "a simple red apple on a white background",
            "model": "sd15",
            "num_inference_steps": 10,  # Reduced steps for faster testing
            "guidance_scale": 7.5
        }
        
        print(f"Sending request: {json.dumps(data, indent=2)}")
        start_time = time.time()
        
        response = requests.post(
            f"{base_url}/generate/",
            json=data,
            timeout=300  # 5 minutes
        )
        
        generation_time = time.time() - start_time
        
        if response.status_code == 200:
            filename = f"debug_test_{int(time.time())}.png"
            with open(filename, "wb") as f:
                f.write(response.content)
            print(f"✅ Image generation successful!")
            print(f"⏱️  Generation time: {generation_time:.2f} seconds")
            print(f"📁 Image saved as: {filename}")
            print(f"📏 Image size: {len(response.content)} bytes")
            return True
        else:
            print(f"❌ Image generation failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"⏱️ Request timed out after {time.time() - start_time:.1f} seconds")
        return False
    except Exception as e:
        print(f"❌ Generation error: {str(e)}")
        return False

def check_modal_logs():
    """Instructions for checking Modal logs"""
    print("\n" + "="*50)
    print("🔍 HOW TO DEBUG MODAL ISSUES:")
    print("="*50)
    print("1. Go to https://modal.com/apps")
    print("2. Find your 'multimodel-image-gen' app")
    print("3. Click on 'Containers' tab")
    print("4. Click on a failed container to see logs")
    print("5. Look for error messages in the logs")
    print("\nCommon issues to look for:")
    print("- Import errors (missing packages)")
    print("- CUDA/GPU errors")
    print("- Model download failures")
    print("- Authentication errors (HF token)")
    print("- Out of memory errors")

if __name__ == "__main__":
    print("🚀 Multi-Model Debug Test")
    print("=" * 40)
    
    success = test_debug_api()
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 DEBUG TEST PASSED!")
        print("The API is working correctly.")
    else:
        print("❌ DEBUG TEST FAILED!")
        print("Check the issues above and Modal logs.")
        check_modal_logs()
    
    print("\n💡 Next steps:")
    if success:
        print("- Deploy the full multi-model version")
        print("- Test other models one by one")
        print("- Update Streamlit UI to use the working API")
    else:
        print("- Check Modal dashboard for container logs")
        print("- Verify Hugging Face secret is set correctly")
        print("- Try deploying the debug version first")