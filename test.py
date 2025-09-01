import requests
import json
import time
import sys

def test_api_endpoints():
    """Test basic API endpoints"""
    print("🧪 Testing API Endpoints...")
    
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
            return True, response.json()
        else:
            print(f"❌ Models endpoint failed: {response.status_code}")
            return False, None
    except Exception as e:
        print(f"❌ Models endpoint error: {str(e)}")
        return False, None

def test_model_generation(model_name, base_url, quick=False):
    """Test image generation for a specific model"""
    print(f"\n🎨 Testing {model_name.upper()} model...")
    
    # Model-specific configurations
    model_configs = {
        "sd15": {
            "steps": 10 if quick else 20,
            "guidance": 7.5,
            "width": 512,
            "height": 512,
            "timeout": 90  # Increased timeout
        },
        "sdxl": {
            "steps": 15 if quick else 25,
            "guidance": 7.5,
            "width": 1024,
            "height": 1024,
            "timeout": 180  # Increased timeout
        },
        "lightning": {
            "steps": 4,
            "guidance": 1.0,
            "width": 1024,
            "height": 1024,
            "timeout": 120  # Much longer timeout for Lightning
        },
        "flux": {
            "steps": 2 if quick else 2,  # Reasonable steps for testing
            "guidance": 3.5,
            "width": 1024,
            "height": 1024,
            "timeout": 180  # 3 minutes should be enough with A10G
        }
    }
    
    if model_name not in model_configs:
        print(f"❌ Unknown model: {model_name}")
        return False
    
    config = model_configs[model_name]
    
    try:
        data = {
            "prompt": f"a simple red apple on a white background, {model_name} style",
            "model": model_name,
            "num_inference_steps": config["steps"],
            "guidance_scale": config["guidance"],
            "width": config["width"],
            "height": config["height"]
        }
        
        print(f"Request: {json.dumps(data, indent=2)}")
        print(f"Expected timeout: {config['timeout']}s")
        
        start_time = time.time()
        
        response = requests.post(
            f"{base_url}/generate/",
            json=data,
            timeout=config["timeout"]
        )
        
        generation_time = time.time() - start_time
        
        if response.status_code == 200:
            filename = f"{model_name}_test_{int(time.time())}.png"
            with open(filename, "wb") as f:
                f.write(response.content)
            
            print(f"✅ {model_name.upper()} generation successful!")
            print(f"⏱️  Generation time: {generation_time:.2f} seconds")
            print(f"📁 Image saved as: {filename}")
            print(f"📏 Image size: {len(response.content)} bytes")
            return True
        else:
            print(f"❌ {model_name.upper()} generation failed: {response.status_code}")
            try:
                error_detail = response.json()
                print(f"Error details: {json.dumps(error_detail, indent=2)}")
            except:
                print(f"Response text: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"⏱️ {model_name.upper()} request timed out after {time.time() - start_time:.1f} seconds")
        return False
    except Exception as e:
        print(f"❌ {model_name.upper()} generation error: {str(e)}")
        return False

def test_all_models(base_url, quick=False):
    """Test all available models"""
    print("\n" + "="*50)
    print("🎯 TESTING ALL MODELS")
    print("="*50)
    
    models = ["sd15", "sdxl", "lightning", "flux"]
    results = {}
    
    for model in models:
        success = test_model_generation(model, base_url, quick)
        results[model] = success
        
        # Wait between tests to avoid overwhelming the API
        if not quick:
            print("⏳ Waiting 10 seconds before next test...")
            time.sleep(10)
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST RESULTS SUMMARY")
    print("="*50)
    
    for model, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{model.upper():12} - {status}")
    
    passed = sum(results.values())
    total = len(results)
    print(f"\nTotal: {passed}/{total} models passed")
    
    return results

def show_deployment_instructions():
    """Show deployment and troubleshooting instructions"""
    print("\n" + "="*50)
    print("🚀 DEPLOYMENT INSTRUCTIONS")
    print("="*50)
    print("1. Make sure you have Modal CLI installed and configured:")
    print("   pip install modal")
    print("   modal setup")
    print()
    print("2. Create Hugging Face secret in Modal dashboard:")
    print("   - Go to https://modal.com/secrets")
    print("   - Create secret named 'huggingface-secret'")
    print("   - Add your HF token from https://huggingface.co/settings/tokens")
    print()
    print("3. Deploy the app:")
    print("   modal deploy app_modal.py")
    print()
    print("4. Individual model testing (if main app fails):")
    print("   modal run app_modal.py::test_sd15")
    print("   modal run app_modal.py::test_sdxl")
    print("   modal run app_modal.py::test_lightning")
    print("   modal run app_modal.py::test_flux")
    print()
    print("📝 Common Issues:")
    print("- GPU availability: A100 needed for FLUX, A10G for SDXL/Lightning")
    print("- Model download time: First run takes longer")
    print("- VRAM: FLUX needs most memory, SD15 needs least")
    print("- HF Token: Some models require authentication")

def main():
    print("🚀 Multi-Model Image Generation Test Suite")
    print("=" * 50)
    
    # Parse command line arguments
    quick_mode = len(sys.argv) > 1 and sys.argv[1] == "quick"
    single_model = None
    
    if len(sys.argv) > 1 and sys.argv[1] in ["sd15", "sdxl", "lightning", "flux"]:
        single_model = sys.argv[1]
    elif len(sys.argv) > 2 and sys.argv[2] in ["sd15", "sdxl", "lightning", "flux"]:
        single_model = sys.argv[2]
    
    base_url = "https://vuhaiandp2017--multimodel-image-gen-fastapi-app.modal.run"
    
    # Test basic API endpoints first
    api_working, models_info = test_api_endpoints()
    
    if not api_working:
        print("\n❌ API endpoints are not working!")
        show_deployment_instructions()
        return False
    
    # Test specific model or all models
    if single_model:
        print(f"\n🎯 Testing single model: {single_model.upper()}")
        success = test_model_generation(single_model, base_url, quick_mode)
        
        if success:
            print(f"\n🎉 {single_model.upper()} TEST PASSED!")
        else:
            print(f"\n❌ {single_model.upper()} TEST FAILED!")
            print("\n🔍 Troubleshooting tips:")
            print(f"1. Check Modal logs for {single_model} model container")
            print("2. Verify GPU requirements are met")
            print("3. Check HuggingFace token permissions")
            
        return success
    else:
        # Test all models
        results = test_all_models(base_url, quick_mode)
        
        passed = sum(results.values())
        total = len(results)
        
        if passed == total:
            print(f"\n🎉 ALL TESTS PASSED! ({passed}/{total})")
            print("\n✅ Your multi-model API is ready to use!")
            print("Update the API_URL in streamlit_app.py and run:")
            print("streamlit run streamlit_app.py")
        else:
            print(f"\n⚠️  PARTIAL SUCCESS: {passed}/{total} models working")
            
            # Show which models failed
            failed_models = [model for model, success in results.items() if not success]
            print(f"Failed models: {', '.join(failed_models)}")
            
            if passed > 0:
                working_models = [model for model, success in results.items() if success]
                print(f"Working models: {', '.join(working_models)}")
                print("\n💡 You can still use the working models!")
            
            print("\n🔍 For failed models, check:")
            print("1. Modal dashboard for container logs")
            print("2. GPU availability (A100 for FLUX, A10G for SDXL)")
            print("3. HuggingFace token permissions")
        
        return passed > 0

def show_usage():
    """Show usage instructions"""
    print("Usage:")
    print("  python test.py                 # Test all models (full)")
    print("  python test.py quick           # Test all models (quick)")
    print("  python test.py sd15            # Test only SD 1.5")
    print("  python test.py sdxl            # Test only SDXL")
    print("  python test.py lightning       # Test only Lightning")
    print("  python test.py flux            # Test only FLUX")
    print("  python test.py quick sd15      # Test SD 1.5 quickly")
    print()
    print("Models:")
    print("  sd15      - Stable Diffusion 1.5 (T4 GPU, fastest)")
    print("  sdxl      - Stable Diffusion XL (A10G GPU, balanced)")
    print("  lightning - SDXL Lightning (A10G GPU, very fast)")
    print("  flux      - FLUX.1-dev (A100 GPU, highest quality)")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ["help", "-h", "--help"]:
        show_usage()
        sys.exit(0)
    
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)