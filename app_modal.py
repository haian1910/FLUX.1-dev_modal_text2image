import modal
app = modal.App("multimodel-image-gen")

# Start with a single working model first, then add others
image = (
    modal.Image.debian_slim()
    .pip_install(
        "numpy==1.24.3",
        "fastapi==0.104.1",
        "uvicorn[standard]==0.24.0",
        "diffusers==0.30.3",
        "transformers==4.35.2",
        "accelerate==0.24.1",
        "safetensors==0.4.1",
        "huggingface_hub==0.25.2",
        "torch==2.1.1",
        "torchvision==0.16.1",
        "pillow==10.1.0",
        "requests==2.31.0"
    )
)

volume = modal.Volume.from_name("multimodel-cache", create_if_missing=True)
MODEL_DIR = "/model"

# Test with SD 1.5 first - it's most reliable and fastest to load
@app.cls(
    gpu="T4",
    image=image,
    volumes={MODEL_DIR: volume},
    timeout=1200,  # 20 minutes timeout
    container_idle_timeout=300,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class SD15Model:
    @modal.enter()
    def setup(self):
        print("🚀 Starting SD 1.5 model setup...")
        try:
            import torch
            from diffusers import StableDiffusionPipeline
            
            print(f"PyTorch version: {torch.__version__}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"CUDA device: {torch.cuda.get_device_name()}")
            
            print("📥 Loading Stable Diffusion 1.5...")
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            
            # Use public model that doesn't require token
            self.pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch_dtype,
                cache_dir=f"{MODEL_DIR}/sd15",
                safety_checker=None,
                requires_safety_checker=False
            )
            
            if torch.cuda.is_available():
                self.pipe = self.pipe.to("cuda")
                print("✅ Model moved to CUDA")
            
            # Enable memory efficient attention
            self.pipe.enable_attention_slicing()
            
            print("✅ SD 1.5 pipeline loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error in setup: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    @modal.method()
    def generate(self, prompt: str, num_inference_steps: int = 20, guidance_scale: float = 7.5, width: int = 512, height: int = 512) -> bytes:
        print(f"🎨 Generating image: '{prompt[:50]}...'")
        try:
            import torch
            import io
            
            generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(42)
            
            result = self.pipe(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=generator
            )
            
            image = result.images[0]
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            
            print("✅ Image generated successfully!")
            return buf.getvalue()
            
        except Exception as e:
            print(f"❌ Error in generation: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

# Simple FastAPI app for testing
@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse
    from pydantic import BaseModel
    import tempfile
    import os
    
    web_app = FastAPI(title="Multi-Model Image Generation API - Debug", version="2.0.0")
    
    # Initialize only SD15 for now
    sd15_model = SD15Model()

    class GenerateRequest(BaseModel):
        prompt: str
        model: str = "sd15"  # Only sd15 for now
        num_inference_steps: int = 20
        guidance_scale: float = 7.5
        width: int = 512
        height: int = 512

    @web_app.get("/")
    async def root():
        return {
            "message": "Multi-Model Image Generation API - Debug Mode",
            "status": "running",
            "available_models": ["sd15"],
            "debug": True
        }

    @web_app.get("/health")
    async def health():
        return {"status": "healthy", "models": ["sd15"]}

    @web_app.get("/models")
    async def list_models():
        return {
            "models": {
                "sd15": {
                    "name": "Stable Diffusion 1.5",
                    "quality": "Medium",
                    "speed": "Fast",
                    "resolution": "512x512",
                    "default_steps": 20,
                    "default_guidance": 7.5,
                    "status": "available"
                }
            }
        }

    @web_app.post("/generate/")
    async def generate_image(request: GenerateRequest):
        print(f"📨 Received request: {request.model} - {request.prompt[:50]}...")
        
        try:
            if request.model != "sd15":
                raise HTTPException(status_code=400, detail="Only 'sd15' model available in debug mode")
            
            print("🔄 Calling SD15 model...")
            image_bytes = sd15_model.generate.remote(
                request.prompt,
                request.num_inference_steps,
                request.guidance_scale,
                request.width,
                request.height
            )
            
            print("💾 Saving temporary file...")
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                tmp_file.write(image_bytes)
                tmp_path = tmp_file.name

            print(f"✅ Generated image saved to {tmp_path}")
            
            return FileResponse(
                tmp_path,
                media_type="image/png",
                filename=f"generated_sd15.png",
                headers={"Content-Disposition": f"attachment; filename=generated_sd15.png"}
            )
            
        except Exception as e:
            print(f"❌ Error in generate_image: {str(e)}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    return web_app

# Simple test function
@app.function(image=image)
def test_model():
    """Test function to verify model loading"""
    try:
        print("🧪 Testing model initialization...")
        model = SD15Model()
        print("✅ Model initialized successfully!")
        
        print("🎨 Testing image generation...")
        result = model.generate.remote("a simple test image")
        print(f"✅ Generation successful! Image size: {len(result)} bytes")
        return "Test passed!"
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise