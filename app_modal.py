import modal
app = modal.App("multimodel-image-gen")

# Enhanced image with all required dependencies
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
        "requests==2.31.0",
        "xformers==0.0.23"  # For memory efficiency
    )
)

volume = modal.Volume.from_name("multimodel-cache", create_if_missing=True)
MODEL_DIR = "/model"

# Stable Diffusion 1.5 Model (Working)
@app.cls(
    gpu="T4",
    image=image,
    volumes={MODEL_DIR: volume},
    timeout=1200,
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
            
            self.pipe.enable_attention_slicing()
            print("✅ SD 1.5 pipeline loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error in setup: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    @modal.method()
    def generate(self, prompt: str, num_inference_steps: int = 20, guidance_scale: float = 7.5, width: int = 512, height: int = 512) -> bytes:
        print(f"🎨 Generating SD15 image: '{prompt[:50]}...'")
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
            
            print("✅ SD15 image generated successfully!")
            return buf.getvalue()
            
        except Exception as e:
            print(f"❌ Error in SD15 generation: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

# Stable Diffusion XL Model
@app.cls(
    gpu="A10G",  # SDXL needs more VRAM
    image=image,
    volumes={MODEL_DIR: volume},
    timeout=1200,
    container_idle_timeout=300,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class SDXLModel:
    @modal.enter()
    def setup(self):
        print("🚀 Starting SDXL model setup...")
        try:
            import torch
            from diffusers import StableDiffusionXLPipeline
            
            print(f"PyTorch version: {torch.__version__}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"CUDA device: {torch.cuda.get_device_name()}")
            
            print("📥 Loading Stable Diffusion XL...")
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch_dtype,
                cache_dir=f"{MODEL_DIR}/sdxl",
                use_safetensors=True
            )
            
            if torch.cuda.is_available():
                self.pipe = self.pipe.to("cuda")
                print("✅ SDXL model moved to CUDA")
            
            # Enable memory efficient attention
            self.pipe.enable_attention_slicing()
            self.pipe.enable_model_cpu_offload()  # Helps with VRAM
            
            print("✅ SDXL pipeline loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error in SDXL setup: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    @modal.method()
    def generate(self, prompt: str, num_inference_steps: int = 25, guidance_scale: float = 7.5, width: int = 1024, height: int = 1024) -> bytes:
        print(f"🎨 Generating SDXL image: '{prompt[:50]}...'")
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
            
            print("✅ SDXL image generated successfully!")
            return buf.getvalue()
            
        except Exception as e:
            print(f"❌ Error in SDXL generation: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

# SDXL Lightning Model (Fast) - Fixed version
@app.cls(
    gpu="A10G",
    image=image,
    volumes={MODEL_DIR: volume},
    timeout=1800,  # Increased timeout for model loading
    container_idle_timeout=600,  # Keep container alive longer
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class SDXLLightningModel:
    @modal.enter()
    def setup(self):
        print("🚀 Starting SDXL Lightning model setup...")
        try:
            import torch
            from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
            
            print(f"PyTorch version: {torch.__version__}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"CUDA device: {torch.cuda.get_device_name()}")
            
            print("📥 Loading SDXL Lightning...")
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            
            # Try alternative Lightning model that's more reliable
            try:
                print("📥 Trying primary Lightning model...")
                self.pipe = StableDiffusionXLPipeline.from_pretrained(
                    "stabilityai/sdxl-turbo",  # More reliable than ByteDance version
                    torch_dtype=torch_dtype,
                    cache_dir=f"{MODEL_DIR}/sdxl_lightning",
                    use_safetensors=True
                )
                print("✅ Loaded SDXL Turbo (Lightning alternative)")
            except Exception as e:
                print(f"⚠️ Primary model failed: {e}")
                print("📥 Falling back to standard SDXL...")
                self.pipe = StableDiffusionXLPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-xl-base-1.0",
                    torch_dtype=torch_dtype,
                    cache_dir=f"{MODEL_DIR}/sdxl_lightning_fallback",
                    use_safetensors=True
                )
                print("✅ Loaded SDXL base as Lightning fallback")
            
            # Use Euler scheduler optimized for few steps
            self.pipe.scheduler = EulerDiscreteScheduler.from_config(
                self.pipe.scheduler.config,
                timestep_spacing="trailing"
            )
            
            if torch.cuda.is_available():
                self.pipe = self.pipe.to("cuda")
                print("✅ Lightning model moved to CUDA")
            
            # Enable all memory optimizations
            self.pipe.enable_attention_slicing()
            self.pipe.enable_model_cpu_offload()
            if hasattr(self.pipe, 'enable_xformers_memory_efficient_attention'):
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                    print("✅ XFormers memory optimization enabled")
                except:
                    print("⚠️ XFormers not available")
            
            print("✅ SDXL Lightning pipeline loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error in SDXL Lightning setup: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    @modal.method()
    def generate(self, prompt: str, num_inference_steps: int = 4, guidance_scale: float = 1.0, width: int = 1024, height: int = 1024) -> bytes:
        print(f"🎨 Generating SDXL Lightning image: '{prompt[:50]}...'")
        try:
            import torch
            import io
            
            generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(42)
            
            # Optimize for speed - use fewer steps
            actual_steps = min(num_inference_steps, 6)  # Cap at 6 steps for speed
            actual_guidance = max(guidance_scale, 1.0)  # Minimum guidance of 1.0
            
            print(f"⚡ Using {actual_steps} steps with guidance {actual_guidance}")
            
            result = self.pipe(
                prompt=prompt,
                num_inference_steps=actual_steps,
                guidance_scale=actual_guidance,
                width=width,
                height=height,
                generator=generator
            )
            
            image = result.images[0]
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            
            print("✅ SDXL Lightning image generated successfully!")
            return buf.getvalue()
            
        except Exception as e:
            print(f"❌ Error in SDXL Lightning generation: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

# FLUX Model - Using your working implementation
@app.cls(
    gpu="A10G",  # Your working config - A10G instead of A100
    image=image,
    volumes={MODEL_DIR: volume},
    timeout=1800,  # 30 minutes should be enough
    container_idle_timeout=600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    allow_concurrent_inputs=1,
)
class FLUXModel:
    @modal.enter()
    def setup(self):
        print("🚀 Starting FLUX model setup (using working implementation)...")
        try:
            import torch
            from diffusers import FluxPipeline
            import os
            
            print(f"PyTorch version: {torch.__version__}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"CUDA device: {torch.cuda.get_device_name()}")
            
            print("📥 Loading FLUX.1-dev...")
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            
            # Use your working approach
            self.pipe = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                torch_dtype=torch_dtype,
                cache_dir=MODEL_DIR,
                use_auth_token=os.environ.get("HF_TOKEN"),
            )
            
            if torch.cuda.is_available():
                self.pipe = self.pipe.to("cuda")
                print("✅ FLUX model moved to CUDA")
            
            print("✅ FLUX.1-dev pipeline loaded and ready")
            
        except Exception as e:
            print(f"❌ Error in FLUX setup: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    @modal.method()
    def generate(self, prompt: str, num_inference_steps: int = 20, guidance_scale: float = 3.5, width: int = 1024, height: int = 1024) -> bytes:
        print(f"🎨 Generating FLUX image: '{prompt[:50]}...'")
        
        try:
            import torch
            import io
            
            # Use your working generation approach
            generator = None
            if torch.cuda.is_available():
                generator = torch.Generator(device="cuda").manual_seed(42)
            
            print(f"⚡ Generating with {num_inference_steps} steps...")
            
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
            
            print("✅ FLUX image generated successfully!")
            return buf.getvalue()
            
        except Exception as e:
            print(f"❌ Error in FLUX generation: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

# FastAPI application
@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse
    from pydantic import BaseModel
    import tempfile
    import os
    
    web_app = FastAPI(title="Multi-Model Image Generation API", version="2.0.0")
    
    # Initialize all models
    sd15_model = SD15Model()
    sdxl_model = SDXLModel()
    lightning_model = SDXLLightningModel()
    flux_model = FLUXModel()

    class GenerateRequest(BaseModel):
        prompt: str
        model: str = "sd15"
        num_inference_steps: int = None  # Will be set based on model
        guidance_scale: float = None     # Will be set based on model
        width: int = None               # Will be set based on model
        height: int = None              # Will be set based on model

    @web_app.get("/")
    async def root():
        return {
            "message": "Multi-Model Image Generation API",
            "status": "running",
            "available_models": ["sd15", "sdxl", "lightning", "flux"],
            "version": "2.0.0"
        }

    @web_app.get("/health")
    async def health():
        return {
            "status": "healthy", 
            "models": ["sd15", "sdxl", "lightning", "flux"]
        }

    @web_app.get("/models")
    async def list_models():
        return {
            "models": {
                "sd15": {
                    "name": "Stable Diffusion 1.5",
                    "quality": "Good",
                    "speed": "Fast",
                    "resolution": "512x512",
                    "default_steps": 20,
                    "default_guidance": 7.5,
                    "status": "available",
                    "gpu_requirement": "T4"
                },
                "sdxl": {
                    "name": "Stable Diffusion XL",
                    "quality": "High",
                    "speed": "Medium",
                    "resolution": "1024x1024",
                    "default_steps": 25,
                    "default_guidance": 7.5,
                    "status": "available",
                    "gpu_requirement": "A10G"
                },
                "lightning": {
                    "name": "SDXL Lightning",
                    "quality": "Medium-High",
                    "speed": "Fastest",
                    "resolution": "1024x1024",
                    "default_steps": 4,
                    "default_guidance": 1.0,
                    "status": "available",
                    "gpu_requirement": "A10G"
                },
                "flux": {
                    "name": "FLUX.1-dev",
                    "quality": "Highest",
                    "speed": "Slowest",
                    "resolution": "1024x1024",
                    "default_steps": 20,
                    "default_guidance": 3.5,
                    "status": "available",
                    "gpu_requirement": "A100"
                }
            }
        }

    @web_app.post("/generate/")
    async def generate_image(request: GenerateRequest):
        print(f"📨 Received request: {request.model} - {request.prompt[:50]}...")
        
        # Model configuration defaults
        model_configs = {
            "sd15": {"steps": 20, "guidance": 7.5, "width": 512, "height": 512},
            "sdxl": {"steps": 25, "guidance": 7.5, "width": 1024, "height": 1024},
            "lightning": {"steps": 4, "guidance": 1.0, "width": 1024, "height": 1024},
            "flux": {"steps": 20, "guidance": 3.5, "width": 1024, "height": 1024}
        }
        
        if request.model not in model_configs:
            raise HTTPException(
                status_code=400, 
                detail=f"Model '{request.model}' not supported. Available: {list(model_configs.keys())}"
            )
        
        # Get model config and apply defaults
        config = model_configs[request.model]
        steps = request.num_inference_steps or config["steps"]
        guidance = request.guidance_scale or config["guidance"]
        width = request.width or config["width"]
        height = request.height or config["height"]
        
        try:
            print(f"🔄 Calling {request.model} model...")
            
            # Route to appropriate model
            if request.model == "sd15":
                image_bytes = sd15_model.generate.remote(
                    request.prompt, steps, guidance, width, height
                )
            elif request.model == "sdxl":
                image_bytes = sdxl_model.generate.remote(
                    request.prompt, steps, guidance, width, height
                )
            elif request.model == "lightning":
                image_bytes = lightning_model.generate.remote(
                    request.prompt, steps, guidance, width, height
                )
            elif request.model == "flux":
                image_bytes = flux_model.generate.remote(
                    request.prompt, steps, guidance, width, height
                )
            
            print("💾 Saving temporary file...")
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                tmp_file.write(image_bytes)
                tmp_path = tmp_file.name

            print(f"✅ Generated image saved to {tmp_path}")
            
            return FileResponse(
                tmp_path,
                media_type="image/png",
                filename=f"generated_{request.model}.png",
                headers={"Content-Disposition": f"attachment; filename=generated_{request.model}.png"}
            )
            
        except Exception as e:
            print(f"❌ Error in generate_image: {str(e)}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    return web_app

# Test functions for each model
@app.function(image=image)
def test_sd15():
    """Test SD 1.5 model"""
    try:
        print("🧪 Testing SD 1.5...")
        model = SD15Model()
        result = model.generate.remote("a simple red apple", num_inference_steps=10)
        print(f"✅ SD 1.5 test passed! Image size: {len(result)} bytes")
        return "SD 1.5 test passed!"
    except Exception as e:
        print(f"❌ SD 1.5 test failed: {str(e)}")
        raise

@app.function(image=image)
def test_sdxl():
    """Test SDXL model"""
    try:
        print("🧪 Testing SDXL...")
        model = SDXLModel()
        result = model.generate.remote("a simple red apple", num_inference_steps=15)
        print(f"✅ SDXL test passed! Image size: {len(result)} bytes")
        return "SDXL test passed!"
    except Exception as e:
        print(f"❌ SDXL test failed: {str(e)}")
        raise

@app.function(image=image)
def test_lightning():
    """Test SDXL Lightning model"""
    try:
        print("🧪 Testing SDXL Lightning...")
        model = SDXLLightningModel()
        result = model.generate.remote("a simple red apple", num_inference_steps=4)
        print(f"✅ Lightning test passed! Image size: {len(result)} bytes")
        return "Lightning test passed!"
    except Exception as e:
        print(f"❌ Lightning test failed: {str(e)}")
        raise

@app.function(image=image)
def test_flux():
    """Test FLUX model"""
    try:
        print("🧪 Testing FLUX...")
        model = FLUXModel()
        result = model.generate.remote("a simple red apple", num_inference_steps=15)
        print(f"✅ FLUX test passed! Image size: {len(result)} bytes")
        return "FLUX test passed!"
    except Exception as e:
        print(f"❌ FLUX test failed: {str(e)}")
        raise