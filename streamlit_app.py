import streamlit as st
import requests
from PIL import Image
import io
import time
import json

# Configure Streamlit page
st.set_page_config(
    page_title="Multi-Model Image Generator",
    page_icon="🎨",
    layout="wide"
)

# App title and description
st.title("🎨 Multi-Model Image Generator")
st.markdown("Generate images using different AI models, each optimized for different use cases.")

# API endpoint - Update this with your Modal URL
API_URL = "https://vuhaiandp2017--multimodel-image-gen-fastapi-app.modal.run"

# Check API availability
@st.cache_data(ttl=60)  # Cache for 1 minute
def check_api_status():
    """Check if API is available and get model information"""
    try:
        # Test basic connectivity
        response = requests.get(f"{API_URL}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            
            # Get available models
            try:
                models_response = requests.get(f"{API_URL}/models", timeout=10)
                if models_response.status_code == 200:
                    models_data = models_response.json()
                    return True, health_data, models_data.get("models", {})
                else:
                    # If models endpoint fails, use basic model info
                    return True, health_data, {"sd15": {"name": "Stable Diffusion 1.5", "status": "available"}}
            except:
                return True, health_data, {"sd15": {"name": "Stable Diffusion 1.5", "status": "available"}}
        else:
            return False, None, {}
    except Exception as e:
        return False, str(e), {}

# Check API status
api_available, api_info, available_models = check_api_status()

if not api_available:
    st.error("🚨 **API Connection Failed**")
    st.error(f"Could not connect to: `{API_URL}`")
    if isinstance(api_info, str):
        st.error(f"Error: {api_info}")
    
    st.markdown("### 🛠️ Troubleshooting Steps:")
    st.markdown("1. Make sure your Modal app is deployed: `modal deploy app_modal.py`")
    st.markdown("2. Check if the API URL is correct")
    st.markdown("3. Verify the Modal app is running in the dashboard")
    st.markdown("4. Try the test script: `python test.py`")
    st.stop()

# Display API status
st.success("✅ **API Connected Successfully**")
if api_info:
    with st.expander("🔍 API Information"):
        st.json(api_info)

# Model information - combine static info with API data
default_model_info = {
    "sd15": {
        "name": "Stable Diffusion 1.5",
        "description": "⚡ Fast, good quality (~5-10s)",
        "default_steps": 20,
        "default_guidance": 7.5,
        "max_steps": 50,
        "resolution": (512, 512),
        "available": True
    },
    "sdxl": {
        "name": "Stable Diffusion XL", 
        "description": "⚖️ Balanced quality/speed (~15-25s)",
        "default_steps": 25,
        "default_guidance": 7.5,
        "max_steps": 50,
        "resolution": (1024, 1024),
        "available": False
    },
    "flux": {
        "name": "FLUX.1-dev",
        "description": "🏆 Highest quality, slowest (~30-60s)",
        "default_steps": 20,
        "default_guidance": 3.5,
        "max_steps": 50,
        "resolution": (1024, 1024),
        "available": False
    },
    "lightning": {
        "name": "SDXL Lightning",
        "description": "🚀 Fastest generation (~2-5s)",
        "default_steps": 4,
        "default_guidance": 1.0,
        "max_steps": 8,
        "resolution": (1024, 1024),
        "available": False
    }
}

# Update availability based on API response
for model_id in default_model_info:
    if model_id in available_models:
        default_model_info[model_id]["available"] = True

# Filter only available models
available_model_keys = [k for k, v in default_model_info.items() if v["available"]]

if not available_model_keys:
    st.error("❌ No models are currently available!")
    st.stop()

# Sidebar for model selection and parameters
st.sidebar.header("Model Configuration")

# Model selection - only show available models
selected_model = st.sidebar.selectbox(
    "Choose Model:",
    options=available_model_keys,
    format_func=lambda x: f"{default_model_info[x]['name']} - {default_model_info[x]['description']}"
)

# Show unavailable models as info
unavailable_models = [k for k, v in default_model_info.items() if not v["available"]]
if unavailable_models:
    with st.sidebar.expander("📋 Unavailable Models (Coming Soon)"):
        for model_id in unavailable_models:
            st.write(f"🔒 {default_model_info[model_id]['name']}")

# Display selected model information
model_info = default_model_info[selected_model]
st.sidebar.markdown(f"**Selected:** {model_info['name']}")
st.sidebar.markdown(f"**Resolution:** {model_info['resolution'][0]}x{model_info['resolution'][1]}")

# Advanced parameters
st.sidebar.subheader("Generation Parameters")

num_inference_steps = st.sidebar.slider(
    "Inference Steps:",
    min_value=1,
    max_value=model_info['max_steps'],
    value=model_info['default_steps'],
    help="More steps = better quality, but slower generation"
)

guidance_scale = st.sidebar.slider(
    "Guidance Scale:",
    min_value=1.0,
    max_value=20.0,
    value=model_info['default_guidance'],
    step=0.5,
    help="Higher values = more adherence to prompt, but may reduce creativity"
)

# Custom resolution (optional)
st.sidebar.subheader("Custom Resolution (Optional)")
use_custom_resolution = st.sidebar.checkbox("Use custom resolution")

if use_custom_resolution:
    col1, col2 = st.sidebar.columns(2)
    with col1:
        custom_width = st.number_input(
            "Width:", 
            min_value=256, 
            max_value=1536, 
            value=model_info['resolution'][0],
            step=64
        )
    with col2:
        custom_height = st.number_input(
            "Height:", 
            min_value=256, 
            max_value=1536, 
            value=model_info['resolution'][1],
            step=64
        )
else:
    custom_width = model_info['resolution'][0]
    custom_height = model_info['resolution'][1]

# Main content area
st.header("Prompt Input")
prompt = st.text_area(
    "Enter your image description:",
    placeholder="A serene landscape with mountains and a lake at sunset...",
    height=100,
    key="prompt_input"
)

# Example prompts
st.subheader("💡 Example Prompts")
example_col1, example_col2, example_col3 = st.columns(3)

with example_col1:
    if st.button("🌅 Landscape"):
        st.session_state.prompt_input = "A breathtaking mountain landscape with a crystal clear lake reflecting the sunset, photorealistic, highly detailed"
        st.rerun()

with example_col2:
    if st.button("🎨 Artistic"):
        st.session_state.prompt_input = "A vibrant abstract painting with flowing colors, oil painting style, modern art"
        st.rerun()

with example_col3:
    if st.button("🐱 Portrait"):
        st.session_state.prompt_input = "A cute fluffy cat sitting by a window, natural lighting, portrait photography style"
        st.rerun()

# Generation section
st.header("Generate Image")

col1, col2 = st.columns([1, 3])

with col1:
    generate_button = st.button("🎨 Generate Image", type="primary")

with col2:
    if generate_button and not prompt.strip():
        st.error("Please enter a prompt first!")
        generate_button = False

# Generation logic
if generate_button and prompt.strip():
    try:
        with st.spinner(f"Generating image with {model_info['name']}... This may take a while."):
            # Show generation parameters
            with st.expander("🔧 Generation Parameters", expanded=False):
                st.json({
                    "model": selected_model,
                    "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                    "steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "resolution": f"{custom_width}x{custom_height}"
                })
            
            # Prepare request data
            request_data = {
                "prompt": prompt,
                "model": selected_model,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "width": custom_width,
                "height": custom_height
            }
            
            # Record start time
            start_time = time.time()
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Make API request
            try:
                status_text.text("🚀 Sending request to API...")
                progress_bar.progress(20)
                
                response = requests.post(
                    f"{API_URL}/generate/",
                    json=request_data,
                    timeout=300  # 5 minutes timeout
                )
                
                progress_bar.progress(80)
                status_text.text("📦 Processing response...")
                
            except requests.exceptions.Timeout:
                progress_bar.empty()
                status_text.empty()
                st.error("⏱️ Request timed out. The image generation is taking longer than expected. Please try again or use a faster model.")
                st.stop()
            except requests.exceptions.ConnectionError:
                progress_bar.empty()
                status_text.empty()
                st.error("🔌 Unable to connect to the API. Please check if the Modal app is deployed and running.")
                st.stop()
            
            # Calculate generation time
            generation_time = time.time() - start_time
            progress_bar.progress(100)
            status_text.text("✅ Generation complete!")
            
            if response.status_code == 200:
                # Display the generated image
                image = Image.open(io.BytesIO(response.content))
                
                progress_bar.empty()
                status_text.empty()
                
                st.success(f"✅ Image generated successfully in {generation_time:.1f} seconds!")
                
                # Create columns for image display
                img_col1, img_col2, img_col3 = st.columns([1, 2, 1])
                
                with img_col2:
                    st.image(image, caption=f"Generated with {model_info['name']}", width=None)
                
                # Display generation info
                st.subheader("Generation Details")
                info_col1, info_col2, info_col3, info_col4 = st.columns(4)
                
                with info_col1:
                    st.metric("Model", model_info['name'])
                with info_col2:
                    st.metric("Generation Time", f"{generation_time:.1f}s")
                with info_col3:
                    st.metric("Steps", num_inference_steps)
                with info_col4:
                    st.metric("Guidance Scale", guidance_scale)
                
                # Download button
                img_buffer = io.BytesIO()
                image.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                
                st.download_button(
                    label="📥 Download Image",
                    data=img_buffer.getvalue(),
                    file_name=f"generated_{selected_model}_{int(time.time())}.png",
                    mime="image/png"
                )
                
                # Store in session state for comparison
                if 'generated_images' not in st.session_state:
                    st.session_state.generated_images = []
                
                st.session_state.generated_images.append({
                    'image': image,
                    'model': selected_model,
                    'prompt': prompt,
                    'time': generation_time,
                    'timestamp': time.time(),
                    'parameters': {
                        'steps': num_inference_steps,
                        'guidance': guidance_scale,
                        'resolution': f"{custom_width}x{custom_height}"
                    }
                })
                
                # Keep only last 10 images
                if len(st.session_state.generated_images) > 10:
                    st.session_state.generated_images = st.session_state.generated_images[-10:]
            
            else:
                progress_bar.empty()
                status_text.empty()
                
                st.error(f"❌ Error generating image: {response.status_code}")
                
                # Try to parse error details
                try:
                    error_detail = response.json()
                    st.error(f"**Details:** {error_detail.get('detail', 'Unknown error')}")
                    
                    # Show debug info
                    with st.expander("🐛 Debug Information"):
                        st.json(error_detail)
                        st.code(f"Request URL: {API_URL}/generate/")
                        st.code(f"Request Data: {json.dumps(request_data, indent=2)}")
                        
                except:
                    st.error(f"**Response:** {response.text}")
    
    except Exception as e:
        st.error(f"❌ An unexpected error occurred: {str(e)}")
        with st.expander("🐛 Error Details"):
            st.exception(e)

# Image history/comparison section
if 'generated_images' in st.session_state and st.session_state.generated_images:
    st.header("🖼️ Recent Generations")
    
    # Display recent images in a grid
    num_images = len(st.session_state.generated_images)
    cols = st.columns(min(num_images, 4))
    
    for i, img_data in enumerate(reversed(st.session_state.generated_images[-4:])):
        with cols[i]:
            st.image(img_data['image'])
            st.caption(f"**{default_model_info[img_data['model']]['name']}**")
            st.caption(f"⏱️ {img_data['time']:.1f}s")
            
            # Show parameters in expander
            with st.expander("Details"):
                st.write(f"**Prompt:** {img_data['prompt'][:100]}{'...' if len(img_data['prompt']) > 100 else ''}")
                st.write(f"**Steps:** {img_data['parameters']['steps']}")
                st.write(f"**Guidance:** {img_data['parameters']['guidance']}")
                st.write(f"**Resolution:** {img_data['parameters']['resolution']}")
                
                # Individual download button
                img_buffer = io.BytesIO()
                img_data['image'].save(img_buffer, format='PNG')
                img_buffer.seek(0)
                
                st.download_button(
                    label="📥 Download",
                    data=img_buffer.getvalue(),
                    file_name=f"generated_{img_data['model']}_{int(img_data['timestamp'])}.png",
                    mime="image/png",
                    key=f"download_{i}"
                )

# Footer
st.markdown("---")

# Add some CSS for better styling
st.markdown("""
<style>
.stButton > button {
    width: 100%;
}
.metric-container {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
.stProgress .stProgress-bar {
    background-color: #ff4b4b;
}
</style>
""", unsafe_allow_html=True)

# Debug section in sidebar
with st.sidebar.expander("🔧 Debug Info"):
    st.write("**API URL:**")
    st.code(API_URL)
    st.write("**Available Models:**")
    st.json(available_models)
    if st.button("🔄 Refresh API Status"):
        st.cache_data.clear()
        st.rerun()