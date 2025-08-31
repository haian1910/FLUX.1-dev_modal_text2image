# 🎨 Multi-Model Image Generation App

A simple image generation system with multiple AI models, featuring a FastAPI backend on Modal and a Streamlit frontend.

## 🤖 Available Models

| Model | Quality | Speed | Use Case |
|-------|---------|-------|----------|
| **FLUX.1-dev** | Highest | Slowest (30-60s) | Professional artwork |
| **Stable Diffusion XL** | High | Medium (15-25s) | Balanced quality/speed |
| **Stable Diffusion 1.5** | Good | Fast (5-10s) | Quick iterations |
| **SDXL Lightning** | Medium-High | Fastest (2-5s) | Rapid prototyping |

## 📁 Project Structure

```
multimodel-image-gen/
├── app_modal.py           # Modal backend with multiple models
├── streamlit_app.py       # Streamlit frontend UI
├── test.py               # API testing
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## ⚙️ Quick Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Modal

```bash
modal setup
```

### 3. Set Up Hugging Face Secret

1. Go to [Modal Dashboard](https://modal.com/secrets)
2. Create a secret named `huggingface-secret`
3. Add your HF token: Key: `HF_TOKEN`, Value: Your token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### 4. Deploy Backend

```bash
modal deploy app_modal.py
```

### 5. Update API URL

Copy your Modal URL and update `API_URL` in `streamlit_app.py`.

### 6. Run the App

```bash
streamlit run streamlit_app.py
```

Access at: `http://localhost:8501`

## 🧪 Testing

Quick test:
```bash
python test.py quick
```

Full test:
```bash
python test.py
```

## 🎯 Usage

### Web Interface
1. Open the Streamlit app
2. Select a model
3. Enter your prompt
4. Click "Generate Image"

### API Example
```python
import requests

response = requests.post(
    "https://your-modal-url.modal.run/generate/",
    json={
        "prompt": "a magical forest",
        "model": "lightning"
    }
)

with open("image.png", "wb") as f:
    f.write(response.content)
```

## 🔧 Model Parameters

- **prompt**: Text description (required)
- **model**: "flux", "sdxl", "sd15", or "lightning" (required)
- **num_inference_steps**: Denoising steps (optional)
- **guidance_scale**: Prompt adherence (optional)

## 🐛 Common Issues

**"Model not found":** Ensure Modal app is deployed with `modal deploy app_modal.py`

**Timeout errors:** Try a faster model like Lightning or SD 1.5

**Auth errors:** Check your Hugging Face token in Modal secrets


