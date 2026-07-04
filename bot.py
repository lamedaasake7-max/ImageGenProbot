def generate_image(prompt: str) -> Optional[bytes]:
    """Generate an image using OpenRouter API."""
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://your-app.railway.app",
        "X-Title": "ImageGenProbot"
    }
    
    payload = {
        "model": "stabilityai/stable-diffusion-xl-1024-v1-0",
        "prompt": prompt,
        "width": 1024,
        "height": 1024,
        "steps": 30,
        "cfg_scale": 7,
    }
    
    response = requests.post(
        "https://openrouter.ai/api/v1/images/generations",
        headers=headers,
        json=payload,
        timeout=60
    )
    
    if response.status_code == 200:
        # Extract image URL and download
        image_url = response.json()["data"][0]["url"]
        image_response = requests.get(image_url)
        return image_response.content
    return None
