import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import requests
from io import BytesIO
import time
from PIL import Image
import gradio as gr

# --- PATCH pentru bug-ul gradio_client (schema bool) ---
import gradio_client.utils as gc_utils

_original__json_schema_to_python_type = gc_utils._json_schema_to_python_type

def _patched__json_schema_to_python_type(schema, defs=None):
    if isinstance(schema, bool):
        return "object"
    return _original__json_schema_to_python_type(schema, defs)

gc_utils._json_schema_to_python_type = _patched__json_schema_to_python_type
# --- END PATCH ---


# ==============================
# Inference API config (Pollinations)
# ==============================

POLLINATIONS_BASE_URL = "https://image.pollinations.ai/prompt"
MODEL_ENDPOINTS = {
    "Flux": "flux",
    "Turbo": "turbo",
}
MAX_RETRIES = 3
REQUEST_TIMEOUT_SECONDS = 120

TATTOO_STYLE_PROMPT = (
    "tattoo design, white background, fine line art, professional tattoo flash, "
    "8k, symmetrical, centered, isolated on white"
)

ASPECT_RATIOS = {
    "Square 1:1": (1024, 1024),
    "Vertical 2:3 (arm)": (768, 1152),
    "Vertical 9:16": (768, 1365),
    "Horizontal 3:2": (1152, 768),
}

BODY_AREAS = [
    "hand",
    "arm",
    "leg",
    "neck",
    "chest",
    "back",
]

TATTOO_TYPES = [
    "minimalist",
    "fine line",
    "traditional",
    "tribal",
    "geometric",
    "realism",
]


def call_sdxl_text2img(user_prompt: str, aspect_label: str, model_label: str):
    full_prompt = (
        f"{user_prompt}, {TATTOO_STYLE_PROMPT}"
        if user_prompt
        else TATTOO_STYLE_PROMPT
    )

    width, height = ASPECT_RATIOS.get(aspect_label, (1024, 1024))
    model_id = MODEL_ENDPOINTS.get(model_label, "flux")
    seed = int.from_bytes(os.urandom(2), "big")
    params = {
        "width": width,
        "height": height,
        "model": model_id,
        "seed": seed,
        "nologo": "true",
    }
    request_url = f"{POLLINATIONS_BASE_URL}/{requests.utils.quote(full_prompt, safe='')}"
    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = requests.get(
                request_url,
                params=params,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            if attempt < MAX_RETRIES:
                wait_seconds = 2 ** attempt
                print(
                    f"[TattooDesigner] Network timeout/connection error. "
                    f"Retry {attempt + 1}/{MAX_RETRIES} in {wait_seconds}s..."
                )
                time.sleep(wait_seconds)
                continue
            last_error = (
                "Connection timeout to Pollinations (port 443). "
                "Please check internet/VPN/firewall and try again."
            )
            print(f"[TattooDesigner] request error: {e}")
            break

        if resp.status_code == 200:
            content_type = resp.headers.get("Content-Type", "")
            if "image" not in content_type.lower():
                raise RuntimeError(f"Pollinations returned non-image response: {content_type}")
            return Image.open(BytesIO(resp.content)).convert("RGB")

        # 429 = provider rate limit; respect Retry-After when available.
        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            if attempt < MAX_RETRIES:
                try:
                    wait_seconds = max(1, int(retry_after)) if retry_after else (2 ** attempt)
                except ValueError:
                    wait_seconds = 2 ** attempt
                print(
                    f"[TattooDesigner] Pollinations rate-limited (429). "
                    f"Retry {attempt + 1}/{MAX_RETRIES} in {wait_seconds}s..."
                )
                time.sleep(wait_seconds)
                continue
            last_error = (
                "Pollinations rate limit reached (HTTP 429). "
                "Please wait 30-60 seconds and try again."
            )
            break

        last_error = f"Pollinations error {resp.status_code}: {resp.text[:500]}"
        break

    raise RuntimeError(last_error or "Unknown error while calling Pollinations.")


def generate_tattoo(
    prompt: str,
    body_photo_path: str,
    body_area: str,
    tattoo_type: str,
    model_label: str,
    aspect_label: str,
):
    if not prompt or not prompt.strip():
        return None

    # Endpointul folosit este text-to-image; poza este păstrată doar ca referință
    # pentru prompt (pentru un img2img ulterior o putem integra).
    _ = body_photo_path
    enhanced_prompt = (
        f"{prompt.strip()}, {tattoo_type} tattoo, placement on {body_area}, "
        f"clean stencil reference"
    )

    try:
        return call_sdxl_text2img(enhanced_prompt, aspect_label, model_label)
    except Exception as e:
        err = str(e)
        print(f"[TattooDesigner] generate_tattoo error: {err}")
        raise gr.Error(f"Generation failed: {err}")


with gr.Blocks(title="TattooDesigner") as demo:
    gr.Markdown(
        """
        # TattooDesigner 🖋️
        Upload a photo of the body area (hand/arm/leg/neck etc.), choose tattoo type, then generate the design.

        **Backend:** Pollinations.ai (no API token required)
        
        **Fixed style automatically added:**
        `tattoo design, white background, fine line art, professional tattoo flash, 8k, symmetrical, centered, isolated on white`
        """
    )

    with gr.Row():
        with gr.Column():
            user_prompt = gr.Textbox(
                label="Describe the tattoo (in English)",
                placeholder="e.g. a small fox with flowers, minimalistic, on the forearm",
                lines=3,
            )
            body_photo = gr.Image(
                label="Upload body area photo (hand/arm/leg/neck)",
                type="filepath",
            )
            body_area = gr.Dropdown(
                label="Body area",
                choices=BODY_AREAS,
                value="arm",
            )
            tattoo_type = gr.Dropdown(
                label="Tattoo type",
                choices=TATTOO_TYPES,
                value="fine line",
            )
            model_choice = gr.Dropdown(
                label="Model",
                choices=list(MODEL_ENDPOINTS.keys()),
                value=list(MODEL_ENDPOINTS.keys())[0],
            )
            aspect = gr.Dropdown(
                label="Aspect Ratio",
                choices=list(ASPECT_RATIOS.keys()),
                value="Vertical 2:3 (arm)",
            )
            btn = gr.Button("Generate tattoo")

        with gr.Column():
            output_image = gr.Image(
                label="Generated design",
                type="pil",
            )

    btn.click(
        fn=generate_tattoo,
        inputs=[user_prompt, body_photo, body_area, tattoo_type, model_choice, aspect],
        outputs=output_image,
    )

if __name__ == "__main__":
    print("TattooDesigner starting...")
    demo.launch(debug=True)












    







