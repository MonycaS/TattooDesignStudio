import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import base64
import requests
from io import BytesIO
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
# Inference API config
# ==============================

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
SDXL_ENDPOINT_URL = os.getenv(
    "SDXL_ENDPOINT_URL",
    "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-xl-base-1.0",
)

# You can add more models/endpoints here; the user will choose from the dropdown.
MODEL_ENDPOINTS = {
    "SDXL base 1.0": SDXL_ENDPOINT_URL,
    # "Another model name": os.getenv("OTHER_MODEL_ENDPOINT_URL", "<endpoint_url_here>"),
}

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
    if not HF_API_TOKEN:
        raise ValueError("HF_API_TOKEN is not set in environment variables.")

    full_prompt = (
        f"{user_prompt}, {TATTOO_STYLE_PROMPT}"
        if user_prompt
        else TATTOO_STYLE_PROMPT
    )

    width, height = ASPECT_RATIOS.get(aspect_label, (1024, 1024))

    endpoint_url = MODEL_ENDPOINTS.get(model_label, SDXL_ENDPOINT_URL)

    payload = {
        "inputs": full_prompt,
        "parameters": {
            "width": width,
            "height": height,
        },
    }

    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

    resp = requests.post(
        endpoint_url,
        headers=headers,
        json=payload,
        timeout=60,
    )

    if resp.status_code != 200:
        raise RuntimeError(
            f"Inference API error {resp.status_code}: {resp.text[:500]}"
        )

    content_type = resp.headers.get("content-type", "")
    if "image" in content_type:
        return Image.open(BytesIO(resp.content)).convert("RGB")

    data = resp.json()
    if isinstance(data, dict) and "generated_image" in data:
        image_b64 = data["generated_image"]
        image_bytes = base64.b64decode(image_b64)
        return Image.open(BytesIO(image_bytes)).convert("RGB")

    if (
        isinstance(data, list)
        and data
        and isinstance(data[0], dict)
        and "image" in data[0]
    ):
        image_b64 = data[0]["image"]
        image_bytes = base64.b64decode(image_b64)
        return Image.open(BytesIO(image_bytes)).convert("RGB")

    raise RuntimeError("Unknown response format from Inference API.")


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