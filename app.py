import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
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

# ==============================
# Licențe & watermark
# ==============================

# DOAR PENTRU TEST. Mai târziu vei pune aici cheile reale de la Gumroad
VALID_KEYS = {
    "ABC-123",
    "DEF-456",
    "6F0E4C97-B72A4E69-A11BF6C4-AF6517E7",
}


def add_watermark(img: Image.Image) -> Image.Image:
    """Adaugă un watermark simplu pentru varianta Free."""
    img = img.copy()
    draw = ImageDraw.Draw(img)
    text = "TattooDesigner"
    font = ImageFont.load_default()

    # nou în loc de draw.textsize(...)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    w, h = img.size
    x = w - text_w - 10
    y = h - text_h - 10
    draw.text((x, y), text, font=font, fill=(0, 0, 0))
    return img


# ==============================
# Pollinations call
# ==============================

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
    resp = requests.get(
        f"{POLLINATIONS_BASE_URL}/{requests.utils.quote(full_prompt, safe='')}",
        params=params,
        timeout=90,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Pollinations error {resp.status_code}: {resp.text[:500]}")
    content_type = resp.headers.get("Content-Type", "")
    if "image" not in content_type.lower():
        raise RuntimeError(f"Pollinations returned non-image response: {content_type}")
    return Image.open(BytesIO(resp.content)).convert("RGB")


# ==============================
# Generate tattoo (Free vs Pro)
# ==============================

def generate_tattoo(
    prompt: str,
    body_photo_path: str,
    body_area: str,
    tattoo_type: str,
    model_label: str,
    aspect_label: str,
    license_key: str,
):
    if not prompt or not prompt.strip():
        return None

    # verificare simplă de PRO (mai târziu vei înlocui cu chei reale de la Gumroad)
    is_pro = bool(license_key and license_key.strip() in VALID_KEYS)

    # pentru Free, forțăm un aspect mai mic (de ex. Square)
    if not is_pro:
        aspect_label = "Square 1:1"

    # Endpoint text-to-image; poza este păstrată doar ca referință în prompt
    _ = body_photo_path
    enhanced_prompt = (
        f"{prompt.strip()}, {tattoo_type} tattoo, placement on {body_area}, "
        f"clean stencil reference"
    )

    try:
        img = call_sdxl_text2img(enhanced_prompt, aspect_label, model_label)

        if not is_pro:
            # Free: micșorăm și punem watermark
            img = img.resize((512, 512))
            img = add_watermark(img)

        return img

    except Exception as e:
        err = str(e)
        print(f"[TattooDesigner] generate_tattoo error: {err}")
        raise gr.Error(f"Generation failed: {err}")


# ==============================
# UI Gradio
# ==============================

with gr.Blocks(title="TattooDesigner") as demo:
    gr.Markdown(
        """
        # TattooDesigner 🖋️

        Upload a photo of the body area (hand/arm/leg/neck etc.), choose tattoo type, then generate the design.

        **Backend:** Pollinations.ai (no API token required)

        **Free vs PRO**
        - Free: lower resolution + watermark
        - PRO: full resolution, no watermark, using a license key from Gumroad

        **Fixed style automatically added:**
        `tattoo design, white background, fine line art, professional tattoo flash, 8k, symmetrical, centered, isolated on white`

        **Usage & Licensing**

        This app uses the Pollinations.AI image API (models such as "flux" / "turbo") as the backend generator.  
        You own the designs you generate with this app, but you are responsible for ensuring that your use complies with:
        - Pollinations.AI Terms and API documentation  
        - The specific license of each underlying model (some models allow commercial use, some do not)

        For more details, see:
        - [https://pollinations.ai/terms](https://pollinations.ai/terms)  
        - [https://raw.githubusercontent.com/pollinations/pollinations/master/APIDOCS.md](https://raw.githubusercontent.com/pollinations/pollinations/master/APIDOCS.md)
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
                label="Aspect Ratio (for PRO users; Free uses Square 1:1)",
                choices=list(ASPECT_RATIOS.keys()),
                value="Vertical 2:3 (arm)",
            )
            license_key = gr.Textbox(
                label="License key (if you bought PRO on Gumroad)",
                placeholder="Paste your Gumroad license key here",
            )
            btn = gr.Button("Generate tattoo")

        with gr.Column():
            output_image = gr.Image(
                label="Generated design",
                type="pil",
            )

    btn.click(
        fn=generate_tattoo,
        inputs=[user_prompt, body_photo, body_area, tattoo_type, model_choice, aspect, license_key],
        outputs=output_image,
    )

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    print("TattooDesigner starting on port", port)
    demo.launch(server_name="0.0.0.0", server_port=port)












    







