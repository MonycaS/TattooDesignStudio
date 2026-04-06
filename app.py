import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import gradio as gr

# --- PATCH for gradio_client bug (schema bool) ---
import gradio_client.utils as gc_utils
_original__json_schema_to_python_type = gc_utils._json_schema_to_python_type

def _patched__json_schema_to_python_type(schema, defs=None):
    if isinstance(schema, bool):
        return "object"
    return _original__json_schema_to_python_type(schema, defs)

gc_utils._json_schema_to_python_type = _patched__json_schema_to_python_type
# --- END PATCH ---

# ==============================
# Configuration
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

BODY_AREAS = ["hand", "arm", "leg", "neck", "chest", "back"]
TATTOO_TYPES = ["minimalist", "fine line", "traditional", "tribal", "geometric", "realism"]

# DOAR PENTRU TEST – înlocuiești cu cheile reale de la Gumroad
VALID_KEYS = {
    "ABC-123",
    "DEF-456",
    "6F0E4C97-B72A4E69-A11BF6C4-AF6517E7",
}

# ==============================
# Image Processing logic
# ==============================

def add_watermark(img: Image.Image) -> Image.Image:
    """Adds a simple watermark for the Free version."""
    img = img.copy()
    draw = ImageDraw.Draw(img)
    text = "TattooDesigner"
    font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    w, h = img.size
    x = w - text_w - 10
    y = h - text_h - 10
    draw.text((x, y), text, font=font, fill=(0, 0, 0))
    return img

def apply_tattoo_to_skin(background_path, tattoo_img, x_pos, y_pos, scale):
    """Overlays the tattoo onto the body photo by removing the white background."""
    bg = Image.open(background_path).convert("RGBA")
    bg_w, bg_h = bg.size

    # Convert tattoo and remove white background
    tattoo_rgba = tattoo_img.convert("RGBA")
    data = tattoo_rgba.getdata()
    new_data = []
    for item in data:
        # If pixel is near white, make it transparent
        if item[0] > 215 and item[1] > 215 and item[2] > 215:
            new_data.append((255, 255, 255, 0))
        else:
            # Opacity set to 215 for a realistic ink look on skin
            new_data.append((item[0], item[1], item[2], 215))
    tattoo_rgba.putdata(new_data)

    # Resize proportionally based on slider
    t_w = int(bg_w * (scale / 100))
    w_percent = (t_w / float(tattoo_rgba.size[0]))
    t_h = int((float(tattoo_rgba.size[1]) * float(w_percent)))
    tattoo_rgba = tattoo_rgba.resize((t_w, t_h), Image.Resampling.LANCZOS)

    # Calculate center position
    actual_x = int(bg_w * (x_pos / 100)) - (t_w // 2)
    actual_y = int(bg_h * (y_pos / 100)) - (t_h // 2)

    # Composite overlay
    canvas = Image.new("RGBA", bg.size, (0, 0, 0, 0))
    canvas.paste(tattoo_rgba, (actual_x, actual_y))
    combined = Image.alpha_composite(bg, canvas)
    return combined.convert("RGB")

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
# Main Generation Function
# ==============================

def generate_tattoo(
    prompt,
    body_photo_path,
    body_area,
    tattoo_type,
    model_label,
    aspect_label,
    license_key,
    x_pos,
    y_pos,
    scale,
):
    if not prompt or not prompt.strip():
        return None

    # verificare simplă de PRO
    is_pro = bool(license_key and license_key.strip() in VALID_KEYS)

    # pentru Free, forțăm un aspect mai mic (de ex. Square)
    if not is_pro:
        aspect_label = "Square 1:1"

    # Endpoint text-to-image; poza e păstrată doar ca referință în prompt
    enhanced_prompt = (
        f"{prompt.strip()}, {tattoo_type} tattoo, placement on {body_area}, "
        f"clean stencil reference"
    )

    try:
        tattoo_design = call_sdxl_text2img(enhanced_prompt, aspect_label, model_label)

        if body_photo_path:
            final_img = apply_tattoo_to_skin(
                body_photo_path, tattoo_design, x_pos, y_pos, scale
            )
        else:
            final_img = tattoo_design

        if not is_pro:
            # Free: micșorăm și punem watermark
            final_img = final_img.resize((512, 512))
            final_img = add_watermark(final_img)

        return final_img

    except Exception as e:
        err = str(e)
        print(f"[TattooDesigner] generate_tattoo error: {err}")
        raise gr.Error(f"Generation failed: {err}")

# ==============================
# UI Gradio (Full English + T&C)
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

            with gr.Group():
                gr.Markdown("### 📍 Overlay Position & Scale")
                x_pos = gr.Slider(0, 100, value=50, label="Horizontal Position (%)")
                y_pos = gr.Slider(0, 100, value=50, label="Vertical Position (%)")
                scale = gr.Slider(5, 100, value=30, label="Tattoo Size (%)")

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
                value="Flux",
            )
            aspect = gr.Dropdown(
                label="Aspect Ratio (for PRO users; Free uses Square 1:1)",
                choices=list(ASPECT_RATIOS.keys()),
                value="Vertical 2:3 (arm)",
            )
            license_key = gr.Textbox(
                label="License key (if you bought PRO on Gumroad)",
                placeholder="Paste your Gumroad license key here",
                type="password",
            )

            gr.Markdown(
                """
                ### PRO access

                Don’t have a license yet?  
                👉 [Get TattooDesigner PRO on Gumroad](https://inkforge0.gumroad.com/l/tattoodesigner-pro)
                """
            )

            btn = gr.Button("Generate and Apply Tattoo", variant="primary")

        with gr.Column():
            output_image = gr.Image(
                label="Final Result",
                type="pil",
            )

    btn.click(
        fn=generate_tattoo,
        inputs=[
            user_prompt,
            body_photo,
            body_area,
            tattoo_type,
            model_choice,
            aspect,
            license_key,
            x_pos,
            y_pos,
            scale,
        ],
        outputs=output_image,
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"TattooDesigner starting on port {port}")
    demo.launch(server_name="0.0.0.0", server_port=port)







