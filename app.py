import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter, ImageChops
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

DEFAULT_X = 50
DEFAULT_Y = 50
AUTO_PLACEMENT = {
    "hand": (52, 58),
    "arm": (50, 45),
    "leg": (50, 48),  # mai sus pe picior fata de 62
    "neck": (50, 38),
    "chest": (50, 42),
    "back": (50, 48),
}

STRICT_SKIN_MODE = True
MIN_SKIN_COVERAGE = 0.12

# DOAR PENTRU TEST – inlocuiesti cu cheile reale de la Gumroad
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

def resolve_position(body_area: str, x_pos: float, y_pos: float):
    """Use auto placement only if sliders remain unchanged."""
    if int(x_pos) == DEFAULT_X and int(y_pos) == DEFAULT_Y:
        return AUTO_PLACEMENT.get(body_area, (x_pos, y_pos))
    return x_pos, y_pos

def _is_skin_pixel(r: int, g: int, b: int) -> bool:
    """Simple RGB skin detector."""
    mx = max(r, g, b)
    mn = min(r, g, b)
    return (
        r > 85 and g > 35 and b > 20 and
        (mx - mn) > 12 and
        abs(r - g) > 10 and
        r > g and r > b
    )

def _build_skin_mask(rgb_img: Image.Image) -> Image.Image:
    """Build full-size mask where skin pixels are white."""
    w, h = rgb_img.size
    src = rgb_img.load()
    mask = Image.new("L", (w, h), 0)
    dst = mask.load()

    for y in range(h):
        for x in range(w):
            r, g, b = src[x, y]
            dst[x, y] = 255 if _is_skin_pixel(r, g, b) else 0

    mask = mask.filter(ImageFilter.GaussianBlur(2))
    return mask

def _snap_to_skin(rgb_img: Image.Image, x: int, y: int, max_radius: int = 220, step: int = 2):
    """Move target point to nearest detected skin pixel."""
    w, h = rgb_img.size
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    px = rgb_img.load()

    r, g, b = px[x, y]
    if _is_skin_pixel(r, g, b):
        return x, y

    for radius in range(step, max_radius + 1, step):
        left = max(0, x - radius)
        right = min(w - 1, x + radius)
        top = max(0, y - radius)
        bottom = min(h - 1, y + radius)

        for xx in range(left, right + 1, step):
            for yy in (top, bottom):
                r, g, b = px[xx, yy]
                if _is_skin_pixel(r, g, b):
                    return xx, yy

        for yy in range(top, bottom + 1, step):
            for xx in (left, right):
                r, g, b = px[xx, yy]
                if _is_skin_pixel(r, g, b):
                    return xx, yy

    return x, y

def prepare_tattoo_ink(tattoo_img: Image.Image) -> Image.Image:
    """
    Convert generated artwork to dark tattoo linework:
    - remove bright background
    - preserve shape details (less aggressive thresholding)
    """
    gray = ImageOps.grayscale(tattoo_img)
    gray = ImageOps.autocontrast(gray, cutoff=1)
    gray = gray.filter(ImageFilter.MedianFilter(3))

    inv = ImageOps.invert(gray)
    alpha = inv.point(lambda p: 0 if p < 55 else min(255, int((p - 55) * 1.6)))
    alpha = alpha.filter(ImageFilter.GaussianBlur(0.6))

    ink = Image.new("RGBA", gray.size, (12, 12, 12, 0))
    ink.putalpha(alpha)

    bbox = ink.getbbox()
    if bbox:
        ink = ink.crop(bbox)

    return ink

def apply_tattoo_to_skin(background_path, tattoo_img, x_pos, y_pos, scale):
    """
    Apply tattoo ONLY on skin pixels:
    1) detect skin in body photo
    2) snap target point to skin
    3) combine tattoo alpha with skin mask
    """
    bg = Image.open(background_path).convert("RGBA")
    bg_rgb = bg.convert("RGB")
    bg_w, bg_h = bg.size

    skin_mask_full = _build_skin_mask(bg_rgb)
    tattoo_rgba = prepare_tattoo_ink(tattoo_img)

    # Resize tattoo from slider
    t_w = max(1, int(bg_w * (scale / 100)))
    ratio = t_w / float(tattoo_rgba.size[0])
    t_h = max(1, int(tattoo_rgba.size[1] * ratio))
    tattoo_rgba = tattoo_rgba.resize((t_w, t_h), Image.Resampling.LANCZOS)

    # Position + snap on skin
    center_x = int(bg_w * (x_pos / 100))
    center_y = int(bg_h * (y_pos / 100))
    center_x, center_y = _snap_to_skin(bg_rgb, center_x, center_y)

    actual_x = center_x - (t_w // 2)
    actual_y = center_y - (t_h // 2)

    # Keep inside photo bounds
    max_x = max(0, bg_w - t_w)
    max_y = max(0, bg_h - t_h)
    actual_x = max(0, min(actual_x, max_x))
    actual_y = max(0, min(actual_y, max_y))

    # Local skin ROI
    roi_skin = skin_mask_full.crop((actual_x, actual_y, actual_x + t_w, actual_y + t_h))

    # Strict mode: ensure enough skin where tattoo will be placed
    if STRICT_SKIN_MODE:
        hist = roi_skin.histogram()
        skin_strength = sum(v * i for i, v in enumerate(hist))
        max_strength = 255 * (t_w * t_h)
        coverage = (skin_strength / max_strength) if max_strength else 0.0
        if coverage < MIN_SKIN_COVERAGE:
            raise gr.Error(
                "Nu detectez suficienta piele in zona selectata. "
                "Muta sliderele sau incarca o poza mai clara."
            )

    # Keep tattoo only where skin exists
    tattoo_alpha = tattoo_rgba.getchannel("A")
    final_alpha = ImageChops.multiply(tattoo_alpha, roi_skin)

    # Realistic opacity without harsh shadow blocks
    final_alpha = final_alpha.point(lambda p: int(p * 0.75))

    tattoo_final = Image.new("RGBA", (t_w, t_h), (12, 12, 12, 0))
    tattoo_final.putalpha(final_alpha)

    layer = Image.new("RGBA", bg.size, (0, 0, 0, 0))
    layer.paste(tattoo_final, (actual_x, actual_y), tattoo_final)

    out = Image.alpha_composite(bg, layer)
    return out.convert("RGB")

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

    # verificare simpla de PRO
    is_pro = bool(license_key and license_key.strip() in VALID_KEYS)

    # pentru Free, fortam un aspect mai mic
    if not is_pro:
        aspect_label = "Square 1:1"

    # Prompt mai strict pentru forme recognoscibile (ex: flowers)
    enhanced_prompt = (
        f"{prompt.strip()}, {tattoo_type} tattoo, placement on {body_area}, "
        "clean tattoo stencil, subject must be clearly recognizable, "
        "black ink linework, crisp contours, simple composition, centered, "
        "no abstract marks, no splatter, no texture, no watercolor, no shading, no 3d render"
    )

    try:
        tattoo_design = call_sdxl_text2img(enhanced_prompt, aspect_label, model_label)

        if body_photo_path:
            x_pos, y_pos = resolve_position(body_area, x_pos, y_pos)
            final_img = apply_tattoo_to_skin(
                body_photo_path, tattoo_design, x_pos, y_pos, scale
            )
        else:
            final_img = tattoo_design

        if not is_pro:
            # Free: micsoram si punem watermark
            final_img = final_img.resize((512, 512))
            final_img = add_watermark(final_img)

        return final_img

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
                placeholder="e.g. single rose flower, black fine line tattoo stencil, no shading",
                lines=3,
            )
            body_photo = gr.Image(
                label="Upload body area photo (hand/arm/leg/neck)",
                type="filepath",
            )

            with gr.Group():
                gr.Markdown("### 📍 Overlay Position & Scale")
                x_pos = gr.Slider(0, 100, value=DEFAULT_X, label="Horizontal Position (%)")
                y_pos = gr.Slider(0, 100, value=DEFAULT_Y, label="Vertical Position (%)")
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
