import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import requests
import numpy as np

from io import BytesIO

from PIL import (
    Image,
    ImageDraw,
    ImageFont,
    ImageFilter,
)

import gradio as gr

# ==============================
# PATCH for gradio_client bug
# ==============================

import gradio_client.utils as gc_utils

_original__json_schema_to_python_type = (
    gc_utils._json_schema_to_python_type
)

def _patched__json_schema_to_python_type(schema, defs=None):
    if isinstance(schema, bool):
        return "object"

    return _original__json_schema_to_python_type(schema, defs)

gc_utils._json_schema_to_python_type = (
    _patched__json_schema_to_python_type
)

# ==============================
# CONFIG
# ==============================

POLLINATIONS_BASE_URL = (
    "https://image.pollinations.ai/prompt"
)

MODEL_ENDPOINTS = {
    "Flux": "flux",
    "Turbo": "turbo",
}

TATTOO_STYLE_PROMPT = (
    "isolated tattoo design, "
    "pure white background, "
    "black ink only, "
    "professional tattoo stencil, "
    "centered composition, "
    "no human, no body, no skin"
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

VALID_KEYS = {
    "ABC-123",
    "DEF-456",
    "6F0E4C97-B72A4E69-A11BF6C4-AF6517E7",
}

# ==============================
# WATERMARK
# ==============================

def add_watermark(img: Image.Image):

    img = img.copy()

    draw = ImageDraw.Draw(img)

    text = "TattooDesigner"

    font = ImageFont.load_default()

    bbox = draw.textbbox(
        (0, 0),
        text,
        font=font
    )

    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    w, h = img.size

    x = w - text_w - 10
    y = h - text_h - 10

    draw.text(
        (x, y),
        text,
        font=font,
        fill=(0, 0, 0),
    )

    return img

# ==============================
# TATTOO OVERLAY
# ==============================

def apply_tattoo_to_skin(
    background_path,
    tattoo_img,
    x_pos,
    y_pos,
    scale,
    rotation=0,
    opacity=160,
):

    bg = Image.open(
        background_path
    ).convert("RGBA")

    bg_w, bg_h = bg.size

    # ==============================
    # CLEAN TATTOO
    # ==============================

    tattoo = tattoo_img.convert("RGBA")

    arr = np.array(tattoo)

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    # remove white background
    white_mask = (
        (r > 235) &
        (g > 235) &
        (b > 235)
    )

    # grayscale tattoo
    gray = (
        0.299 * r +
        0.587 * g +
        0.114 * b
    ).astype(np.uint8)

    # darker tattoo ink
    dark = np.clip(
        gray * 0.38,
        0,
        255
    ).astype(np.uint8)

    arr[:, :, 0] = dark
    arr[:, :, 1] = dark
    arr[:, :, 2] = dark

    # transparency
    arr[:, :, 3] = np.where(
        white_mask,
        0,
        opacity
    )

    tattoo = Image.fromarray(arr)

    # soften edges slightly
    tattoo = tattoo.filter(
        ImageFilter.GaussianBlur(0.2)
    )

    # ==============================
    # RESIZE
    # ==============================

    t_w = int(
        bg_w * (scale / 100)
    )

    ratio = t_w / tattoo.size[0]

    t_h = int(
        tattoo.size[1] * ratio
    )

    tattoo = tattoo.resize(
        (t_w, t_h),
        Image.Resampling.LANCZOS
    )

    # ==============================
    # ROTATE
    # ==============================

    tattoo = tattoo.rotate(
        rotation,
        expand=True
    )

    t_w, t_h = tattoo.size

    # ==============================
    # POSITION
    # ==============================

    actual_x = (
        int(bg_w * (x_pos / 100))
        - (t_w // 2)
    )

    actual_y = (
        int(bg_h * (y_pos / 100))
        - (t_h // 2)
    )

    # ==============================
    # TATTOO LAYER
    # ==============================

    tattoo_layer = Image.new(
        "RGBA",
        bg.size,
        (0, 0, 0, 0)
    )

    tattoo_layer.paste(
        tattoo,
        (actual_x, actual_y),
        tattoo
    )

    # ==============================
    # COMPOSITE
    # ==============================

    combined = Image.alpha_composite(
        bg,
        tattoo_layer
    )

    return combined.convert("RGB")

# ==============================
# POLLINATIONS API
# ==============================

def call_sdxl_text2img(
    user_prompt: str,
    aspect_label: str,
    model_label: str,
):

    full_prompt = (
        f"{user_prompt}, "
        f"{TATTOO_STYLE_PROMPT}"
    )

    width, height = ASPECT_RATIOS.get(
        aspect_label,
        (1024, 1024)
    )

    model_id = MODEL_ENDPOINTS.get(
        model_label,
        "flux"
    )

    seed = int.from_bytes(
        os.urandom(2),
        "big"
    )

    params = {
        "width": width,
        "height": height,
        "model": model_id,
        "seed": seed,
        "nologo": "true",
    }

    resp = requests.get(
        f"{POLLINATIONS_BASE_URL}/"
        f"{requests.utils.quote(full_prompt, safe='')}",
        params=params,
        timeout=90,
    )

    if resp.status_code != 200:

        raise RuntimeError(
            f"Pollinations error "
            f"{resp.status_code}: "
            f"{resp.text[:500]}"
        )

    content_type = resp.headers.get(
        "Content-Type",
        ""
    )

    if "image" not in content_type.lower():

        raise RuntimeError(
            f"Pollinations returned "
            f"non-image response: "
            f"{content_type}"
        )

    return Image.open(
        BytesIO(resp.content)
    ).convert("RGB")

# ==============================
# MAIN GENERATION
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
    rotation,
    opacity,
):

    if not prompt or not prompt.strip():
        return None

    is_pro = bool(
        license_key and
        license_key.strip() in VALID_KEYS
    )

    if not is_pro:
        aspect_label = "Square 1:1"

    enhanced_prompt = (
        f"{prompt.strip()}, "
        f"{tattoo_type} tattoo design, "
        f"isolated tattoo, "
        f"black ink only, "
        f"pure white background, "
        f"no body, no skin, no human, "
        f"professional tattoo stencil, "
        f"centered design"
    )

    try:

        tattoo_design = call_sdxl_text2img(
            enhanced_prompt,
            aspect_label,
            model_label,
        )

        if body_photo_path:

            final_img = apply_tattoo_to_skin(
                body_photo_path,
                tattoo_design,
                x_pos,
                y_pos,
                scale,
                rotation,
                opacity,
            )

        else:

            final_img = tattoo_design

        if not is_pro:

            final_img = final_img.resize(
                (512, 512)
            )

            final_img = add_watermark(
                final_img
            )

        return final_img

    except Exception as e:

        err = str(e)

        print(
            f"[TattooDesigner] "
            f"generate_tattoo error: {err}"
        )

        raise gr.Error(
            f"Generation failed: {err}"
        )

# ==============================
# UI
# ==============================

with gr.Blocks(
    title="TattooDesigner"
) as demo:

    gr.Markdown(
        """
# TattooDesigner 🖋️

Generate realistic tattoo previews directly on body photos.
"""
    )

    with gr.Row():

        with gr.Column():

            user_prompt = gr.Textbox(
                label="Describe the tattoo",
                placeholder=(
                    "e.g. black spider tattoo stencil"
                ),
                lines=3,
            )

            body_photo = gr.Image(
                label="Upload body area photo",
                type="filepath",
            )

            gr.Markdown(
                "### Placement"
            )

            x_pos = gr.Slider(
                0,
                100,
                value=50,
                label="Horizontal Position (%)",
            )

            y_pos = gr.Slider(
                0,
                100,
                value=50,
                label="Vertical Position (%)",
            )

            scale = gr.Slider(
                5,
                100,
                value=30,
                label="Tattoo Size (%)",
            )

            rotation = gr.Slider(
                -180,
                180,
                value=0,
                label="Rotation",
            )

            opacity = gr.Slider(
                30,
                255,
                value=160,
                label="Tattoo Opacity",
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
                choices=list(
                    MODEL_ENDPOINTS.keys()
                ),
                value="Flux",
            )

            aspect = gr.Dropdown(
                label="Aspect Ratio",
                choices=list(
                    ASPECT_RATIOS.keys()
                ),
                value="Vertical 2:3 (arm)",
            )

            license_key = gr.Textbox(
                label="PRO License Key",
                type="password",
            )

            btn = gr.Button(
                "Generate Tattoo",
                variant="primary",
            )

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
            rotation,
            opacity,
        ],
        outputs=output_image,
    )

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 7860)
    )

    print(
        f"TattooDesigner starting "
        f"on port {port}"
    )

    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )
