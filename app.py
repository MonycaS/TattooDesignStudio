import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

from io import BytesIO
import random
import requests
import gradio as gr

from PIL import (
    Image,
    ImageDraw,
    ImageFont,
    ImageOps,
    ImageFilter,
)

# ==============================
# PATCH for gradio_client bool schema bug
# ==============================

import gradio_client.utils as gc_utils

_original__json_schema_to_python_type = (
    gc_utils._json_schema_to_python_type
)

def _patched__json_schema_to_python_type(schema, defs=None):
    if isinstance(schema, bool):
        return "object"

    return _original__json_schema_to_python_type(
        schema,
        defs
    )

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

LEG_PLACEMENTS = [
    "auto",
    "ankle",
    "mid",
    "upper",
]

DEFAULT_X = 50
DEFAULT_Y = 50

AUTO_PLACEMENT = {
    "hand": (52, 58),
    "arm": (50, 45),
    "leg": (50, 50),
    "neck": (50, 38),
    "chest": (50, 42),
    "back": (50, 48),
}

LEG_PLACEMENT_Y = {
    "auto": None,
    "ankle": 65,
    "mid": 50,
    "upper": 36,
}

VALID_KEYS = {
    "ABC-123",
    "DEF-456",
    "6F0E4C97-B72A4E69-A11BF6C4-AF6517E7",
}

# ==============================
# TATTOO STYLE PROMPT
# ==============================

TATTOO_STYLE_PROMPT = (
    "professional black tattoo stencil, "
    "high detail tattoo flash, "
    "single centered tattoo design, "
    "isolated on pure white background, "
    "bold clean outlines, "
    "sharp black ink, "
    "tattoo studio quality, "
    "high contrast, "
    "minimal composition, "
    "symmetrical, "
    "vector tattoo style, "
    "real tattoo reference, "
    "no skin, "
    "no human, "
    "no body, "
    "no background scene, "
    "no abstract art, "
    "no smoke, "
    "no watercolor, "
    "no gray shading, "
    "no texture, "
    "no extra objects"
)

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

    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    w, h = img.size

    draw.text(
        (w - tw - 10, h - th - 10),
        text,
        font=font,
        fill=(0, 0, 0)
    )

    return img

# ==============================
# AUTO POSITION
# ==============================

def resolve_position(
    body_area,
    leg_placement,
    x_pos,
    y_pos,
):
    if (
        int(x_pos) == DEFAULT_X and
        int(y_pos) == DEFAULT_Y
    ):
        ax, ay = AUTO_PLACEMENT.get(
            body_area,
            (x_pos, y_pos)
        )

        if (
            body_area == "leg" and
            LEG_PLACEMENT_Y.get(
                leg_placement
            ) is not None
        ):
            ay = LEG_PLACEMENT_Y[
                leg_placement
            ]

        return ax, ay

    if (
        body_area == "leg" and
        LEG_PLACEMENT_Y.get(
            leg_placement
        ) is not None
    ):
        return x_pos, LEG_PLACEMENT_Y[
            leg_placement
        ]

    return x_pos, y_pos

# ==============================
# CLEAN TATTOO EXTRACTION
# ==============================

def prepare_tattoo_rgba(
    tattoo_img: Image.Image
):
    rgb = tattoo_img.convert("RGB")
    gray = ImageOps.grayscale(rgb)

    gray = ImageOps.autocontrast(
        gray,
        cutoff=1
    )

    inv = ImageOps.invert(gray)

    # Raised threshold (42) to reduce dirty blobs
    alpha = inv.point(
        lambda p: (
            0 if p < 42
            else min(255, int((p - 42) * 1.7))
        )
    )

    alpha = alpha.filter(
        ImageFilter.MedianFilter(3)
    )

    alpha = alpha.filter(
        ImageFilter.GaussianBlur(0.5)
    )

    ink = Image.new(
        "RGBA",
        rgb.size,
        (10, 10, 10, 0)
    )

    ink.putalpha(alpha)

    bbox = ink.getbbox()
    if bbox:
        ink = ink.crop(bbox)

    return ink

# ==============================
# GENERATE IMAGE
# ==============================

def call_text2img(
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

    seed = random.randint(
        0,
        65535
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
            f"{resp.text[:400]}"
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
# APPLY TATTOO
# ==============================

def apply_tattoo_stable(
    background_path,
    tattoo_img,
    x_pos,
    y_pos,
    scale,
    realism_strength,
    ink_darkness,
    edge_blur,
):
    bg = Image.open(
        background_path
    ).convert("RGBA")

    bw, bh = bg.size

    tattoo = prepare_tattoo_rgba(
        tattoo_img
    )

    tw = max(
        1,
        int(bw * (scale / 100))
    )

    ratio = tw / max(
        1,
        tattoo.size[0]
    )

    th = max(
        1,
        int(tattoo.size[1] * ratio)
    )

    tattoo = tattoo.resize(
        (tw, th),
        Image.Resampling.LANCZOS
    )

    if edge_blur > 0:
        a = tattoo.getchannel("A").filter(
            ImageFilter.GaussianBlur(
                edge_blur
            )
        )
        tattoo.putalpha(a)

    cx = int(bw * (x_pos / 100))
    cy = int(bh * (y_pos / 100))

    ax = max(
        0,
        min(cx - tw // 2, bw - tw)
    )

    ay = max(
        0,
        min(cy - th // 2, bh - th)
    )

    gain = (
        0.55 +
        (realism_strength / 100.0) * 0.55
    )

    # Added alpha floor cutoff to reduce haze
    a = tattoo.getchannel("A").point(
        lambda p: (
            0 if p < 12
            else int(max(0, min(255, p * gain)))
        )
    )

    tattoo.putalpha(a)

    dark = int(
        max(
            0,
            min(
                60,
                60 - int(ink_darkness)
            )
        )
    )

    tone = Image.new(
        "RGBA",
        tattoo.size,
        (dark, dark, dark, 0)
    )

    tone.putalpha(
        tattoo.getchannel("A")
    )

    layer = Image.new(
        "RGBA",
        bg.size,
        (0, 0, 0, 0)
    )

    layer.paste(
        tone,
        (ax, ay),
        tone
    )

    out = Image.alpha_composite(
        bg,
        layer
    ).convert("RGB")

    return out

# ==============================
# MAIN GENERATION
# ==============================

def generate_tattoo(
    prompt,
    body_photo_path,
    body_area,
    leg_placement,
    tattoo_type,
    model_label,
    aspect_label,
    license_key,
    x_pos,
    y_pos,
    scale,
    realism_strength,
    ink_darkness,
    edge_blur,
):
    if not prompt or not prompt.strip():
        raise gr.Error(
            "Please enter a prompt."
        )

    is_pro = bool(
        license_key and
        license_key.strip() in VALID_KEYS
    )

    if not is_pro:
        aspect_label = "Square 1:1"

    enhanced_prompt = (
        f"{prompt.strip()}, "
        f"{tattoo_type} professional tattoo design, "
        "premium tattoo flash, "
        "single subject tattoo, "
        "clean black ink stencil, "
        "sharp outlines, "
        "tattoo artist quality, "
        "isolated centered composition, "
        "pure white background, "
        "high detail, "
        "high contrast, "
        "recognizable subject, "
        "minimal tattoo composition, "
        "no body, "
        "no human, "
        "no realistic scene, "
        "no extra symbols, "
        "no abstract elements, "
        "ugly, blurry, duplicate, poster, frame, paper"
    )

    tattoo_design = call_text2img(
        enhanced_prompt,
        aspect_label,
        model_label,
    )

    if body_photo_path:
        rx, ry = resolve_position(
            body_area,
            leg_placement,
            x_pos,
            y_pos
        )

        final_img = apply_tattoo_stable(
            background_path=body_photo_path,
            tattoo_img=tattoo_design,
            x_pos=rx,
            y_pos=ry,
            scale=scale,
            realism_strength=realism_strength,
            ink_darkness=ink_darkness,
            edge_blur=edge_blur,
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

# ==============================
# UI
# ==============================

with gr.Blocks(
    title="TattooDesigner Stable"
) as demo:

    gr.Markdown(
        """
# TattooDesigner Stable

Generate realistic tattoo previews directly on body photos.
"""
    )

    with gr.Row():

        with gr.Column():

            user_prompt = gr.Textbox(
                label="Describe the tattoo (EN)",
                placeholder=(
                    "e.g. black spider tattoo stencil"
                ),
                lines=3,
            )

            body_photo = gr.Image(
                label="Upload body area photo",
                type="filepath",
            )

            body_area = gr.Dropdown(
                label="Body area",
                choices=BODY_AREAS,
                value="leg"
            )

            leg_placement = gr.Dropdown(
                label="Leg placement",
                choices=LEG_PLACEMENTS,
                value="auto"
            )

            tattoo_type = gr.Dropdown(
                label="Tattoo type",
                choices=TATTOO_TYPES,
                value="fine line"
            )

            model_choice = gr.Dropdown(
                label="Model",
                choices=list(
                    MODEL_ENDPOINTS.keys()
                ),
                value="Flux"
            )

            aspect = gr.Dropdown(
                label="Aspect Ratio",
                choices=list(
                    ASPECT_RATIOS.keys()
                ),
                value="Vertical 2:3 (arm)",
            )

            with gr.Group():

                gr.Markdown(
                    "### Position / Size"
                )

                x_pos = gr.Slider(
                    0,
                    100,
                    value=DEFAULT_X,
                    label="Horizontal Position (%)"
                )

                y_pos = gr.Slider(
                    0,
                    100,
                    value=DEFAULT_Y,
                    label="Vertical Position (%)"
                )

                scale = gr.Slider(
                    5,
                    100,
                    value=28,
                    label="Tattoo Size (%)"
                )

            with gr.Group():

                gr.Markdown(
                    "### Visibility Controls"
                )

                realism_strength = gr.Slider(
                    0,
                    100,
                    value=85,
                    label="Realism strength"
                )

                ink_darkness = gr.Slider(
                    0,
                    100,
                    value=90,
                    label="Ink darkness"
                )

                edge_blur = gr.Slider(
                    0.0,
                    2.0,
                    value=0.0,
                    step=0.1,
                    label="Edge blur"
                )

            license_key = gr.Textbox(
                label="PRO license key",
                placeholder=(
                    "Paste your Gumroad key"
                ),
                type="password",
            )

            btn = gr.Button(
                "Generate & Apply",
                variant="primary"
            )

        with gr.Column():

            output_image = gr.Image(
                label="Result",
                type="pil"
            )

    btn.click(
        fn=generate_tattoo,
        inputs=[
            user_prompt,
            body_photo,
            body_area,
            leg_placement,
            tattoo_type,
            model_choice,
            aspect,
            license_key,
            x_pos,
            y_pos,
            scale,
            realism_strength,
            ink_darkness,
            edge_blur,
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
