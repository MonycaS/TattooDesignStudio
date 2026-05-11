import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

from io import BytesIO
import random
import requests
import gradio as gr
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter, ImageChops

# --- PATCH for gradio_client bool schema bug ---
import gradio_client.utils as gc_utils
_original__json_schema_to_python_type = gc_utils._json_schema_to_python_type

def _patched__json_schema_to_python_type(schema, defs=None):
    if isinstance(schema, bool):
        return "object"
    return _original__json_schema_to_python_type(schema, defs)

gc_utils._json_schema_to_python_type = _patched__json_schema_to_python_type
# --- END PATCH ---

# ==============================
# Config
# ==============================
POLLINATIONS_BASE_URL = "https://image.pollinations.ai/prompt"

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

BODY_AREAS = ["hand", "arm", "leg", "neck", "chest", "back"]
TATTOO_TYPES = ["minimalist", "fine line", "traditional", "tribal", "geometric", "realism"]
LEG_PLACEMENTS = ["auto", "ankle", "mid", "upper"]

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

STRICT_SKIN_DEFAULT = True

VALID_KEYS = {
    "ABC-123",
    "DEF-456",
    "6F0E4C97-B72A4E69-A11BF6C4-AF6517E7",
}

TATTOO_STYLE_PROMPT = (
    "tattoo flash, clean stencil, centered, high contrast, white background, black ink linework, "
    "recognizable single subject, isolated, no background elements, no extra symbols, "
    "no watercolor, no splatter, no abstract texture, no gray wash, no 3d"
)

# ==============================
# Helpers
# ==============================
def add_watermark(img: Image.Image) -> Image.Image:
    img = img.copy()
    draw = ImageDraw.Draw(img)
    text = "TattooDesigner"
    font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    w, h = img.size
    draw.text((w - tw - 10, h - th - 10), text, font=font, fill=(0, 0, 0))
    return img

def resolve_position(body_area, leg_placement, x_pos, y_pos):
    if int(x_pos) == DEFAULT_X and int(y_pos) == DEFAULT_Y:
        ax, ay = AUTO_PLACEMENT.get(body_area, (x_pos, y_pos))
        if body_area == "leg" and LEG_PLACEMENT_Y.get(leg_placement) is not None:
            ay = LEG_PLACEMENT_Y[leg_placement]
        return ax, ay

    if body_area == "leg" and LEG_PLACEMENT_Y.get(leg_placement) is not None:
        return x_pos, LEG_PLACEMENT_Y[leg_placement]

    return x_pos, y_pos

def _is_skin_pixel(r, g, b):
    mx = max(r, g, b)
    mn = min(r, g, b)
    return (
        r > 85 and g > 35 and b > 20 and
        (mx - mn) > 12 and
        abs(r - g) > 10 and
        r > g and r > b
    )

def build_skin_mask(rgb_img: Image.Image) -> Image.Image:
    w, h = rgb_img.size
    src = rgb_img.load()
    mask = Image.new("L", (w, h), 0)
    dst = mask.load()

    for y in range(h):
        for x in range(w):
            r, g, b = src[x, y]
            dst[x, y] = 255 if _is_skin_pixel(r, g, b) else 0

    mask = mask.filter(ImageFilter.MinFilter(3))
    mask = mask.filter(ImageFilter.MaxFilter(5))
    mask = mask.point(lambda p: 255 if p >= 128 else 0)
    return mask

def snap_to_skin(rgb_img: Image.Image, x, y, max_radius=220, step=2):
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
                rr, gg, bb = px[xx, yy]
                if _is_skin_pixel(rr, gg, bb):
                    return xx, yy

        for yy in range(top, bottom + 1, step):
            for xx in (left, right):
                rr, gg, bb = px[xx, yy]
                if _is_skin_pixel(rr, gg, bb):
                    return xx, yy

    return x, y

def _remove_border_connected(alpha: Image.Image, threshold=8) -> Image.Image:
    alpha = alpha.copy().convert("L")
    w, h = alpha.size
    px = alpha.load()

    visited = [[False] * h for _ in range(w)]
    stack = []

    for x in range(w):
        if px[x, 0] > threshold:
            stack.append((x, 0))
        if px[x, h - 1] > threshold:
            stack.append((x, h - 1))
    for y in range(h):
        if px[0, y] > threshold:
            stack.append((0, y))
        if px[w - 1, y] > threshold:
            stack.append((w - 1, y))

    while stack:
        x, y = stack.pop()
        if x < 0 or x >= w or y < 0 or y >= h:
            continue
        if visited[x][y]:
            continue
        visited[x][y] = True
        if px[x, y] <= threshold:
            continue

        px[x, y] = 0
        stack.append((x + 1, y))
        stack.append((x - 1, y))
        stack.append((x, y + 1))
        stack.append((x, y - 1))

    return alpha

def prepare_tattoo_alpha(tattoo_img: Image.Image) -> Image.Image:
    """
    Shape-preserving mask:
    - remove bright background
    - keep full dark subject structure
    """
    gray = ImageOps.grayscale(tattoo_img)
    gray = ImageOps.autocontrast(gray, cutoff=1)

    inv = ImageOps.invert(gray)
    alpha = inv.point(lambda p: 0 if p < 38 else min(255, int((p - 38) * 1.45)))

    alpha = alpha.filter(ImageFilter.MedianFilter(3))
    alpha = _remove_border_connected(alpha, threshold=8)

    bbox = alpha.getbbox()
    if bbox:
        alpha = alpha.crop(bbox)
    else:
        alpha = Image.new("L", (16, 16), 0)
        alpha.putpixel((8, 8), 255)

    return alpha

def call_text2img(user_prompt: str, aspect_label: str, model_label: str):
    full_prompt = f"{user_prompt}, {TATTOO_STYLE_PROMPT}" if user_prompt else TATTOO_STYLE_PROMPT
    width, height = ASPECT_RATIOS.get(aspect_label, (1024, 1024))
    model_id = MODEL_ENDPOINTS.get(model_label, "flux")
    seed = random.randint(0, 65535)

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
        raise RuntimeError(f"Pollinations error {resp.status_code}: {resp.text[:400]}")

    content_type = resp.headers.get("Content-Type", "")
    if "image" not in content_type.lower():
        raise RuntimeError(f"Pollinations returned non-image response: {content_type}")

    return Image.open(BytesIO(resp.content)).convert("RGB")

def apply_tattoo_stable(
    background_path,
    tattoo_img,
    x_pos,
    y_pos,
    scale,
    realism_strength,
    ink_darkness,
    edge_blur,
    strict_skin_mode,
):
    bg_rgb = Image.open(background_path).convert("RGB")
    bg_w, bg_h = bg_rgb.size

    skin_mask_full = build_skin_mask(bg_rgb)
    tattoo_alpha = prepare_tattoo_alpha(tattoo_img)

    # Resize by slider
    t_w = max(1, int(bg_w * (scale / 100)))
    ratio = t_w / float(tattoo_alpha.size[0])
    t_h = max(1, int(tattoo_alpha.size[1] * ratio))
    tattoo_alpha = tattoo_alpha.resize((t_w, t_h), Image.Resampling.LANCZOS)

    if edge_blur > 0:
        tattoo_alpha = tattoo_alpha.filter(ImageFilter.GaussianBlur(edge_blur))

    center_x = int(bg_w * (x_pos / 100))
    center_y = int(bg_h * (y_pos / 100))
    center_x, center_y = snap_to_skin(bg_rgb, center_x, center_y)

    actual_x = max(0, min(center_x - t_w // 2, max(0, bg_w - t_w)))
    actual_y = max(0, min(center_y - t_h // 2, max(0, bg_h - t_h)))

    roi_skin = skin_mask_full.crop((actual_x, actual_y, actual_x + t_w, actual_y + t_h))
    roi_skin_hard = roi_skin.point(lambda p: 255 if p >= 128 else 0)

    if strict_skin_mode:
        hist = roi_skin_hard.histogram()
        skin_pixels = hist[255]
        coverage = skin_pixels / float(t_w * t_h)
        if coverage < 0.10:
            raise gr.Error("Not enough skin detected in selected area.")

    # Keep tattoo only on skin
    final_alpha = ImageChops.multiply(tattoo_alpha, roi_skin_hard)

    # Keep fine details
    final_alpha = final_alpha.point(lambda p: 0 if p < 12 else p)

    # Stronger visibility
    alpha_gain = 0.50 + (realism_strength / 100.0) * 0.60
    final_alpha = final_alpha.point(lambda p: int(max(0, min(255, p * alpha_gain))))

    roi_bg = bg_rgb.crop((actual_x, actual_y, actual_x + t_w, actual_y + t_h))

    dark = int(max(0, min(60, 60 - int(ink_darkness))))
    tattoo_tone = Image.new("RGB", (t_w, t_h), (dark, dark, dark))

    multiplied = ImageChops.multiply(roi_bg, tattoo_tone)
    roi_out = Image.composite(multiplied, roi_bg, final_alpha)

    out = bg_rgb.copy()
    out.paste(roi_out, (actual_x, actual_y))
    return out

# ==============================
# Main
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
    strict_skin_mode,
):
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a prompt.")

    is_pro = bool(license_key and license_key.strip() in VALID_KEYS)
    if not is_pro:
        aspect_label = "Square 1:1"

    enhanced_prompt = (
        f"{prompt.strip()}, {tattoo_type} tattoo, {body_area} placement, "
        "subject clearly recognizable, black fine line tattoo stencil, clean contours, "
        "minimal composition, single subject only, isolated, "
        "no abstract marks, no splatter, no shading, no extra symbols"
    )

    tattoo_design = call_text2img(enhanced_prompt, aspect_label, model_label)

    if body_photo_path:
        rx, ry = resolve_position(body_area, leg_placement, x_pos, y_pos)
        final_img = apply_tattoo_stable(
            background_path=body_photo_path,
            tattoo_img=tattoo_design,
            x_pos=rx,
            y_pos=ry,
            scale=scale,
            realism_strength=realism_strength,
            ink_darkness=ink_darkness,
            edge_blur=edge_blur,
            strict_skin_mode=strict_skin_mode,
        )
    else:
        final_img = tattoo_design

    if not is_pro:
        final_img = final_img.resize((512, 512))
        final_img = add_watermark(final_img)

    return final_img

# ==============================
# UI
# ==============================
with gr.Blocks(title="TattooDesigner Stable Mode") as demo:
    gr.Markdown(
        """
# TattooDesigner Stable Mode

This version prioritizes clean, complete tattoo shapes and avoids rectangle/line-loss artifacts.
"""
    )

    with gr.Row():
        with gr.Column():
            user_prompt = gr.Textbox(
                label="Describe the tattoo (EN)",
                placeholder="e.g. single rose flower, black fine line tattoo stencil",
                lines=3,
            )
            body_photo = gr.Image(
                label="Upload body area photo",
                type="filepath",
            )

            body_area = gr.Dropdown(label="Body area", choices=BODY_AREAS, value="leg")
            leg_placement = gr.Dropdown(label="Leg placement", choices=LEG_PLACEMENTS, value="auto")
            tattoo_type = gr.Dropdown(label="Tattoo type", choices=TATTOO_TYPES, value="fine line")
            model_choice = gr.Dropdown(label="Model", choices=list(MODEL_ENDPOINTS.keys()), value="Flux")
            aspect = gr.Dropdown(
                label="Aspect Ratio (PRO only; Free is forced to Square)",
                choices=list(ASPECT_RATIOS.keys()),
                value="Vertical 2:3 (arm)",
            )

            with gr.Group():
                gr.Markdown("### Position / Size")
                x_pos = gr.Slider(0, 100, value=DEFAULT_X, label="Horizontal Position (%)")
                y_pos = gr.Slider(0, 100, value=DEFAULT_Y, label="Vertical Position (%)")
                scale = gr.Slider(5, 100, value=28, label="Tattoo Size (%)")

            with gr.Group():
                gr.Markdown("### Visibility Controls")
                realism_strength = gr.Slider(0, 100, value=75, label="Realism strength")
                ink_darkness = gr.Slider(0, 100, value=85, label="Ink darkness")
                edge_blur = gr.Slider(0.0, 2.0, value=0.0, step=0.1, label="Edge blur")
                strict_skin_mode = gr.Checkbox(value=STRICT_SKIN_DEFAULT, label="Strict skin mode")

            license_key = gr.Textbox(
                label="PRO license key (Gumroad)",
                placeholder="Paste your Gumroad key",
                type="password",
            )

            btn = gr.Button("Generate & Apply", variant="primary")

        with gr.Column():
            output_image = gr.Image(label="Result", type="pil")

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
            strict_skin_mode,
        ],
        outputs=output_image,
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"TattooDesigner starting on port {port}")
    demo.launch(server_name="0.0.0.0", server_port=port)
