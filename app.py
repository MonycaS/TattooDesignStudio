import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

from io import BytesIO
import math
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

VALID_KEYS = {
    "ABC-123",
    "DEF-456",
    "6F0E4C97-B72A4E69-A11BF6C4-AF6517E7",
}

STRICT_SKIN_DEFAULT = True

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

    # curatare fara "scurgeri" pe fundal
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

def _remove_border_connected(alpha: Image.Image, threshold=10) -> Image.Image:
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
    gray = ImageOps.grayscale(tattoo_img)
    gray = ImageOps.autocontrast(gray, cutoff=1)
    gray = gray.filter(ImageFilter.MedianFilter(3))

    inv = ImageOps.invert(gray)
    alpha = inv.point(lambda p: 255 if p > 135 else 0)
    alpha = alpha.filter(ImageFilter.MinFilter(3))
    alpha = alpha.filter(ImageFilter.MaxFilter(3))
    alpha = _remove_border_connected(alpha, threshold=10)

    bbox = alpha.getbbox()
    if bbox:
        alpha = alpha.crop(bbox)

    return alpha

def make_radial_falloff_mask(size, softness=0.25):
    w, h = size
    cx = w / 2.0
    cy = h / 2.0
    max_d = math.sqrt(cx * cx + cy * cy)

    m = Image.new("L", (w, h), 0)
    px = m.load()
    soft = max(0.01, min(0.8, softness))

    for y in range(h):
        for x in range(w):
            dx = x - cx
            dy = y - cy
            d = math.sqrt(dx * dx + dy * dy) / max_d
            v = 1.0 - (d ** (1.0 / soft))
            v = max(0.0, min(1.0, v))
            px[x, y] = int(v * 255)
    return m

def warp_cylindrical(alpha: Image.Image, strength=0.12) -> Image.Image:
    """
    Simuleaza curbura pielii prin compresie pe margini.
    strength: 0..0.4
    """
    strength = max(0.0, min(0.4, strength))
    src = alpha.convert("L")
    w, h = src.size
    dst = Image.new("L", (w, h), 0)

    src_px = src.load()
    dst_px = dst.load()
    cx = (w - 1) / 2.0

    for y in range(h):
        row_factor = 1.0 - strength * (0.7 + 0.3 * math.sin((y / max(1, h - 1)) * math.pi))
        row_factor = max(0.55, min(1.0, row_factor))
        for x in range(w):
            nx = (x - cx) / max(1.0, cx)
            sx = nx / row_factor
            src_x = int(round((sx * cx) + cx))
            if 0 <= src_x < w:
                dst_px[x, y] = src_px[src_x, y]

    return dst

def extract_texture_overlay(roi_bg: Image.Image, amount=0.16):
    """
    Transfer subtil de textura din piele peste tatuaj.
    """
    amount = max(0.0, min(0.5, amount))
    gray = ImageOps.grayscale(roi_bg)
    blur = gray.filter(ImageFilter.GaussianBlur(2.0))
    detail = ImageChops.subtract(gray, blur, scale=1.0, offset=128)  # high-pass style
    detail_rgb = Image.merge("RGB", (detail, detail, detail))
    # normalize around 128 then blend lightly
    base = Image.new("RGB", roi_bg.size, (128, 128, 128))
    mixed = Image.blend(base, detail_rgb, amount)
    return mixed

def apply_tattoo_realistic(
    background_path,
    tattoo_img,
    x_pos,
    y_pos,
    scale,
    realism_strength,
    ink_darkness,
    edge_blur,
    strict_skin_mode,
    curvature_strength,
):
    bg_rgb = Image.open(background_path).convert("RGB")
    bg_w, bg_h = bg_rgb.size

    skin_mask_full = build_skin_mask(bg_rgb)
    tattoo_alpha = prepare_tattoo_alpha(tattoo_img)

    # size
    t_w = max(1, int(bg_w * (scale / 100)))
    ratio = t_w / float(tattoo_alpha.size[0])
    t_h = max(1, int(tattoo_alpha.size[1] * ratio))
    tattoo_alpha = tattoo_alpha.resize((t_w, t_h), Image.Resampling.LANCZOS)

    # curvature warp (important for "inkhunter-like" look)
    tattoo_alpha = warp_cylindrical(tattoo_alpha, strength=curvature_strength)

    # edge soften minimal
    if edge_blur > 0:
        tattoo_alpha = tattoo_alpha.filter(ImageFilter.GaussianBlur(edge_blur))

    # placement
    center_x = int(bg_w * (x_pos / 100))
    center_y = int(bg_h * (y_pos / 100))
    center_x, center_y = snap_to_skin(bg_rgb, center_x, center_y)

    actual_x = max(0, min(center_x - t_w // 2, max(0, bg_w - t_w)))
    actual_y = max(0, min(center_y - t_h // 2, max(0, bg_h - t_h)))

    roi_skin = skin_mask_full.crop((actual_x, actual_y, actual_x + t_w, actual_y + t_h))

    if strict_skin_mode:
        hist = roi_skin.histogram()
        skin_strength = sum(v * i for i, v in enumerate(hist))
        max_strength = 255 * (t_w * t_h)
        coverage = (skin_strength / max_strength) if max_strength else 0.0
        if coverage < 0.10:
            raise gr.Error("Nu detectez suficienta piele in zona selectata. Muta pozitia sau schimba poza.")

    # hard clip on skin
    roi_skin_hard = roi_skin.point(lambda p: 255 if p >= 128 else 0)
    final_alpha = ImageChops.multiply(tattoo_alpha, roi_skin_hard)

    # remove weak spill
    final_alpha = final_alpha.point(lambda p: 0 if p < 35 else p)

    # adaptive realism
    alpha_gain = 0.22 + (realism_strength / 100.0) * 0.62
    final_alpha = final_alpha.point(lambda p: int(max(0, min(255, p * alpha_gain))))

    # edge attenuation (subtle)
    falloff = make_radial_falloff_mask((t_w, t_h), softness=0.30)
    falloff = falloff.filter(ImageFilter.GaussianBlur(1.2))
    final_alpha = ImageChops.multiply(final_alpha, falloff)

    roi_bg = bg_rgb.crop((actual_x, actual_y, actual_x + t_w, actual_y + t_h))

    # preserve highlights: reduce alpha in bright skin areas
    lum = ImageOps.grayscale(roi_bg)
    highlight_suppress = lum.point(lambda p: int(255 - max(0, p - 175) * 1.6))
    final_alpha = ImageChops.multiply(final_alpha, highlight_suppress)

    # tattoo tone
    dark = int(max(0, min(65, 65 - int(ink_darkness))))  # bigger ink_darkness -> darker
    tattoo_tone = Image.new("RGB", (t_w, t_h), (dark, dark, dark))

    # multiply blend
    multiplied = ImageChops.multiply(roi_bg, tattoo_tone)
    roi_out = Image.composite(multiplied, roi_bg, final_alpha)

    # texture transfer
    tex = extract_texture_overlay(roi_bg, amount=0.14)
    roi_out = ImageChops.multiply(roi_out, tex)

    # tiny grain to avoid sticker look
    if realism_strength > 35:
        noise = Image.effect_noise((t_w, t_h), sigma=4).convert("L")
        noise = noise.point(lambda p: int(120 + (p - 128) * 0.35))
        noise_rgb = Image.merge("RGB", (noise, noise, noise))
        roi_out = ImageChops.multiply(roi_out, noise_rgb)

    out = bg_rgb.copy()
    out.paste(roi_out, (actual_x, actual_y))
    return out

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

# ==============================
# Main generate
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
    curvature_strength,
    strict_skin_mode,
):
    if not prompt or not prompt.strip():
        raise gr.Error("Te rog scrie un prompt.")

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
        final_img = apply_tattoo_realistic(
            body_photo_path=body_photo_path,
            tattoo_img=tattoo_design,
            x_pos=rx,
            y_pos=ry,
            scale=scale,
            realism_strength=realism_strength,
            ink_darkness=ink_darkness,
            edge_blur=edge_blur,
            strict_skin_mode=strict_skin_mode,
            curvature_strength=curvature_strength,
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
with gr.Blocks(title="TattooDesigner Pro-Look") as demo:
    gr.Markdown(
        """
# TattooDesigner Pro-Look

Aplicatie Gradio cu aplicare tatuaj mai realista (skin-aware + curvature + lighting + texture transfer).

**Prompt recomandat (exemplu):**  
`single rose flower, black fine line tattoo stencil, clean contours, centered, no shading`
"""
    )

    with gr.Row():
        with gr.Column():
            user_prompt = gr.Textbox(
                label="Descrie tatuajul (EN)",
                placeholder="e.g. single rose flower, black fine line tattoo stencil",
                lines=3,
            )
            body_photo = gr.Image(
                label="Upload poza zona corp",
                type="filepath",
            )

            body_area = gr.Dropdown(label="Body area", choices=BODY_AREAS, value="leg")
            leg_placement = gr.Dropdown(label="Leg placement", choices=LEG_PLACEMENTS, value="auto")
            tattoo_type = gr.Dropdown(label="Tattoo type", choices=TATTOO_TYPES, value="fine line")
            model_choice = gr.Dropdown(label="Model", choices=list(MODEL_ENDPOINTS.keys()), value="Flux")
            aspect = gr.Dropdown(
                label="Aspect Ratio (PRO only; Free forced to Square)",
                choices=list(ASPECT_RATIOS.keys()),
                value="Vertical 2:3 (arm)",
            )

            with gr.Group():
                gr.Markdown("### Pozitie / marime")
                x_pos = gr.Slider(0, 100, value=DEFAULT_X, label="Horizontal Position (%)")
                y_pos = gr.Slider(0, 100, value=DEFAULT_Y, label="Vertical Position (%)")
                scale = gr.Slider(5, 100, value=28, label="Tattoo Size (%)")

            with gr.Group():
                gr.Markdown("### Realism controls")
                realism_strength = gr.Slider(0, 100, value=72, label="Realism strength")
                ink_darkness = gr.Slider(0, 100, value=80, label="Ink darkness")
                edge_blur = gr.Slider(0.0, 2.0, value=0.1, step=0.1, label="Edge blur")
                curvature_strength = gr.Slider(0.0, 0.4, value=0.14, step=0.01, label="Curvature strength")
                strict_skin_mode = gr.Checkbox(value=STRICT_SKIN_DEFAULT, label="Strict skin mode")

            license_key = gr.Textbox(
                label="License key PRO (Gumroad)",
                placeholder="Paste your Gumroad key",
                type="password",
            )

            btn = gr.Button("Generate & Apply", variant="primary")

        with gr.Column():
            output_image = gr.Image(label="Rezultat", type="pil")

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
            curvature_strength,
            strict_skin_mode,
        ],
        outputs=output_image,
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"TattooDesigner starting on port {port}")
    demo.launch(server_name="0.0.0.0", server_port=port)
