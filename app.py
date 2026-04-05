import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import gradio as gr

# --- PATCH for gradio_client bug ---
import gradio_client.utils as gc_utils
_original__json_schema_to_python_type = gc_utils._json_schema_to_python_type
def _patched__json_schema_to_python_type(schema, defs=None):
    if isinstance(schema, bool):
        return "object"
    return _original__json_schema_to_python_type(schema, defs)
gc_utils._json_schema_to_python_type = _patched__json_schema_to_python_type

# ==============================
# Configuration
# ==============================
POLLINATIONS_BASE_URL = "https://image.pollinations.ai/prompt"
MODEL_ENDPOINTS = {"Flux": "flux", "Turbo": "turbo"}
TATTOO_STYLE_PROMPT = "tattoo design, white background, fine line art, professional tattoo flash, 8k, symmetrical, centered, isolated on white"
ASPECT_RATIOS = {
    "Square 1:1": (1024, 1024),
    "Vertical 2:3 (arm)": (768, 1152),
    "Vertical 9:16": (768, 1365),
    "Horizontal 3:2": (1152, 768),
}
BODY_AREAS = ["hand", "arm", "leg", "neck", "chest", "back"]
TATTOO_TYPES = ["minimalist", "fine line", "traditional", "tribal", "geometric", "realism"]
VALID_KEYS = {"ABC-123", "6F0E4C97-B72A4E69-A11BF6C4-AF6517E7"}

# ==============================
# Image Processing
# ==============================
def add_watermark(img: Image.Image) -> Image.Image:
    img = img.copy()
    draw = ImageDraw.Draw(img)
    text = "TattooDesigner"
    font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    w, h = img.size
    draw.text((w - (bbox[2]-bbox[0]) - 10, h - (bbox[3]-bbox[1]) - 10), text, font=font, fill=(0, 0, 0))
    return img

def apply_tattoo_to_skin(background_path, tattoo_img, x_pos, y_pos, scale):
    bg = Image.open(background_path).convert("RGBA")
    bg_w, bg_h = bg.size
    tattoo_rgba = tattoo_img.convert("RGBA")
    data = tattoo_rgba.getdata()
    new_data = []
    for item in data:
        if item[0] > 215 and item[1] > 215 and item[2] > 215:
            new_data.append((255, 255, 255, 0))
        else:
            new_data.append((item[0], item[1], item[2], 215))
    tattoo_rgba.putdata(new_data)
    t_w = int(bg_w * (scale / 100))
    w_percent = (t_w / float(tattoo_rgba.size[0]))
    t_h = int((float(tattoo_rgba.size[1]) * float(w_percent)))
    tattoo_rgba = tattoo_rgba.resize((t_w, t_h), Image.Resampling.LANCZOS)
    actual_x = int(bg_w * (x_pos / 100)) - (t_w // 2)
    actual_y = int(bg_h * (y_pos / 100)) - (t_h // 2)
    canvas = Image.new("RGBA", bg.size, (0, 0, 0, 0))
    canvas.paste(tattoo_rgba, (actual_x, actual_y))
    combined = Image.alpha_composite(bg, canvas)
    return combined.convert("RGB")

def call_sdxl_text2img(user_prompt: str, aspect_label: str, model_label: str):
    full_prompt = f"{user_prompt}, {TATTOO_STYLE_PROMPT}"
    width, height = ASPECT_RATIOS.get(aspect_label, (1024, 1024))
    model_id = MODEL_ENDPOINTS.get(model_label, "flux")
    params = {"width": width, "height": height, "model": model_id, "seed": int.from_bytes(os.urandom(2), "big"), "nologo": "true"}
    resp = requests.get(f"{POLLINATIONS_BASE_URL}/{requests.utils.quote(full_prompt)}", params=params, timeout=90)
    if resp.status_code != 200: raise RuntimeError("API Error")
    return Image.open(BytesIO(resp.content)).convert("RGB")

# ==============================
# Generation Logic
# ==============================
def generate_tattoo(prompt, body_photo_path, body_area, tattoo_type, model_label, aspect_label, license_key, x_pos, y_pos, scale):
    if not prompt or not prompt.strip(): return None
    is_pro = bool(license_key and license_key.strip() in VALID_KEYS)
    if not is_pro: aspect_label = "Square 1:1"

    enhanced_prompt = f"{prompt.strip()}, {tattoo_type} tattoo, placement on {body_area}, clean stencil reference"
    try:
        tattoo_design = call_sdxl_text2img(enhanced_prompt, aspect_label, model_label)
        if body_photo_path:
            final_img = apply_tattoo_to_skin(body_photo_path, tattoo_design, x_pos, y_pos, scale)
        else:
            final_img = tattoo_design
        if not is_pro:
            final_img = final_img.resize((512, 512))
            final_img = add_watermark(final_img)
        return final_img
    except Exception as e:
        raise gr.Error(f"Generation failed: {str(e)}")

# ==============================
# UI
# ==============================
with gr.Blocks(title="TattooDesigner") as demo:
    gr.Markdown(
        """
        # TattooDesigner 🖋️
        Upload a photo of your skin area, configure position, and generate your tattoo design.
        **Backend:** Pollinations.ai
        
        **License:** Free (watermarked) vs PRO (full resolution, no watermark).
        """
    )

    with gr.Row():
        with gr.Column():
            user_prompt = gr.Textbox(label="Describe your tattoo idea", placeholder="e.g. minimalist rose on forearm", lines=3)
            body_photo = gr.Image(label="Upload skin area photo", type="filepath")
            
            with gr.Group():
                gr.Markdown("### 📍 Overlay Settings")
                x_pos = gr.Slider(0, 100, value=50, label="Horizontal Position (%)")
                y_pos = gr.Slider(0, 100, value=50, label="Vertical Position (%)")
                scale = gr.Slider(5, 100, value=30, label="Tattoo Scale (%)")

            body_area = gr.Dropdown(label="Body area", choices=BODY_AREAS, value="arm")
            tattoo_type = gr.Dropdown(label="Tattoo style", choices=TATTOO_TYPES, value="fine line")
            model_choice = gr.Dropdown(label="AI Model", choices=list(MODEL_ENDPOINTS.keys()), value="Flux")
            aspect = gr.Dropdown(label="Aspect Ratio", choices=list(ASPECT_RATIOS.keys()), value="Square 1:1")
            license_key = gr.Textbox(label="License key", placeholder="Paste your PRO key here", type="password")

            gr.Markdown("### Get PRO access\n[Buy TattooDesigner PRO on Gumroad](https://inkforge0.gumroad.com/l/tattoodesigner-pro)")
            btn = gr.Button("Generate and Apply", variant="primary")

        with gr.Column():
            output_image = gr.Image(label="Final Design Preview", type="pil")

    btn.click(
        fn=generate_tattoo,
        inputs=[user_prompt, body_photo, body_area, tattoo_type, model_choice, aspect, license_key, x_pos, y_pos, scale],
        outputs=output_image,
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)import os
# Dezactivăm analiticele pentru a evita erori de conexiune inutile pe server
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import gradio as gr

# --- PATCH pentru bug-ul gradio_client (esențial pentru stabilitate) ---
import gradio_client.utils as gc_utils
_original__json_schema_to_python_type = gc_utils._json_schema_to_python_type
def _patched__json_schema_to_python_type(schema, defs=None):
    if isinstance(schema, bool):
        return "object"
    return _original__json_schema_to_python_type(schema, defs)
gc_utils._json_schema_to_python_type = _patched__json_schema_to_python_type

# ==============================
# Configurații API & Stil
# ==============================
POLLINATIONS_BASE_URL = "https://image.pollinations.ai/prompt"
MODEL_ENDPOINTS = {"Flux": "flux", "Turbo": "turbo"}
TATTOO_STYLE_PROMPT = "tattoo design, white background, fine line art, professional tattoo flash, 8k, symmetrical, centered, isolated on white"
ASPECT_RATIOS = {"Square 1:1": (1024, 1024), "Vertical 2:3 (arm)": (768, 1152)}
BODY_AREAS = ["hand", "arm", "leg", "neck", "chest", "back"]
TATTOO_TYPES = ["minimalist", "fine line", "traditional", "tribal", "geometric", "realism"]
VALID_KEYS = {"ABC-123", "6F0E4C97-B72A4E69-A11BF6C4-AF6517E7"}

# ==============================
# Funcții Procesare Imagine
# ==============================

def add_watermark(img: Image.Image) -> Image.Image:
    img = img.copy()
    draw = ImageDraw.Draw(img)
    text = "TattooDesigner Free"
    font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    w, h = img.size
    draw.text((w - (bbox[2]-bbox[0]) - 10, h - (bbox[3]-bbox[1]) - 10), text, font=font, fill=(128, 128, 128))
    return img

def apply_tattoo_to_skin(background_path, tattoo_img, x_pos, y_pos, scale):
    """Suprapune tatuajul pe poza cu mâna eliminând fundalul alb."""
    bg = Image.open(background_path).convert("RGBA")
    bg_w, bg_h = bg.size

    # Convertim tatuajul și scoatem albul (transparență)
    tattoo_rgba = tattoo_img.convert("RGBA")
    data = tattoo_rgba.getdata()
    new_data = []
    for item in data:
        # Dacă pixelul e alb, îl facem transparent
        if item[0] > 215 and item[1] > 215 and item[2] > 215:
            new_data.append((255, 255, 255, 0))
        else:
            # Opacitate 210 pentru a lăsa textura pielii să se vadă puțin (realism)
            new_data.append((item[0], item[1], item[2], 210))
    tattoo_rgba.putdata(new_data)

    # Redimensionare proporțională
    t_w = int(bg_w * (scale / 100))
    w_percent = (t_w / float(tattoo_rgba.size[0]))
    t_h = int((float(tattoo_rgba.size[1]) * float(w_percent)))
    tattoo_rgba = tattoo_rgba.resize((t_w, t_h), Image.Resampling.LANCZOS)

    # Poziționare procentuală
    actual_x = int(bg_w * (x_pos / 100)) - (t_w // 2)
    actual_y = int(bg_h * (y_pos / 100)) - (t_h // 2)

    # Suprapunere finală
    canvas = Image.new("RGBA", bg.size, (0, 0, 0, 0))
    canvas.paste(tattoo_rgba, (actual_x, actual_y))
    combined = Image.alpha_composite(bg, canvas)
    return combined.convert("RGB")

def call_sdxl_text2img(user_prompt: str, aspect_label: str, model_label: str):
    full_prompt = f"{user_prompt}, {TATTOO_STYLE_PROMPT}"
    width, height = ASPECT_RATIOS.get(aspect_label, (1024, 1024))
    model_id = MODEL_ENDPOINTS.get(model_label, "flux")
    params = {"width": width, "height": height, "model": model_id, "seed": int.from_bytes(os.urandom(2), "big"), "nologo": "true"}
    
    resp = requests.get(f"{POLLINATIONS_BASE_URL}/{requests.utils.quote(full_prompt)}", params=params, timeout=90)
    if resp.status_code != 200:
        raise RuntimeError("Eroare API Pollinations")
    return Image.open(BytesIO(resp.content)).convert("RGB")

# ==============================
# Logica Principală Gradio
# ==============================

def generate_tattoo(prompt, body_photo_path, body_area, tattoo_type, model_label, aspect_label, license_key, x_pos, y_pos, scale):
    if not prompt: return None
    is_pro = bool(license_key and license_key.strip() in VALID_KEYS)
    
    enhanced_prompt = f"{prompt.strip()}, {tattoo_type} tattoo, stencil, black ink, white background"
    
    try:
        tattoo_design = call_sdxl_text2img(enhanced_prompt, aspect_label, model_label)
        
        if body_photo_path:
            final_img = apply_tattoo_to_skin(body_photo_path, tattoo_design, x_pos, y_pos, scale)
        else:
            final_img = tattoo_design

        if not is_pro:
            final_img = final_img.resize((700, 700))
            final_img = add_watermark(final_img)
            
        return final_img
    except Exception as e:
        raise gr.Error(f"Eroare: {str(e)}")

# ==============================
# Interfața (UI)
# ==============================
with gr.Blocks(title="TattooDesigner PRO") as demo:
    gr.Markdown("# TattooDesigner 🖋️\nÎncarcă poza cu zona corpului și generează design-ul!")
    
    with gr.Row():
        with gr.Column():
            user_prompt = gr.Textbox(label="Descriere (Engleză)", placeholder="Ex: minimalist geometric forest")
            body_photo = gr.Image(label="Poza cu mâna/brațul", type="filepath")
            
            with gr.Group():
                gr.Markdown("### Control Poziție Tatuaj")
                x_pos = gr.Slider(0, 100, value=50, label="Orizontal (%)")
                y_pos = gr.Slider(0, 100, value=50, label="Vertical (%)")
                scale = gr.Slider(5, 100, value=30, label="Dimensiune (%)")
            
            with gr.Accordion("Opțiuni Avansate", open=False):
                body_area = gr.Dropdown(choices=BODY_AREAS, value="arm", label="Zonă")
                tattoo_type = gr.Dropdown(choices=TATTOO_TYPES, value="fine line", label="Stil")
                model_choice = gr.Dropdown(choices=list(MODEL_ENDPOINTS.keys()), value="Flux", label="Model")
                aspect = gr.Dropdown(choices=list(ASPECT_RATIOS.keys()), value="Square 1:1", label="Aspect")
            
            license_key = gr.Textbox(label="Licență PRO", type="password")
            btn = gr.Button("Generează Tatuaj", variant="primary")

        with gr.Column():
            output_image = gr.Image(label="Previzualizare", type="pil")

    btn.click(
        fn=generate_tattoo,
        inputs=[user_prompt, body_photo, body_area, tattoo_type, model_choice, aspect, license_key, x_pos, y_pos, scale],
        outputs=output_image
    )

# ==============================
# Lansare (Configurat pentru Render)
# ==============================
if __name__ == "__main__":
    # Luăm portul de la Render sau folosim 7860 local
    port = int(os.environ.get("PORT", 7860))
    print(f"Lansare aplicație pe portul {port}...")
    
    # server_name="0.0.0.0" permite accesul extern (Render)
    demo.launch(
        server_name="0.0.0.0", 
        server_port=port
    )   







