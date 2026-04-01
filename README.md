---
title: InkGenerator
emoji: 🎭
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.44.1"
python_version: "3.9"
app_file: app.py
pinned: false
---





# FunnyAvatar 🎭

O aplicație Gradio care transformă fotografiile în caricaturi 3D în stil Pixar folosind modelul SDXL prin Hugging Face Inference API.

## 🎯 Funcționalități

- **Img2img workflow**: Transformă imagini existente în caricaturi
- **System prompt fix**: "3D caricature, big head, small body, Pixar style, exaggerated facial features, highly detailed, cute, funny, 8k render"
- **Denoising Strength slider**: Control fin al transformării (default 0.65)
- **Autentificare**: Protecție cu user și parolă
- **Prompt suplimentar**: Posibilitatea de a adăuga instrucțiuni artistice personalizate

## 🚀 Instalare

### Local
1. Clonează repository-ul:
```bash
git clone <repository-url>
cd ai_caricature_studio
```

2. Instalează dependențele:
```bash
pip install -r requirements.txt
```

3. Setează variabilele de mediu:
```bash
# Windows
set HF_API_TOKEN="token-ul_tău_de_la_huggingface"
set APP_USERNAME="user_dorit"
set APP_PASSWORD="parola_dorită"

# Linux/Mac
export HF_API_TOKEN="token-ul_tău_de_la_huggingface"
export APP_USERNAME="user_dorit"
export APP_PASSWORD="parola_dorită"
```

4. Pornește aplicația:
```bash
python app.py
```

### Hugging Face Spaces
Aplicația este gata să fie deployată pe Hugging Face Spaces. Doar upload-ează fișierele și setează secretul `HF_API_TOKEN` în Settings → Secrets.

## 🔧 Variabile de mediu

- `HF_API_TOKEN`: Token-ul tău de la Hugging Face (obligatoriu)
- `APP_USERNAME`: Nume utilizator pentru autentificare (default: "admin")
- `APP_PASSWORD`: Parolă pentru autentificare (default: "caricature123")
- `SDXL_ENDPOINT_URL`: URL custom pentru SDXL (opțional, default: modelul stabilityai/stable-diffusion-xl-base-1.0)

## 📱 Utilizare

1. **Autentificare**: Introdu user și parolă când accesezi aplicația
2. **Încarcă imagine**: Alege o fotografie din calculator
3. **Personalizează**: Adaugă un prompt suplimentar (opțional)
4. **Ajustează**: Modifică Denoising Strength dacă este necesar
   - **Valori mici (0.1-0.3)**: Transformare subtilă, păstrează multe detalii
   - **Valori medii (0.4-0.7)**: Echilibru între original și caricatură
   - **Valori mari (0.8-1.0)**: Transformare drastică, mai puține detalii originale
5. **Generează**: Apasă "Generează caricatură" și așteaptă procesarea

## 🎨 Exemple de prompt-uri suplimentare

- "businessman, office lighting, professional"
- "pirate, eyepatch, tropical background"
- "astronaut, space suit, stars"
- "chef, white hat, kitchen background"
- "rockstar, leather jacket, stage lights"

## ⚡ Performanță

- Timp mediu de procesare: 10-30 secunde
- Rezoluție recomandată: 512x512 sau 768x768
- Formate acceptate: PNG, JPG, JPEG

## 🛠️ Tehnologii

- **Gradio 4.44.0**: Interfață web
- **Pillow 10.2.0**: Procesare imagini
- **Requests 2.31.0**: API calls
- **Stable Diffusion XL**: Model AI
- **Hugging Face Inference API**: Infrastructură AI

## 🔒 Securitate

- Autentificare obligatorie
- Token API securizat
- Procesare locală a imaginilor
- Nu se stochează imaginile pe server

## 🐛 Depanare

### Probleme comune:
1. **"HF_API_TOKEN nu este setat"**: Setează variabila de mediu corect
2. **Eroare 429**: Prea multe cereri - așteaptă și încearcă din nou
3. **Imaginea nu se încarcă**: Verifică formatul și dimensiunea (max 10MB)
4. **Timeout**: Crește timeout-ul în cod sau încearcă cu imagine mai mică

### Debug mode:
Pentru debugging, poți comenta linia cu autentificarea în `app.py`:
```python
# demo.launch(auth=(USERNAME, PASSWORD))
demo.launch()  # Fără autentificare pentru development
```

## 📝 License

MIT License - poți folosi codul liber pentru proiecte personale și comerciale.

## 🤝 Contribuții

Contribuțiile sunt binevenite! Deschide un issue sau pull request pentru îmbunătățiri.

---

**Transformă-ți pozele în caricaturi amuzante cu FunnyAvatar! 🎭**

