import os
import re
import time
import asyncio
import torch
import requests
import edge_tts

from PIL import Image, ImageDraw, ImageFont
from diffusers import AutoPipelineForText2Image
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips

# ============================================================
# 1. CONFIGURATION & STATE
# ============================================================

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:3b"
VOICE = "en-US-ChristopherNeural"

OUTPUT_DIR = "output"
AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio")
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

COUNTER_FILE = "counter.txt"
try:
    with open(COUNTER_FILE, "r") as f:
        CURRENT_COUNT = int(f.read().strip()) + 1
except Exception:
    CURRENT_COUNT = 1

with open(COUNTER_FILE, "w") as f:
    f.write(str(CURRENT_COUNT))

print(f"🚀 STARTING PERFECT-SYNC TIMELINE PIPELINE FOR EPISODE #{CURRENT_COUNT}")


# ============================================================
# 2. LOCAL AI IMAGE ENGINE (HIGH-QUALITY CPU OPTIMIZED)
# ============================================================

print("🧠 Loading local SD-Turbo model with high-quality CPU settings...")
pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sd-turbo",
    torch_dtype=torch.float32
)
pipe.to("cpu")
pipe.enable_attention_slicing()

def get_system_font(size):
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    ]
    for p in font_paths:
        if os.path.exists(p):
            try: return ImageFont.truetype(p, size)
            except: pass
    return ImageFont.load_default()

def generate_local_scene_image(sentence_text, output_path, is_thumbnail=False):
    # Har sentence ke base par image banegi
    # Strict style lock taaki creepy models ya faces na aayein
    clean_prompt = (
        f"clean corporate flat vector art, minimalist infographic style, {sentence_text}, "
        f"crisp bold black outlines, vibrant solid color background, no shading, "
        f"faceless stick figure animation style, high quality UI UX illustration"
    )

    # 3 steps aur 1.5 scale quality ko boost karenge bina CPU ko crash kiye
    image = pipe(
        prompt=clean_prompt,
        num_inference_steps=3, 
        guidance_scale=1.5,
        height=512,
        width=512,
    ).images[0]

    # Widescreen me fit karna bina side bars ke
    widescreen = image.resize((1280, 720), Image.Resampling.LANCZOS)

    if is_thumbnail:
        draw = ImageDraw.Draw(widescreen)
        font = get_system_font(65)
        badge = "STOP NOW!"
        bbox = draw.textbbox((0, 0), badge, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        bx, by, pad_x, pad_y = 60, 50, 30, 15
        
        draw.rounded_rectangle([(bx+8, by+8), (bx+tw+pad_x*2+8, by+th+pad_y*2+8)], radius=15, fill=(0,0,0,200))
        draw.rounded_rectangle([(bx, by), (bx+tw+pad_x*2, by+th+pad_y*2)], radius=15, fill=(255,200,0), outline=(0,0,0), width=6)
        draw.text((bx+pad_x, by+pad_y), badge, fill=(0,0,0), font=font)

    widescreen.save(output_path, format="PNG")


# ============================================================
# 3. TEXT & SCRIPT HELPERS
# ============================================================

def request_llm(prompt_text):
    response = requests.post(
        OLLAMA_URL, 
        json={"model": OLLAMA_MODEL, "prompt": prompt_text, "stream": False}, 
        timeout=None
    )
    response.raise_for_status()
    return response.json()["response"].strip()

async def generate_audio(text, output_path):
    communicate = edge_tts.Communicate(text, VOICE)
    await communicate.save(output_path)


# ============================================================
# 4. SCRIPT GENERATION & TIMELINE SPLITTING
# ============================================================

print("\n✍️ Generating master script...")

script_prompt = f"""
Write a 5-paragraph deep-dive YouTube script about US Personal Finance (Episode {CURRENT_COUNT}).
Topic: The hidden traps of debt and the power of index funds.
Do not use lists, bullet points, or complex formatting. Write pure, conversational paragraphs.
Each paragraph must be detailed and punchy.
"""
raw_script = request_llm(script_prompt)

# Sentence Splitting Engine: Ye logic script ko line-by-line tod dega
# (e.g. "You are a loan fighter." ban jayega ek akela scene)
sentences = re.split(r'(?<=[.!?]) +|\n+', raw_script)
sentences = [s.strip() for s in sentences if len(s.strip()) > 15]

TOTAL_SCENES = len(sentences)
print(f"🎬 Script split into exactly {TOTAL_SCENES} dynamic timeline scenes!")


# ============================================================
# 5. ASSET GENERATION (PERFECT SYNC LOOP)
# ============================================================

print("\n🖼️ Generating Thumbnail...")
generate_local_scene_image("Stickman facing a giant red financial graph going down", os.path.join(OUTPUT_DIR, f"thumbnail_{CURRENT_COUNT}.png"), is_thumbnail=True)

async def build_timeline_assets():
    for idx, sentence in enumerate(sentences):
        scene_id = str(idx).zfill(4)
        audio_path = os.path.join(AUDIO_DIR, f"scene_{scene_id}.mp3")
        image_path = os.path.join(IMAGE_DIR, f"scene_{scene_id}.png")

        print(f"⚙️ Rendering Scene {idx+1}/{TOTAL_SCENES}: {sentence[:50]}...")
        
        # Audio aur image dono ek hi 'sentence' ke basis par banenge
        await generate_audio(sentence, audio_path)
        generate_local_scene_image(sentence, image_path)

asyncio.run(build_timeline_assets())


# ============================================================
# 6. PERFECT SYNC VIDEO ASSEMBLY
# ============================================================

print("\n🎞️ ASSEMBLING PERFECT SYNC VIDEO...")

clips = []
try:
    for idx in range(TOTAL_SCENES):
        scene_id = str(idx).zfill(4)
        audio_path = os.path.join(AUDIO_DIR, f"scene_{scene_id}.mp3")
        image_path = os.path.join(IMAGE_DIR, f"scene_{scene_id}.png")

        if os.path.exists(audio_path) and os.path.exists(image_path):
            audio_clip = AudioFileClip(audio_path)
            
            # CORE LOGIC: Image ki duration utni hi hogi jitni uss sentence ke audio ki hai
            image_clip = (
                ImageClip(image_path)
                .set_duration(audio_clip.duration)
                .set_audio(audio_clip)
            )
            clips.append(image_clip)

    final_video = concatenate_videoclips(clips, method="compose")
    video_path = os.path.join(OUTPUT_DIR, f"video_{CURRENT_COUNT}.mp4")

    # fps=15 rakha hai taaki GitHub Actions ka RAM (7GB) crash na ho video export karte waqt
    final_video.write_videofile(
        video_path,
        fps=15, 
        codec="libx264",
        audio_codec="aac",
        threads=2,
        preset="ultrafast",
        logger="bar"
    )

finally:
    for clip in clips:
        try: clip.close()
        except: pass
    try: final_video.close()
    except: pass

# SEO File
seo_path = os.path.join(OUTPUT_DIR, f"seo_{CURRENT_COUNT}.txt")
with open(seo_path, "w") as f:
    f.write(f"TITLE: Perfect Sync Finance Episode #{CURRENT_COUNT}\n")
    f.write(f"TOTAL_SCENES: {TOTAL_SCENES}\n")

print(f"\n✅ PERFECT TIMELINE VIDEO #{CURRENT_COUNT} CREATED SUCCESSFULLY!")
