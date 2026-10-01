import os
import json
import asyncio
import torch
import requests
import edge_tts

from PIL import Image, ImageDraw, ImageFont
from diffusers import AutoPipelineForText2Image
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips


# ============================================================
# CONFIG & AUTO COUNTER TRACKING
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
if os.path.exists(COUNTER_FILE):
    try:
        with open(COUNTER_FILE, "r") as f:
            CURRENT_COUNT = int(f.read().strip()) + 1
    except Exception:
        CURRENT_COUNT = 1
else:
    CURRENT_COUNT = 1

with open(COUNTER_FILE, "w") as f:
    f.write(str(CURRENT_COUNT))

print(f"🎯 Running pipeline for Episode #{CURRENT_COUNT}")


# ============================================================
# LOCAL AI IMAGE ENGINE (SD-TURBO ON VM CPU)
# ============================================================

print("🧠 Loading local SD-Turbo model onto VM CPU...")
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
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def generate_local_ai_image(prompt_text, output_path, overlay_text=None):
    full_prompt = (
        f"cartoon stickman finance illustration, {prompt_text}, "
        f"clean bold black lines, high contrast, flat 2d vector style, expressive actions, "
        f"minimalist art, youtube stickman animation style, plain white background"
    )

    image = pipe(
        prompt=full_prompt,
        num_inference_steps=1,
        guidance_scale=0.0,
        height=512,
        width=512
    ).images[0]

    canvas = Image.new("RGB", (1280, 720), (255, 255, 255))
    image = image.resize((720, 720))
    canvas.paste(image, ((1280 - 720) // 2, 0))

    if overlay_text:
        draw = ImageDraw.Draw(canvas)
        font = get_system_font(52)
        clean_badge = overlay_text.upper().strip()

        text_bbox = draw.textbbox((0, 0), clean_badge, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        bx, by = 60, 50
        padding_x, padding_y = 30, 15
        draw.rectangle(
            [(bx, by), (bx + text_w + padding_x * 2, by + text_h + padding_y * 2)],
            fill=(255, 221, 0),
            outline=(0, 0, 0),
            width=5
        )
        draw.text((bx + padding_x, by + padding_y), clean_badge, fill=(0, 0, 0), font=font)

    canvas.save(output_path, format="PNG")
    print(f"   ✅ Local AI Image created: {output_path}")


# ============================================================
# HELPERS
# ============================================================

def clean_json_text(raw_text):
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Qwen ne valid JSON object return nahi kiya.")
    return raw_text[start:end + 1]


def request_json(url, payload):
    response = requests.post(url, json=payload, timeout=None)
    response.raise_for_status()
    return response.json()


async def generate_audio(text, output_path):
    communicate = edge_tts.Communicate(text, VOICE)
    await communicate.save(output_path)
    print(f"   ✅ Audio created: {output_path}")


# ============================================================
# SCRIPT GENERATION
# ============================================================

print("=" * 60)
print(f"🧠 ASKING QWEN TO CREATE STICKMAN FINANCE EPISODE #{CURRENT_COUNT}")
print("=" * 60)

prompt = f"""
You are an expert viral YouTube scriptwriter for a Stickman Finance channel.
Generate Episode #{CURRENT_COUNT}.

GOAL: Extreme retention, fast-paced storytelling, and high CTR clickbait.

RULES:
- Hook in first 3 seconds: No intros, no greetings. Start with a shocking mistake or urgent rule.
- Short, punchy sentences (Grade 4-5 level simple English).
- Fast rhythm: 6 to 8 scenes. Each scene must be only 1 to 2 short sentences.
- Clickbait thumbnail text: 2-3 words ONLY in ALL CAPS (e.g. STOP THIS!, BIG LIE!, SAVE $10,000).

Return ONLY valid JSON:
{{
  "video_title": "Curiosity driven title under 55 characters",
  "seo_tags": "finance, money tips, investing, debt, wealth, habits",
  "description": "Engaging YouTube description with hook, actionable lessons, and viral hashtags.",
  "thumbnail_text": "2-3 WORDS ALL CAPS",
  "thumbnail_prompt": "Shocked stickman pointing at burning wallet, dramatic facial expression, simple clean cartoon",
  "script_sections": [
    {{
      "text": "1-2 punchy spoken lines.",
      "image_prompt": "Clear stickman action, e.g. stickman drowning under a giant credit card"
    }}
  ]
}}
"""

payload = {
    "model": OLLAMA_MODEL,
    "prompt": prompt,
    "stream": False,
    "format": "json"
}

qwen_response = request_json(OLLAMA_URL, payload)
clean_json = clean_json_text(qwen_response["response"])
data = json.loads(clean_json)

final_title = f"{data['video_title']} #{CURRENT_COUNT}"
sections = data["script_sections"]
SCENE_COUNT = len(sections)

print(f"🎬 TITLE: {final_title}")
print(f"🎞️ SCENE COUNT: {SCENE_COUNT}")


# ============================================================
# GENERATE CLICKBAIT THUMBNAIL
# ============================================================

print(f"\n🖼️ Generating AI Thumbnail #{CURRENT_COUNT}...")
thumbnail_path = os.path.join(OUTPUT_DIR, f"thumbnail_{CURRENT_COUNT}.png")
generate_local_ai_image(
    data["thumbnail_prompt"],
    thumbnail_path,
    overlay_text=data.get("thumbnail_text", "DON'T DO THIS!")
)


# ============================================================
# ASSETS GENERATION
# ============================================================

async def generate_all_assets():
    for index, section in enumerate(sections):
        scene_number = str(index).zfill(3)
        audio_path = os.path.join(AUDIO_DIR, f"scene_{scene_number}.mp3")
        image_path = os.path.join(IMAGE_DIR, f"scene_{scene_number}.png")

        print(f"\n🎬 SCENE {index + 1}/{SCENE_COUNT}")
        await generate_audio(section["text"], audio_path)
        generate_local_ai_image(section["image_prompt"], image_path)

asyncio.run(generate_all_assets())


# ============================================================
# NUMBERED SEO FILE
# ============================================================

seo_path = os.path.join(OUTPUT_DIR, f"seo_{CURRENT_COUNT}.txt")
with open(seo_path, "w", encoding="utf-8") as file:
    file.write(f"TITLE:\n{final_title}\n\n")
    file.write(f"DESCRIPTION:\n{data['description']}\n\n")
    file.write(f"TAGS:\n{data['seo_tags']}\n\n")
    file.write(f"EPISODE_NUMBER:\n{CURRENT_COUNT}\n")


# ============================================================
# ASSEMBLE FINAL VIDEO
# ============================================================

print("\n" + "=" * 60)
print(f"🎞️ ASSEMBLING VIDEO {CURRENT_COUNT}")
print("=" * 60)

clips = []
final_video = None

try:
    for index in range(SCENE_COUNT):
        scene_number = str(index).zfill(3)
        audio_path = os.path.join(AUDIO_DIR, f"scene_{scene_number}.mp3")
        image_path = os.path.join(IMAGE_DIR, f"scene_{scene_number}.png")

        audio_clip = AudioFileClip(audio_path)
        image_clip = (
            ImageClip(image_path)
            .set_duration(audio_clip.duration)
            .set_audio(audio_clip)
        )
        clips.append(image_clip)

    final_video = concatenate_videoclips(clips, method="compose")
    video_path = os.path.join(OUTPUT_DIR, f"video_{CURRENT_COUNT}.mp4")

    final_video.write_videofile(
        video_path,
        fps=24,
        codec="libx264",
        audio_codec="aac",
        threads=2,
        logger="bar"
    )

finally:
    for clip in clips:
        try:
            clip.close()
        except Exception:
            pass
    if final_video is not None:
        try:
            final_video.close()
        except Exception:
            pass

print("\n" + "=" * 60)
print(f"✅ EPISODE #{CURRENT_COUNT} CREATED IN 'output/'")
print("=" * 60)
