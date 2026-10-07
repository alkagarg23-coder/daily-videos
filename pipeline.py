import os
import json
import time
import urllib.parse
import asyncio
import torch
import requests
import edge_tts

from PIL import Image, ImageDraw, ImageFont
from diffusers import AutoPipelineForText2Image
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips


# ============================================================
# CONFIG & AUTO COUNTER TRACKING (NO API KEYS REQUIRED)
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

print(f"🎯 Starting Long-Form Engine (20-30 Min) for Episode #{CURRENT_COUNT}")


# ============================================================
# LOCAL AI IMAGE ENGINE (SD-TURBO ON VM CPU - NO API KEYS)
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


def generate_local_scene_image(prompt_text, output_path):
    clean_prompt = (
        f"clean flat vector minimalist stickman line art, {prompt_text}, "
        f"crisp bold black ink outlines, clean modern graphic design, solid plain light grey background, "
        f"no realistic human skin, 2D infographic whiteboard animation style, high clarity"
    )

    image = pipe(
        prompt=clean_prompt,
        num_inference_steps=1,
        guidance_scale=0.0,
        height=512,
        width=512
    ).images[0]

    widescreen = image.resize((1280, 720), Image.Resampling.LANCZOS)
    widescreen.save(output_path, format="PNG")
    print(f"   ✅ Scene Image created: {output_path}")


def generate_high_ctr_thumbnail(prompt_text, badge_text, color_theme, output_path):
    print("🎨 Generating 16:9 Thumbnail via Flux...")

    flux_prompt = (
        f"viral YouTube thumbnail, 16:9 widescreen, {prompt_text}, {color_theme} lighting and atmosphere, "
        f"modern 3D stylized cartoon character finance concept, expressive dramatic posture, "
        f"cinematic rim light, high contrast volumetric lighting, 8k resolution, trending on Artstation"
    )

    encoded = urllib.parse.quote(flux_prompt)
    seed = int(time.time()) + CURRENT_COUNT * 77
    flux_url = f"https://image.pollinations.ai/prompt/{encoded}?width=1280&height=720&model=flux&nologo=true&seed={seed}"

    img = None
    try:
        res = requests.get(flux_url, timeout=35)
        res.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(res.content)
        img = Image.open(output_path).convert("RGB")
    except Exception as e:
        print(f"⚠️ Flux cloud request timed out, switching to high-res local fallback: {e}")
        raw = pipe(
            prompt=flux_prompt,
            num_inference_steps=1,
            guidance_scale=0.0,
            height=512,
            width=512
        ).images[0]
        img = raw.resize((1280, 720), Image.Resampling.LANCZOS)

    draw = ImageDraw.Draw(img)
    font = get_system_font(56)
    badge = str(badge_text).upper().strip()

    bbox = draw.textbbox((0, 0), badge, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    bx, by = 60, 50
    pad_x, pad_y = 30, 14

    draw.rounded_rectangle(
        [(bx + 6, by + 6), (bx + tw + pad_x * 2 + 6, by + th + pad_y * 2 + 6)],
        radius=14,
        fill=(0, 0, 0, 200)
    )
    draw.rounded_rectangle(
        [(bx, by), (bx + tw + pad_x * 2, by + th + pad_y * 2)],
        radius=14,
        fill=(255, 215, 0),
        outline=(0, 0, 0),
        width=5
    )
    draw.text((bx + pad_x, by + pad_y), badge, fill=(0, 0, 0), font=font)

    img.save(output_path, format="PNG")
    print(f"✅ Authentic 16:9 Thumbnail saved: {output_path}")


# ============================================================
# HELPERS
# ============================================================

def clean_json_text(raw_text):
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Qwen did not return a valid JSON object.")
    return raw_text[start:end + 1]


def request_json(url, payload):
    response = requests.post(url, json=payload, timeout=None)
    response.raise_for_status()
    return response.json()


async def generate_audio(text, output_path):
    communicate = edge_tts.Communicate(text, VOICE)
    await communicate.save(output_path)
    print(f"   ✅ Audio ready: {output_path}")


# ============================================================
# STEP 1: GENERATE OVERALL VIDEO PLAN
# ============================================================

print("=" * 60)
print(f"🧠 STEP 1: PLANNING 20-30 MINUTE MASTERCLASS #{CURRENT_COUNT}")
print("=" * 60)

outline_prompt = f"""
You are an elite YouTube creator making a 25-minute deep-dive Masterclass on US Personal Finance Episode #{CURRENT_COUNT}.
Pick a distinct theme (e.g., Hidden Bank Fees, Credit Score Manipulation, Stock Market Psychology, Retirement Traps, Tax Loopholes).

Return ONLY valid JSON:
{{
  "video_title": "Unique high-CTR title under 60 characters",
  "seo_tags": "finance, investing, money tips, wealth, 401k, us economy, debt free",
  "description": "Engaging description with hooks and timestamps.",
  "thumbnail_badge": "2-3 WORDS ALL CAPS",
  "thumbnail_visual": "Detailed concept visual of finance dilemma",
  "thumbnail_color_theme": "neon crimson red and pitch black dramatic contrast",
  "chapters": [
    "Chapter 1: The First Unspoken Rule",
    "Chapter 2: The Math They Hide From You",
    "Chapter 3: The Danger of Normal Habits",
    "Chapter 4: The Strategic Exit Plan",
    "Chapter 5: Long-Term Compounding Reality"
  ]
}}
"""

response = request_json(OLLAMA_URL, {"model": OLLAMA_MODEL, "prompt": outline_prompt, "stream": False, "format": "json"})
plan_data = json.loads(clean_json_text(response["response"]))

video_title = f"{plan_data.get('video_title', 'The Wealth Blueprint')} #{CURRENT_COUNT}"
thumbnail_visual = plan_data.get("thumbnail_visual", "Stickman standing on pile of gold facing stormy financial clouds")
thumbnail_badge = plan_data.get("thumbnail_badge", "WAKE UP!")
color_theme = plan_data.get("thumbnail_color_theme", "deep moody blue with radiant gold highlights")
chapters = plan_data.get("chapters", [
    "Chapter 1: Cashflow Traps",
    "Chapter 2: The Debt Illusion",
    "Chapter 3: High Yield Foundations",
    "Chapter 4: Index Fund Realities",
    "Chapter 5: The Wealth Endgame"
])

print(f"🎬 Title: {video_title}")
print(f"📚 Chapters: {len(chapters)}")


# ============================================================
# STEP 2: GENERATE THUMBNAIL
# ============================================================

print("\n🖼️ Generating High-CTR Thumbnail...")
thumbnail_path = os.path.join(OUTPUT_DIR, f"thumbnail_{CURRENT_COUNT}.png")
generate_high_ctr_thumbnail(thumbnail_visual, thumbnail_badge, color_theme, thumbnail_path)


# ============================================================
# STEP 3: DEEP-DIVE SCRIPT GENERATION FOR EACH CHAPTER
# ============================================================

all_sections = []

for c_idx, chapter_title in enumerate(chapters):
    print("\n" + "=" * 60)
    print(f"📖 WRITING CHAPTER {c_idx + 1}/{len(chapters)}: {chapter_title}")
    print("=" * 60)

    chapter_prompt = f"""
Write Chapter {c_idx + 1}: "{chapter_title}" for a deep-dive finance video.
Target: US viewers seeking practical wealth management.
Provide 7 engaging scenes.
Each scene must have 60 to 90 words of clear, conversational English with real examples.

Return ONLY valid JSON:
{{
  "scenes": [
    {{
      "narration": "Detailed conversational spoken paragraph (60-90 words)...",
      "image_prompt": "Action of stickman illustrating the concept"
    }}
  ]
}}
"""
    chap_res = request_json(OLLAMA_URL, {"model": OLLAMA_MODEL, "prompt": chapter_prompt, "stream": False, "format": "json"})
    chap_data = json.loads(clean_json_text(chap_res["response"]))
    scenes_list = chap_data.get("scenes") or chap_data.get("script_sections") or []
    all_sections.extend(scenes_list)

TOTAL_SCENES = len(all_sections)
print(f"\n🎬 Total scenes generated for full video: {TOTAL_SCENES}")


# ============================================================
# STEP 4: GENERATE ASSETS WITH SAFE KEYS
# ============================================================

async def generate_assets():
    for idx, scene in enumerate(all_sections):
        scene_id = str(idx).zfill(3)
        audio_path = os.path.join(AUDIO_DIR, f"scene_{scene_id}.mp3")
        image_path = os.path.join(IMAGE_DIR, f"scene_{scene_id}.png")

        if isinstance(scene, dict):
            narration_text = (
                scene.get("narration")
                or scene.get("text")
                or scene.get("script")
                or "Understanding financial habits is the single most important skill for building sustainable wealth."
            )
            image_prompt_text = (
                scene.get("image_prompt")
                or scene.get("prompt")
                or scene.get("visual")
                or scene.get("image")
                or narration_text[:60]
            )
        else:
            narration_text = str(scene)
            image_prompt_text = "Stickman managing cashflow and growing assets"

        print(f"\n⚙️ Rendering Scene {idx + 1}/{TOTAL_SCENES}...")
        await generate_audio(narration_text, audio_path)
        generate_local_scene_image(image_prompt_text, image_path)

asyncio.run(generate_assets())


# ============================================================
# STEP 5: SAVE SEO & DESCRIPTION
# ============================================================

seo_path = os.path.join(OUTPUT_DIR, f"seo_{CURRENT_COUNT}.txt")
with open(seo_path, "w", encoding="utf-8") as f:
    f.write(f"TITLE:\n{video_title}\n\n")
    f.write(f"DESCRIPTION:\n{plan_data.get('description', '')}\n\n")
    f.write(f"TAGS:\n{plan_data.get('seo_tags', '')}\n\n")
    f.write(f"EPISODE:\n#{CURRENT_COUNT}\n")
    f.write(f"TOTAL_SCENES:\n{TOTAL_SCENES}\n")


# ============================================================
# STEP 6: ASSEMBLE 20-30 MINUTE VIDEO (MOVIEPY)
# ============================================================

print("\n" + "=" * 60)
print(f"🎞️ ASSEMBLING FULL MASTERCLASS VIDEO ({TOTAL_SCENES} SCENES)")
print("=" * 60)

clips = []
final_video = None

try:
    for idx in range(TOTAL_SCENES):
        scene_id = str(idx).zfill(3)
        audio_path = os.path.join(AUDIO_DIR, f"scene_{scene_id}.mp3")
        image_path = os.path.join(IMAGE_DIR, f"scene_{scene_id}.png")

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
        preset="ultrafast",
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
print(f"✅ FULL-LENGTH MASTERCLASS #{CURRENT_COUNT} ASSEMBLED LOCALLY!")
print("=" * 60)
