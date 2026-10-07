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

print(f"🎯 Starting Long-Form Engine (20-30 Min) for Episode #{CURRENT_COUNT}")


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
        clean_badge = str(overlay_text).upper().strip()

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
    print(f"   ✅ Image ready: {output_path}")


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
# STEP 1: GENERATE OVERALL VIDEO PLAN & OUTLINE
# ============================================================

print("=" * 60)
print(f"🧠 STEP 1: PLANNING 20-30 MINUTE MASTERCLASS #{CURRENT_COUNT}")
print("=" * 60)

outline_prompt = f"""
You are an elite YouTube creator making a 25-minute deep-dive Masterclass on Personal Finance Episode #{CURRENT_COUNT}.
Target Audience: US Citizens.
Plan 5 comprehensive chapters that cover every aspect in depth (Saving, Investing, Psychology, Debt Traps, Building Wealth).

Return ONLY valid JSON:
{{
  "video_title": "Ultimate Financial Masterclass: From Broke to Wealthy",
  "seo_tags": "personal finance, investing 101, build wealth, financial freedom, money mastery, 401k, roth ira",
  "description": "A comprehensive 25-minute masterclass covering everything you need to know about money management.",
  "thumbnail_text": "DON'T BE POOR!",
  "thumbnail_prompt": "Shocked stickman holding an empty wallet vs wealthy stickman sitting on gold coins, dramatic cartoon",
  "chapters": [
    "The Dark Psychology of Consumerism & Spending Traps",
    "The Emergency Fund Blueprint and Cashflow System",
    "Understanding Debt: Good Debt vs Toxic Debt",
    "Index Funds & Compound Interest: The Real Math",
    "Building Multi-Stream Wealth & Long-Term Freedom"
  ]
}}
"""

response = request_json(OLLAMA_URL, {"model": OLLAMA_MODEL, "prompt": outline_prompt, "stream": False, "format": "json"})
plan_data = json.loads(clean_json_text(response["response"]))

video_title = f"{plan_data.get('video_title', 'Finance Masterclass')} #{CURRENT_COUNT}"
thumbnail_prompt = plan_data.get("thumbnail_prompt", "Stickman holding money bag with upward financial charts")
thumbnail_badge = plan_data.get("thumbnail_text", "DON'T BE POOR!")
chapters = plan_data.get("chapters", [
    "Chapter 1: The Money Basics",
    "Chapter 2: Managing Debt",
    "Chapter 3: Saving Cashflow",
    "Chapter 4: Smart Investing",
    "Chapter 5: Wealth Freedom"
])

print(f"🎬 Title: {video_title}")
print(f"📚 Chapters: {len(chapters)}")


# ============================================================
# STEP 2: GENERATE THUMBNAIL
# ============================================================

print("\n🖼️ Generating Masterclass AI Thumbnail...")
thumbnail_path = os.path.join(OUTPUT_DIR, f"thumbnail_{CURRENT_COUNT}.png")
generate_local_ai_image(thumbnail_prompt, thumbnail_path, overlay_text=thumbnail_badge)


# ============================================================
# STEP 3: DEEP-DIVE SCRIPT GENERATION FOR EACH CHAPTER
# ============================================================

all_sections = []

for c_idx, chapter_title in enumerate(chapters):
    print("\n" + "=" * 60)
    print(f"📖 WRITING CHAPTER {c_idx + 1}/{len(chapters)}: {chapter_title}")
    print("=" * 60)

    chapter_prompt = f"""
Write Chapter {c_idx + 1} for a 25-minute deep-dive YouTube video: "{chapter_title}".
Target Audience: US audience.
Provide 7 detailed scenes for this chapter.
Each scene MUST contain a long, detailed, and educational explanation (around 60 to 90 words per scene).
Simple conversational English, full of real examples.

Return ONLY valid JSON:
{{
  "scenes": [
    {{
      "narration": "Deep detailed spoken paragraph (60-90 words)...",
      "image_prompt": "Stickman character demonstrating this exact concept"
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
# STEP 4: GENERATE ASSETS (AUDIO + AI IMAGES WITH SAFE KEYS)
# ============================================================

async def generate_assets():
    for idx, scene in enumerate(all_sections):
        scene_id = str(idx).zfill(3)
        audio_path = os.path.join(AUDIO_DIR, f"scene_{scene_id}.mp3")
        image_path = os.path.join(IMAGE_DIR, f"scene_{scene_id}.png")

        # Bulletproof safe extraction taaki KeyError 'image_prompt' dobara na aaye
        if isinstance(scene, dict):
            narration_text = (
                scene.get("narration") 
                or scene.get("text") 
                or scene.get("script") 
                or "Understanding money and smart financial habits is key to building wealth."
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
            image_prompt_text = "Stickman managing finances and growing money"

        print(f"\n⚙️ Rendering Scene {idx + 1}/{TOTAL_SCENES}...")
        await generate_audio(narration_text, audio_path)
        generate_local_ai_image(image_prompt_text, image_path)

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
print(f"✅ FULL-LENGTH 20-30 MIN MASTERCLASS #{CURRENT_COUNT} ASSEMBLED LOCALLY!")
print("=" * 60)
