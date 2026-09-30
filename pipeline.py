import os
import json
import asyncio
import torch
import requests
import edge_tts

from PIL import Image
from diffusers import AutoPipelineForText2Image
from mega import Mega
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips


# ============================================================
# CONFIG
# ============================================================

MEGA_EMAIL = os.environ.get("MEGA_EMAIL")
MEGA_PASSWORD = os.environ.get("MEGA_PASSWORD")

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:3b"

VOICE = "en-US-ChristopherNeural"

OUTPUT_DIR = "output"
AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio")
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

if not MEGA_EMAIL or not MEGA_PASSWORD:
    raise RuntimeError("MEGA_EMAIL aur MEGA_PASSWORD secrets set karna zaroori hai.")


# ============================================================
# LOCAL AI IMAGE ENGINE (SD-TURBO ON VM CPU)
# ============================================================

print("🧠 Loading local SD-Turbo model onto GitHub VM CPU...")
pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sd-turbo",
    torch_dtype=torch.float32
)
pipe.to("cpu")
pipe.enable_attention_slicing()


def generate_local_ai_image(prompt_text, output_path):
    """Bina kisi external API ke local VM CPU par AI drawing banata hai."""
    full_prompt = (
        f"stickman finance cartoon, {prompt_text}, minimalist black ink stick figure on clean white paper, "
        f"bold outlines, 2d vector style, simple, sharp, high quality"
    )

    image = pipe(
        prompt=full_prompt,
        num_inference_steps=1,
        guidance_scale=0.0,
        height=512,
        width=512
    ).images[0]

    # 16:9 canvas (1280x720) mein fit karna
    canvas = Image.new("RGB", (1280, 720), (255, 255, 255))
    image = image.resize((720, 720))
    canvas.paste(image, ((1280 - 720) // 2, 0))
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
# QWEN SCRIPT GENERATION
# ============================================================

print("=" * 60)
print("🧠 ASKING QWEN TO CREATE STICKMAN FINANCE VIDEO")
print("=" * 60)

prompt = """
You are an automated YouTube video engine for a Stickman Finance channel.

Create one highly engaging YouTube video about personal finance,
money psychology, investing basics, saving, debt, income,
financial mistakes, wealth building, or behavioral finance.

STYLE:
- Easy English
- Short sentences
- 5 to 7 script sections

Return ONLY valid JSON.
No markdown.

Required structure:
{
  "video_title": "YouTube title under 70 characters",
  "seo_tags": "tag1, tag2, tag3, tag4",
  "description": "YouTube description with hashtags.",
  "thumbnail_prompt": "Detailed description of stickman finance thumbnail",
  "script_sections": [
    {
      "text": "Narration for this scene.",
      "image_prompt": "Detailed stickman action scene description"
    }
  ]
}
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

sections = data["script_sections"]
SCENE_COUNT = len(sections)

print(f"🎬 TITLE: {data['video_title']}")
print(f"🎞️️ SCENE COUNT: {SCENE_COUNT}")


# ============================================================
# GENERATE THUMBNAIL
# ============================================================

print("🖼️ Generating AI Thumbnail...")
thumbnail_path = os.path.join(OUTPUT_DIR, "thumbnail.png")
generate_local_ai_image(data["thumbnail_prompt"], thumbnail_path)


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
# SEO FILE
# ============================================================

seo_path = os.path.join(OUTPUT_DIR, "seo_and_description.txt")
with open(seo_path, "w", encoding="utf-8") as file:
    file.write(f"TITLE:\n{data['video_title']}\n\n")
    file.write(f"DESCRIPTION:\n{data['description']}\n\n")
    file.write(f"TAGS:\n{data['seo_tags']}\n\n")
    file.write(f"SCENE_COUNT:\n{SCENE_COUNT}\n")


# ============================================================
# ASSEMBLE VIDEO
# ============================================================

print("\n" + "=" * 60)
print("🎞️ ASSEMBLING FINAL VIDEO")
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
    video_path = os.path.join(OUTPUT_DIR, "video.mp4")

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


# ============================================================
# UPLOAD TO MEGA
# ============================================================

mega = Mega()
m = mega.login(MEGA_EMAIL, MEGA_PASSWORD)
folder_name = "Latest_YouTube_Video"

try:
    old_folder = m.find(folder_name)
    if old_folder:
        for folder in old_folder:
            try:
                m.destroy(folder)
            except Exception:
                pass
except Exception:
    pass

folder = m.create_folder(folder_name)
folder_id = folder[folder_name]

m.upload(video_path, folder_id)
m.upload(thumbnail_path, folder_id)
m.upload(seo_path, folder_id)

print("\n" + "=" * 60)
print("✅ AUTOMATION COMPLETE")
print("=" * 60)
