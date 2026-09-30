import os
import json
import asyncio
import urllib.parse
import time
import requests
import edge_tts

from io import BytesIO
from PIL import Image
from mega import Mega
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips


# ============================================================
# CONFIG
# ============================================================

MEGA_EMAIL = os.environ.get("MEGA_EMAIL")
MEGA_PASSWORD = os.environ.get("MEGA_PASSWORD")

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:3b"

IMAGE_BASE_URL = "https://image.pollinations.ai/prompt/"
VOICE = "en-US-ChristopherNeural"

OUTPUT_DIR = "output"
AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio")
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)


# ============================================================
# BASIC VALIDATION
# ============================================================

if not MEGA_EMAIL or not MEGA_PASSWORD:
    raise RuntimeError(
        "MEGA_EMAIL and MEGA_PASSWORD GitHub Secrets are required."
    )


# ============================================================
# HELPERS
# ============================================================

def clean_json_text(raw_text):
    """Extract JSON object even if model adds accidental text."""
    start = raw_text.find("{")
    end = raw_text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("Qwen did not return a valid JSON object.")

    return raw_text[start:end + 1]


def request_json(url, payload, retries=3):
    """POST JSON with retries."""
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=600
            )

            response.raise_for_status()
            return response.json()

        except Exception as exc:
            last_error = exc
            print(
                f"⚠️ Request failed "
                f"(attempt {attempt}/{retries}): {exc}"
            )

            if attempt < retries:
                time.sleep(5)

    raise RuntimeError(
        f"Request failed after {retries} attempts: {last_error}"
    )


def download_valid_png(prompt, output_path, retries=5):
    """
    Download image and make absolutely sure it is a real PNG.
    This prevents MoviePy/Pillow from receiving HTML/error bytes.
    """

    encoded_prompt = urllib.parse.quote(prompt, safe="")

    url = (
        f"{IMAGE_BASE_URL}{encoded_prompt}"
        "?width=1280"
        "&height=720"
        "&nologo=true"
        "&model=flux"
    )

    last_error = None

    for attempt in range(1, retries + 1):
        try:
            print(
                f"   🖼️ Image download "
                f"attempt {attempt}/{retries}"
            )

            response = requests.get(
                url,
                timeout=180,
                headers={
                    "User-Agent": "Mozilla/5.0"
                }
            )

            response.raise_for_status()

            content = response.content

            if len(content) < 1000:
                raise ValueError(
                    f"Image response too small: {len(content)} bytes"
                )

            # Decode image from memory.
            with Image.open(BytesIO(content)) as img:
                img.load()

                # Always convert to RGB/RGBA-compatible PNG.
                if img.mode not in ("RGB", "RGBA"):
                    img = img.convert("RGB")

                img.save(
                    output_path,
                    format="PNG"
                )

            # Final validation.
            with Image.open(output_path) as check:
                check.verify()

            print(f"   ✅ Valid PNG created: {output_path}")
            return

        except Exception as exc:
            last_error = exc

            print(
                f"   ⚠️ Image generation failed: {exc}"
            )

            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except Exception:
                    pass

            if attempt < retries:
                wait_time = attempt * 5
                print(
                    f"   ⏳ Retrying in {wait_time} seconds..."
                )
                time.sleep(wait_time)

    raise RuntimeError(
        f"Could not create valid image after "
        f"{retries} attempts: {last_error}"
    )


async def generate_audio(text, output_path, retries=5):
    """Generate Edge TTS audio with retries."""

    last_error = None

    for attempt in range(1, retries + 1):
        try:
            print(
                f"   🔊 Audio attempt {attempt}/{retries}"
            )

            communicate = edge_tts.Communicate(
                text,
                VOICE
            )

            await communicate.save(output_path)

            if not os.path.exists(output_path):
                raise RuntimeError(
                    "Audio file was not created."
                )

            if os.path.getsize(output_path) < 1000:
                raise RuntimeError(
                    "Audio file is suspiciously small."
                )

            print(f"   ✅ Audio created: {output_path}")
            return

        except Exception as exc:
            last_error = exc

            print(
                f"   ⚠️ Audio generation failed: {exc}"
            )

            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except Exception:
                    pass

            if attempt < retries:
                wait_time = attempt * 3
                print(
                    f"   ⏳ Retrying in {wait_time} seconds..."
                )
                await asyncio.sleep(wait_time)

    raise RuntimeError(
        f"Could not generate audio after "
        f"{retries} attempts: {last_error}"
    )


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

The audience should be ordinary people who want to understand money
in a simple and entertaining way.

STYLE:
- Stickman Finance
- Simple 2D stickman cartoon
- White/light background
- Bold black outlines
- Flat colors
- Minimal visual clutter
- Clear visual storytelling
- No realistic humans
- No gradients
- No photographic style
- No complicated charts unless absolutely necessary

NARRATION:
- Calm but engaging
- Easy English
- Short sentences
- Second-person style where appropriate
- Each section should normally contain 1-3 short sentences
- Every section must move the story forward

IMPORTANT:
DO NOT use a fixed scene count.

Choose the number of script_sections naturally based on the story.
It may be 8, 11, 17, 23, 30, or any other reasonable number.

The automation will create EXACTLY:
1 audio file per script section
1 image per script section
1 video clip per script section

Therefore every script_sections item must contain both text and image_prompt.

Return ONLY valid JSON.
No markdown.
No explanation before or after JSON.

Required structure:

{
  "video_title": "YouTube title under 70 characters",
  "seo_tags": "tag1, tag2, tag3, tag4",
  "description": "A compelling YouTube description with a hook, useful context, CTA, and hashtags.",
  "thumbnail_prompt": "Stickman Finance thumbnail scene representing the title, bold black outlines, flat colors, dramatic but clean composition, YouTube thumbnail style",
  "script_sections": [
    {
      "text": "Narration for this scene.",
      "image_prompt": "Detailed Stickman Finance visual scene for this narration."
    }
  ]
}

Make the story complete from beginning to end.
Do not create empty sections.
Do not number the sections.
Do not put multiple scenes inside one script_sections item.
"""

payload = {
    "model": OLLAMA_MODEL,
    "prompt": prompt,
    "stream": False,
    "format": "json"
}

print("⏳ Qwen is generating the script...")

qwen_response = request_json(
    OLLAMA_URL,
    payload,
    retries=3
)

if "response" not in qwen_response:
    raise RuntimeError(
        "Ollama response did not contain 'response'."
    )

raw_text = qwen_response["response"]

clean_json = clean_json_text(raw_text)

try:
    data = json.loads(clean_json)
except json.JSONDecodeError as exc:
    raise RuntimeError(
        f"Qwen returned invalid JSON: {exc}"
    )


# ============================================================
# VALIDATE SCRIPT
# ============================================================

required_fields = [
    "video_title",
    "seo_tags",
    "description",
    "thumbnail_prompt",
    "script_sections"
]

for field in required_fields:
    if field not in data:
        raise RuntimeError(
            f"Qwen JSON is missing required field: {field}"
        )


sections = data["script_sections"]

if not isinstance(sections, list):
    raise RuntimeError(
        "script_sections must be a list."
    )

if len(sections) == 0:
    raise RuntimeError(
        "Qwen generated zero script sections."
    )


for index, section in enumerate(sections):
    if not isinstance(section, dict):
        raise RuntimeError(
            f"Scene {index} is not a JSON object."
        )

    if not section.get("text"):
        raise RuntimeError(
            f"Scene {index} has no narration text."
        )

    if not section.get("image_prompt"):
        raise RuntimeError(
            f"Scene {index} has no image prompt."
        )


SCENE_COUNT = len(sections)

print()
print("=" * 60)
print(f"🎬 TITLE: {data['video_title']}")
print(f"🎞️ DYNAMIC SCENE COUNT: {SCENE_COUNT}")
print("=" * 60)
print()


# ============================================================
# GENERATE THUMBNAIL
# ============================================================

print("🖼️ Generating thumbnail...")

thumbnail_path = os.path.join(
    OUTPUT_DIR,
    "thumbnail.png"
)

download_valid_png(
    data["thumbnail_prompt"],
    thumbnail_path,
    retries=5
)


# ============================================================
# GENERATE EXACTLY N AUDIO + IMAGE FILES
# ============================================================

async def generate_all_assets():

    for index, section in enumerate(sections):

        scene_number = str(index).zfill(3)

        audio_path = os.path.join(
            AUDIO_DIR,
            f"scene_{scene_number}.mp3"
        )

        image_path = os.path.join(
            IMAGE_DIR,
            f"scene_{scene_number}.png"
        )

        print()
        print(
            f"🎬 SCENE {index + 1}/{SCENE_COUNT}"
        )
        print("-" * 50)

        await generate_audio(
            section["text"],
            audio_path
        )

        download_valid_png(
            section["image_prompt"],
            image_path
        )

        print(
            f"✅ Scene {index + 1}/{SCENE_COUNT} complete"
        )


print("=" * 60)
print("🎨 GENERATING AUDIO + IMAGES")
print("=" * 60)

asyncio.run(generate_all_assets())


# ============================================================
# SEO FILE
# ============================================================

print()
print("📝 Creating SEO file...")

seo_path = os.path.join(
    OUTPUT_DIR,
    "seo_and_description.txt"
)

with open(
    seo_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        f"TITLE:\n{data['video_title']}\n\n"
    )

    file.write(
        f"DESCRIPTION:\n{data['description']}\n\n"
    )

    file.write(
        f"TAGS:\n{data['seo_tags']}\n\n"
    )

    file.write(
        f"SCENE_COUNT:\n{SCENE_COUNT}\n"
    )


# ============================================================
# FINAL VALIDATION BEFORE MOVIEPY
# ============================================================

print()
print("=" * 60)
print("🔍 VALIDATING GENERATED ASSETS")
print("=" * 60)

for index in range(SCENE_COUNT):

    scene_number = str(index).zfill(3)

    audio_path = os.path.join(
        AUDIO_DIR,
        f"scene_{scene_number}.mp3"
    )

    image_path = os.path.join(
        IMAGE_DIR,
        f"scene_{scene_number}.png"
    )

    if not os.path.isfile(audio_path):
        raise RuntimeError(
            f"Missing audio: {audio_path}"
        )

    if not os.path.isfile(image_path):
        raise RuntimeError(
            f"Missing image: {image_path}"
        )

    if os.path.getsize(audio_path) < 1000:
        raise RuntimeError(
            f"Invalid/empty audio: {audio_path}"
        )

    try:
        with Image.open(image_path) as img:
            img.verify()
    except Exception as exc:
        raise RuntimeError(
            f"Invalid PNG: {image_path}: {exc}"
        )

    print(
        f"✅ Scene {index + 1}/{SCENE_COUNT} assets valid"
    )


# ============================================================
# BUILD VIDEO
# ============================================================

print()
print("=" * 60)
print("🎞️ ASSEMBLING FINAL VIDEO")
print("=" * 60)

clips = []
final_video = None

try:

    for index in range(SCENE_COUNT):

        scene_number = str(index).zfill(3)

        audio_path = os.path.join(
            AUDIO_DIR,
            f"scene_{scene_number}.mp3"
        )

        image_path = os.path.join(
            IMAGE_DIR,
            f"scene_{scene_number}.png"
        )

        print(
            f"🎞️ Building clip "
            f"{index + 1}/{SCENE_COUNT}"
        )

        audio_clip = AudioFileClip(
            audio_path
        )

        image_clip = (
            ImageClip(image_path)
            .set_duration(audio_clip.duration)
            .set_audio(audio_clip)
        )

        clips.append(image_clip)


    print("🔗 Joining all scenes...")

    final_video = concatenate_videoclips(
        clips,
        method="compose"
    )

    video_path = os.path.join(
        OUTPUT_DIR,
        "video.mp4"
    )

    print("💾 Writing video.mp4...")

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
# FINAL VIDEO CHECK
# ============================================================

video_path = os.path.join(
    OUTPUT_DIR,
    "video.mp4"
)

if not os.path.isfile(video_path):
    raise RuntimeError(
        "Final video was not created."
    )

if os.path.getsize(video_path) < 10000:
    raise RuntimeError(
        "Final video file is suspiciously small."
    )


# ============================================================
# UPLOAD TO MEGA
# ============================================================

print()
print("=" * 60)
print("☁️ UPLOADING TO MEGA")
print("=" * 60)

mega = Mega()

print("🔐 Logging into Mega...")

m = mega.login(
    MEGA_EMAIL,
    MEGA_PASSWORD
)

folder_name = "Latest_YouTube_Video"

print("🧹 Removing previous Latest_YouTube_Video folder...")

try:

    old_folder = m.find(folder_name)

    if old_folder:
        for folder in old_folder:
            try:
                m.destroy(folder)
                print("🗑️ Old folder deleted.")
            except Exception as exc:
                print(
                    f"⚠️ Could not delete old folder: {exc}"
                )

except Exception as exc:

    print(
        f"⚠️ Old folder lookup skipped: {exc}"
    )


print("📁 Creating new Mega folder...")

folder = m.create_folder(
    folder_name
)

folder_id = folder[folder_name]


print("🚀 Uploading video...")

m.upload(
    video_path,
    folder_id
)

print("🚀 Uploading thumbnail...")

m.upload(
    thumbnail_path,
    folder_id
)

print("🚀 Uploading SEO file...")

m.upload(
    seo_path,
    folder_id
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 60)
print("✅ AUTOMATION COMPLETE")
print("=" * 60)
print(f"🎬 Title: {data['video_title']}")
print(f"🎞️ Scenes: {SCENE_COUNT}")
print("🔊 Audio files: generated dynamically")
print("🖼️ Images: generated dynamically")
print("🎞️ Video clips: generated dynamically")
print("☁️ Mega upload: complete")
print("=" * 60)
