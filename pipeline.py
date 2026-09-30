import os
import json
import time
import asyncio
import urllib.parse
from pathlib import Path

import requests
import edge_tts
from PIL import Image
from mega import Mega
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips


# ============================================================
# 1. CONFIGURATION
# ============================================================

MEGA_EMAIL = os.environ.get("MEGA_EMAIL")
MEGA_PASSWORD = os.environ.get("MEGA_PASSWORD")

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:3b"
IMAGE_BASE_URL = "https://image.pollinations.ai/prompt/"
VOICE = "en-US-ChristopherNeural"

OUTPUT_DIR = Path("output")
AUDIO_DIR = OUTPUT_DIR / "audio"
IMAGE_DIR = OUTPUT_DIR / "images"

for folder in (OUTPUT_DIR, AUDIO_DIR, IMAGE_DIR):
    folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. QWEN PROMPT
# ============================================================

QWEN_PROMPT = r"""
You are an automated YouTube video engine for a Stickman Finance channel.

NICHE:
Educational, entertaining personal-finance and money-psychology content
explained with simple stickman characters.

Choose ONE highly engaging topic. Possible topics include:
- budgeting
- saving
- debt
- compound interest
- inflation
- investing concepts
- financial mistakes
- money psychology
- lifestyle inflation
- emergency funds
- credit
- scams and financial traps
- wealth-building principles
- financial independence

Do not provide personalized financial advice.
Do not promise profits or guaranteed investment returns.

NARRATION:
- Calm, conversational second-person narration.
- Simple English.
- Short sentences.
- Each section should normally contain 1-3 short sentences.
- Build a clear story: hook -> explanation -> examples -> conclusion.

VISUAL STYLE:
- Hand-drawn 2D stickman finance cartoon.
- White/light background.
- Bold black outlines.
- Flat colors.
- Simple objects and backgrounds.
- Clear visual storytelling.
- No gradients.
- No photorealism.
- No realistic humans.
- No text-heavy images.
- 16:9 YouTube composition.

SCENE COUNT:
There is NO fixed scene count.
Choose however many scenes are actually needed to tell the story properly.
Do NOT target 15.
Do NOT target 20.
Do NOT pad or truncate the story to hit a fixed number.

Every item in script_sections becomes exactly one audio clip,
one image, and one video scene.

Return ONLY valid JSON, with no markdown and no explanation.

Required JSON structure:
{
  "video_title": "Viral title under 70 characters",
  "seo_tags": "tag1, tag2, tag3",
  "description": "YouTube description with hook, useful context, CTA, and relevant hashtags",
  "thumbnail_prompt": "Detailed stickman finance thumbnail prompt",
  "script_sections": [
    {
      "text": "Narration for this scene",
      "image_prompt": "Detailed visual prompt for this scene"
    }
  ]
}

Requirements:
- script_sections must contain at least 1 item.
- Every item must have non-empty text.
- Every item must have non-empty image_prompt.
- Each section should move the story forward.
"""


# ============================================================
# 3. HELPERS
# ============================================================

def fail(message):
    raise RuntimeError(message)


def scene_paths(index):
    scene = str(index).zfill(3)
    return (
        AUDIO_DIR / f"scene_{scene}.mp3",
        IMAGE_DIR / f"scene_{scene}.png",
    )


def parse_qwen_json(raw_text):
    raw_text = (raw_text or "").strip()
    start = raw_text.find("{")
    end = raw_text.rfind("}") + 1

    if start < 0 or end <= start:
        fail(
            "Qwen did not return a JSON object.\n"
            f"Response:\n{raw_text[:3000]}"
        )

    try:
        data = json.loads(raw_text[start:end])
    except json.JSONDecodeError as exc:
        fail(
            f"Qwen returned invalid JSON: {exc}\n"
            f"Response:\n{raw_text[:3000]}"
        )

    required = (
        "video_title",
        "seo_tags",
        "description",
        "thumbnail_prompt",
        "script_sections",
    )
    missing = [key for key in required if key not in data]
    if missing:
        fail(f"Qwen JSON is missing: {missing}")

    sections = data["script_sections"]
    if not isinstance(sections, list) or not sections:
        fail("Qwen returned an empty script_sections list.")

    clean_sections = []
    for number, section in enumerate(sections, start=1):
        if not isinstance(section, dict):
            fail(f"Scene {number} is not a JSON object.")

        text = str(section.get("text", "")).strip()
        image_prompt = str(section.get("image_prompt", "")).strip()

        if not text:
            fail(f"Scene {number} has empty narration.")
        if not image_prompt:
            fail(f"Scene {number} has an empty image prompt.")

        clean_sections.append({
            "text": text,
            "image_prompt": image_prompt,
        })

    data["script_sections"] = clean_sections
    return data


def download_and_normalize_image(prompt_text, destination, width=1280, height=720, attempts=4):
    encoded = urllib.parse.quote(prompt_text, safe="")
    url = (
        f"{IMAGE_BASE_URL}{encoded}"
        f"?width={width}&height={height}&nologo=true"
    )

    temp = destination.with_suffix(".download")
    last_error = None

    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(
                url,
                timeout=(30, 180),
                headers={"User-Agent": "Mozilla/5.0"},
            )
            response.raise_for_status()

            content = response.content
            content_type = response.headers.get("Content-Type", "").lower()

            if len(content) < 1024:
                raise ValueError(
                    f"response too small ({len(content)} bytes), "
                    f"content-type={content_type}"
                )

            temp.write_bytes(content)

            # Decode whatever format the service returned and write
            # a genuine PNG. This prevents MoviePy/Pillow format errors.
            with Image.open(temp) as source:
                source.load()
                if source.width < 2 or source.height < 2:
                    raise ValueError("image dimensions are invalid")

                converted = (
                    source.convert("RGBA")
                    if source.mode in ("RGBA", "LA", "P")
                    else source.convert("RGB")
                )
                converted.save(temp, format="PNG")

            with Image.open(temp) as check:
                check.load()
                if check.format != "PNG":
                    raise ValueError("normalized file is not PNG")

            temp.replace(destination)
            return

        except Exception as exc:
            last_error = exc
            if temp.exists():
                temp.unlink()
            print(
                f"   ⚠️ image attempt {attempt}/{attempts} failed: {exc}",
                flush=True,
            )
            if attempt < attempts:
                time.sleep(attempt * 3)

    fail(f"Could not create valid image after {attempts} attempts: {last_error}")


async def generate_audio(text, destination, attempts=3):
    last_error = None

    for attempt in range(1, attempts + 1):
        try:
            if destination.exists():
                destination.unlink()

            communicate = edge_tts.Communicate(text, VOICE)
            await communicate.save(str(destination))

            if not destination.exists() or destination.stat().st_size < 1000:
                raise ValueError("audio file is missing or suspiciously small")

            return

        except Exception as exc:
            last_error = exc
            print(
                f"   ⚠️ audio attempt {attempt}/{attempts} failed: {exc}",
                flush=True,
            )
            if attempt < attempts:
                await asyncio.sleep(attempt * 2)

    fail(f"Could not create audio after {attempts} attempts: {last_error}")


# ============================================================
# 4. GENERATE SCRIPT WITH OLLAMA
# ============================================================

def generate_script():
    print("🧠 Asking Qwen for a Stickman Finance script...", flush=True)

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": QWEN_PROMPT,
        "stream": False,
        "format": "json",
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=900,
        )
        response.raise_for_status()
        result = response.json()
    except Exception as exc:
        fail(f"Ollama/Qwen request failed: {exc}")

    raw = result.get("response", "")
    data = parse_qwen_json(raw)

    print(f"🎬 Topic: {data['video_title']}", flush=True)
    print(
        f"🎞️ Qwen generated {len(data['script_sections'])} scenes. "
        "ALL will be processed.",
        flush=True,
    )

    return data


# ============================================================
# 5. GENERATE THUMBNAIL + ALL SCENE ASSETS
# ============================================================

async def generate_assets(data):
    print("🖼️ Generating thumbnail...", flush=True)
    download_and_normalize_image(
        data["thumbnail_prompt"],
        OUTPUT_DIR / "thumbnail.png",
    )

    total = len(data["script_sections"])

    for index, section in enumerate(data["script_sections"]):
        audio_path, image_path = scene_paths(index)

        print(
            f"🎨 Scene {index + 1}/{total}: "
            "generating audio + image...",
            flush=True,
        )

        await generate_audio(section["text"], audio_path)
        download_and_normalize_image(
            section["image_prompt"],
            image_path,
        )


# ============================================================
# 6. SEO FILE
# ============================================================

def create_seo_file(data):
    print("📝 Creating SEO file...", flush=True)

    seo_path = OUTPUT_DIR / "seo_and_description.txt"
    with seo_path.open("w", encoding="utf-8") as file:
        file.write(f"TITLE:\n{data['video_title']}\n\n")
        file.write(f"DESCRIPTION:\n{data['description']}\n\n")
        file.write(f"TAGS:\n{data['seo_tags']}\n\n")
        file.write(
            f"SCENE_COUNT:\n{len(data['script_sections'])}\n"
        )


# ============================================================
# 7. BUILD VIDEO FROM EXACT SCENE COUNT
# ============================================================

def build_video(data):
    print("🎞️ Assembling final video...", flush=True)

    clips = []
    final_video = None

    try:
        total = len(data["script_sections"])

        for index in range(total):
            audio_path, image_path = scene_paths(index)

            if not audio_path.exists():
                fail(f"Missing audio: {audio_path}")
            if not image_path.exists():
                fail(f"Missing image: {image_path}")

            # Validate the actual image immediately before MoviePy.
            try:
                with Image.open(image_path) as image:
                    image.load()
            except Exception as exc:
                fail(f"Invalid/corrupt image {image_path}: {exc}")

            audio_clip = AudioFileClip(str(audio_path))
            image_clip = (
                ImageClip(str(image_path))
                .set_duration(audio_clip.duration)
                .set_audio(audio_clip)
            )
            clips.append(image_clip)

        if not clips:
            fail("No clips were created.")

        final_video = concatenate_videoclips(clips, method="compose")
        final_video.write_videofile(
            str(OUTPUT_DIR / "video.mp4"),
            fps=24,
            codec="libx264",
            audio_codec="aac",
            logger=None,
        )

        print(f"✅ Video assembled from exactly {total} scenes.", flush=True)

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
# 8. UPLOAD TO MEGA
# ============================================================

def upload_to_mega():
    if not MEGA_EMAIL or not MEGA_PASSWORD:
        fail("MEGA_EMAIL or MEGA_PASSWORD is not set.")

    print("☁️ Connecting to Mega...", flush=True)

    mega = Mega()
    client = mega.login(MEGA_EMAIL, MEGA_PASSWORD)

    folder_name = "Latest_YouTube_Video"

    print("🧹 Removing previous output folder if present...", flush=True)
    try:
        old_folder = client.find(folder_name)
        if old_folder:
            client.destroy(old_folder[0])
            print("🗑️ Previous folder deleted.", flush=True)
    except Exception as exc:
        print(f"ℹ️ Old folder cleanup skipped: {exc}", flush=True)

    print("📁 Creating new Mega folder...", flush=True)
    folder = client.create_folder(folder_name)
    folder_id = folder[folder_name]

    print("🚀 Uploading video, thumbnail and SEO file...", flush=True)

    client.upload(str(OUTPUT_DIR / "video.mp4"), folder_id)
    client.upload(str(OUTPUT_DIR / "thumbnail.png"), folder_id)
    client.upload(str(OUTPUT_DIR / "seo_and_description.txt"), folder_id)

    print("✅ Mega upload complete.", flush=True)


# ============================================================
# 9. MAIN
# ============================================================

def main():
    data = generate_script()

    asyncio.run(generate_assets(data))
    create_seo_file(data)
    build_video(data)
    upload_to_mega()

    print(
        f"🎉 WORKFLOW COMPLETE — "
        f"{len(data['script_sections'])} scenes generated.",
        flush=True,
    )


if __name__ == "__main__":
    main()
