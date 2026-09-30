import os
import json
import asyncio
import requests
import edge_tts

from PIL import Image, ImageDraw, ImageFont
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


# ============================================================
# BASIC VALIDATION
# ============================================================

if not MEGA_EMAIL or not MEGA_PASSWORD:
    raise RuntimeError("MEGA_EMAIL and MEGA_PASSWORD GitHub Secrets are required.")


# ============================================================
# HELPERS
# ============================================================

def clean_json_text(raw_text):
    """Extract JSON object even if model adds markdown or explanation."""
    start = raw_text.find("{")
    end = raw_text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("Qwen did not return a valid JSON object.")

    return raw_text[start:end + 1]


def request_json(url, payload, retries=3):
    """POST JSON with retries for Ollama."""
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            response = requests.post(url, json=payload, timeout=600)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            print(f"⚠️ Ollama request failed (attempt {attempt}/{retries}): {exc}")

    raise RuntimeError(f"Ollama request failed after {retries} attempts: {last_error}")


def get_font(size):
    """Loads a readable TrueType font available on Ubuntu runner."""
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


def generate_procedural_scene(title_text, output_path, is_thumbnail=False):
    """
    Renders 1280x720 Stickman Finance illustration locally using Pillow.
    No network requests, zero timeouts, takes ~0.05s.
    """
    width, height = 1280, 720
    bg_color = (245, 247, 250) if is_thumbnail else (255, 255, 255)
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    black = (30, 30, 30)
    green = (34, 139, 34)
    blue = (0, 102, 204)
    line_w = 7

    # Ground line
    ground_y = 560
    draw.line([(60, ground_y), (width - 60, ground_y)], fill=black, width=line_w)

    # Stickman
    cx, head_y = 380, 270
    head_r = 45

    # Head & Face
    draw.ellipse([cx - head_r, head_y - head_r, cx + head_r, head_y + head_r], outline=black, width=line_w)
    draw.ellipse([cx - 16, head_y - 10, cx - 8, head_y - 2], fill=black)
    draw.ellipse([cx + 8, head_y - 10, cx + 16, head_y - 2], fill=black)
    draw.arc([cx - 18, head_y, cx + 18, head_y + 22], start=0, end=180, fill=black, width=4)

    # Body
    spine_bottom = 440
    draw.line([(cx, head_y + head_r), (cx, spine_bottom)], fill=black, width=line_w)

    # Arms
    draw.line([(cx, head_y + 70), (cx - 70, head_y + 120)], fill=black, width=line_w)
    draw.line([(cx, head_y + 70), (cx + 80, head_y + 40)], fill=black, width=line_w)

    # Legs
    draw.line([(cx, spine_bottom), (cx - 50, ground_y)], fill=black, width=line_w)
    draw.line([(cx, spine_bottom), (cx + 50, ground_y)], fill=black, width=line_w)

    # Presentation / Finance Board
    board_box = [(580, 180), (1180, 500)]
    draw.rectangle(board_box, fill=(245, 248, 252), outline=black, width=5)

    # Board Stand
    draw.line([(880, 500), (880, ground_y)], fill=black, width=line_w)
    draw.line([(830, ground_y), (930, ground_y)], fill=black, width=line_w)

    # Chart Line (Upward trend)
    chart_pts = [(620, 440), (750, 360), (870, 400), (1050, 240)]
    for i in range(len(chart_pts) - 1):
        draw.line([chart_pts[i], chart_pts[i + 1]], fill=green, width=8)

    # Arrow Head
    draw.polygon([(1050, 240), (1020, 240), (1050, 270)], fill=green)

    # Money Bag
    bx, by = 1080, 430
    draw.ellipse([bx - 40, by - 30, bx + 40, by + 50], fill=(225, 245, 225), outline=green, width=4)
    draw.polygon([(bx - 15, by - 30), (bx + 15, by - 30), (bx, by - 48)], fill=green)
    draw.line([(bx, by - 15), (bx, by + 25)], fill=green, width=4)

    # Header Card for narration / prompt summary
    header_box = [(60, 40), (width - 60, 120)]
    draw.rectangle(header_box, fill=(255, 255, 255), outline=blue if is_thumbnail else black, width=4)

    title_clean = title_text.strip().replace("\n", " ")
    if len(title_clean) > 65:
        title_clean = title_clean[:62] + "..."

    font = get_font(26 if not is_thumbnail else 30)
    draw.text((90, 62), title_clean, fill=(0, 51, 102) if is_thumbnail else black, font=font)

    img.save(output_path, format="PNG")
    print(f"   ✅ Local PNG created: {output_path}")


async def generate_audio(text, output_path, retries=5):
    """Generate Edge TTS audio with retries."""
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            print(f"   🔊 Audio attempt {attempt}/{retries}")
            communicate = edge_tts.Communicate(text, VOICE)
            await communicate.save(output_path)

            if not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
                raise RuntimeError("Audio file invalid or missing.")

            print(f"   ✅ Audio created: {output_path}")
            return

        except Exception as exc:
            last_error = exc
            print(f"   ⚠️ Audio generation failed: {exc}")
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except Exception:
                    pass

            if attempt < retries:
                await asyncio.sleep(attempt * 3)

    raise RuntimeError(f"Could not generate audio after {retries} attempts: {last_error}")


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
- Short, clear sentences
- 5 to 8 script sections

Return ONLY valid JSON.
No markdown or intro text.

Required structure:
{
  "video_title": "YouTube title under 70 characters",
  "seo_tags": "tag1, tag2, tag3, tag4",
  "description": "Engaging description with key lessons and hashtags.",
  "thumbnail_prompt": "Title text to show on the thumbnail",
  "script_sections": [
    {
      "text": "Narration for this scene.",
      "image_prompt": "Short title or scene summary (max 8 words)"
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

print("⏳ Qwen is generating the script...")

qwen_response = request_json(OLLAMA_URL, payload, retries=3)

if "response" not in qwen_response:
    raise RuntimeError("Ollama response did not contain 'response'.")

clean_json = clean_json_text(qwen_response["response"])
data = json.loads(clean_json)

sections = data["script_sections"]
SCENE_COUNT = len(sections)

print()
print("=" * 60)
print(f"🎬 TITLE: {data['video_title']}")
print(f"🎞️ SCENE COUNT: {SCENE_COUNT}")
print("=" * 60)


# ============================================================
# GENERATE THUMBNAIL
# ============================================================

print("🖼️ Generating thumbnail...")
thumbnail_path = os.path.join(OUTPUT_DIR, "thumbnail.png")
generate_procedural_scene(
    data.get("video_title", "Stickman Finance"),
    thumbnail_path,
    is_thumbnail=True
)


# ============================================================
# GENERATE AUDIO + LOCAL IMAGES
# ============================================================

async def generate_all_assets():
    for index, section in enumerate(sections):
        scene_number = str(index).zfill(3)
        audio_path = os.path.join(AUDIO_DIR, f"scene_{scene_number}.mp3")
        image_path = os.path.join(IMAGE_DIR, f"scene_{scene_number}.png")

        print(f"\n🎬 SCENE {index + 1}/{SCENE_COUNT}")
        print("-" * 50)

        await generate_audio(section["text"], audio_path)
        generate_procedural_scene(
            section.get("image_prompt", f"Point {index + 1}"),
            image_path
        )


print("=" * 60)
print("🎨 GENERATING AUDIO + IMAGES")
print("=" * 60)

asyncio.run(generate_all_assets())


# ============================================================
# SEO FILE
# ============================================================

print("\n📝 Creating SEO file...")
seo_path = os.path.join(OUTPUT_DIR, "seo_and_description.txt")
with open(seo_path, "w", encoding="utf-8") as file:
    file.write(f"TITLE:\n{data['video_title']}\n\n")
    file.write(f"DESCRIPTION:\n{data['description']}\n\n")
    file.write(f"TAGS:\n{data['seo_tags']}\n\n")
    file.write(f"SCENE_COUNT:\n{SCENE_COUNT}\n")


# ============================================================
# BUILD VIDEO
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

print("\n" + "=" * 60)
print("☁️ UPLOADING TO MEGA")
print("=" * 60)

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
