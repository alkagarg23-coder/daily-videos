import os
import re
import time
import requests
import subprocess
import gc
import io
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips
from PIL import Image

# ============================================================
# CONFIGURATION & CONSTANTS
# ============================================================

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:3b"
PIPER_EXEC = "./piper/piper"
PIPER_MODEL_PATH = "models/voice.onnx"
HF_TOKEN = os.getenv("HF_TOKEN", "")

# SDXL API Endpoint
HF_API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"

OUTPUT_DIR = "output"
AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio")
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

# Counter Tracker
try:
    with open("counter.txt", "r") as f:
        CURRENT_COUNT = int(f.read().strip()) + 1
except:
    CURRENT_COUNT = 1
with open("counter.txt", "w") as f:
    f.write(str(CURRENT_COUNT))

# ============================================================
# ZERO-ERROR AUDIO ENGINE (PIPER TTS)
# ============================================================

def generate_local_audio(text, output_path):
    # subprocess.run with input bytes prevents crash if text contains quotes/special characters
    process = subprocess.run(
        [PIPER_EXEC, "--model", PIPER_MODEL_PATH, "--output_file", output_path],
        input=text.encode('utf-8'),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    if process.returncode != 0:
        print(f"⚠️ Piper Warning on text: {text[:30]}... Retrying with safe characters.")
        safe_text = re.sub(r'[^a-zA-Z0-9\s\.,]', '', text)
        subprocess.run(
            [PIPER_EXEC, "--model", PIPER_MODEL_PATH, "--output_file", output_path],
            input=safe_text.encode('utf-8'),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

# ============================================================
# CLOUD IMAGE ENGINE (WITH SMART FALLBACK)
# ============================================================

def create_fallback_image(output_path):
    """Creates a solid dark cinematic background if API totally fails to prevent pipeline crash."""
    img = Image.new('RGB', (1280, 720), color=(15, 20, 30))
    img.save(output_path)

def generate_cloud_image(prompt_text, output_path):
    if not HF_TOKEN:
        create_fallback_image(output_path)
        return

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    clean_prompt = (
        f"clean modern vector art, 2D minimalist stickman illustration, {prompt_text}, "
        f"bold clean black outlines, crisp solid light backdrop, corporate doodle infographic, 8k resolution"
    )
    payload = {"inputs": clean_prompt, "parameters": {"guidance_scale": 7.5}}
    
    for attempt in range(4): # Smart Retries
        try:
            response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=40)
            if response.status_code == 200:
                image = Image.open(io.BytesIO(response.content))
                widescreen = image.resize((1280, 720), Image.Resampling.LANCZOS)
                widescreen.save(output_path, format="PNG")
                time.sleep(3) # Safe delay to prevent ban
                return
        except Exception:
            pass
        time.sleep(8)
            
    print(f"⚠️ Image API limit reached for scene. Using fallback visual.")
    create_fallback_image(output_path)

# ============================================================
# TEXT GENERATION & TIMING HELPERS
# ============================================================

def request_llm(prompt_text):
    response = requests.post(
        OLLAMA_URL, 
        json={"model": OLLAMA_MODEL, "prompt": prompt_text, "stream": False}, 
        timeout=None
    )
    return response.json()["response"].strip()

def format_srt_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

# ============================================================
# STEP 1: SCRIPT & ASSET CREATION
# ============================================================

print(f"🚀 GENERATING 30-MIN MASTERCLASS #{CURRENT_COUNT}")

script_prompt = f"Write an engaging 10-chapter YouTube script about escaping the middle-class trap. Use punchy, practical paragraphs. No bullet points or special characters."
raw_script = request_llm(script_prompt)

# Clean and split into distinct timeline cuts
raw_script = raw_script.replace('"', '').replace('\n', ' ')
sentences = re.split(r'(?<=[.!?]) +', raw_script)
sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
TOTAL_SCENES = len(sentences)

print(f"🎬 Total Perfect-Sync Cuts: {TOTAL_SCENES}")

for idx, sentence in enumerate(sentences):
    scene_id = str(idx).zfill(4)
    audio_path = os.path.join(AUDIO_DIR, f"scene_{scene_id}.wav")
    image_path = os.path.join(IMAGE_DIR, f"scene_{scene_id}.png")
    
    print(f"⚙️ Rendering Asset {idx+1}/{TOTAL_SCENES}...")
    generate_local_audio(sentence, audio_path)
    generate_cloud_image(sentence[:90], image_path)

# ============================================================
# STEP 2: ALTERNATING ZOOM & TIMELINE ASSEMBLY (RAM SAFE)
# ============================================================

print("\n🎞️ BUILDING TIMELINE & KEN BURNS MOTION...")
clips = []
srt_content = ""
current_time = 0.0

for idx in range(TOTAL_SCENES):
    scene_id = str(idx).zfill(4)
    audio_path = os.path.join(AUDIO_DIR, f"scene_{scene_id}.wav")
    image_path = os.path.join(IMAGE_DIR, f"scene_{scene_id}.png")

    if os.path.exists(audio_path) and os.path.exists(image_path):
        audio_clip = AudioFileClip(audio_path)
        dur = audio_clip.duration
        
        # Subtitle Generation
        start_str = format_srt_time(current_time)
        end_str = format_srt_time(current_time + dur)
        display_text = sentences[idx]
        if len(display_text) > 42:
            mid = len(display_text) // 2
            split_at = display_text.find(" ", mid)
            if split_at != -1:
                display_text = display_text[:split_at] + "\n" + display_text[split_at+1:]
        
        srt_content += f"{idx+1}\n{start_str} --> {end_str}\n{display_text}\n\n"
        current_time += dur
        
        # Alternating Zoom: Even = Zoom-In, Odd = Zoom-Out
        base_clip = ImageClip(image_path).set_duration(dur)
        is_even = (idx % 2 == 0)
        
        def make_zoom(even_flag):
            if even_flag:
                return lambda t: 1.0 + 0.05 * (t / dur)
            return lambda t: 1.05 - 0.05 * (t / dur)

        zoomed_clip = (
            base_clip.resize(make_zoom(is_even))
            .crop(x_center=640, y_center=360, width=1280, height=720)
            .set_audio(audio_clip)
        )
        clips.append(zoomed_clip)

srt_path = os.path.join(OUTPUT_DIR, "subtitles.srt")
with open(srt_path, "w", encoding="utf-8") as f:
    f.write(srt_content)

base_video_path = os.path.join(OUTPUT_DIR, f"raw_video_{CURRENT_COUNT}.mp4")

if clips:
    final_video = concatenate_videoclips(clips, method="compose")
    final_video.write_videofile(
        base_video_path,
        fps=15,
        codec="libx264",
        audio_codec="aac",
        threads=2, # Keeps CPU memory stable on GitHub Actions
        preset="ultrafast",
        logger="bar"
    )

# RAM Cleanup before FFmpeg
try:
    final_video.close()
    for c in clips:
        c.close()
except:
    pass
gc.collect() 

# ============================================================
# STEP 3: FFMPEG HARDCODED SUBTITLES
# ============================================================

print("\n🔥 BURNING KINETIC SUBTITLES...")
final_video_path = os.path.join(OUTPUT_DIR, f"final_video_{CURRENT_COUNT}.mp4")

# Using DejaVu Sans, bold yellow text, black outline for high readability
ffmpeg_cmd = [
    "ffmpeg", "-y",
    "-i", base_video_path,
    "-vf", f"subtitles={srt_path}:force_style='FontName=DejaVu Sans,FontSize=24,PrimaryColour=&H0000FFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=1,Alignment=2,MarginV=25'",
    "-c:a", "copy",
    final_video_path
]

subprocess.run(ffmpeg_cmd, check=True)
print(f"\n✅ FULL MASTERCLASS #{CURRENT_COUNT} CREATED SUCCESSFULLY!")
