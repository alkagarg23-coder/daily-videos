import os
import re
import time
import requests
import subprocess
import gc
import io
import random
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
HF_API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"

OUTPUT_DIR = "output"
AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio")
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

try:
    with open("counter.txt", "r") as f:
        CURRENT_COUNT = int(f.read().strip()) + 1
except:
    CURRENT_COUNT = 1
with open("counter.txt", "w") as f:
    f.write(str(CURRENT_COUNT))

# ============================================================
# MULTI-DIMENSIONAL RANDOM ENGINE
# ============================================================

ART_STYLES = [
    "clean modern corporate flat vector art, minimalist infographic style",
    "high quality 3D Pixar-style render, vibrant studio lighting, highly detailed animated movie style",
    "dramatic cinematic comic book illustration, heavy shadows, vivid neon accents",
    "elegant watercolor painting, soft pastel colors, atmospheric documentary style",
    "futuristic cyberpunk aesthetic, glowing holographic financial charts, dark moody background",
    "minimalist continuous line art drawing, elegant luxury corporate aesthetic",
    "vintage 1920s newspaper editorial sketch, cross-hatch illustration style",
    "isometric 3D low poly architectural illustration, soft studio clay render"
]

SUBTITLE_COLORS = [
    "&H0000FFFF",  # Bold Yellow
    "&H00FFFF00",  # Cyan
    "&H0000FF00",  # Neon Green
    "&H00FFFFFF"   # Clean White
]

CURRENT_VIDEO_STYLE = random.choice(ART_STYLES)
CURRENT_SUB_COLOR = random.choice(SUBTITLE_COLORS)

print(f"🎨 EPISODE #{CURRENT_COUNT} SELECTED VISUAL STYLE: {CURRENT_VIDEO_STYLE}")
print(f"🔤 SELECTED SUBTITLE COLOR: {CURRENT_SUB_COLOR}")

# ============================================================
# LOCAL NEURAL TTS (PIPER)
# ============================================================

def generate_local_audio(text, output_path):
    process = subprocess.run(
        [PIPER_EXEC, "--model", PIPER_MODEL_PATH, "--output_file", output_path],
        input=text.encode('utf-8'),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    if process.returncode != 0:
        safe_text = re.sub(r'[^a-zA-Z0-9\s\.,]', '', text)
        subprocess.run(
            [PIPER_EXEC, "--model", PIPER_MODEL_PATH, "--output_file", output_path],
            input=safe_text.encode('utf-8'),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

# ============================================================
# CLOUD IMAGE ENGINE
# ============================================================

def create_fallback_image(output_path):
    img = Image.new('RGB', (1280, 720), color=(15, 20, 30))
    img.save(output_path)

def generate_cloud_image(prompt_text, output_path):
    if not HF_TOKEN:
        create_fallback_image(output_path)
        return

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    clean_prompt = f"{CURRENT_VIDEO_STYLE}, visually showing {prompt_text}, no text, masterpiece, 8k resolution"
    payload = {"inputs": clean_prompt, "parameters": {"guidance_scale": 7.5}}
    
    for attempt in range(4): 
        try:
            response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=45)
            if response.status_code == 200:
                image = Image.open(io.BytesIO(response.content))
                widescreen = image.resize((1280, 720), Image.Resampling.LANCZOS)
                widescreen.save(output_path, format="PNG")
                time.sleep(3) 
                return
        except Exception:
            pass
        time.sleep(8)
            
    print(f"⚠️ Image generation fallback used for: {output_path}")
    create_fallback_image(output_path)

# ============================================================
# LLM & TIMING FUNCTIONS
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
# STEP 1: SCRIPT & ASSETS GENERATION
# ============================================================

print(f"🚀 GENERATING MASTERCLASS EPISODE #{CURRENT_COUNT}")

TOPICS = [
    "escaping the rat race and building wealth",
    "the psychology of money and financial freedom",
    "why saving money keeps you broke and how investing changes everything",
    "the brutal truth about debt and economic cycles"
]
chosen_topic = random.choice(TOPICS)

# 30-minute script prompt (Forcing the LLM to write more)
script_prompt = (
    f"Write a very long, highly detailed, 30-minute documentary script about {chosen_topic}. "
    f"It MUST be at least 4000 words. Divide it into 15 chapters. Each chapter MUST have at least 3 deep, "
    f"punchy paragraphs explaining concepts in detail with examples. "
    f"Do NOT use bullet points or lists. Make it sound like a premium finance documentary."
)
raw_script = request_llm(script_prompt)

raw_script = raw_script.replace('"', '').replace('\n', ' ')
sentences = re.split(r'(?<=[.!?]) +', raw_script)
sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
TOTAL_SCENES = len(sentences)

print(f"🎬 Total Perfect-Sync Timeline Cuts: {TOTAL_SCENES}")

for idx, sentence in enumerate(sentences):
    scene_id = str(idx).zfill(4)
    audio_path = os.path.join(AUDIO_DIR, f"scene_{scene_id}.wav")
    image_path = os.path.join(IMAGE_DIR, f"scene_{scene_id}.png")
    
    generate_local_audio(sentence, audio_path)
    generate_cloud_image(sentence[:90], image_path)

# ============================================================
# STEP 2: ALTERNATING ZOOM & TIMELINE COMPOSITION
# ============================================================

print("\n🎞️ COMPOSING TIMELINE WITH ALTERNATING KEN BURNS MOTION...")
clips = []
srt_content = ""
current_time = 0.0
subtitle_index = 1

for idx in range(TOTAL_SCENES):
    scene_id = str(idx).zfill(4)
    audio_path = os.path.join(AUDIO_DIR, f"scene_{scene_id}.wav")
    image_path = os.path.join(IMAGE_DIR, f"scene_{scene_id}.png")

    if os.path.exists(audio_path) and os.path.exists(image_path):
        audio_clip = AudioFileClip(audio_path)
        dur = audio_clip.duration
        
        # Word-by-word (Kinetic) Subtitles Logic
        words = sentences[idx].split()
        num_words = len(words)
        
        if num_words > 0:
            word_duration = dur / num_words
            word_time = current_time
            
            for word in words:
                start_str = format_srt_time(word_time)
                end_str = format_srt_time(word_time + word_duration)
                srt_content += f"{subtitle_index}\n{start_str} --> {end_str}\n{word}\n\n"
                subtitle_index += 1
                word_time += word_duration

        current_time += dur
        
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
        threads=2, 
        preset="ultrafast",
        logger="bar"
    )

try:
    final_video.close()
    for c in clips:
        c.close()
except:
    pass
gc.collect()

# ============================================================
# STEP 3: FFMPEG SUBTITLE BURN-IN WITH ANIMATION
# ============================================================

print("\n🔥 BURNING KINETIC SUBTITLES...")
final_video_path = os.path.join(OUTPUT_DIR, f"final_video_{CURRENT_COUNT}.mp4")

# Added simple 'pop' animation using ASS tags for subtitles
ffmpeg_cmd = [
    "ffmpeg", "-y",
    "-i", base_video_path,
    "-vf", f"subtitles={srt_path}:force_style='FontName=DejaVu Sans,FontSize=48,PrimaryColour={CURRENT_SUB_COLOR},OutlineColour=&H00000000,BorderStyle=1,Outline=3,Shadow=2,Alignment=2,MarginV=35'",
    "-c:a", "copy",
    final_video_path
]

subprocess.run(ffmpeg_cmd, check=True)
print(f"\n✅ EPISODE #{CURRENT_COUNT} CREATED SUCCESSFULLY!")
