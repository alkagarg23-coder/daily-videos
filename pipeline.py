import os
import re
import time
import requests
import subprocess
import gc
import io
import random
from PIL import Image
from kokoro_onnx import Kokoro
import soundfile as sf

# ============================================================
# CONFIGURATION & CONSTANTS
# ============================================================

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "llama3.2"
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"

OUTPUT_DIR = "output"
AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio")
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
SCENES_DIR = os.path.join(OUTPUT_DIR, "scenes")

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(SCENES_DIR, exist_ok=True)

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
    "futuristic cyberpunk aesthetic, glowing holographic financial charts",
    "isometric 3D low poly architectural illustration, soft studio clay render"
]

SUBTITLE_COLORS = ["&H0000FFFF", "&H00FFFF00", "&H0000FF00", "&H00FFFFFF"]

CURRENT_VIDEO_STYLE = random.choice(ART_STYLES)
CURRENT_SUB_COLOR = random.choice(SUBTITLE_COLORS)

print(f"🎨 EPISODE #{CURRENT_COUNT} STYLE: {CURRENT_VIDEO_STYLE}")
print(f"🔤 SUBTITLE COLOR: {CURRENT_SUB_COLOR}")

# ============================================================
# KOKORO TTS (LOADED ONCE)
# ============================================================
print("🧠 Loading Kokoro ONNX Engine (am_michael)...")
kokoro_tts = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")

def generate_local_audio(text, output_path):
    try:
        samples, sample_rate = kokoro_tts.create(text, voice="am_michael", speed=1.0, lang="en-us")
        sf.write(output_path, samples, sample_rate)
        return len(samples) / sample_rate
    except Exception as e:
        safe_text = re.sub(r'[^a-zA-Z0-9\s\.,]', '', text)
        samples, sample_rate = kokoro_tts.create(safe_text, voice="am_michael", speed=1.0, lang="en-us")
        sf.write(output_path, samples, sample_rate)
        return len(samples) / sample_rate

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
            
    create_fallback_image(output_path)

# ============================================================
# LLM & TIMING HELPERS
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
# STEP 1: SCRIPT, ASSETS & FFMPEG CHUNKS (ZERO-RAM ENGINE)
# ============================================================

print(f"🚀 GENERATING MASTERCLASS EPISODE #{CURRENT_COUNT}")

TOPICS = [
    "escaping the rat race and building wealth",
    "the psychology of money and financial freedom",
    "why saving money keeps you broke and how investing changes everything"
]
chosen_topic = random.choice(TOPICS)

full_script = ""
total_chapters = 8

print("📝 Writing Script locally with Llama 3.2...")

for chapter_num in range(1, total_chapters + 1):
    print(f"✍️ Gen Chapter {chapter_num}/{total_chapters}...")
    chapter_prompt = (
        f"You are writing a YouTube finance documentary about {chosen_topic}. "
        f"Write ONLY Chapter {chapter_num}. Make it highly detailed. "
        f"Write at least 3 long paragraphs. No bullet points or special characters."
    )
    full_script += request_llm(chapter_prompt) + " "

raw_script = full_script.replace('"', '').replace('\n', ' ')
sentences = re.split(r'(?<=[.!?]) +', raw_script)
sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
TOTAL_SCENES = len(sentences)

print(f"🎬 Total Cuts to Render: {TOTAL_SCENES}")

srt_content = ""
current_time = 0.0
subtitle_index = 1
concat_lines = []
concat_file_path = os.path.join(OUTPUT_DIR, "concat.txt")

for idx, sentence in enumerate(sentences):
    scene_id = str(idx).zfill(4)
    audio_path = os.path.join(AUDIO_DIR, f"scene_{scene_id}.wav")
    image_path = os.path.join(IMAGE_DIR, f"scene_{scene_id}.png")
    scene_video_path = os.path.join(SCENES_DIR, f"scene_{scene_id}.mp4")
    
    dur = generate_local_audio(sentence, audio_path)
    generate_cloud_image(sentence[:90], image_path)
    
    words = sentence.split()
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
    
    frames = int(dur * 15) + 5
    ffmpeg_chunk_cmd = [
        "ffmpeg", "-y", "-loop", "1",
        "-i", image_path,
        "-i", audio_path,
        "-vf", f"zoompan=z='zoom+0.0008':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s=1280x720,framerate=15",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-shortest",
        scene_video_path
    ]
    
    subprocess.run(ffmpeg_chunk_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if os.path.exists(scene_video_path):
        concat_lines.append(f"file 'scenes/scene_{scene_id}.mp4'")
        print(f"✅ Rendered Cut {idx+1}/{TOTAL_SCENES}")

# ============================================================
# STEP 2: FAST CONCATENATION & SUBTITLES (FFMPEG ONLY)
# ============================================================

print("\n🎞️ JOINING CLIPS (RAM-SAFE CONCATENATION)...")
with open(concat_file_path, "w") as f:
    f.write("\n".join(concat_lines))

srt_path = os.path.join(OUTPUT_DIR, "subtitles.srt")
with open(srt_path, "w", encoding="utf-8") as f:
    f.write(srt_content)

raw_video_path = os.path.join(OUTPUT_DIR, f"raw_video_{CURRENT_COUNT}.mp4")

ffmpeg_concat_cmd = [
    "ffmpeg", "-y", "-f", "concat", "-safe", "0",
    "-i", concat_file_path,
    "-c", "copy",
    raw_video_path
]
subprocess.run(ffmpeg_concat_cmd, check=True)

print("\n🔥 BURNING KINETIC SUBTITLES (FINAL PASS)...")
final_video_path = os.path.join(OUTPUT_DIR, f"final_video_{CURRENT_COUNT}.mp4")

ffmpeg_sub_cmd = [
    "ffmpeg", "-y",
    "-i", raw_video_path,
    "-vf", f"subtitles={srt_path}:force_style='FontName=DejaVu Sans,FontSize=48,PrimaryColour={CURRENT_SUB_COLOR},OutlineColour=&H00000000,BorderStyle=1,Outline=3,Shadow=2,Alignment=2,MarginV=35'",
    "-c:a", "copy",
    final_video_path
]
subprocess.run(ffmpeg_sub_cmd, check=True)

print(f"\n✅ FULL 30-MIN MASTERCLASS #{CURRENT_COUNT} CREATED SAFELY!")
