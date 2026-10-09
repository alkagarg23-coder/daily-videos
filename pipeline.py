import os
import sys
import re
import time
import random
import requests
import subprocess
import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont
from kokoro_onnx import Kokoro

# =====================================================================
# 1. PIPELINE INITIALIZATION & I/O MANAGEMENT
# =====================================================================
directories = ["output/audio", "output/images", "output/scenes"]
for d in directories:
    os.makedirs(d, exist_ok=True)

print("🚀 INITIATING AUTONOMOUS RENDER ENGINE (30-MINUTE TARGET)", flush=True)

# Initialize Kokoro TTS engine (Requires kokoro-v1.0.onnx and voices-v1.0.bin)
try:
    tts = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load Kokoro ONNX model. {e}")
    sys.exit(1)

# =====================================================================
# 2. LOCAL AI GENERATION ENGINES (TEXT & VISION)
# =====================================================================
def llm(prompt):
    """
    Interfaces with Ollama. Contains aggressive exponential backoff
    to prevent thermal throttling or context timeout on cloud CPUs.
    """
    for attempt in range(5):
        try:
            res = requests.post(
                "http://127.0.0.1:11434/api/generate", 
                json={"model": "gemma2:2b", "prompt": prompt, "stream": False}, 
                timeout=300
            )
            return res.json()["response"].strip()
        except Exception as e:
            print(f"⚠️ LLM Timeout (Attempt {attempt+1}/5). Cooling down CPU... {e}")
            time.sleep(15)
    return "Building wealth is a systematic process of consistent discipline."

def clean_txt(text):
    """Sanitizes LLM output for safe TTS processing and subtitle rendering."""
    c = re.sub(r'#|\*|_|`|~|>|\[.*?\]|\(.*?\)|http\S+', '', text)
    c = c.replace('"', '').replace('\n', ' ').replace('-', ' ')
    return re.sub(r'\s+', ' ', c).strip()

def get_local_image(prompt, path):
    """
    Generates imagery using stable-diffusion.cpp. 
    Optimized to 10 steps for CPU speed. Includes a PIL fallback to 
    prevent missing-file errors that cause black screens in FFmpeg.
    """
    safe_prompt = re.sub(r'[^a-zA-Z0-9\s]', '', prompt)[:120]
    sd_cmd = [
        "./sd", "-m", "model.safetensors",
        "-p", f"masterpiece, highly detailed, black and white minimalist stickman drawing on a clean whiteboard. {safe_prompt}. authentic youtube explainer style.",
        "-n", "text, bad anatomy, blurry, messy lines, realistic, colorful", 
        "--steps", "10", "--cfg-scale", "5.0", 
        "-W", "768", "-H", "512", "-o", "temp_out.png"
    ]
    try:
        # 180s timeout prevents the 6-hour CI/CD action from failing due to hangs
        subprocess.run(sd_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)
        img = Image.open("temp_out.png").convert("RGB")
        img.resize((1280, 720), Image.Resampling.LANCZOS).save(path, format="JPEG", quality=95)
    except Exception as e:
        print(f"⚠️ Image generation fallback triggered for prompt '{safe_prompt}': {e}")
        # Generate an off-white fallback canvas to ensure the video keeps rendering
        Image.new('RGB', (1280, 720), (245, 245, 245)).save(path, format="JPEG")

def format_ass_time(seconds):
    """Formats float seconds into ASS timestamp format: H:MM:SS.cs"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = min(99, int((seconds - int(seconds)) * 100))
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

# =====================================================================
# 3. LONG-FORM SCRIPT GENERATION (SCALED FOR 30 MINUTES)
# =====================================================================
topic = random.choice([
    "The dark psychology of debt and escaping the rat race", 
    "Why saving money keeps you poor", 
    "The hidden secrets of the top 1% wealth builders"
])

print(f"📝 Generating script for topic: {topic}")
script = ""

# 30 iterations * ~150 words = ~4,500 words. (~30 minutes of speech at 150wpm)
for i in range(1, 31):
    if i == 1:
        prompt = f"Write the INTRO chapter for a 30-minute YouTube finance documentary on '{topic}'. Start with a shocking hook ('99% of people are lied to...'). Make it highly suspenseful. 3 long paragraphs. Plain English."
    else:
        prompt = f"Write Chapter {i}/30 of a finance documentary on '{topic}'. Provide 3 highly detailed, insightful paragraphs. Plain English. Do not include titles or headings."
    
    chunk = llm(prompt)
    script += chunk + " "
    print(f"   Generated chapter {i}/30")

# Split script into larger semantic chunks (approx 2 sentences) to reduce SD.cpp calls.
raw_sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', script) if len(s.strip()) > 15]
chunks_text = []
for i in range(0, len(raw_sentences), 2):
    chunks_text.append(" ".join(raw_sentences[i:i+2]))
    
total_chunks = len(chunks_text)
print(f"Total visual scenes extracted: {total_chunks}")

# =====================================================================
# 4. TIMELINE ASSEMBLY, TTS SYNC, AND KARAOKE MATHEMATICS
# =====================================================================
curr_time = 0.0
ass_events = []
video_segments = []
thumb_img = None

print("🎬 Compiling Audio, Video, and Subtitle synchronizations...")
for idx, text_chunk in enumerate(chunks_text):
    sid = f"{idx:04d}"
    wav = f"output/audio/{sid}.wav"
    jpg = f"output/images/{sid}.jpg"
    mp4 = f"output/scenes/{sid}.mp4"
    
    safe_text = clean_txt(text_chunk)
    if not safe_text: continue
        
    # Generate Audio via Kokoro-ONNX (24000 Hz)
    try: 
        samples, sr = tts.create(safe_text, voice="am_michael", speed=1.0)
    except: 
        # Ultimate fallback stripping all non-alphanumeric characters
        stripped = re.sub(r'[^a-zA-Z0-9\s\.,]', '', safe_text)
        samples, sr = tts.create(stripped, voice="am_michael", speed=1.0)
        
    sf.write(wav, samples, sr)
    dur = len(samples) / sr
    
    # Skip microscopic audio artifacts that break FFmpeg concat demuxing
    if dur < 1.0: continue

    # Generate visual asset (Only pass the first 80 chars to SD to prevent token overflow)
    get_local_image(safe_text[:80], jpg)
    if not thumb_img and os.path.exists(jpg) and idx == 3: 
        thumb_img = jpg

    # --- KARAOKE SUBTITLE GENERATION (THE \kf ALGORITHM) ---
    words = safe_text.split()
    # Weight characters to estimate spoken duration. Punctuation takes longer to speak.
    weighted_chars = [max(len(w) + (6 if w.endswith(',') or w.endswith('.') or w.endswith('?') else 0), 1) for w in words]
    total_weight = max(sum(weighted_chars), 1)
    time_per_weight = dur / total_weight
    
    ass_text_payload = ""
    for i, w in enumerate(words):
        w_dur = weighted_chars[i] * time_per_weight
        # Convert duration to centiseconds for the \kf tag
        cs_dur = int(w_dur * 100)
        # Escape ASS special characters
        clean_w = w.replace('"', '').replace("'", "\\'")
        ass_text_payload += f"{{\\kf{cs_dur}}}{clean_w} "
    
    # Create the single synchronized line event
    ass_events.append(
        f"Dialogue: 0,{format_ass_time(curr_time)},{format_ass_time(curr_time + dur)},"
        f"Default,,0,0,0,,{ass_text_payload.strip()}"
    )
    
    # --- FFMPEG BLACK-SCREEN PREVENTION & ENCODING ---
    # format=yuv420p MUST be at the end of the filterchain to prevent libx264 color space crashes.
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-loop", "1", "-framerate", "15", 
        "-i", jpg, "-i", wav, 
        "-vf", f"scale=8000:-1,zoompan=z='min(zoom+0.0005,1.10)':d={int(dur*15)+15}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1280x720,format=yuv420p", 
        "-c:v", "libx264", "-preset", "ultrafast", "-profile:v", "main",
        "-c:a", "aac", "-b:a", "128k", "-ar", "24000",
        "-shortest", "-fps_mode", "cfr", "-t", str(dur), mp4
    ]
    
    subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    video_segments.append(f"file 'scenes/{sid}.mp4'")
    
    curr_time += dur
    if idx % 5 == 0:
        print(f"   Processed {idx}/{total_chunks} scenes. Current runtime: {curr_time/60:.2f} mins")

# =====================================================================
# 5. FINAL MASTER RENDER & METADATA
# =====================================================================
print("📦 Assembling Final 30-Minute Masterpiece...")
with open("output/concat.txt", "w") as f: 
    f.write("\n".join(video_segments))

# --- ASS HEADER SPECIFICATION ---
# SecondaryColour = &H00FFFFFF (White Base)
# PrimaryColour = &H0000FFFF (Yellow Sweep)
# The \kf tag pre-fills with White and sweeps to Yellow over the word's duration.
ass_head = """[Script Info]
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 720
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,55,&H0000FFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,2,2,10,10,60,1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

with open("output/subs.ass", "w", encoding="utf-8") as f: 
    f.write(ass_head + "\n".join(ass_events))

# Concat pass (stream copy ensures exact synchronization preservation)
subprocess.run(
    ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "output/concat.txt", "-c", "copy", "output/raw_video.mp4"], 
    check=True
)

# Subtitle burn-in pass (Applying the generated ASS file)
subprocess.run(
    ["ffmpeg", "-y", "-i", "output/raw_video.mp4", "-vf", "ass='output/subs.ass'", "-c:a", "copy", "output/final_video.mp4"], 
    check=True
)

# Generating SEO metadata
with open("output/seo_metadata.txt", "w", encoding="utf-8") as f: 
    seo_prompt = f"Generate SEO metadata for a 30-minute YouTube video on: '{topic}'. FORMAT EXACTLY:\nTITLE: [Clickbait title]\nDESCRIPTION: [500 chars with emojis 🚨🔥💰. Call to subscribe.]\nTAGS: [tags]"
    f.write(llm(seo_prompt))

# Dynamic Thumbnail Generation
if thumb_img:
    try:
        img = Image.open(thumb_img).convert("RGBA")
        grad = Image.new('RGBA', img.size, (0,0,0,0))
        d = ImageDraw.Draw(grad)
        # Apply cinematic gradient fade to the left side
        for x in range(int(img.width * 0.6)):
            d.line([(x, 0), (x, img.height)], fill=(20, 20, 20, int(255 * (1 - (x / (img.width * 0.6))))))
        
        final_thumb = Image.alpha_composite(img, grad).convert("RGB")
        try: 
            font1 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 95)
            font2 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 60)
        except: 
            font1 = font2 = ImageFont.load_default()
            
        draw = ImageDraw.Draw(final_thumb)
        draw.text((60, 280), "STOP", fill=(239, 68, 68), font=font1)
        draw.text((60, 390), "DOING THIS", fill=(250, 204, 21), font=font1)
        draw.text((60, 500), "FINANCE SECRETS", fill=(255, 255, 255), font=font2)
        final_thumb.save("output/thumbnail.jpg", format="JPEG", quality=100)
    except Exception as e:
        print(f"Thumbnail generation skipped: {e}")

print("✅ EXECUTION COMPLETE. Artifacts stored in /output.", flush=True)
