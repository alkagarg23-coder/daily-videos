import os, sys, re, time, random, requests, subprocess
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from kokoro_onnx import Kokoro
import soundfile as sf

# --- 1. SETUP & CONFIG ---
for f in ["output/audio", "output/images", "output/scenes"]: os.makedirs(f, exist_ok=True)
HF_TOKEN = os.getenv("HF_TOKEN", "")

try:
    with open("counter.txt", "r") as f: EP = int(f.read().strip()) + 1
except: EP = 1
with open("counter.txt", "w") as f: f.write(str(EP))

# ⚡ HIGH-RETENTION STICKMAN/WHITEBOARD STYLE
STYLE = "high quality minimalist stickman drawing on a whiteboard, engaging infographic style, clean lines, authentic YouTube explainer aesthetic"
print(f"🚀 STARTING EPISODE {EP} | STYLE: STICKMAN EXPLAINER", flush=True)

print("🧠 Loading Kokoro ONNX TTS...", flush=True)
tts = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")

# --- 2. FAST API & TEXT ENGINES ---
def llm(prompt):
    for attempt in range(3):
        try:
            res = requests.post("http://127.0.0.1:11434/api/generate", json={"model": "gemma2:2b", "prompt": prompt, "stream": False}, timeout=180)
            return res.json()["response"].strip()
        except: time.sleep(5)
    return "The secret to wealth is starting today. Let's dive in."

def clean_txt(text):
    c = re.sub(r'#|\*|_|`|~|>|\[.*?\]|\(.*?\)|http\S+', '', text).replace('"', '').replace('\n', ' ').replace('-', ' ')
    return re.sub(r'\s+', ' ', c).strip()

def get_image(prompt, path):
    if not HF_TOKEN: return fallback_img(path)
    
    # ⚡ USING SDXL-TURBO / FAST API FOR LIGHTNING SPEED (Under 6 Hours Guarantee)
    API_URL = "https://api-inference.huggingface.co/models/stabilityai/sdxl-turbo"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    for _ in range(4):
        try:
            # Turbo models need fewer steps, ensuring blazingly fast generation
            res = requests.post(API_URL, headers=headers, json={"inputs": f"{STYLE}. Visual context: {prompt}. No text.", "parameters": {"guidance_scale": 0.0, "num_inference_steps": 1}}, timeout=45)
            if res.status_code == 200:
                img = Image.open(BytesIO(res.content)).convert("RGB")
                r = 16 / 9
                if img.width / img.height > r: img = img.crop(((img.width - img.height*r)//2, 0, (img.width + img.height*r)//2, img.height))
                else: img = img.crop((0, (img.height - img.width/r)//2, img.width, (img.height + img.width/r)//2))
                img.resize((1280, 720), Image.Resampling.LANCZOS).save(path, format="JPEG", quality=95)
                time.sleep(1)
                return
        except: time.sleep(3)
    fallback_img(path)

def fallback_img(path):
    Image.new('RGB', (1280, 720), (245, 245, 245)).save(path, format="JPEG") # Whiteboard Background

def format_ass(seconds):
    h, m, s = int(seconds // 3600), int((seconds % 3600) // 60), int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}.{min(99, int((seconds - int(seconds)) * 100)):02d}"

# --- 3. HIGH-RETENTION SCRIPTWRITING ---
topics = [
    "The dark psychology of debt and how banks trap you forever",
    "Why saving money is a scam and the real way the top 1% builds wealth",
    "The brutal truth about escaping the middle-class rat race"
]
topic = random.choice(topics)
print(f"📝 Topic: {topic}", flush=True)

script = ""
for i in range(1, 9): # 8 Chapters for ~25-30 mins
    print(f"✍️ Writing Chapter {i}/8...", flush=True)
    if i == 1:
        # ⚡ THE HOOK: Chapter 1 forces a high-retention suspense intro
        prompt = f"Write the INTRO chapter for a YouTube finance video about '{topic}'. Start with a shocking, controversial hook (e.g. '99% of people are being lied to about money...'). Make it suspenseful so the viewer cannot click away. 3 paragraphs. Plain English. No markdown."
    else:
        prompt = f"Write Chapter {i}/8 of the YouTube finance documentary about '{topic}'. Deep, engaging storytelling. 3 long paragraphs. NO hashtags or markdown. Plain English."
    
    script += llm(prompt) + " "

sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', script) if len(s.strip()) > 15]
total = len(sentences)
print(f"🎬 Total Scenes: {total}", flush=True)

# --- 4. ASSET GENERATION & TIMELINE ---
curr_time, ass_events, chunks = 0.0, [], []
thumb_img = None

for idx, sen in enumerate(sentences):
    sid = f"{idx:04d}"
    wav, jpg, mp4 = f"output/audio/{sid}.wav", f"output/images/{sid}.jpg", f"output/scenes/{sid}.mp4"
    
    safe_sen = clean_txt(sen) or "Listen closely."
    try: samples, sr = tts.create(safe_sen, voice="am_michael", speed=1.0)
    except: samples, sr = tts.create(re.sub(r'[^a-zA-Z0-9\s\.,]', '', safe_sen), voice="am_michael", speed=1.0)
    sf.write(wav, samples, sr)
    dur = len(samples) / sr
    if dur < 0.5: continue

    get_image(safe_sen[:80], jpg)
    if not thumb_img and os.path.exists(jpg) and idx == 2: thumb_img = jpg # Pick a scene slightly into the video for better context

    # ⚡ PUNCTUATION-AWARE PERFECT SUBTITLE SYNC
    words = safe_sen.split()
    weighted_chars = [len(w) + (5 if w.endswith(',') or w.endswith('.') else 0) for w in words]
    tpc = dur / max(sum(weighted_chars), 1)
    
    w_start = curr_time
    for i, w in enumerate(words):
        w_end = w_start + (weighted_chars[i] * tpc)
        clean_w = w.replace('"', '').replace("'", "\\'")
        # Faster kinetic pop (100ms) for snappy, highly engaging subtitles
        ass_events.append(f"Dialogue: 0,{format_ass(w_start)},{format_ass(w_end)},Default,,0,0,0,,{{\\fscx70\\fscy70\\t(0,100,\\fscx100\\fscy100)}}{clean_w}")
        w_start = w_end
    curr_time += dur

    # FFmpeg Scene Assembly
    subprocess.run(["ffmpeg", "-y", "-loop", "1", "-framerate", "15", "-i", jpg, "-i", wav, 
                    "-vf", f"format=yuv420p,scale=8000:-1,zoompan=z='min(zoom+0.0008,1.2)':d={int(dur*15)+5}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1280x720,fps=15", 
                    "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-b:a", "128k", "-t", str(dur), mp4], 
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    chunks.append(f"file 'scenes/{sid}.mp4'")
    if (idx + 1) % 10 == 0 or (idx + 1) == total: print(f"✅ Rendered {idx+1}/{total}", flush=True)

# --- 5. MASTER RENDER, SEO & AUTHENTIC THUMBNAIL ---
print("🎞️ Assembling Final Video...", flush=True)
with open("output/concat.txt", "w") as f: f.write("\n".join(chunks))

# Black Outline, Pure White Text for maximum readability
ass_head = "[Script Info]\nScriptType: v4.00+\nPlayResX: 1280\nPlayResY: 720\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Default,Arial,65,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,5,3,2,10,10,50,1\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
with open("output/subs.ass", "w", encoding="utf-8") as f: f.write(ass_head + "\n".join(ass_events))

subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "output/concat.txt", "-c", "copy", f"output/raw_{EP}.mp4"], check=True)
subprocess.run(["ffmpeg", "-y", "-i", f"output/raw_{EP}.mp4", "-vf", "ass='output/subs.ass'", "-c:a", "copy", f"output/final_video_{EP}.mp4"], check=True)

print("📈 Generating Viral SEO...", flush=True)
seo_prompt = f"Generate highly engaging YouTube SEO for: '{topic}'. FORMAT EXACTLY:\nTITLE: [Clickbait but true title]\nDESCRIPTION: [Write exactly 500 chars with emojis like 🚨🔥💰. Add a suspenseful hook in the first line. Call to subscribe.]\nTAGS: [comma separated tags]"
with open(f"output/seo_metadata_{EP}.txt", "w", encoding="utf-8") as f: f.write(llm(seo_prompt))

print("🎨 Creating High-Retention Thumbnail...", flush=True)
if thumb_img:
    try:
        # Authentic Thumbnail: Image on right, Dark Fade on left, Bold YouTube text
        img = Image.open(thumb_img).convert("RGBA")
        grad = Image.new('RGBA', img.size, (0,0,0,0))
        d = ImageDraw.Draw(grad)
        for x in range(int(img.width * 0.6)):
            d.line([(x, 0), (x, img.height)], fill=(20, 20, 20, int(255 * (1 - (x / (img.width * 0.6))))))
        
        final_thumb = Image.alpha_composite(img, grad).convert("RGB")
        try: 
            font1 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 95)
            font2 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 75)
        except: 
            font1 = font2 = ImageFont.load_default()
            
        draw = ImageDraw.Draw(final_thumb)
        draw.text((50, 280), "STOP", fill=(239, 68, 68), font=font1) # Red text
        draw.text((50, 390), "DOING THIS", fill=(250, 204, 21), font=font1) # Yellow text
        draw.text((50, 500), f"EP.{EP} MASTERCLASS", fill=(255, 255, 255), font=font2)
        
        final_thumb.save(f"output/thumbnail_{EP}.jpg", format="JPEG", quality=100)
    except Exception as e: print(f"Thumbnail error: {e}")

print(f"🎉 SUCCESS! Cinematic Episode {EP} is fully ready.", flush=True)
