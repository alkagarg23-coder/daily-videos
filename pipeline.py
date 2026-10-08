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

STYLE = random.choice(["ultra-realistic 8k cinematic photography", "high quality 3D render Pixar style", "clean modern corporate vector art"])
print(f"🚀 STARTING EPISODE {EP} | STYLE: {STYLE}", flush=True)

print("🧠 Loading Kokoro ONNX...", flush=True)
tts = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")

# --- 2. CORE FUNCTIONS ---
def llm(prompt):
    for _ in range(3):
        try:
            res = requests.post("http://127.0.0.1:11434/api/generate", json={"model": "gemma2:2b", "prompt": prompt, "stream": False}, timeout=180)
            return res.json()["response"].strip()
        except: time.sleep(5)
    return "Building true wealth takes time, discipline, and strategic investing."

def clean_txt(text):
    c = re.sub(r'#|\*|_|`|~|>|\[.*?\]|\(.*?\)|http\S+', '', text).replace('"', '').replace('\n', ' ').replace('-', ' ')
    return re.sub(r'\s+', ' ', c).strip()

def get_image(prompt, path):
    if not HF_TOKEN: return fallback_img(path)
    for _ in range(4):
        try:
            res = requests.post("https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0", 
                                headers={"Authorization": f"Bearer {HF_TOKEN}"}, 
                                json={"inputs": f"{STYLE}, visual representation of: {prompt}, highly detailed, no text", "parameters": {"guidance_scale": 7.5}}, timeout=60)
            if res.status_code == 200:
                img = Image.open(BytesIO(res.content)).convert("RGB")
                r = 16 / 9
                if img.width / img.height > r: img = img.crop(((img.width - img.height*r)//2, 0, (img.width + img.height*r)//2, img.height))
                else: img = img.crop((0, (img.height - img.width/r)//2, img.width, (img.height + img.width/r)//2))
                img.resize((1280, 720), Image.Resampling.LANCZOS).save(path, format="JPEG", quality=95)
                time.sleep(2)
                return
        except: time.sleep(5)
    fallback_img(path)

def fallback_img(path):
    Image.new('RGB', (1280, 720), (20, 25, 35)).save(path, format="JPEG")

def format_ass(seconds):
    h, m, s = int(seconds // 3600), int((seconds % 3600) // 60), int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}.{min(99, int((seconds - int(seconds)) * 100)):02d}"

# --- 3. MAIN PIPELINE ---
topic = random.choice(["the dark reality of debt and escaping the rat race", "psychology of money and building wealth", "why saving keeps you poor and investing makes you rich"])
print(f"📝 Topic: {topic}", flush=True)

script = ""
for i in range(1, 9):
    print(f"✍️ Writing Chapter {i}/8...", flush=True)
    script += llm(f"Write Chapter {i}/8 of a YouTube finance documentary about '{topic}'. Write 3 long paragraphs. NO hashtags, asterisks, bullet points or formatting. Plain English.") + " "

sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', script) if len(s.strip()) > 15]
total = len(sentences)
print(f"🎬 Total Scenes: {total}", flush=True)

curr_time, ass_events, chunks = 0.0, [], []
thumb_img = None

for idx, sen in enumerate(sentences):
    sid = f"{idx:04d}"
    wav, jpg, mp4 = f"output/audio/{sid}.wav", f"output/images/{sid}.jpg", f"output/scenes/{sid}.mp4"
    
    # Audio
    safe_sen = clean_txt(sen) or "Moving forward."
    try: samples, sr = tts.create(safe_sen, voice="am_michael", speed=1.0)
    except: samples, sr = tts.create(re.sub(r'[^a-zA-Z0-9\s\.,]', '', safe_sen), voice="am_michael", speed=1.0)
    sf.write(wav, samples, sr)
    dur = len(samples) / sr
    if dur < 0.5: continue

    # Image
    get_image(safe_sen[:80], jpg)
    if not thumb_img and os.path.exists(jpg): thumb_img = jpg

    # Subtitles (Character-Weighted Sync + Kinetic Pop)
    words = safe_sen.split()
    tpc = dur / max(sum(len(w) for w in words), 1)
    w_start = curr_time
    for w in words:
        w_end = w_start + (len(w) * tpc)
        clean_w = w.replace('"', '').replace("'", "\\'")
        ass_events.append(f"Dialogue: 0,{format_ass(w_start)},{format_ass(w_end)},Default,,0,0,0,,{{\\fscx70\\fscy70\\t(0,150,\\fscx100\\fscy100)}}{clean_w}")
        w_start = w_end
    curr_time += dur

    # FFmpeg Render (Zero RAM, No Black Screen)
    subprocess.run(["ffmpeg", "-y", "-loop", "1", "-framerate", "15", "-i", jpg, "-i", wav, 
                    "-vf", f"format=yuv420p,scale=8000:-1,zoompan=z='min(zoom+0.0005,1.15)':d={int(dur*15)+5}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1280x720,fps=15", 
                    "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-b:a", "128k", "-t", str(dur), mp4], 
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    chunks.append(f"file 'scenes/{sid}.mp4'")
    if (idx + 1) % 10 == 0 or (idx + 1) == total: print(f"✅ Rendered {idx+1}/{total}", flush=True)

# --- 4. ASSEMBLY, SEO & THUMBNAIL ---
print("🎞️ Assembling Final Video...", flush=True)
with open("output/concat.txt", "w") as f: f.write("\n".join(chunks))

ass_head = "[Script Info]\nScriptType: v4.00+\nPlayResX: 1280\nPlayResY: 720\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Default,Arial,60,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,2,2,10,10,60,1\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
with open("output/subs.ass", "w", encoding="utf-8") as f: f.write(ass_head + "\n".join(ass_events))

subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "output/concat.txt", "-c", "copy", f"output/raw_{EP}.mp4"], check=True)
subprocess.run(["ffmpeg", "-y", "-i", f"output/raw_{EP}.mp4", "-vf", "ass='output/subs.ass'", "-c:a", "copy", f"output/final_video_{EP}.mp4"], check=True)

print("📈 Generating SEO...", flush=True)
seo_prompt = f"Generate YouTube SEO for: '{topic}'. EXACT format:\nTITLE: [Title]\nDESCRIPTION: [~500 chars with emojis 🚀💰. Call to subscribe.]\nTAGS: [comma separated tags]"
with open(f"output/seo_metadata_{EP}.txt", "w", encoding="utf-8") as f: f.write(llm(seo_prompt))

if thumb_img:
    try:
        img = Image.alpha_composite(Image.open(thumb_img).convert("RGBA"), Image.new('RGBA', (1280, 720), (0, 0, 0, 100))).convert("RGB")
        try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 100)
        except: font = ImageFont.load_default()
        ImageDraw.Draw(img).text((80, 80), f"FINANCE\nMASTERCLASS\nEP.{EP}", fill=(255, 255, 0), font=font)
        img.save(f"output/thumbnail_{EP}.jpg", format="JPEG", quality=100)
    except: pass

print(f"🎉 SUCCESS! Episode {EP} is fully ready.", flush=True)
