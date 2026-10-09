import os, sys, re, time, random, requests, subprocess
from PIL import Image, ImageDraw, ImageFont
from kokoro_onnx import Kokoro
import soundfile as sf

# --- 1. SETUP ---
for f in ["output/audio", "output/images", "output/scenes"]: os.makedirs(f, exist_ok=True)

print("🚀 STARTING HIGH-QUALITY LOCAL RENDER (STANDALONE VIDEO)", flush=True)
tts = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")

# --- 2. FAST LOCAL ENGINES ---
def llm(prompt):
    for _ in range(3):
        try:
            res = requests.post("http://127.0.0.1:11434/api/generate", json={"model": "gemma2:2b", "prompt": prompt, "stream": False}, timeout=180)
            return res.json()["response"].strip()
        except: time.sleep(5)
    return "Building wealth takes time. Let's start."

def clean_txt(text):
    c = re.sub(r'#|\*|_|`|~|>|\[.*?\]|\(.*?\)|http\S+', '', text).replace('"', '').replace('\n', ' ').replace('-', ' ')
    return re.sub(r'\s+', ' ', c).strip()

def get_local_image(prompt, path):
    sd_cmd = [
        "./sd", "-m", "model.safetensors",
        "-p", f"masterpiece, extremely detailed, black and white minimalist stickman drawing on a clean whiteboard. {prompt}. authentic youtube explainer style.",
        "-n", "text, bad anatomy, blurry, messy lines, realistic", 
        "--steps", "15", "--cfg-scale", "7.0", 
        "-W", "768", "-H", "512", "-o", "temp_out.png"
    ]
    try:
        subprocess.run(sd_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=240)
        img = Image.open("temp_out.png").convert("RGB")
        img.resize((1280, 720), Image.Resampling.LANCZOS).save(path, format="JPEG", quality=100)
    except Exception as e:
        print(f"Image generation fallback: {e}")
        Image.new('RGB', (1280, 720), (245, 245, 245)).save(path, format="JPEG")

def format_ass(seconds):
    h, m, s = int(seconds // 3600), int((seconds % 3600) // 60), int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}.{min(99, int((seconds - int(seconds)) * 100)):02d}"

# --- 3. HIGH-RETENTION SCRIPT ---
topic = random.choice(["The dark psychology of debt and escaping the rat race", "Why saving money keeps you poor", "The hidden secrets of the top 1% wealth builders"])
script = ""
for i in range(1, 9):
    if i == 1:
        prompt = f"Write INTRO chapter for a YouTube finance video on '{topic}'. Start with a shocking hook ('99% of people are lied to...'). Make it suspenseful so the viewer watches the whole video. 3 paragraphs. Plain English."
    else:
        prompt = f"Write Chapter {i}/8 of finance documentary on '{topic}'. 3 long paragraphs. Plain English."
    script += llm(prompt) + " "

sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', script) if len(s.strip()) > 15]
total = len(sentences)

# --- 4. TIMELINE ASSEMBLY & SYNC ---
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

    get_local_image(safe_sen[:60], jpg)
    if not thumb_img and os.path.exists(jpg) and idx == 2: thumb_img = jpg

    words = safe_sen.split()
    weighted_chars = [len(w) + (5 if w.endswith(',') or w.endswith('.') else 0) for w in words]
    tpc = dur / max(sum(weighted_chars), 1)
    
    w_start = curr_time
    for i, w in enumerate(words):
        w_end = w_start + (weighted_chars[i] * tpc)
        clean_w = w.replace('"', '').replace("'", "\\'")
        ass_events.append(f"Dialogue: 0,{format_ass(w_start)},{format_ass(w_end)},Default,,0,0,0,,{{\\fscx70\\fscy70\\t(0,100,\\fscx100\\fscy100)}}{clean_w}")
        w_start = w_end
    curr_time += dur

    subprocess.run(["ffmpeg", "-y", "-loop", "1", "-framerate", "15", "-i", jpg, "-i", wav, 
                    "-vf", f"format=yuv420p,scale=8000:-1,zoompan=z='min(zoom+0.0008,1.2)':d={int(dur*15)+5}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1280x720,fps=15", 
                    "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-b:a", "128k", "-t", str(dur), mp4], 
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    chunks.append(f"file 'scenes/{sid}.mp4'")

# --- 5. RENDER, SEO & THUMBNAIL ---
with open("output/concat.txt", "w") as f: f.write("\n".join(chunks))

ass_head = "[Script Info]\nScriptType: v4.00+\nPlayResX: 1280\nPlayResY: 720\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Default,Arial,65,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,5,3,2,10,10,50,1\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
with open("output/subs.ass", "w", encoding="utf-8") as f: f.write(ass_head + "\n".join(ass_events))

subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "output/concat.txt", "-c", "copy", "output/raw_video.mp4"], check=True)
subprocess.run(["ffmpeg", "-y", "-i", "output/raw_video.mp4", "-vf", "ass='output/subs.ass'", "-c:a", "copy", "output/final_video.mp4"], check=True)

with open("output/seo_metadata.txt", "w", encoding="utf-8") as f: 
    f.write(llm(f"Generate SEO for: '{topic}'. FORMAT EXACTLY:\nTITLE: [Clickbait title]\nDESCRIPTION: [500 chars with emojis 🚨🔥💰. Call to subscribe.]\nTAGS: [tags]"))

if thumb_img:
    try:
        img = Image.open(thumb_img).convert("RGBA")
        grad = Image.new('RGBA', img.size, (0,0,0,0))
        d = ImageDraw.Draw(grad)
        for x in range(int(img.width * 0.6)):
            d.line([(x, 0), (x, img.height)], fill=(20, 20, 20, int(255 * (1 - (x / (img.width * 0.6))))))
        
        final_thumb = Image.alpha_composite(img, grad).convert("RGB")
        try: 
            font1 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 95)
            font2 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 60)
        except: 
            font1 = font2 = ImageFont.load_default()
            
        draw = ImageDraw.Draw(final_thumb)
        draw.text((50, 280), "STOP", fill=(239, 68, 68), font=font1)
        draw.text((50, 390), "DOING THIS", fill=(250, 204, 21), font=font1)
        draw.text((50, 500), "FINANCE SECRETS", fill=(255, 255, 255), font=font2)
        final_thumb.save("output/thumbnail.jpg", format="JPEG", quality=100)
    except: pass
