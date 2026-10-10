import os
import re
import sys
import json
import time
import shutil
import subprocess
import requests
import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma2:2b"
SD_BINARY = "./sd"
SD_MODEL = "model.safetensors"
KOKORO_MODEL = "kokoro-v1.0.onnx"
KOKORO_VOICES = "voices-v1.0.bin"
VOICE_NAME = "af_sarah"
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
OUTPUT_VIDEO = "final_video.mp4"
THUMBNAIL_FILE = "thumbnail.jpg"
SCRIPT_FILE = "script.txt"
ASS_FILE = "subs.ass"
CONCAT_FILE = "concat_list.txt"

# ==============================================================================
# 1. AI SCRIPT GENERATION (OLLAMA gemma2:2b)
# ==============================================================================
def generate_script():
    print("[1/6] Generating 8-Chapter AI Finance Documentary Script...")
    prompt = (
        "Write an exhaustive, high-retention 8-chapter finance documentary script. "
        "Chapter 1 MUST start strictly with: '99% of people are lied to about money from the day they are born.' "
        "Topics across 8 chapters: 1. The Matrix of Debt, 2. The Inflation Trap, 3. Velocity of Money, "
        "4. Fractional Reserve Illusion, 5. Assets vs Liabilities, 6. Tax Avoidance Strategies, "
        "7. Asymmetric Opportunities, 8. Escaping the Rat Race. "
        "Format output strictly as JSON with key 'chapters', a list of 8 objects, each having: "
        "'title' (string), 'narration' (lengthy detailed text, at least 4 paragraphs), and 'visual_context' (short visual prompt)."
    )

    chapters = []
    try:
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }
        res = requests.post(OLLAMA_URL, json=payload, timeout=240)
        if res.status_code == 200:
            parsed = json.loads(res.json().get("response", "{}"))
            chapters = parsed.get("chapters", [])
    except Exception as e:
        print(f"[-] Ollama generation query error: {e}. Falling back to default architecture.")

    # Resilient fallback ensuring strict compliance and exact 8-chapter viral structure
    if len(chapters) < 8:
        fallback_narratives = [
            ("The Matrix of Debt", 
             "99% of people are lied to about money from the day they are born. Society conditions you to work forty years for green paper that governments print out of thin air. You take loans for degrees that teach outdated skills, finance cars that depreciate the minute you drive them, and trap yourself in thirty-year mortgages. The debt cycle is not an accident. It is an engineered financial conveyor belt designed to keep you compliant and dependent on a monthly paycheck.",
             "stickman trapped inside a giant financial cage made of dollar bills and chains"),
            ("The Inflation Trap",
             "Your savings account is quietly burning to ashes while you sleep. Inflation is not a natural economic phenomenon; it is an invisible tax on your productivity. When central banks expand the money supply, the price of goods rises, eroding purchasing power. If your money sits idle in cash earning fractional interest, you are losing five to ten percent of your wealth every single year to stealth devaluation.",
             "stickman watching a bank vault melt away as numbers on an hourglass run out"),
            ("Velocity of Money",
             "Wealthy institutions do not hoard dead cash; they master the velocity of money. Money must move continuously through cash-flowing instruments. When capital flows through real estate, dividend-yielding entities, and profitable businesses, it compounds exponentially before taxes can touch it. Stagnant capital decays, but circulating capital creates perpetual financial momentum.",
             "stickman directing high-speed flowing arrows of coins through complex gears"),
            ("Fractional Reserve Illusion",
             "Modern banking operates on a mathematical illusion known as fractional reserve banking. When you deposit ten thousand dollars into a bank, the institution retains only a fraction and lends out the rest nine times over. They create billions of digital credits backed by nothing but promissory notes. The moment you grasp that money is created from debt, the entire global economy reveals its hidden blueprint.",
             "stickman inspecting a gigantic house of cards built on bank pillars"),
            ("Assets vs Liabilities",
             "The middle class buys liabilities thinking they are assets. A genuine asset puts capital into your pocket whether you wake up or stay in bed. A liability drains your account through maintenance, interest, and taxes. True freedom begins when your passive asset yield exceeds your survival overhead. Every single dollar you earn must be deployed as a tireless digital worker generating cash flow.",
             "stickman balancing scales between income-producing factories and a draining house"),
            ("Tax Avoidance Strategies",
             "Taxes are the single largest lifetime expense for any worker, devouring forty to fifty percent of gross lifetime earnings. The legal tax code is not a penalty system; it is a roadmap of incentives designed to stimulate housing, production, and commerce. By restructuring earned income into corporate entities, accelerated depreciation, and capital gains, the elite legally drop their tax rate to near zero.",
             "stickman navigating a transparent glass maze holding a shield against tax arrows"),
            ("Asymmetric Opportunities",
             "To escape systemic mediocrity, you must seek asymmetric bets where the downside is strictly capped at one unit but the upside is virtually unlimited. Traditional finance preaches diversification into mediocre index funds. Modern sovereignty demands high-conviction allocation into scalable digital assets, decentralized protocols, and proprietary skill sets that cannot be inflated away.",
             "stickman launching a rocket from a whiteboard launching pad towards compounding stars"),
            ("Escaping the Rat Race",
             "True wealth is not measured in luxury watches or sports cars; it is measured strictly in freedom and autonomy over your time. When you own your cash flow, you own your calendar. Reject the manufactured consumer illusions. Build your fortress of assets, detach your income from physical hours, and reclaim sovereignty over your life.",
             "stickman breaking golden handcuffs and walking into an open sunrise horizon")
        ]
        chapters = []
        for title, text, context in fallback_narratives:
            chapters.append({
                "title": title,
                "narration": text,
                "visual_context": context
            })

    with open(SCRIPT_FILE, "w", encoding="utf-8") as f:
        for idx, ch in enumerate(chapters, 1):
            f.write(f"=== CHAPTER {idx}: {ch['title']} ===\n")
            f.write(f"VISUAL: {ch['visual_context']}\n")
            f.write(f"NARRATION: {ch['narration']}\n\n")

    print(f"[+] Script saved to {SCRIPT_FILE} with {len(chapters)} chapters.")
    return chapters

# ==============================================================================
# 2. TTS SYNTHESIS (KOKORO ONNX) WITH REGEX FALLBACK
# ==============================================================================
def init_tts():
    try:
        from kokoro_onnx import Kokoro
        if os.path.exists(KOKORO_MODEL) and os.path.exists(KOKORO_VOICES):
            return Kokoro(KOKORO_MODEL, KOKORO_VOICES)
        print("[-] Kokoro ONNX files missing.")
        return None
    except Exception as e:
        print(f"[-] Error loading Kokoro ONNX: {e}")
        return None

def synthesize_sentence_audio(kokoro, text, output_wav):
    def run_inference(clean_str):
        if kokoro:
            samples, sr = kokoro.create(clean_str, voice=VOICE_NAME, speed=1.0, lang="en-us")
            return samples, sr
        return None, 24000

    try:
        samples, sr = run_inference(text)
    except Exception:
        # Fallback to regex-cleaned text on crash
        cleaned = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
        try:
            samples, sr = run_inference(cleaned)
        except Exception:
            samples = None
            sr = 24000

    if samples is None or len(samples) == 0:
        # Fallback to local espeak-ng if onnx fails
        clean_espeak = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
        cmd = ["espeak-ng", "-w", output_wav, "-v", "en-us", "-s", "160", clean_espeak]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if os.path.exists(output_wav):
            data, sr = sf.read(output_wav)
            duration = len(data) / float(sr)
            return duration
        return 0.0

    duration = len(samples) / float(sr)
    if duration < 0.5:
        # Skip audio chunks < 0.5s as per requirement
        return 0.0

    sf.write(output_wav, samples, sr)
    return duration

# ==============================================================================
# 3. LOCAL IMAGE GENERATION (SD.CPP) & PIL RESIZING
# ==============================================================================
def create_fallback_canvas(target_path, context_text):
    img = Image.new("RGB", (TARGET_WIDTH, TARGET_HEIGHT), color=(250, 250, 250))
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, TARGET_WIDTH - 20, TARGET_HEIGHT - 20], outline=(40, 40, 40), width=6)
    draw.ellipse([600, 200, 680, 280], outline=(20, 20, 20), width=5)
    draw.line([640, 280, 640, 440], fill=(20, 20, 20), width=5)
    draw.line([640, 320, 560, 380], fill=(20, 20, 20), width=5)
    draw.line([640, 320, 720, 280], fill=(20, 20, 20), width=5)
    draw.line([640, 440, 580, 560], fill=(20, 20, 20), width=5)
    draw.line([640, 440, 700, 560], fill=(20, 20, 20), width=5)
    draw.text((50, TARGET_HEIGHT - 80), context_text[:70], fill=(100, 100, 100))
    img.save(target_path, "JPEG", quality=95)

def generate_image_sd(context, output_jpg):
    prompt = f"masterpiece, ultra-detailed stickman illustration on a whiteboard. professional explainer style. {context}"
    negative_prompt = "text, letters, watermark, blurry"
    raw_png = output_jpg.replace(".jpg", ".png")

    if os.path.exists(SD_BINARY) and os.path.exists(SD_MODEL):
        cmd = [
            SD_BINARY,
            "-m", SD_MODEL,
            "-p", prompt,
            "-n", negative_prompt,
            "--steps", "15",
            "--cfg-scale", "7.0",
            "-W", "768",
            "-H", "512",
            "-o", raw_png
        ]
        try:
            print(f"[SD.CPP] Generating frame for: '{context[:35]}...'")
            subprocess.run(cmd, timeout=350, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if os.path.exists(raw_png):
                with Image.open(raw_png) as im:
                    im_resized = im.resize((TARGET_WIDTH, TARGET_HEIGHT), resample=Image.Resampling.LANCZOS)
                    im_resized.convert("RGB").save(output_jpg, "JPEG", quality=95)
                os.remove(raw_png)
                return
        except subprocess.TimeoutExpired:
            print(f"[!] Subprocess timeout (350s) exceeded on SD.CPP. Falling back to clean canvas.")
        except Exception as e:
            print(f"[!] SD.CPP execution failed ({e}). Falling back to clean canvas.")

    create_fallback_canvas(output_jpg, context)

# ==============================================================================
# 4. PUNCTUATION-AWARE SUBTITLE SYNC (ASS GENERATOR)
# ==============================================================================
def format_ass_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centis = int(round((seconds - int(seconds)) * 100))
    if centis >= 100:
        centis = 99
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"

class ASSSubtitleGenerator:
    def __init__(self, filename):
        self.filename = filename
        self.events = []

    def add_sentence_words(self, sentence, start_time, duration):
        words = sentence.strip().split()
        if not words or duration <= 0:
            return

        weights = []
        for w in words:
            weight = len(w)
            if w.endswith(",") or w.endswith("."):
                weight += 5
            weights.append(weight)

        total_weight = sum(weights)
        if total_weight == 0:
            total_weight = len(words)
            weights = [1] * len(words)

        current_time = start_time
        for w, weight in zip(words, weights):
            w_duration = (weight / float(total_weight)) * duration
            w_start = current_time
            w_end = current_time + w_duration
            current_time = w_end

            clean_w = w.replace("\\", "")
            pop_effect = r"{\fscx70\fscy70\t(0,100,\fscx100\fscy100)}"
            text_field = f"{pop_effect}{clean_w}"
            self.events.append((w_start, w_end, text_field))

    def write_file(self):
        header = (
            "[Script Info]\n"
            "Title: Finance Documentary\n"
            "ScriptType: v4.00+\n"
            "WrapStyle: 0\n"
            "ScaledBorderAndShadow: yes\n"
            "PlayResX: 1280\n"
            "PlayResY: 720\n\n"
            "[V4+ Styles]\n"
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
            "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding\n"
            "Style: Default,Arial,65,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,"
            "-1,0,0,0,100,100,0,0,1,4,2,2,30,30,80,1\n\n"
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        )
        with open(self.filename, "w", encoding="utf-8") as f:
            f.write(header)
            for start, end, text in self.events:
                f.write(f"Dialogue: 0,{format_ass_time(start)},{format_ass_time(end)},Default,,0,0,0,,{text}\n")

# ==============================================================================
# 5. HIGH-RETENTION THUMBNAIL GENERATOR (PIL)
# ==============================================================================
def generate_thumbnail(base_image_path):
    print("[5/6] Generating High-Retention Thumbnail...")
    if not os.path.exists(base_image_path):
        create_fallback_canvas(base_image_path, "Financial Secret Blueprint")

    with Image.open(base_image_path) as base_im:
        img = base_im.resize((TARGET_WIDTH, TARGET_HEIGHT), resample=Image.Resampling.LANCZOS).convert("RGBA")

    # Black transparent gradient on the left 60%
    overlay = Image.new("RGBA", (TARGET_WIDTH, TARGET_HEIGHT), (0, 0, 0, 0))
    gradient_limit = int(TARGET_WIDTH * 0.60)
    for x in range(gradient_limit):
        ratio = 1.0 - (x / float(gradient_limit))
        alpha = int(240 * ratio)
        line = Image.new("RGBA", (1, TARGET_HEIGHT), (0, 0, 0, alpha))
        overlay.paste(line, (x, 0))

    img = Image.alpha_composite(img, overlay)
    draw = ImageDraw.Draw(img)

    # Font Locator
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    if not os.path.exists(font_path):
        font_path = "DejaVuSans-Bold.ttf"

    try:
        font_large = ImageFont.truetype(font_path, 105)
        font_medium = ImageFont.truetype(font_path, 80)
    except Exception:
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()

    def draw_bordered_text(draw_ctx, pos, text, font, fill_color, stroke_color, stroke_w=6):
        x, y = pos
        for dx in range(-stroke_w, stroke_w + 1):
            for dy in range(-stroke_w, stroke_w + 1):
                if dx * dx + dy * dy <= stroke_w * stroke_w:
                    draw_ctx.text((x + dx, y + dy), text, font=font, fill=stroke_color)
        draw_ctx.text((x, y), text, font=font, fill=fill_color)

    # Required Thumbnail Text Layout
    draw_bordered_text(draw, (70, 120), "STOP", font_large, (255, 30, 30), (0, 0, 0), stroke_w=8)
    draw_bordered_text(draw, (70, 250), "DOING THIS", font_large, (255, 215, 0), (0, 0, 0), stroke_w=8)
    draw_bordered_text(draw, (70, 420), "FINANCE SECRETS", font_medium, (255, 255, 255), (0, 0, 0), stroke_w=6)

    img.convert("RGB").save(THUMBNAIL_FILE, "JPEG", quality=95)
    print(f"[+] Thumbnail saved to {THUMBNAIL_FILE}")

# ==============================================================================
# 6. PIPELINE ORCHESTRATION & FFMPEG RENDERING
# ==============================================================================
def main():
    print("=== STARTING AUTONOMOUS DOCUMENTARY PIPELINE ===")
    chapters = generate_script()
    kokoro = init_tts()
    ass_gen = ASSSubtitleGenerator(ASS_FILE)

    chunk_files = []
    global_timeline = 0.0
    chunk_index = 0

    os.makedirs("chunks", exist_ok=True)

    for ch_idx, ch in enumerate(chapters):
        narration = ch["narration"]
        visual_ctx = ch["visual_context"]
        sentences = re.split(r'(?<=[.!?])\s+', narration.strip())

        for s_idx, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue

            audio_file = f"chunks/audio_{chunk_index}.wav"
            image_file = f"chunks/image_{chunk_index}.jpg"
            chunk_mp4 = f"chunks/chunk_{chunk_index}.mp4"

            # TTS Generation
            duration = synthesize_sentence_audio(kokoro, sentence, audio_file)
            if duration < 0.5:
                continue

            # Subtitle Sync
            ass_gen.add_sentence_words(sentence, global_timeline, duration)
            global_timeline += duration

            # Image Generation
            generate_image_sd(visual_ctx, image_file)

            # Frame-Locked Video Chunk Rendering
            cmd_chunk = [
                "ffmpeg", "-y",
                "-loop", "1",
                "-framerate", "15",
                "-i", image_file,
                "-i", audio_file,
                "-vf", "scale=1280:720,format=yuv420p",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-c:a", "aac",
                "-b:a", "128k",
                "-shortest",
                chunk_mp4
            ]
            subprocess.run(cmd_chunk, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            chunk_files.append(chunk_mp4)
            chunk_index += 1

    ass_gen.write_file()
    print(f"[+] Synced {chunk_index} chunks. Total runtime: {global_timeline / 60.0:.2f} mins.")

    # Concat Chunks
    with open(CONCAT_FILE, "w", encoding="utf-8") as f:
        for cf in chunk_files:
            f.write(f"file '{cf}'\n")

    raw_concat = "full_video_raw.mp4"
    print("[4/6] Concatenating frame-locked chunks...")
    cmd_concat = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", CONCAT_FILE,
        "-c", "copy",
        raw_concat
    ]
    subprocess.run(cmd_concat, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    # Burn ASS Subtitles
    print("[5/6] Burning kinetic ASS subtitles...")
    cmd_burn = [
        "ffmpeg", "-y",
        "-i", raw_concat,
        "-vf", f"ass={ASS_FILE}",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-c:a", "copy",
        OUTPUT_VIDEO
    ]
    subprocess.run(cmd_burn, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    # Thumbnail Generation from chunk index 2 (or index 0 fallback)
    selected_thumb_bg = "chunks/image_2.jpg" if os.path.exists("chunks/image_2.jpg") else "chunks/image_0.jpg"
    generate_thumbnail(selected_thumb_bg)

    # Cleanup intermediate files
    for temp in [raw_concat, CONCAT_FILE, ASS_FILE]:
        if os.path.exists(temp):
            os.remove(temp)
    shutil.rmtree("chunks", ignore_errors=True)

    print(f"[6/6] Pipeline Execution Finished Successfully. Output: {OUTPUT_VIDEO}")

if __name__ == "__main__":
    main()