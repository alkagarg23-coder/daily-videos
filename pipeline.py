"""
================================================================================
YOUTUBE AUTOMATION PIPELINE - ENTERPRISE EDITION (ZERO CRASH GUARANTEE)
================================================================================
Features:
- Robust Error Handling & Logging
- Advanced ASS Subtitle Generation with Kinetic Pop Animations
- Cloud Image Verification & Fallback Mechanisms
- Memory-Safe FFmpeg Direct Chunking (Zero MoviePy RAM Leaks)
- SEO Metadata Generation
================================================================================
"""

import os
import re
import time
import requests
import subprocess
import io
import sys
import random
import traceback
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from kokoro_onnx import Kokoro
import soundfile as sf

# ==============================================================================
# 1. CONFIGURATION & CONSTANTS
# ==============================================================================

class Config:
    # API Endpoints
    OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
    OLLAMA_MODEL = "gemma2:2b"
    HF_TOKEN = os.getenv("HF_TOKEN", "")
    HF_API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
    
    # Directory Structure
    OUTPUT_DIR = "output"
    AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio")
    IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
    SCENES_DIR = os.path.join(OUTPUT_DIR, "scenes")
    
    # Video Settings
    RESOLUTION = (1280, 720)
    FPS = 15
    TARGET_CHAPTERS = 6
    
    # Styling
    SUBTITLE_COLOR_PRIMARY = "&H00FFFFFF"  # Pure White in ASS format
    SUBTITLE_COLOR_OUTLINE = "&H00000000"  # Pure Black outline
    
    ART_STYLES = [
        "clean modern corporate flat vector art, minimalist infographic style",
        "high quality 3D Pixar-style render, vibrant studio lighting, cinematic",
        "dramatic cinematic comic book illustration, heavy shadows, neon accents",
        "elegant watercolor painting, soft pastel colors, atmospheric documentary"
    ]

# Initialize Directories
for directory in [Config.AUDIO_DIR, Config.IMAGE_DIR, Config.SCENES_DIR]:
    os.makedirs(directory, exist_ok=True)

# ==============================================================================
# 2. ADVANCED LOGGING SYSTEM
# ==============================================================================

class Logger:
    @staticmethod
    def info(msg):
        print(f"[INFO] {datetime.now().strftime('%H:%M:%S')} - {msg}", flush=True)

    @staticmethod
    def success(msg):
        print(f"✅ [SUCCESS] {datetime.now().strftime('%H:%M:%S')} - {msg}", flush=True)

    @staticmethod
    def warning(msg):
        print(f"⚠️ [WARNING] {datetime.now().strftime('%H:%M:%S')} - {msg}", flush=True)

    @staticmethod
    def error(msg, exc=None):
        print(f"❌ [ERROR] {datetime.now().strftime('%H:%M:%S')} - {msg}", flush=True)
        if exc:
            traceback.print_exc()

# ==============================================================================
# 3. TEXT SANITIZATION ENGINE
# ==============================================================================

class TextCleaner:
    @staticmethod
    def sanitize_for_tts(text):
        """Removes markdown, brackets, and special chars so TTS doesn't read them."""
        try:
            # Remove Markdown headers, bold, italic
            clean = re.sub(r'#|\*|_|`|~', '', text)
            # Remove text in brackets
            clean = re.sub(r'\[.*?\]|\(.*?\)', '', clean)
            # Replace common problematic characters
            clean = clean.replace('"', '').replace('\n', ' ')
            clean = clean.replace('-', ' ')
            # Fix multiple spaces
            clean = re.sub(r'\s+', ' ', clean).strip()
            return clean
        except Exception as e:
            Logger.error("Failed to sanitize text, returning safe default.", e)
            return "Continuing the journey."

    @staticmethod
    def split_into_sentences(text):
        """Splits text into timeline-friendly sentences."""
        try:
            sentences = re.split(r'(?<=[.!?]) +', text)
            return [s.strip() for s in sentences if len(s.strip()) > 10]
        except Exception as e:
            Logger.error("Failed to split sentences.", e)
            return [text]

# ==============================================================================
# 4. LLM SCRIPT GENERATOR
# ==============================================================================

class AIWriter:
    @staticmethod
    def generate_chapter(chapter_num, total_chapters, topic):
        prompt = (
            f"You are a professional YouTube scriptwriter creating a highly detailed "
            f"finance documentary about '{topic}'.\n"
            f"Write ONLY Chapter {chapter_num} of {total_chapters}.\n"
            f"Write exactly 3 long, engaging paragraphs. Do not use hashtags, asterisks, "
            f"bullet points, or intro/outro text. Write in plain conversational English."
        )
        
        for attempt in range(3):
            try:
                response = requests.post(
                    Config.OLLAMA_URL, 
                    json={"model": Config.OLLAMA_MODEL, "prompt": prompt, "stream": False}, 
                    timeout=120
                )
                response.raise_for_status()
                return response.json()["response"].strip()
            except Exception as e:
                Logger.warning(f"LLM Generation failed (Attempt {attempt+1}/3): {str(e)}")
                time.sleep(5)
        
        Logger.error(f"Critical failure generating chapter {chapter_num}.")
        return "Financial growth requires patience and strategic planning."

    @staticmethod
    def generate_seo(topic):
        prompt = (
            f"Generate YouTube SEO metadata for a documentary about '{topic}'.\n"
            f"Format exactly like this:\n\n"
            f"TITLE: [Catchy Title]\n"
            f"DESCRIPTION: [3 sentence description]\n"
            f"TAGS: [comma separated tags]"
        )
        try:
            response = requests.post(
                Config.OLLAMA_URL, 
                json={"model": Config.OLLAMA_MODEL, "prompt": prompt, "stream": False}, 
                timeout=60
            )
            return response.json()["response"].strip()
        except Exception as e:
            Logger.warning("Failed to generate SEO metadata.")
            return f"TITLE: Financial Masterclass\nDESCRIPTION: Learn about {topic}.\nTAGS: finance, money"

# ==============================================================================
# 5. HIGH-QUALITY IMAGE ENGINE (WITH VALIDATION)
# ==============================================================================

class ImageEngine:
    def __init__(self, style):
        self.style = style

    def create_solid_fallback(self, output_path):
        """Creates a safe RGB fallback image to prevent FFmpeg black screen crashes."""
        try:
            img = Image.new('RGB', Config.RESOLUTION, color=(15, 23, 42))
            img.save(output_path, format="PNG")
            Logger.info(f"Fallback image created at {output_path}")
        except Exception as e:
            Logger.error("Failed to create fallback image.", e)

    def verify_image(self, image_path):
        """Verifies if the downloaded image is valid and not corrupted."""
        try:
            with Image.open(image_path) as img:
                img.verify()
            return True
        except Exception:
            return False

    def generate(self, prompt_text, output_path):
        if not Config.HF_TOKEN:
            Logger.warning("HF_TOKEN not found. Using fallback image.")
            self.create_solid_fallback(output_path)
            return

        headers = {"Authorization": f"Bearer {Config.HF_TOKEN}"}
        clean_prompt = f"{self.style}, visually showing {prompt_text}, masterpiece, 8k resolution, highly detailed, no text"
        payload = {"inputs": clean_prompt, "parameters": {"guidance_scale": 7.5}}
        
        for attempt in range(4): 
            try:
                response = requests.post(Config.HF_API_URL, headers=headers, json=payload, timeout=60)
                if response.status_code == 200:
                    image = Image.open(io.BytesIO(response.content)).convert('RGB')
                    widescreen = image.resize(Config.RESOLUTION, Image.Resampling.LANCZOS)
                    widescreen.save(output_path, format="PNG")
                    
                    if self.verify_image(output_path):
                        time.sleep(2) # Respect rate limits
                        return
                    else:
                        Logger.warning(f"Image verification failed for {output_path}")
                elif response.status_code == 429:
                    Logger.warning("Rate limit hit. Sleeping for 15 seconds.")
                    time.sleep(15)
                else:
                    Logger.warning(f"HF API Error: {response.status_code}")
            except Exception as e:
                Logger.warning(f"Image generation exception: {str(e)}")
            
            time.sleep(5)
                
        Logger.error(f"All attempts failed for image. Generating fallback.")
        self.create_solid_fallback(output_path)

# ==============================================================================
# 6. LOCAL TTS ENGINE (KOKORO ONNX)
# ==============================================================================

class TTSEngine:
    def __init__(self):
        Logger.info("Initializing Kokoro ONNX TTS Engine...")
        try:
            self.kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
            Logger.success("Kokoro loaded successfully.")
        except Exception as e:
            Logger.error("Failed to load Kokoro ONNX.", e)
            sys.exit(1)

    def generate_audio(self, text, output_path):
        spoken_text = TextCleaner.sanitize_for_tts(text)
        if not spoken_text:
            spoken_text = "Let us continue."
            
        try:
            samples, sample_rate = self.kokoro.create(spoken_text, voice="am_michael", speed=1.0, lang="en-us")
            sf.write(output_path, samples, sample_rate)
            duration = len(samples) / sample_rate
            return duration
        except Exception as e:
            Logger.warning(f"TTS generation failed on primary text. Trying ultra-safe mode. Error: {str(e)}")
            safe_text = re.sub(r'[^a-zA-Z0-9\s\.]', '', spoken_text)
            if not safe_text.strip(): safe_text = "Financial concept."
            
            try:
                samples, sample_rate = self.kokoro.create(safe_text, voice="am_michael", speed=1.0, lang="en-us")
                sf.write(output_path, samples, sample_rate)
                return len(samples) / sample_rate
            except Exception as e2:
                Logger.error("Total TTS Failure.", e2)
                return 0.0

# ==============================================================================
# 7. ADVANCED SUBTITLE ENGINE (ASS FORMAT WITH KINETIC ANIMATIONS)
# ==============================================================================

class SubtitleEngine:
    def __init__(self):
        self.events = []
        
    def add_word(self, word, start_time, end_time):
        """
        Adds a single word with a POP-UP animation.
        {\fscx80\fscy80\t(0,100,\fscx100\fscy100)} = Start at 80% scale, zoom to 100% in 100ms
        """
        start_ass = self.format_ass_time(start_time)
        end_ass = self.format_ass_time(end_time)
        
        # Clean subtitle word for display
        clean_word = word.replace('"', '').replace("'", "\\'")
        
        # Kinetic Pop Animation Tag
        animation_tag = r"{\fscx70\fscy70\t(0,120,\fscx100\fscy100)}"
        event_line = f"Dialogue: 0,{start_ass},{end_ass},Default,,0,0,0,,{animation_tag}{clean_word}"
        self.events.append(event_line)

    @staticmethod
    def format_ass_time(seconds):
        """Formats time to ASS format: H:MM:SS.cs"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        centi = int((seconds - int(seconds)) * 100)
        return f"{hours}:{minutes:02d}:{secs:02d}.{centi:02d}"

    def build_file(self, output_path):
        """Builds the final .ass file with strict styling rules."""
        header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 720
WrapStyle: 1

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,60,{Config.SUBTITLE_COLOR_PRIMARY},&H000000FF,{Config.SUBTITLE_COLOR_OUTLINE},&H80000000,-1,0,0,0,100,100,0,0,1,4,2,2,10,10,60,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(header)
                f.write("\n".join(self.events))
            Logger.success(f"Animated Subtitles saved to {output_path}")
        except Exception as e:
            Logger.error("Failed to write ASS subtitle file.", e)

# ==============================================================================
# 8. FFMPEG ZERO-RAM VIDEO ENGINE
# ==============================================================================

class VideoEngine:
    @staticmethod
    def render_chunk(image_path, audio_path, output_path, duration):
        """Renders a single scene using direct FFmpeg hardware-safe processing."""
        # Using d=1 in zoompan forces exactly 1 frame mapping, killing the black screen bug
        # Scale to 10x height internally for smooth slow zoom, then crop back to 720p
        cmd = [
            "ffmpeg", "-y", "-loop", "1", "-framerate", str(Config.FPS),
            "-i", image_path,
            "-i", audio_path,
            "-vf", f"scale=-2:10*ih,zoompan=z='min(zoom+0.001,1.5)':d=1:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={Config.RESOLUTION[0]}x{Config.RESOLUTION[1]},fps={Config.FPS}",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", 
            "-t", str(duration),
            output_path
        ]
        
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return True
        except subprocess.CalledProcessError as e:
            Logger.error(f"FFmpeg chunk render failed for {output_path}", e)
            return False

    @staticmethod
    def concatenate_chunks(concat_file_path, output_path):
        """Joins all chunks instantaneously without re-encoding (RAM Safe)."""
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", concat_file_path,
            "-c", "copy",
            output_path
        ]
        try:
            subprocess.run(cmd, check=True)
            Logger.success("Chunks concatenated successfully.")
        except subprocess.CalledProcessError as e:
            Logger.error("Failed to concatenate video chunks.", e)
            sys.exit(1)

    @staticmethod
    def burn_subtitles(input_video, ass_subtitle_file, final_output):
        """Burns the animated .ass subtitles into the final video."""
        cmd = [
            "ffmpeg", "-y",
            "-i", input_video,
            "-vf", f"ass='{ass_subtitle_file}'",
            "-c:a", "copy",
            final_output
        ]
        try:
            subprocess.run(cmd, check=True)
            Logger.success(f"Final Video rendered completely: {final_output}")
        except subprocess.CalledProcessError as e:
            Logger.error("Failed to burn animated subtitles.", e)
            sys.exit(1)

# ==============================================================================
# 9. MAIN EXECUTION PIPELINE
# ==============================================================================

def get_current_episode():
    try:
        with open("counter.txt", "r") as f:
            count = int(f.read().strip()) + 1
    except:
        count = 1
    with open("counter.txt", "w") as f:
        f.write(str(count))
    return count

def main():
    Logger.info("Starting Enterprise Video Pipeline...")
    episode_num = get_current_episode()
    Logger.info(f"Producing Episode #{episode_num}")

    # Initialize Core Engines
    selected_style = random.choice(Config.ART_STYLES)
    Logger.info(f"Selected Visual Style: {selected_style}")
    
    img_engine = ImageEngine(selected_style)
    tts_engine = TTSEngine()
    sub_engine = SubtitleEngine()

    # Define Topic
    topics = [
        "the brutal reality of escaping the rat race and building real wealth",
        "the psychology of money and true long-term financial freedom",
        "why hoarding cash keeps you broke and how smart investing changes your life"
    ]
    chosen_topic = random.choice(topics)
    Logger.info(f"Topic Selected: {chosen_topic}")

    # Phase 1: Script Generation
    Logger.info("Generating Full Script (Iterative Chapter Loop)...")
    full_script = ""
    for chapter in range(1, Config.TARGET_CHAPTERS + 1):
        Logger.info(f"Generating Chapter {chapter}/{Config.TARGET_CHAPTERS}...")
        chapter_text = AIWriter.generate_chapter(chapter, Config.TARGET_CHAPTERS, chosen_topic)
        full_script += chapter_text + " "

    # Phase 2: Processing and Asset Generation
    sentences = TextCleaner.split_into_sentences(full_script)
    total_scenes = len(sentences)
    Logger.info(f"Script compiled. Total Scenes to process: {total_scenes}")

    current_timeline_time = 0.0
    concat_lines = []
    concat_file_path = os.path.join(Config.OUTPUT_DIR, "concat.txt")

    for idx, sentence in enumerate(sentences):
        scene_id = str(idx).zfill(4)
        audio_path = os.path.join(Config.AUDIO_DIR, f"scene_{scene_id}.wav")
        image_path = os.path.join(Config.IMAGE_DIR, f"scene_{scene_id}.png")
        video_chunk_path = os.path.join(Config.SCENES_DIR, f"scene_{scene_id}.mp4")

        # 1. Generate Audio
        duration = tts_engine.generate_audio(sentence, audio_path)
        if duration <= 0.1:
            Logger.warning(f"Scene {idx+1} audio generation failed. Skipping.")
            continue

        # 2. Generate Image
        prompt_snippet = sentence[:100] if len(sentence) > 100 else sentence
        img_engine.generate(prompt_snippet, image_path)

        # 3. Calculate Character-Weighted Subtitles
        words = sentence.split()
        if words:
            total_chars = sum(len(w) for w in words)
            time_per_char = duration / max(total_chars, 1)
            
            word_start = current_timeline_time
            for word in words:
                word_dur = len(word) * time_per_char
                word_end = word_start + word_dur
                sub_engine.add_word(word, word_start, word_end)
                word_start = word_end

        current_timeline_time += duration

        # 4. Render FFmpeg Chunk
        success = VideoEngine.render_chunk(image_path, audio_path, video_chunk_path, duration)
        if success:
            concat_lines.append(f"file 'scenes/scene_{scene_id}.mp4'")
            if (idx + 1) % 5 == 0 or (idx + 1) == total_scenes:
                Logger.info(f"Rendered {idx+1}/{total_scenes} chunks...")

    # Phase 3: Assembly
    Logger.info("Building ASS Subtitle File...")
    ass_path = os.path.join(Config.OUTPUT_DIR, "animated_subtitles.ass")
    sub_engine.build_file(ass_path)

    Logger.info("Writing Concat Demuxer Map...")
    with open(concat_file_path, "w") as f:
        f.write("\n".join(concat_lines))

    raw_video = os.path.join(Config.OUTPUT_DIR, f"raw_video_{episode_num}.mp4")
   