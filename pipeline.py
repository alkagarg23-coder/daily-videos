import os
import sys
import re
import time
import random
import requests
import subprocess
import traceback
from io import BytesIO
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from kokoro_onnx import Kokoro
import soundfile as sf

# ==============================================================================
# 1. CORE SYSTEM CONFIGURATION
# ==============================================================================

class CoreConfig:
    ROOT_DIR = os.getcwd()
    OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
    AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio")
    IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
    SCENES_DIR = os.path.join(OUTPUT_DIR, "scenes")
    
    OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
    OLLAMA_MODEL = "gemma2:2b"
    HF_TOKEN = os.getenv("HF_TOKEN", "")
    HF_API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
    
    RESOLUTION = (1280, 720)
    FPS = 15
    CHAPTERS = 8
    
    SUB_FONT = "Arial"
    SUB_SIZE = 55
    SUB_PRIMARY_COLOR = "&H00FFFFFF"
    SUB_OUTLINE_COLOR = "&H00000000"
    
    STYLES = [
        "ultra-realistic 8k cinematic photography, documentary style, highly detailed",
        "clean vector illustration, modern corporate finance aesthetic, minimalist",
        "high quality 3D render, vibrant studio lighting, Pixar-style animation",
        "moody cyberpunk aesthetic, neon lights, dark background, futuristic finance",
        "classic vintage newspaper illustration, cross-hatch style, dramatic"
    ]

for folder in [CoreConfig.OUTPUT_DIR, CoreConfig.AUDIO_DIR, CoreConfig.IMAGE_DIR, CoreConfig.SCENES_DIR]:
    os.makedirs(folder, exist_ok=True)

def update_and_get_counter():
    try:
        with open("counter.txt", "r") as f:
            count = int(f.read().strip()) + 1
    except:
        count = 1
    with open("counter.txt", "w") as f:
        f.write(str(count))
    return count

CURRENT_EPISODE = update_and_get_counter()
CURRENT_STYLE = random.choice(CoreConfig.STYLES)

# ==============================================================================
# 2. ADVANCED LOGGING & ERROR HANDLING
# ==============================================================================

class Logger:
    @staticmethod
    def _timestamp():
        return datetime.now().strftime("%H:%M:%S")

    @staticmethod
    def info(msg):
        print(f"🔹 [{Logger._timestamp()}] [INFO] {msg}", flush=True)

    @staticmethod
    def success(msg):
        print(f"✅ [{Logger._timestamp()}] [SUCCESS] {msg}", flush=True)

    @staticmethod
    def warn(msg):
        print(f"⚠️ [{Logger._timestamp()}] [WARN] {msg}", flush=True)

    @staticmethod
    def error(msg, exc=None):
        print(f"❌ [{Logger._timestamp()}] [ERROR] {msg}", flush=True)
        if exc:
            traceback.print_exc()

# ==============================================================================
# 3. TEXT SANITIZATION & SYLLABLE ENGINE
# ==============================================================================

class TextProcessor:
    @staticmethod
    def clean_for_speech(text):
        try:
            clean = re.sub(r'#|\*|_|`|~|>', '', text)
            clean = re.sub(r'\[.*?\]|\(.*?\)', '', clean)
            clean = re.sub(r'http\S+', '', clean)
            clean = clean.replace('"', '').replace('\n', ' ').replace('-', ' ')
            clean = re.sub(r'\s+', ' ', clean).strip()
            return clean
        except Exception as e:
            Logger.error("Error sanitizing text", e)
            return "Continuing our discussion on wealth."

    @staticmethod
    def chunk_into_sentences(text):
        try:
            sentences = re.split(r'(?<=[.!?]) +', text)
            valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
            return valid_sentences
        except Exception as e:
            Logger.error("Failed to split text", e)
            return [text]

# ==============================================================================
# 4. LLM SCRIPTWRITER (OLLAMA LOCAL ENGINE)
# ==============================================================================

class LLMEngine:
    @staticmethod
    def execute_prompt(prompt, retries=3):
        payload = {"model": CoreConfig.OLLAMA_MODEL, "prompt": prompt, "stream": False}
        for attempt in range(retries):
            try:
                response = requests.post(CoreConfig.OLLAMA_URL, json=payload, timeout=180)
                response.raise_for_status()
                return response.json()["response"].strip()
            except Exception as e:
                Logger.warn(f"LLM Engine failed (Attempt {attempt+1}/{retries}): {e}")
                time.sleep(5)
        Logger.error("LLM Engine completely failed. Using emergency fallback text.")
        return "Building true wealth takes time, discipline, and strategic investing. " * 5

    @staticmethod
    def generate_chapter(chapter_index, total_chapters, topic):
        prompt = (
            f"You are writing an engaging, deep YouTube finance documentary about: '{topic}'.\n"
            f"Write ONLY Chapter {chapter_index} out of {total_chapters}.\n"
            f"Write at least 3 very long, detailed paragraphs filled with examples and storytelling.\n"
            f"IMPORTANT RULE: DO NOT use hashtags, asterisks, bullet points, numbers, or special formatting. "
            f"Write in pure, flowing conversational English paragraphs only."
        )
        return LLMEngine.execute_prompt(prompt)

    @staticmethod
    def generate_seo_metadata(topic):
        prompt = (
            f"Generate YouTube SEO metadata for a deep documentary about: '{topic}'.\n"
            f"Respond EXACTLY in this format, nothing else:\n\n"
            f"TITLE: [Your highly clickable, SEO-optimized title here]\n"
            f"DESCRIPTION: [Write a highly engaging description of exactly around 500 characters. Use plenty of relevant emojis like 🚀💰📉🔥. Include a call to action to subscribe.]\n"
            f"TAGS: [comma, separated, high-volume, tags, here]"
        )
        return LLMEngine.execute_prompt(prompt, retries=2)

# ==============================================================================
# 5. CLOUD VISION ENGINE (THE BLACK SCREEN KILLER)
# ==============================================================================

class CloudVisionEngine:
    @staticmethod
    def _create_failsafe_image(output_path):
        try:
            img = Image.new('RGB', CoreConfig.RESOLUTION, color=(20, 25, 35))
            img.save(output_path, format="JPEG", quality=95)
            Logger.info(f"Fallback image safely created: {output_path}")
        except Exception as e:
            Logger.error("Failed to create failsafe image.", e)

    @staticmethod
    def _process_and_save_image(image_bytes, output_path):
        try:
            img = Image.open(BytesIO(image_bytes)).convert("RGB")
            target_ratio = 16 / 9
            img_ratio = img.width / img.height
            
            if img_ratio > target_ratio:
                new_width = int(img.height * target_ratio)
                offset = (img.width - new_width) // 2
                img = img.crop((offset, 0, offset + new_width, img.height))
            elif img_ratio < target_ratio:
                new_height = int(img.width / target_ratio)
                offset = (img.height - new_height) // 2
                img = img.crop((0, offset, img.width, offset + new_height))
                
            final_img = img.resize(CoreConfig.RESOLUTION, Image.Resampling.LANCZOS)
            final_img.save(output_path, format="JPEG", quality=95)
            return True
        except Exception as e:
            Logger.error("Image processing pipeline crashed", e)
            return False

    @staticmethod
    def fetch_image(context_text, output_path):
        if not CoreConfig.HF_TOKEN:
            Logger.warn("No HuggingFace token provided. Bypassing cloud vision.")
            CloudVisionEngine._create_failsafe_image(output_path)
            return

        headers = {"Authorization": f"Bearer {CoreConfig.HF_TOKEN}"}
        prompt = f"{CURRENT_STYLE}, visual representation of: {context_text}, highly detailed, masterpiece, no text"
        payload = {"inputs": prompt, "parameters": {"guidance_scale": 7.5}}

        for attempt in range(4):
            try:
                response = requests.post(CoreConfig.HF_API_URL, headers=headers, json=payload, timeout=60)
                if response.status_code == 200:
                    if CloudVisionEngine._process_and_save_image(response.content, output_path):
                        time.sleep(2)
                        return
                elif response.status_code == 429:
                    Logger.warn("Vision API Rate Limit. Resting for 15s...")
                    time.sleep(15)
            except Exception as e:
                pass
            time.sleep(5)
            
        Logger.warn(f"Cloud Vision completely failed for scene. Using Failsafe.")
        CloudVisionEngine._create_failsafe_image(output_path)

# ==============================================================================
# 6. HIGH-FIDELITY TTS ENGINE (LOCAL)
# ==============================================================================

class LocalAudioEngine:
    def __init__(self):
        Logger.info("Initializing Kokoro ONNX Neural Voice Engine...")
        try:
            self.model = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
            Logger.success("Neural Voice Engine Online.")
        except Exception as e:
            Logger.error("Failed to load Kokoro Engine.", e)
            sys.exit(1)

    def synthesize_speech(self, text, output_path):
        clean_text = TextProcessor.clean_for_speech(text)
        if not clean_text:
            clean_text = "Moving on to the next point."
            
        try:
            samples, sample_rate = self.model.create(clean_text, voice="am_michael", speed=1.0, lang="en-us")
            sf.write(output_path, samples, sample_rate)
            duration = len(samples) / sample_rate
            return duration, clean_text
        except Exception as e:
            Logger.warn("Standard synthesis failed. Running emergency fallback synthesis.")
            safe_text = re.sub(r'[^a-zA-Z0-9\s\.,]', '', clean_text)
            try:
                samples, sample_rate = self.model.create(safe_text, voice="am_michael", speed=1.0, lang="en-us")
                sf.write(output_path, samples, sample_rate)
                return len(samples) / sample_rate, safe_text
            except Exception as critical_e:
                Logger.error("Total Synthesis Failure.", critical_e)
                return 0.0, ""

# ==============================================================================
# 7. ADVANCED SUBTITLE ENGINE (KINETIC POP & CHARACTER-WEIGHTED SYNC)
# ==============================================================================

class SubtitleEngine:
    def __init__(self):
        self.events = []
        
    @staticmethod
    def _format_time(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        cs = int(round((seconds - int(seconds)) * 100))
        if cs == 100:
            cs = 99
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

    def build_scene_subtitles(self, words, scene_duration, scene_start_time):
        if not words: return
        total_chars = sum(len(w) for w in words)
        time_per_char = scene_duration / max(total_chars, 1)
        current_word_start = scene_start_time
        
        for word in words:
            word_duration = len(word) * time_per_char
            word_end = current_word_start + word_duration
            clean_word = word.replace('"', '').replace("'", "\\'")
            start_str = self._format_time(current_word_start)
            end_str = self._format_time(word_end)
            
            pop_anim = r"{\fscx70\fscy70\t(0,150,\fscx100\fscy100)}"
            event = f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{pop_anim}{clean_word}"
            
            self.events.append(event)
            current_word_start = word_end

    def generate_ass_file(self, filepath):
        header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 720
WrapStyle: 1

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{CoreConfig.SUB_FONT},{CoreConfig.SUB_SIZE},{CoreConfig.SUB_PRIMARY_COLOR},&H000000FF,{CoreConfig.SUB_OUTLINE_COLOR},&H80000000,-1,0,0,0,100,100,0,0,1,3,2,2,10,10,70,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(header)
                f.write("\n".join(self.events))
            Logger.success(f"Advanced Animated Subtitles written to {filepath}")
        except Exception as e:
            Logger.error("Failed to generate ASS file.", e)

# ==============================================================================
# 8. FFMPEG RENDER ENGINE (MEMORY SAFE & BLACK-SCREEN PROOF)
# ==============================================================================

class RenderEngine:
    @staticmethod
    def render_scene_chunk(image_path, audio_path, output_path, duration):
        cmd = [
            "ffmpeg", "-y", 
            "-loop", "1", 
            "-framerate", str(CoreConfig.FPS),
            "-i", image_path,
            "-i", audio_path,
            "-vf", f"format=yuv420p,scale=8000:-1,zoompan=z='min(zoom+0.0005,1.15)':d={int(duration*CoreConfig.FPS)+5}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1280x720,fps={CoreConfig.FPS}",
            "-c:v", "libx264", "-preset", "ultrafast",
            "-c:a", "aac", "-b:a", "128k",
            "-t", str(duration),
            output_path
        ]
        
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return os.path.exists(output_path)
        except subprocess.CalledProcessError as e:
            Logger.error(f"FFmpeg chunk failure for {output_path}", e)
            return False

    @staticmethod
    def execute_concat(concat_file, output_path):
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", concat_file, "-c", "copy", output_path
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            Logger.success("Chunks assembled successfully.")
        except Exception as e:
            Logger.error("Concatenation failed.", e)

    @staticmethod
    def burn_kinetic_subtitles(input_video, ass_file, output_video):
        cmd = [
            "ffmpeg", "-y",
            "-i", input_video,
            "-vf", f"ass='{ass_file}'",
            "-c:a", "copy",
            output_video
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            Logger.success("Kinetic Subtitles burnt successfully.")
        except Exception as e:
            Logger.error("Subtitle burning failed.", e)

    @staticmethod
    def generate_youtube_thumbnail(image_path, output_path, episode_num):
        try:
            img = Image.open(image_path).convert("RGBA")
            overlay = Image.new('RGBA', img.size, (0, 0, 0, 100))
            img = Image.alpha_composite(img, overlay).convert("RGB")
            draw = ImageDraw.Draw(img)
            text = f"FINANCE\nMASTERCLASS\nEP.{episode_num}"
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 100)
            except:
                font = ImageFont.load_default()
                
            draw.text((80, 80), text, fill=(255, 255, 0), font=font)
            img.save(output_path, format="JPEG", quality=100)
            Logger.success(f"YouTube Thumbnail created: {output_path}")
        except Exception as e:
            Logger.warn(f"Failed to generate thumbnail: {e}")

# ==============================================================================
# 9. PIPELINE ORCHESTRATOR (THE MAIN BRAIN)
# ==============================================================================

class MasterclassPipeline:
    def __init__(self):
        Logger.info(f"--- BOOTING PIPELINE ENGINE FOR EPISODE {CURRENT_EPISODE} ---")
        self.audio_engine = LocalAudioEngine()
        self.sub_engine = SubtitleEngine()
        
    def run(self):
        topics = [
            "the dark reality of debt and how to escape the financial rat race forever",
            "the hidden psychology of money and how rich people truly build lasting wealth",
            "why saving your money keeps you poor and the exact blueprint to start investing"
        ]
        chosen_topic = random.choice(topics)
        Logger.info(f"Target Subject: {chosen_topic}")

        # --- PHASE 1: INTELLIGENT SCRIPTING ---
        Logger.info("Initiating Deep Script Generation...")
        full_script = ""
        for i in range(1, CoreConfig.CHAPTERS + 1):
            Logger.info(f"Drafting Chapter {i}/{CoreConfig.CHAPTERS}...")
            full_script += LLMEngine.generate_chapter(i, CoreConfig.CHAPTERS, chosen_topic) + " "

        sentences = TextProcessor.chunk_into_sentences(full_script)
        total_scenes = len(sentences)
        Logger.info(f"Script Finalized. Total Processing Scenes: {total_scenes}")

        # --- PHASE 2: ASSET GENERATION & TIMELINE ASSEMBLY ---
        timeline_position = 0.0
        valid_chunks = []
        thumbnail_base_image = None
        
        for idx, sentence in enumerate(sentences):
            scene_id = f"{idx:04d}"
            wav_path = os.path.join(CoreConfig.AUDIO_DIR, f"scene_{scene_id}.wav")
            jpg_path = os.path.join(CoreConfig.IMAGE_DIR, f"scene_{scene_id}.jpg")
            mp4_path = os.path.join(CoreConfig.SCENES_DIR, f"scene_{scene_id}.mp4")

            # 1. Synthesize Audio & Get Real Text Used
            duration, spoken_text = self.audio_engine.synthesize_speech(sentence, wav_path)
            if duration < 0.5:
                continue

            # 2. Generate Contextual Vision
            vision_context = spoken_text[:80]
            CloudVisionEngine.fetch_image(vision_context, jpg_path)
            
            if not thumbnail_base_image and os.path.exists(jpg_path):
                thumbnail_base_image = jpg_path

            # 3. Synchronize Subtitles (Character-Weighted)
            words = spoken_text.split()
            self.sub_engine.build_scene_subtitles(words, duration, timeline_position)

            # 4. Render Hardware-Safe Chunk
            if RenderEngine.render_scene_chunk(jpg_path, wav_path, mp4_path, duration):
                valid_chunks.append(f"file 'scenes/scene_{scene_id}.mp4'")
                timeline_position += duration

            if (idx + 1) % 10 == 0 or (idx + 1) == total_scenes:
                percent = int(((idx + 1) / total_scenes) * 100)
                Logger.info(f"Rendering Progress: {percent}% ({idx+1}/{total_scenes})")

        # --- PHASE 3: MASTER CONCATENATION ---
        Logger.info("Initiating Final Master Assembly...")
        concat_file = os.path.join(CoreConfig.OUTPUT_DIR, "concat.txt")
        with open(concat_file, "w") as f:
            f.write("\n".join(valid_chunks))

        ass_file = os.path.join(CoreConfig.OUTPUT_DIR, "master_subs.ass")
        self.sub_engine.generate_ass_file(ass_file)

        raw_vid = os.path.join(CoreConfig.OUTPUT_DIR, f"raw_vid_{CURRENT_EPISODE}.mp4")
        final_vid = os.path.join(CoreConfig.OUTPUT_DIR, f"final_video_{CURRENT_EPISODE}.mp4")

                RenderEngine.execute_concat(concat_file, raw_vid)
        RenderEngine.burn_kinetic_subtitles(raw_vid, ass_file, final_vid)

        # --- PHASE 4: METADATA & EXTRAS ---
        Logger.info("Compiling SEO Metadata & Thumbnail...")
        seo_text = LLMEngine.generate_seo_metadata(chosen_topic)
        with open(os.path.join(CoreConfig.OUTPUT_DIR, f"seo_metadata_{CURRENT_EPISODE}.txt"), "w", encoding="utf-8") as f:
            f.write(seo_text)
            
        if thumbnail_base_image:
            thumb_path = os.path.join(CoreConfig.OUTPUT_DIR, f"thumbnail_{CURRENT_EPISODE}.jpg")
            RenderEngine.generate_youtube_thumbnail(thumbnail_base_image, thumb_path, CURRENT_EPISODE)

        Logger.success(f"PIPELINE COMPLETE! Episode {CURRENT_EPISODE} successfully constructed.")

if __name__ == "__main__":
    try:
        pipeline = MasterclassPipeline()
        pipeline.run()
    except Exception as fatal_error:
        Logger.error("CRITICAL PIPELINE FAILURE", fatal_error)
        sys.exit(1)
