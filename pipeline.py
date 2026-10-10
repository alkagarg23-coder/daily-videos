#!/usr/bin/env python3
"""100% local finance documentary pipeline (Ollama + Kokoro ONNX + stable-diffusion.cpp + FFmpeg)."""

import datetime
import gc
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import requests
import soundfile as sf
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps, ImageStat

# --------------------------------------------------------------------------- #
# Paths and configuration
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work"
OUT = ROOT / "output"
IMG_DIR = WORK / "images"
AUDIO_DIR = WORK / "audio"
CHUNK_DIR = WORK / "chunks"

SD_BIN = ROOT / "sd"
SD_MODEL = ROOT / "model.safetensors"
KOKORO_MODEL = ROOT / "kokoro-v1.0.onnx"
KOKORO_VOICES = ROOT / "voices-v1.0.bin"
LORA_DIR = ROOT / "loras"
LORA_NAME = "lcm-lora-sdv1.5"

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "gemma2:2b"

CHAPTERS = 8
CHAPTER_WORDS = int(os.environ.get("CHAPTER_WORDS", "600"))
MAX_PARTS_PER_CHAPTER = int(os.environ.get("MAX_PARTS_PER_CHAPTER", "6"))

VOICE = "am_michael"
SPEED = 1.0
SCENE_TARGET_SEC = 10.0
SCENE_MIN_TAIL_SEC = 6.0
MIN_SCENE_SEC = 0.5
SENTENCE_PAD_SEC = 0.18
SILENCE_THRESHOLD = 0.01
SILENCE_KEEP_SEC = 0.04

SD_TIMEOUT = 350
# Keep SD sizes multiples of 64 (576x320 is close to 16:9 and safe for SD 1.5).
SD_WIDTH = int(os.environ.get("SD_WIDTH", "576"))
SD_HEIGHT = int(os.environ.get("SD_HEIGHT", "320"))
SD_STEPS = os.environ.get("SD_STEPS", "5")
SD_CFG = os.environ.get("SD_CFG", "1.5")
# Minutes (measured from pipeline start) after which no new images are generated.
IMAGE_DEADLINE_MIN = int(os.environ.get("IMAGE_DEADLINE_MIN", "230"))

W, H = 1280, 720
# Scene images are stored at 2x the video size so zoompan can move without visible jitter.
SRC_W, SRC_H = W * 2, H * 2
FPS = 15

# Fade at the start and end of every scene (seconds).
FADE_SEC = 0.25
# Total zoom gained in a zoom scene, and the fixed zoom used while panning.
ZOOM_RANGE = 0.18
PAN_ZOOM = 1.15
# Number of words shown together in each subtitle group.
WORDS_PER_GROUP = 4

SD_PROMPT_PREFIX = (
    "masterpiece, ultra-detailed black and white stickman illustration on a clean whiteboard. "
    "professional youtube explainer animation style. "
)
SD_NEGATIVE = "text, letters, watermark, blurry, deformed, messy lines"

PUNCT_WEIGHT = ",.;:!?"
POP_TAG = r"{\fscx70\fscy70\t(0,100,\fscx100\fscy100)}"

START = time.time()

TOPICS = [
    "how inflation silently destroys your savings",
    "why most people never build real wealth",
    "how credit cards and debt keep people trapped",
    "the truth about index funds and compound interest",
    "how banks really make money from your money",
    "the psychology of money and spending habits",
    "how hidden fees and taxes drain your retirement",
    "why the middle class keeps falling behind",
    "the biggest money mistakes people make in their twenties and thirties",
    "how the wealthy actually think about money",
]

DEFAULT_TITLES = [
    "The Truth They Hide From You",
    "The Hidden Problem",
    "How The System Really Works",
    "The Biggest Mistakes People Make",
    "What The Wealthy Do Differently",
    "The Math That Changes Everything",
    "Your Step By Step Action Plan",
    "The Final Truth",
]

STYLE_RULES = (
    "You are a world-class YouTube finance documentary scriptwriter. "
    "Write ONLY the spoken narration as plain flowing paragraphs. "
    "No headings, no bullet points, no markdown, no emojis, no stage directions, no speaker labels, "
    "and never write the word 'Chapter'. Mix short punchy sentences with longer ones and speak directly to "
    "the viewer using 'you'. Do not invent precise statistics or cite made-up studies, and do not give "
    "personalized investment advice. Keep it educational."
)

OUTRO_TEXT = (
    "If this changed the way you think about money, subscribe and watch the next video. "
    "Everything in this video is for educational purposes only and is not financial advice."
)


def log(message):
    print(f"[{(time.time() - START) / 60.0:7.1f} min] {message}", flush=True)


# --------------------------------------------------------------------------- #
# Ollama helpers
# --------------------------------------------------------------------------- #
def llm(prompt, num_predict=450, temperature=0.8, retries=3):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "30m",
        "options": {
            "temperature": temperature,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "num_predict": num_predict,
            "num_ctx": 4096,
        },
    }
    for attempt in range(1, retries + 1):
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=1500)
            response.raise_for_status()
            text = response.json().get("response", "").strip()
            if text:
                return text
            log(f"LLM returned empty text (attempt {attempt})")
        except Exception as exc:
            log(f"LLM request failed (attempt {attempt}): {exc}")
        time.sleep(5)
    return ""


def unload_llm():
    try:
        requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": "", "keep_alive": 0, "stream": False},
            timeout=60,
        )
        log("Ollama model unloaded to free RAM")
    except Exception as exc:
        log(f"Could not unload Ollama model: {exc}")


def word_count(text):
    return len(text.split())


def _dollars(match):
    number = match.group(1).rstrip(".,")
    unit = match.group(2)
    return f"{number} {unit} dollars" if unit else f"{number} dollars"


def clean_narration(text):
    text = text.replace("\r", "")
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    kept = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if re.match(r"^[-*_=#\s]{3,}$", line):
            continue
        if re.match(r"^(okay|ok|sure|certainly|here is|here's|here are)\b.*[:!]$", line, re.I):
            continue
        line = re.sub(r"^\s*(?:[-\u2022*]|\d+[.)])\s+", "", line)
        line = re.sub(r"^(chapter|narrator|voiceover|voice over|scene)\b[^:]{0,70}:\s*", "", line, flags=re.I)
        kept.append(line)
    text = " ".join(kept)
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\[[^\]]*\]", " ", text)
    replacements = {
        "\u2019": "'",
        "\u2018": "'",
        "\u201c": "",
        "\u201d": "",
        "\u2026": "...",
        "\u2014": ", ",
        "\u2013": ", ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"[*_#`>~|\"]", "", text)
    text = re.sub(
        r"\$\s?(\d[\d,.]*)\s*(thousand|million|billion|trillion)?",
        _dollars,
        text,
        flags=re.I,
    )
    text = text.replace("%", " percent").replace("&", " and ")
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --------------------------------------------------------------------------- #
# Stage 1: script + SEO
# --------------------------------------------------------------------------- #
def get_chapter_titles(topic):
    prompt = (
        f"List exactly {CHAPTERS} short chapter titles (maximum 6 words each) for a 30-minute YouTube "
        f"finance documentary about: {topic}. Output only the {CHAPTERS} titles, one per line, numbered."
    )
    raw = llm(prompt, num_predict=200, temperature=0.7)
    titles = []
    for line in raw.split("\n"):
        line = line.strip()
        line = re.sub(r"^\s*(?:[-\u2022*]|\d+[.):])\s*", "", line)
        line = re.sub(r"[*_#`\"]", "", line)
        line = re.sub(r"^(chapter|part)\s*\d+\s*[:\-]?\s*", "", line, flags=re.I).strip()
        if 3 <= len(line) <= 70 and not line.endswith(":"):
            titles.append(line)
    if len(titles) < CHAPTERS:
        titles = titles + DEFAULT_TITLES[len(titles):]
    return titles[:CHAPTERS]


def generate_chapter(topic, titles, index):
    title = titles[index]
    parts = []
    for part in range(MAX_PARTS_PER_CHAPTER):
        if part == 0 and index == 0:
            prompt = (
                f"{STYLE_RULES}\n\n"
                f"Write the OPENING HOOK of a 30-minute YouTube documentary about: {topic}.\n"
                "Make it extremely viral, controversial and suspenseful. Open with a shocking claim in the spirit of "
                "'99 percent of people are being lied to about money', then build tension with open loops: tease the "
                "secrets the viewer is about to discover and promise a big reveal later in the video. "
                "Use short punchy sentences and speak directly to 'you'. About 220 words. Output only the narration."
            )
        elif part == 0:
            prompt = (
                f"{STYLE_RULES}\n\n"
                f"Write the narration for part {index + 1} of {CHAPTERS} of a documentary about: {topic}. "
                f"This part is titled '{title}'. Start with a strong curiosity hook, then explain the idea with "
                "vivid everyday examples and simple analogies. About 220 words. Output only the narration."
            )
        else:
            tail = " ".join(" ".join(parts).split()[-120:])
            prompt = (
                f"{STYLE_RULES}\n\n"
                f"Documentary topic: {topic}. Current part: '{title}'.\n"
                f"Narration so far (end of it): {tail}\n\n"
                "Continue the narration smoothly with about 220 NEW words. Do not repeat earlier points, "
                "do not summarize, and do not conclude the video. Output only the new narration."
            )
        text = clean_narration(llm(prompt))
        if word_count(text) < 20:
            log(f"Chapter {index + 1} part {part + 1} was too short, skipping")
            continue
        parts.append(text)
        total_words = word_count(" ".join(parts))
        log(f"Chapter {index + 1}/{CHAPTERS} part {part + 1}: {total_words} words so far")
        if total_words >= CHAPTER_WORDS:
            break
    chapter_text = " ".join(parts)
    if index == CHAPTERS - 1:
        chapter_text = (chapter_text + " " + OUTRO_TEXT).strip()
    return chapter_text


def generate_script():
    topic = TOPICS[datetime.date.today().toordinal() % len(TOPICS)]
    log(f"Topic: {topic}")
    titles = get_chapter_titles(topic)
    log("Chapter titles: " + " | ".join(titles))
    chapters = []
    for index in range(CHAPTERS):
        text = generate_chapter(topic, titles, index)
        chapters.append({"title": titles[index], "text": text})
    total_words = sum(word_count(c["text"]) for c in chapters)
    log(f"Script complete: {total_words} words")
    if total_words < 300:
        raise RuntimeError("LLM produced too little script text; aborting.")
    (WORK / "script.json").write_text(json.dumps({"topic": topic, "chapters": chapters}, indent=2), encoding="utf-8")
    return topic, chapters


def generate_seo(topic, chapters):
    outline = "; ".join(c["title"] for c in chapters)
    prompt = (
        "You are a viral YouTube growth expert. "
        f"Video topic: {topic}. Chapters: {outline}.\n"
        "Write an ultra-clickbait YouTube title, a 500-character highly engaging description packed with emojis "
        "(🚨🔥💰), and 20 viral tags. Reply in EXACTLY this format and nothing else:\n"
        "TITLE: <title, max 90 characters>\n"
        "DESCRIPTION: <description>\n"
        "TAGS: <20 tags separated by commas>"
    )
    raw = llm(prompt, num_predict=500, temperature=0.9)

    title_match = re.search(r"TITLE:\s*(.+)", raw)
    desc_match = re.search(r"DESCRIPTION:\s*(.+?)(?=\n\s*TAGS:|\Z)", raw, re.S)
    tags_match = re.search(r"TAGS:\s*(.+)", raw, re.S)

    title = title_match.group(1).strip() if title_match else ""
    title = re.sub(r"[*_#`\"]", "", title).strip()
    if len(title) < 10:
        title = f"🚨 The Shocking Truth About {topic.title()} (Nobody Tells You This)"
    title = title[:100]

    description = desc_match.group(1).strip() if desc_match else ""
    description = re.sub(r"[*_#`]", "", description)
    description = re.sub(r"\s+", " ", description).strip()
    if len(description) < 80:
        description = (
            f"🚨 Everything you were never taught about {topic}. 🔥 In this deep-dive documentary we expose how the "
            "system really works, the mistakes that quietly cost people years of progress, and the simple steps "
            "you can start using today. 💰 Watch until the end for the final twist, and subscribe so you never "
            "miss the next one!"
        )
    if len(description) > 500:
        description = description[:500].rsplit(" ", 1)[0]
    description += "\n\n⚠️ Educational content only. This is not financial advice."

    tags = []
    if tags_match:
        for tag in re.split(r"[,\n]", tags_match.group(1)):
            tag = re.sub(r"[#*_`\"]", "", tag).strip()
            if 2 <= len(tag) <= 40 and tag.lower() not in [t.lower() for t in tags]:
                tags.append(tag)
    fallback_tags = [
        "finance", "money", "investing", "personal finance", "wealth", "financial freedom", "saving money",
        "budgeting", "passive income", "stock market", "inflation", "debt", "retirement", "banking",
        "money mistakes", "financial education", "compound interest", "index funds", "credit score",
        "money tips",
    ]
    for tag in fallback_tags:
        if len(tags) >= 20:
            break
        if tag.lower() not in [t.lower() for t in tags]:
            tags.append(tag)
    tags = tags[:20]
    while len(", ".join(tags)) > 480 and len(tags) > 5:
        tags.pop()

    payload = f"TITLE:\n{title}\n\nDESCRIPTION:\n{description}\n\nTAGS:\n{', '.join(tags)}\n"
    (OUT / "seo_metadata.txt").write_text(payload, encoding="utf-8")
    log("SEO metadata saved")


# --------------------------------------------------------------------------- #
# Stage 2: TTS
# --------------------------------------------------------------------------- #
def split_long(sentence, limit=220):
    if len(sentence) <= limit:
        return [sentence]
    pieces, current = [], ""
    for chunk in re.split(r"(?<=[,;:])\s+", sentence):
        if current and len(current) + len(chunk) + 1 > limit:
            pieces.append(current.strip())
            current = chunk
        else:
            current = f"{current} {chunk}".strip()
    if current:
        pieces.append(current.strip())
    final = []
    for piece in pieces:
        while len(piece) > limit * 1.5:
            cut = piece.rfind(" ", 0, limit)
            if cut <= 0:
                cut = limit
            final.append(piece[:cut].strip())
            piece = piece[cut:].strip()
        if piece:
            final.append(piece)
    return final


def split_sentences(text):
    sentences = []
    for raw in re.split(r"(?<=[.!?])\s+", text):
        raw = raw.strip()
        if not raw:
            continue
        for piece in split_long(raw):
            if re.search(r"[A-Za-z0-9]", piece):
                sentences.append(piece)
    return sentences


def synth(tts, text):
    try:
        samples, sample_rate = tts.create(text, voice=VOICE, speed=SPEED, lang="en-us")
        return np.asarray(samples, dtype=np.float32).flatten(), int(sample_rate)
    except Exception as exc:
        log(f"TTS failed on sentence ({exc}); retrying with cleaned text")
    cleaned = re.sub(r"[^A-Za-z0-9\s.,!?'\-]", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not re.search(r"[A-Za-z0-9]", cleaned):
        return None
    try:
        samples, sample_rate = tts.create(cleaned, voice=VOICE, speed=SPEED, lang="en-us")
        return np.asarray(samples, dtype=np.float32).flatten(), int(sample_rate)
    except Exception as exc:
        log(f"TTS fallback also failed, skipping sentence: {exc}")
        return None


def trim_silence(samples, sr):
    """Cut leading/trailing silence so subtitle word timing is spread over real speech only."""
    loud = np.where(np.abs(samples) > SILENCE_THRESHOLD)[0]
    if loud.size == 0:
        return samples
    keep = int(SILENCE_KEEP_SEC * sr)
    start = max(int(loud[0]) - keep, 0)
    end = min(int(loud[-1]) + keep, len(samples))
    return samples[start:end]


def build_scenes(chapters):
    from kokoro_onnx import Kokoro

    tts = Kokoro(str(KOKORO_MODEL), str(KOKORO_VOICES))
    scenes = []
    sample_rate = None

    for chapter_index, chapter in enumerate(chapters):
        items = []
        for sentence in split_sentences(chapter["text"]):
            result = synth(tts, sentence)
            if result is None:
                continue
            samples, sr = result
            if sample_rate is None:
                sample_rate = sr
            if sr != sample_rate or samples.size < int(0.2 * sr):
                continue
            samples = trim_silence(samples, sr)
            if samples.size < int(0.2 * sr):
                continue
            padded = np.concatenate([samples, np.zeros(int(sr * SENTENCE_PAD_SEC), dtype=np.float32)])
            items.append(
                {
                    "text": sentence,
                    "samples": padded,
                    "dur": len(padded) / float(sr),
                    "speech": len(samples) / float(sr),
                }
            )

        groups, current, current_dur = [], [], 0.0
        for item in items:
            current.append(item)
            current_dur += item["dur"]
            if current_dur >= SCENE_TARGET_SEC:
                groups.append(current)
                current, current_dur = [], 0.0
        if current:
            if groups and current_dur < SCENE_MIN_TAIL_SEC:
                groups[-1].extend(current)
            else:
                groups.append(current)

        for group in groups:
            audio = np.concatenate([g["samples"] for g in group])
            duration = len(audio) / float(sample_rate)
            if duration < MIN_SCENE_SEC:
                continue
            scene_index = len(scenes)
            wav_path = AUDIO_DIR / f"scene_{scene_index:03d}.wav"
            sf.write(str(wav_path), audio, sample_rate, subtype="PCM_16")
            sentences, offset = [], 0.0
            for g in group:
                sentences.append({"text": g["text"], "start": offset, "dur": g["dur"], "speech": g["speech"]})
                offset += g["dur"]
            scenes.append(
                {
                    "index": scene_index,
                    "chapter": chapter_index,
                    "title": chapter["title"],
                    "text": " ".join(g["text"] for g in group),
                    "wav": wav_path,
                    "duration": duration,
                    "sentences": sentences,
                }
            )
        log(f"TTS chapter {chapter_index + 1}/{CHAPTERS} done, scenes so far: {len(scenes)}")
        del items, groups
        gc.collect()

    del tts
    gc.collect()
    total = sum(s["duration"] for s in scenes)
    log(f"TTS complete: {len(scenes)} scenes, {total / 60.0:.1f} minutes of audio")
    if not scenes:
        raise RuntimeError("No audio scenes were generated.")
    return scenes


# --------------------------------------------------------------------------- #
# Stage 3: images
# --------------------------------------------------------------------------- #
def load_font(size, bold=True):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


def image_is_valid(path):
    try:
        img = Image.open(path).convert("RGB").resize((64, 64))
        stat = ImageStat.Stat(img)
        mean = sum(stat.mean) / 3.0
        std = sum(stat.stddev) / 3.0
        return mean > 20 and std > 4
    except Exception:
        return False


def to_zoom_source(img):
    """Cover-crop to 16:9, upscale to SRC_W x SRC_H (LANCZOS), then clean up the line art."""
    img = img.convert("RGB")
    width, height = img.size
    scale = max(SRC_W / width, SRC_H / height)
    resized = img.resize((round(width * scale), round(height * scale)), Image.Resampling.LANCZOS)
    left = (resized.width - SRC_W) // 2
    top = (resized.height - SRC_H) // 2
    frame = resized.crop((left, top, left + SRC_W, top + SRC_H))
    frame = ImageOps.autocontrast(frame, cutoff=1)
    return frame.filter(ImageFilter.UnsharpMask(radius=2, percent=110, threshold=3))


def make_fallback_image(title):
    img = Image.new("RGB", (W, H), (250, 250, 250))
    draw = ImageDraw.Draw(img)
    for x in range(0, W, 80):
        draw.line([(x, 0), (x, H)], fill=(236, 236, 236), width=2)
    for y in range(0, H, 80):
        draw.line([(0, y), (W, y)], fill=(236, 236, 236), width=2)
    ink = (25, 25, 25)
    cx, cy = 640, 300
    draw.ellipse((cx - 45, cy - 150, cx + 45, cy - 60), outline=ink, width=8)
    draw.line((cx, cy - 60, cx, cy + 80), fill=ink, width=8)
    draw.line((cx, cy - 30, cx - 90, cy + 35), fill=ink, width=8)
    draw.line((cx, cy - 30, cx + 90, cy - 90), fill=ink, width=8)
    draw.line((cx, cy + 80, cx - 60, cy + 190), fill=ink, width=8)
    draw.line((cx, cy + 80, cx + 60, cy + 190), fill=ink, width=8)
    draw.ellipse((cx + 95, cy - 160, cx + 175, cy - 80), outline=ink, width=8)
    draw.text((cx + 135, cy - 120), "$", font=load_font(48), fill=ink, anchor="mm")
    font = load_font(54)
    words, lines, line = title.split(), [], ""
    for word in words:
        trial = f"{line} {word}".strip()
        if draw.textbbox((0, 0), trial, font=font)[2] > W - 160 and line:
            lines.append(line)
            line = word
        else:
            line = trial
    if line:
        lines.append(line)
    y = 530
    for text_line in lines[:2]:
        draw.text((W // 2, y), text_line, font=font, fill=ink, anchor="mm")
        y += 70
    return img


def scene_context(scene):
    first_sentence = scene["sentences"][0]["text"] if scene["sentences"] else scene["text"]
    first_words = " ".join(first_sentence.split()[:14])
    context = f"{scene['title']}, {first_words}"
    context = re.sub(r"[^A-Za-z0-9, ]", " ", context)
    return re.sub(r"\s+", " ", context).strip()


def build_sd_cmd(prompt, out_path, seed, width, height, steps):
    return [
        str(SD_BIN),
        "-m", str(SD_MODEL),
        "--lora-model-dir", str(LORA_DIR),
        "-p", f"{prompt} <lora:{LORA_NAME}:1>",
        "-n", SD_NEGATIVE,
        "--sampling-method", "lcm",
        "--steps", str(steps),
        "--cfg-scale", SD_CFG,
        "-W", str(width),
        "-H", str(height),
        "-t", "2",
        "-s", str(seed),
        "-o", str(out_path),
    ]


def sd_preflight():
    """Make one quick image first so a wrong sd flag or LoRA problem fails in a minute, not after hours."""
    test_path = IMG_DIR / "preflight.png"
    if test_path.exists():
        test_path.unlink()
    cmd = build_sd_cmd(SD_PROMPT_PREFIX + "a stickman holding a coin", test_path, 1234, SD_WIDTH, SD_HEIGHT, 2)
    started = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=SD_TIMEOUT, cwd=str(ROOT))
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"sd preflight timed out after {SD_TIMEOUT}s")
    if result.returncode != 0 or not test_path.exists():
        tail = (result.stderr or result.stdout or "")[-800:]
        raise RuntimeError(f"sd preflight failed (rc={result.returncode}): {tail}")
    test_path.unlink()
    log(f"sd preflight ok ({time.time() - started:.0f}s for a 2-step image)")


def generate_scene_image(prompt, index, base_seed):
    raw_path = IMG_DIR / f"raw_{index:03d}.png"
    for attempt in range(2):
        if raw_path.exists():
            raw_path.unlink()
        seed = base_seed + attempt * 7919
        cmd = build_sd_cmd(prompt, raw_path, seed, SD_WIDTH, SD_HEIGHT, SD_STEPS)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=350, cwd=str(ROOT))
        except subprocess.TimeoutExpired:
            log(f"Image {index}: sd timed out after {SD_TIMEOUT}s")
            return None, "timeout"
        if result.returncode == 0 and raw_path.exists() and image_is_valid(raw_path):
            return raw_path, "ok"
        tail = (result.stderr or result.stdout or "")[-300:].replace("\n", " ")
        log(f"Image {index}: attempt {attempt + 1} failed or came out black (rc={result.returncode}) {tail}")
    return None, "failed"


def build_images(scenes):
    deadline = START + IMAGE_DEADLINE_MIN * 60
    image_times = []
    credit = 0.0
    last_good = None
    reuse_count = 0
    generated = set()

    for i, scene in enumerate(scenes):
        out_path = IMG_DIR / f"scene_{i:03d}.jpg"
        remaining_scenes = len(scenes) - i
        time_left = deadline - time.time()
        generate = False

        if not image_times:
            generate = time_left > SD_TIMEOUT
        else:
            avg = sum(image_times) / len(image_times)
            if time_left > avg * 1.1:
                ratio = min(1.0, time_left / (avg * remaining_scenes))
                credit += ratio
                if credit >= 1.0:
                    generate = True
                    credit -= 1.0

        done = False
        if generate:
            prompt = SD_PROMPT_PREFIX + scene_context(scene)
            started = time.time()
            raw_path, status = generate_scene_image(prompt, i, random.randint(1, 2**31 - 1))
            elapsed = time.time() - started
            image_times.append(SD_TIMEOUT if status == "timeout" else elapsed)
            if raw_path is not None:
                to_zoom_source(Image.open(raw_path)).save(out_path, "JPEG", quality=92)
                last_good = out_path
                generated.add(i)
                done = True
                log(f"Image {i + 1}/{len(scenes)} generated in {elapsed:.0f}s")
            try:
                raw = IMG_DIR / f"raw_{i:03d}.png"
                if raw.exists():
                    raw.unlink()
            except Exception:
                pass

        if not done:
            if last_good is not None:
                frame = Image.open(last_good).convert("RGB")
                reuse_count += 1
                if reuse_count % 2 == 1:
                    frame = frame.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                frame.save(out_path, "JPEG", quality=92)
            else:
                to_zoom_source(make_fallback_image(scene["title"])).save(out_path, "JPEG", quality=92)
        scene["image"] = out_path
        scene["generated"] = i in generated

    log(f"Images ready: {len(generated)} generated, {len(scenes) - len(generated)} reused or fallback")
    return scenes


# --------------------------------------------------------------------------- #
# Stage 4: subtitles (punctuation-aware sync, 4 words per group)
# --------------------------------------------------------------------------- #
ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 720
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,65,&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,4,1,2,60,60,50,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def ass_time(seconds):
    cs = max(0, int(round(seconds * 100)))
    hours = cs // 360000
    minutes = (cs // 6000) % 60
    secs = (cs // 100) % 60
    centis = cs % 100
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def escape_ass(text):
    return text.replace("\\", "").replace("{", "").replace("}", "").replace("\n", " ")


def sentence_events(text, t0, t1, speech_end):
    words = text.split()
    if not words or t1 - t0 < 0.05:
        return []
    weights = [len(word) + (5 if word[-1] in PUNCT_WEIGHT else 0) for word in words]
    bounds = [0.0]
    for weight in weights:
        bounds.append(bounds[-1] + weight)
    total = float(bounds[-1])

    # Fixed groups of WORDS_PER_GROUP words; a lone trailing word joins the previous group.
    groups = [
        list(range(i, min(i + WORDS_PER_GROUP, len(words))))
        for i in range(0, len(words), WORDS_PER_GROUP)
    ]
    if len(groups) > 1 and len(groups[-1]) == 1:
        groups[-2].extend(groups.pop())

    # Spread the words over the real speech only (silence trimmed); the last group stays until the sentence ends.
    span = max(speech_end - t0, 0.05)
    events = []
    for number, group in enumerate(groups):
        start = t0 + span * bounds[group[0]] / total
        end = t0 + span * bounds[group[-1] + 1] / total
        if number == len(groups) - 1:
            end = max(end, t1)
        if end - start < 0.03:
            continue
        phrase = escape_ass(" ".join(words[p] for p in group))
        events.append(f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Default,,0,0,0,,{POP_TAG}{phrase}")
    return events


def build_ass(scenes):
    lines = [ASS_HEADER.rstrip("\n")]
    for scene in scenes:
        # Audio plays at real speed, so no time scaling: just clamp to the scene end.
        scene_end = scene["timeline_start"] + scene["timeline_dur"]
        for sentence in scene["sentences"]:
            t0 = scene["timeline_start"] + sentence["start"]
            t1 = min(t0 + sentence["dur"], scene_end)
            speech_end = min(t0 + sentence["speech"], t1)
            lines.extend(sentence_events(sentence["text"], t0, t1, speech_end))
    (WORK / "subs.ass").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("subs.ass written")


# --------------------------------------------------------------------------- #
# Stage 5: FFmpeg assembly
# --------------------------------------------------------------------------- #
def probe_duration(path):
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        capture_output=True,
        text=True,
    )
    try:
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def zoom_filter(index, frames):
    """Per-scene zoompan filter. Neighbouring scenes get different moves; the frame count is exact so audio sync holds."""
    n = max(int(frames), 2)
    move = index % 6
    zr = ZOOM_RANGE
    pz = PAN_ZOOM
    mid_y = "(ih-ih/zoom)*0.5"
    if move == 0:  # slow zoom in, centre
        z, x, y = f"1+{zr}*on/{n}", "(iw-iw/zoom)*0.5", mid_y
    elif move == 1:  # pan left to right
        z, x, y = f"{pz}", f"(iw-iw/zoom)*on/{n}", mid_y
    elif move == 2:  # slow zoom out, centre
        z, x, y = f"{1 + zr}-{zr}*on/{n}", "(iw-iw/zoom)*0.5", mid_y
    elif move == 3:  # pan right to left
        z, x, y = f"{pz}", f"(iw-iw/zoom)*(1-on/{n})", mid_y
    elif move == 4:  # zoom in towards the upper left
        z, x, y = f"1+{zr}*on/{n}", "(iw-iw/zoom)*0.2", "(ih-ih/zoom)*0.25"
    else:  # zoom in towards the lower right
        z, x, y = f"1+{zr}*on/{n}", "(iw-iw/zoom)*0.8", "(ih-ih/zoom)*0.75"
    return f"zoompan=z='{z}':x='{x}':y='{y}':d={n}:s={W}x{H}:fps={FPS}"


def assemble(scenes):
    valid = []
    cursor = 0.0
    for scene in scenes:
        chunk = CHUNK_DIR / f"chunk_{scene['index']:03d}.mp4"

        # zoompan makes the frames itself from ONE input frame (no -loop, or the frame count would multiply).
        # The count is derived from the audio length, then -t / -shortest trim to the exact audio duration.
        frames = int(np.ceil(scene["duration"] * FPS)) + 4
        fade_out_start = max(scene["duration"] - FADE_SEC, 0.0)
        video_filter = (
            f"{zoom_filter(scene['index'], frames)},"
            "format=yuv420p,"
            f"fade=t=in:st=0:d={FADE_SEC},"
            f"fade=t=out:st={fade_out_start:.2f}:d={FADE_SEC}"
        )

        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(scene["image"]),
            "-i", str(scene["wav"]),
            "-vf", video_filter,
            "-r", str(FPS),
            "-c:v", "libx264", "-preset", "ultrafast",
            "-c:a", "aac", "-b:a", "128k",
            "-t", f"{scene['duration']:.3f}",
            "-shortest", str(chunk),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        actual = probe_duration(chunk) if chunk.exists() else 0.0
        if result.returncode != 0 or actual < MIN_SCENE_SEC:
            log(f"Chunk {scene['index']} failed, dropping scene: {(result.stderr or '')[-200:]}")
            continue
        scene["chunk"] = chunk
        scene["timeline_start"] = cursor
        scene["timeline_dur"] = actual
        cursor += actual
        valid.append(scene)
        if scene["index"] % 10 == 0:
            log(f"Chunk {scene['index'] + 1}/{len(scenes)} encoded")

    if not valid:
        raise RuntimeError("No video chunks could be encoded.")

    concat_file = WORK / "concat.txt"
    concat_file.write_text(
        "".join(f"file '{scene['chunk'].resolve()}'\n" for scene in valid),
        encoding="utf-8",
    )
    merged = WORK / "merged.mp4"
    result = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "concat", "-safe", "0", "-i", str(concat_file),
            "-c", "copy", str(merged),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not merged.exists():
        raise RuntimeError(f"Concat failed: {result.stderr[-500:]}")
    log(f"Concatenated {len(valid)} chunks, total {cursor / 60.0:.1f} minutes")

    build_ass(valid)

    final_path = OUT / "final_video.mp4"
    result = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", "merged.mp4",
            "-vf", "ass=subs.ass",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "24", "-pix_fmt", "yuv420p",
            "-c:a", "copy", "-movflags", "+faststart",
            str(final_path),
        ],
        capture_output=True,
        text=True,
        cwd=str(WORK),
    )
    if result.returncode != 0 or not final_path.exists():
        raise RuntimeError(f"Subtitle burn-in failed: {result.stderr[-500:]}")
    size_mb = final_path.stat().st_size / (1024 * 1024)
    log(f"Final video written: {size_mb:.0f} MB")
    return valid


# --------------------------------------------------------------------------- #
# Stage 6: thumbnail
# --------------------------------------------------------------------------- #
def make_thumbnail(scenes):
    source = None
    if len(scenes) > 2 and scenes[2].get("generated"):
        source = scenes[2]["image"]
    else:
        for scene in scenes:
            if scene.get("generated"):
                source = scene["image"]
                break
        if source is None:
            source = scenes[min(2, len(scenes) - 1)]["image"]

    base = Image.open(source).convert("RGB").resize((W, H), Image.Resampling.LANCZOS).convert("RGBA")

    grad_w = int(W * 0.6)
    ramp = np.linspace(215, 0, grad_w).astype(np.uint8)
    alpha = np.zeros((H, W), dtype=np.uint8)
    alpha[:, :grad_w] = ramp[None, :]
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    overlay.putalpha(Image.fromarray(alpha))
    composed = Image.alpha_composite(base, overlay)

    draw = ImageDraw.Draw(composed)
    big = load_font(95)
    small = load_font(65)
    draw.text((50, 280), "STOP", font=big, fill="#EF4444", stroke_width=5, stroke_fill="black")
    draw.text((50, 390), "DOING THIS", font=big, fill="#FACC15", stroke_width=5, stroke_fill="black")
    draw.text((50, 500), "FINANCE SECRETS", font=small, fill="#FFFFFF", stroke_width=4, stroke_fill="black")

    composed.convert("RGB").save(OUT / "thumbnail.jpg", "JPEG", quality=92, optimize=True)
    log("Thumbnail saved")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def check_prerequisites():
    lora_file = LORA_DIR / f"{LORA_NAME}.safetensors"
    missing = [p.name for p in (SD_BIN, SD_MODEL, KOKORO_MODEL, KOKORO_VOICES, lora_file) if not p.exists()]
    if missing:
        raise RuntimeError(f"Missing required files: {', '.join(missing)}")


def main():
    for directory in (WORK, OUT, IMG_DIR, AUDIO_DIR, CHUNK_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    check_prerequisites()
    sd_preflight()

    log("Stage 1/6: script and SEO")
    topic, chapters = generate_script()
    generate_seo(topic, chapters)
    unload_llm()
    gc.collect()

    log("Stage 2/6: text to speech")
    scenes = build_scenes(chapters)

    log("Stage 3/6: image generation")
    scenes = build_images(scenes)

    log("Stage 4-5/6: FFmpeg assembly and subtitles")
    scenes = assemble(scenes)

    log("Stage 6/6: thumbnail")
    make_thumbnail(scenes)

    log("Pipeline finished successfully")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        log(f"FATAL: {error}")
        sys.exit(1)
