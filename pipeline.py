#!/usr/bin/env python3
"""
Stickman Finance YouTube Documentary Pipeline
Local AI only: Ollama (gemma2:2b) + SDXL-Turbo + Kokoro (am_puck)
Test Run = exactly 12 scenes / ~60s | Full Run = target 360 scenes / ~30min
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import subprocess
import sys
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont
from tenacity import retry, stop_after_attempt, wait_exponential

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(level: str = "INFO", log_file: Optional[Path] = None) -> logging.Logger:
    logger = logging.getLogger("stickman_finance")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger

log = setup_logging()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class Config:
    mode: str = "test"                          # test | full
    topic: str = ""
    target_duration_sec: int = 60
    scene_interval_sec: float = 5.0
    image_width: int = 1920
    image_height: int = 1080
    thumb_width: int = 1280
    thumb_height: int = 720
    text_model: str = "gemma2:2b"
    image_model: str = "stabilityai/sdxl-turbo"
    kokoro_model_path: str = "models/kokoro-v1.0.onnx"
    kokoro_voices_path: str = "models/voices-v1.0.bin"
    kokoro_voice: str = "am_puck"
    sample_rate: int = 24000
    max_subtitle_words: int = 4
    output_dir: Path = Path("output")
    models_dir: Path = Path("models")
    enable_youtube: bool = False
    youtube_privacy: str = "unlisted"
    retry_count: int = 3
    log_level: str = "INFO"
    seed: Optional[int] = None
    force: bool = False

    def __post_init__(self):
        if self.mode == "test":
            self.target_duration_sec = 60
            self._expected_scenes = 12
        else:
            self.target_duration_sec = 1800
            self._expected_scenes = 360
        self.output_dir = Path(self.output_dir)
        self.models_dir = Path(self.models_dir)

    @property
    def expected_scenes(self) -> int:
        return getattr(self, "_expected_scenes", 12)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class Scene:
    scene_id: str
    scene_number: int
    narration_text: str
    image_prompt: str
    negative_prompt: str = "blurry, low quality, text, watermark, realistic photo, 3d render"
    start_time: float = 0.0
    end_time: float = 5.0
    image_path: Optional[str] = None
    seed: int = 0
    validated: bool = False
    hash: Optional[str] = None

@dataclass
class SubtitleCue:
    index: int
    start: float
    end: float
    text: str

@dataclass
class RunManifest:
    run_id: str
    mode: str
    topic: str
    video_slug: str
    created_at: str
    config: Dict[str, Any]
    script: str = ""
    scenes: List[Dict] = field(default_factory=list)
    audio_path: Optional[str] = None
    audio_duration: float = 0.0
    subtitle_ass: Optional[str] = None
    subtitle_srt: Optional[str] = None
    thumbnail_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    video_path: Optional[str] = None
    video_duration: float = 0.0
    youtube_video_id: Optional[str] = None
    status: str = "started"
    errors: List[str] = field(default_factory=list)
    checkpoints: Dict[str, bool] = field(default_factory=dict)

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def sanitize_slug(text: str, max_len: int = 60) -> str:
    text = re.sub(r"[^\w\s-]", "", text.lower())
    text = re.sub(r"[-\s]+", "-", text).strip("-")
    return text[:max_len] or f"video-{uuid.uuid4().hex[:8]}"

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def image_perceptual_hash(path: Path) -> str:
    try:
        import imagehash
        return str(imagehash.phash(Image.open(path)))
    except Exception:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

def check_disk_space(min_gb: float = 5.0) -> Tuple[bool, float]:
    free_gb = psutil.disk_usage("/").free / (1 << 30)
    return free_gb >= min_gb, free_gb

def run_cmd(cmd: List[str], timeout: int = 300, check: bool = True) -> subprocess.CompletedProcess:
    log.debug("CMD: %s", " ".join(cmd))
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=check)

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------
def preflight(cfg: Config) -> None:
    log.info("=== PRE-FLIGHT CHECKS ===")
    if sys.version_info < (3, 10):
        raise RuntimeError(f"Python >= 3.10 required, found {sys.version}")
    log.info("Python: %s", sys.version.split()[0])

    ok, free = check_disk_space(8.0 if cfg.mode == "test" else 40.0)
    log.info("Free disk: %.1f GB", free)
    if not ok:
        raise RuntimeError(f"Insufficient disk space ({free:.1f} GB free)")

    try:
        r = run_cmd(["ffmpeg", "-version"], timeout=10)
        log.info("FFmpeg: %s", r.stdout.split("\n")[0])
    except Exception as e:
        raise RuntimeError(f"FFmpeg not available: {e}") from e

    ensure_dir(cfg.models_dir)
    ensure_dir(cfg.output_dir)
    log.info("Pre-flight OK")

# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=30))
def download_file(url: str, dest: Path, expected_min_bytes: int = 1000) -> None:
    import requests
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s -> %s", url, dest)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
        if tmp.stat().st_size < expected_min_bytes:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"Download too small ({tmp.stat().st_size} bytes)")
        tmp.rename(dest)
    log.info("Downloaded %s (%d bytes)", dest.name, dest.stat().st_size)

def ensure_kokoro_models(cfg: Config) -> None:
    model = Path(cfg.kokoro_model_path)
    voices = Path(cfg.kokoro_voices_path)
    if model.exists() and voices.exists() and model.stat().st_size > 50_000_000:
        return
    base = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
    if not model.exists() or model.stat().st_size < 50_000_000:
        download_file(f"{base}/kokoro-v1.0.onnx", model, 50_000_000)
    if not voices.exists() or voices.stat().st_size < 1_000_000:
        download_file(f"{base}/voices-v1.0.bin", voices, 1_000_000)

# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------
def ensure_ollama(cfg: Config) -> None:
    try:
        r = run_cmd(["ollama", "--version"], timeout=10, check=False)
        if r.returncode != 0:
            raise FileNotFoundError
        log.info("Ollama present: %s", r.stdout.strip())
    except Exception:
        log.info("Installing Ollama...")
        run_cmd(["bash", "-c", "curl -fsSL https://ollama.com/install.sh | sh"], timeout=180)

    try:
        import requests
        requests.get("http://127.0.0.1:11434/api/tags", timeout=3)
        log.info("Ollama already serving")
    except Exception:
        log.info("Starting Ollama serve...")
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        for _ in range(30):
            try:
                import requests
                requests.get("http://127.0.0.1:11434/api/tags", timeout=2)
                log.info("Ollama ready")
                break
            except Exception:
                time.sleep(2)
        else:
            raise RuntimeError("Ollama failed to start")

    log.info("Ensuring model %s ...", cfg.text_model)
    run_cmd(["ollama", "pull", cfg.text_model], timeout=600)

def ollama_generate(prompt: str, model: str, system: str = "", json_mode: bool = False) -> str:
    import ollama
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    kwargs = {"model": model, "messages": messages, "options": {"temperature": 0.7, "num_predict": 4096}}
    if json_mode:
        kwargs["format"] = "json"
    return ollama.chat(**kwargs)["message"]["content"].strip()

# ---------------------------------------------------------------------------
# Script
# ---------------------------------------------------------------------------
def generate_script(cfg: Config, topic: str) -> str:
    if cfg.mode == "test":
        structure = (
            "Write a complete 55-65 second English educational documentary script "
            "for a USA audience about the given personal-finance topic. "
            "Use a clear hook, one core concept, 2-3 practical points, and a short conclusion. "
            "No fabricated statistics. Natural spoken style. Output ONLY the spoken narration text."
        )
        word_target = 140
    else:
        structure = (
            "Write a complete ~30-minute English educational documentary script "
            "(target 4200-4500 spoken words) for a USA audience. "
            "Structure: strong hook, relatable problem, core concept & incentives, "
            "step-by-step explanation with examples, common mistakes/trade-offs, "
            "practical conclusion + soft CTA. Factually honest. No invented numbers or guarantees. "
            "Output ONLY the spoken narration text, no stage directions or chapter titles."
        )
        word_target = 4300

    system = (
        "You are an expert financial educator and documentary writer specializing in "
        "clear, engaging Stickman-Finance style narration for American viewers. "
        "Never invent statistics or promise returns."
    )
    prompt = f"{structure}\n\nTopic: {topic}\nTarget spoken words: ~{word_target}"
    log.info("Generating script with %s ...", cfg.text_model)
    script = ollama_generate(prompt, cfg.text_model, system=system)
    script = re.sub(r"\n{3,}", "\n\n", script).strip()
    log.info("Script generated: %d words", len(script.split()))
    return script

# ---------------------------------------------------------------------------
# Scene planning (dynamic – no hard 15-scene limit)
# ---------------------------------------------------------------------------
def plan_scenes(cfg: Config, script: str, audio_duration: float) -> List[Scene]:
    n_scenes = max(1, int(round(audio_duration / cfg.scene_interval_sec)))
    if cfg.mode == "test":
        n_scenes = 12
    log.info("Planning %d scenes for %.1f s audio", n_scenes, audio_duration)

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", script) if s.strip()] or [script]
    remaining = script.split()
    words_per_scene = max(1, len(remaining) // n_scenes)
    rng = random.Random(cfg.seed or int(time.time()))
    scenes: List[Scene] = []

    for i in range(n_scenes):
        start = i * cfg.scene_interval_sec
        end = min((i + 1) * cfg.scene_interval_sec, audio_duration)
        chunk_words = remaining[:words_per_scene] if i < n_scenes - 1 else remaining
        remaining = remaining[words_per_scene:]
        narr = " ".join(chunk_words) if chunk_words else sentences[i % len(sentences)]

        vis_prompt = (
            f"Simple stickman illustration, clean line art, white background, "
            f"educational finance style, illustrating: {narr[:120]}. "
            f"Stick figure character, clear metaphor, high contrast, 16:9 composition."
        )
        scenes.append(Scene(
            scene_id=f"s{i+1:04d}",
            scene_number=i + 1,
            narration_text=narr,
            image_prompt=vis_prompt,
            start_time=start,
            end_time=end,
            seed=rng.randint(0, 2**31 - 1),
        ))
    return scenes

# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------
_pipe = None

def get_image_pipeline(cfg: Config):
    global _pipe
    if _pipe is not None:
        return _pipe
    import torch
    from diffusers import AutoPipelineForText2Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    log.info("Loading SDXL-Turbo on %s (%s)...", device, dtype)
    _pipe = AutoPipelineForText2Image.from_pretrained(
        cfg.image_model,
        torch_dtype=dtype,
        variant="fp16" if device == "cuda" else None,
        cache_dir=str(cfg.models_dir / "hf"),
    ).to(device)
    if device == "cpu":
        try:
            _pipe.enable_attention_slicing()
        except Exception:
            pass
    log.info("Image pipeline ready")
    return _pipe

def generate_image(cfg: Config, scene: Scene, out_path: Path) -> Path:
    pipe = get_image_pipeline(cfg)
    import torch
    generator = torch.Generator(device=pipe.device).manual_seed(scene.seed)
    result = pipe(
        prompt=scene.image_prompt,
        negative_prompt=scene.negative_prompt,
        num_inference_steps=1 if cfg.mode == "test" else 2,
        guidance_scale=0.0,
        width=512,
        height=512,
        generator=generator,
    )
    img = result.images[0].convert("RGB")
    target = (cfg.image_width, cfg.image_height)
    img.thumbnail(target, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", target, (255, 255, 255))
    canvas.paste(img, ((target[0] - img.width) // 2, (target[1] - img.height) // 2))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, "PNG", optimize=True)
    scene.image_path = str(out_path)
    scene.hash = image_perceptual_hash(out_path)
    scene.validated = True
    return out_path

def generate_all_images(cfg: Config, scenes: List[Scene], run_dir: Path, manifest: RunManifest) -> None:
    scenes_dir = ensure_dir(run_dir / "scenes")
    seen_hashes: List[str] = []
    for i, scene in enumerate(scenes):
        out = scenes_dir / f"{scene.scene_id}.png"
        if out.exists() and out.stat().st_size > 1000 and not cfg.force:
            log.info("Scene %d/%d already exists, skipping", i + 1, len(scenes))
            scene.image_path = str(out)
            scene.validated = True
            scene.hash = image_perceptual_hash(out)
            continue
        log.info("Generating image %d/%d (seed=%d)", i + 1, len(scenes), scene.seed)
        for attempt in range(cfg.retry_count):
            try:
                generate_image(cfg, scene, out)
                if scene.hash and scene.hash in seen_hashes[-5:]:
                    log.warning("Possible duplicate, retrying with new seed")
                    scene.seed = random.randint(0, 2**31 - 1)
                    continue
                seen_hashes.append(scene.hash or "")
                break
            except Exception as e:
                log.error("Image gen failed (attempt %d): %s", attempt + 1, e)
                if attempt == cfg.retry_count - 1:
                    raise
                time.sleep(2 ** attempt)
        if (i + 1) % 5 == 0:
            manifest.scenes = [asdict(s) for s in scenes]
            save_manifest(run_dir, manifest)

# ---------------------------------------------------------------------------
# Kokoro TTS
# ---------------------------------------------------------------------------
def generate_narration(cfg: Config, script: str, audio_path: Path) -> float:
    ensure_kokoro_models(cfg)
    from kokoro_onnx import Kokoro

    log.info("Loading Kokoro (voice=%s)...", cfg.kokoro_voice)
    kokoro = Kokoro(cfg.kokoro_model_path, cfg.kokoro_voices_path)
    voices = kokoro.get_voices()
    if cfg.kokoro_voice not in voices:
        raise RuntimeError(f"Voice '{cfg.kokoro_voice}' not found. Available: {list(voices)[:20]}...")
    log.info("Synthesizing narration...")
    samples, sr = kokoro.create(script, voice=cfg.kokoro_voice, speed=1.0)
    if samples is None or len(samples) == 0:
        raise RuntimeError("Kokoro returned empty audio")
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(audio_path), samples, sr)
    duration = len(samples) / sr
    log.info("Audio written: %.2f s @ %d Hz", duration, sr)
    return duration

# ---------------------------------------------------------------------------
# Subtitles (max 4 words)
# ---------------------------------------------------------------------------
def split_into_chunks(text: str, max_words: int = 4) -> List[str]:
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words) if words[i:i + max_words]]

def build_subtitles(script: str, audio_duration: float, max_words: int = 4) -> List[SubtitleCue]:
    chunks = split_into_chunks(script, max_words)
    if not chunks:
        return []
    total_chars = sum(len(c) for c in chunks) or 1
    cues: List[SubtitleCue] = []
    t = 0.0
    for idx, chunk in enumerate(chunks):
        dur = max(0.8, (len(chunk) / total_chars) * audio_duration)
        end = min(t + dur, audio_duration)
        if end <= t:
            end = min(t + 0.5, audio_duration)
        cues.append(SubtitleCue(index=idx + 1, start=t, end=end, text=chunk))
        t = end
        if t >= audio_duration:
            break
    if cues and cues[-1].end < audio_duration - 0.05:
        cues[-1].end = audio_duration
    return cues

def write_srt(cues: List[SubtitleCue], path: Path) -> None:
    def ts(s: float) -> str:
        h, m, sec = int(s // 3600), int((s % 3600) // 60), int(s % 60)
        ms = int((s % 1) * 1000)
        return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"
    lines = []
    for c in cues:
        lines += [str(c.index), f"{ts(c.start)} --> {ts(c.end)}", c.text, ""]
    path.write_text("\n".join(lines), encoding="utf-8")

def write_ass(cues: List[SubtitleCue], path: Path, width: int = 1920, height: int = 1080) -> None:
    header = f"""[Script Info]
Title: Stickman Finance
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,64,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,1,2,80,80,60,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    def ass_ts(s: float) -> str:
        h, m = int(s // 3600), int((s % 3600) // 60)
        return f"{h}:{m:02d}:{s % 60:05.2f}"

    events = []
    for c in cues:
        text = r"{\fad(80,60)\t(0,150,\fscx110\fscy110)\t(150,300,\fscx100\fscy100)}" + c.text
        events.append(f"Dialogue: 0,{ass_ts(c.start)},{ass_ts(c.end)},Default,,0,0,0,,{text}")
    path.write_text(header + "\n".join(events), encoding="utf-8")

# ---------------------------------------------------------------------------
# Thumbnail
# ---------------------------------------------------------------------------
def create_thumbnail(cfg: Config, topic: str, headline: str, out_path: Path, seed: int) -> None:
    pipe = get_image_pipeline(cfg)
    import torch
    prompt = (
        f"Bold stickman finance thumbnail illustration, high contrast, "
        f"simple clean line art, stick figure with money or chart metaphor for '{topic}', "
        f"strong focal point, 16:9, white and bright colors, educational YouTube style"
    )
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    img = pipe(prompt=prompt, num_inference_steps=2, guidance_scale=0.0,
               width=512, height=512, generator=generator).images[0].convert("RGB")
    img = img.resize((cfg.thumb_width, cfg.thumb_height), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
    except Exception:
        font = ImageFont.load_default()
    bar_h = 110
    draw.rectangle([0, cfg.thumb_height - bar_h, cfg.thumb_width, cfg.thumb_height], fill=(0, 0, 0))
    words = headline.upper().split()
    lines, cur = [], []
    for w in words:
        test = " ".join(cur + [w])
        if font.getlength(test) < cfg.thumb_width - 40:
            cur.append(w)
        else:
            if cur:
                lines.append(" ".join(cur))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))
    y = cfg.thumb_height - bar_h + 20
    for line in lines[:2]:
        bbox = draw.textbbox((0, 0), line, font=font)
        x = (cfg.thumb_width - (bbox[2] - bbox[0])) // 2
        draw.text((x, y), line, font=font, fill=(255, 255, 0))
        y += 50
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "JPEG", quality=92)
    log.info("Thumbnail saved: %s", out_path)

# ---------------------------------------------------------------------------
# SEO metadata
# ---------------------------------------------------------------------------
def generate_metadata(cfg: Config, script: str, topic: str, duration: float) -> Dict:
    system = "You are a YouTube SEO specialist for educational finance content. Output valid JSON only."
    prompt = f"""
Generate YouTube metadata for this Stickman Finance documentary.
Topic: {topic}
Duration seconds: {duration:.0f}
Script excerpt: {script[:800]}

Return JSON with keys:
title (compelling, accurate, <100 chars),
description (2-3 paragraphs),
tags (list of 8-12 relevant tags),
hashtags (list of 5),
thumbnail_headline (short, 3-6 words, uppercase),
category_id (usually 27 for Education),
language (en)
"""
    raw = ollama_generate(prompt, cfg.text_model, system=system, json_mode=True)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = json.loads(ollama_generate(f"Fix this into valid JSON only:\n{raw}", cfg.text_model, json_mode=True))
    data["actual_duration_seconds"] = round(duration, 2)
    data["video_slug"] = sanitize_slug(data.get("title", topic))
    data["chapters"] = []
    return data

# ---------------------------------------------------------------------------
# Video render
# ---------------------------------------------------------------------------
def render_video(cfg: Config, scenes: List[Scene], audio_path: Path, ass_path: Path, out_path: Path) -> float:
    list_file = out_path.parent / "concat.txt"
    with open(list_file, "w") as f:
        for sc in scenes:
            if not sc.image_path or not Path(sc.image_path).exists():
                raise RuntimeError(f"Missing image for scene {sc.scene_id}")
            f.write(f"file '{Path(sc.image_path).resolve()}'\n")
            f.write(f"duration {sc.end_time - sc.start_time:.3f}\n")
        if scenes:
            f.write(f"file '{Path(scenes[-1].image_path).resolve()}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", str(list_file),
        "-i", str(audio_path),
        "-vf", f"ass={ass_path}",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k",
        "-pix_fmt", "yuv420p",
        "-shortest",
        "-movflags", "+faststart",
        str(out_path),
    ]
    log.info("Rendering video...")
    r = run_cmd(cmd, timeout=600)
    if r.returncode != 0:
        log.error("FFmpeg stderr: %s", r.stderr[-2000:])
        raise RuntimeError("FFmpeg render failed")

    probe = run_cmd(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(out_path)],
        timeout=30,
    )
    duration = float(probe.stdout.strip())
    log.info("Video rendered: %.2f s -> %s", duration, out_path)
    return duration

# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
def save_manifest(run_dir: Path, manifest: RunManifest) -> None:
    with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(asdict(manifest), f, indent=2, default=str)

# ---------------------------------------------------------------------------
# YouTube (optional)
# ---------------------------------------------------------------------------
def upload_to_youtube(cfg: Config, video_path: Path, thumb_path: Path, meta: Dict) -> Optional[str]:
    if not cfg.enable_youtube:
        log.info("YouTube upload disabled")
        return None
    client_id = os.environ.get("YOUTUBE_CLIENT_ID")
    client_secret = os.environ.get("YOUTUBE_CLIENT_SECRET")
    refresh_token = os.environ.get("YOUTUBE_REFRESH_TOKEN")
    if not all([client_id, client_secret, refresh_token]):
        log.warning("YouTube credentials incomplete – skipping upload (Release will still be created)")
        return None

    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    creds = Credentials(
        token=None,
        refresh_token=refresh_token,
        client_id=client_id,
        client_secret=client_secret,
        token_uri="https://oauth2.googleapis.com/token",
    )
    youtube = build("youtube", "v3", credentials=creds)

    body = {
        "snippet": {
            "title": meta.get("title", "Stickman Finance")[:100],
            "description": meta.get("description", "")[:5000],
            "tags": meta.get("tags", [])[:500],
            "categoryId": str(meta.get("category_id", "27")),
            "defaultLanguage": meta.get("language", "en"),
        },
        "status": {"privacyStatus": cfg.youtube_privacy, "selfDeclaredMadeForKids": False},
    }
    media = MediaFileUpload(str(video_path), mimetype="video/mp4", resumable=True)
    log.info("Uploading to YouTube (privacy=%s)...", cfg.youtube_privacy)
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            log.info("Upload progress: %.1f%%", status.progress() * 100)
    video_id = response["id"]
    log.info("Video uploaded: https://youtu.be/%s", video_id)

    try:
        youtube.thumbnails().set(
            videoId=video_id,
            media_body=MediaFileUpload(str(thumb_path), mimetype="image/jpeg"),
        ).execute()
        log.info("Thumbnail set")
    except Exception as e:
        log.error("Thumbnail set failed: %s", e)
    return video_id

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_pipeline(cfg: Config) -> int:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    topic = cfg.topic or (
        "The Power of Compound Interest for Everyday Americans"
        if cfg.mode == "test"
        else "How Ordinary People Build Wealth Through Simple Habits"
    )
    slug = sanitize_slug(topic)
    run_dir = ensure_dir(cfg.output_dir / f"{slug}-{run_id}")
    global log
    log = setup_logging(cfg.log_level, run_dir / "logs" / "pipeline.log")

    log.info("=== STICKMAN FINANCE PIPELINE ===")
    log.info("Mode: %s | Topic: %s | Run: %s", cfg.mode, topic, run_id)

    manifest = RunManifest(
        run_id=run_id, mode=cfg.mode, topic=topic, video_slug=slug,
        created_at=datetime.now(timezone.utc).isoformat(), config=asdict(cfg),
    )
    save_manifest(run_dir, manifest)

    try:
        preflight(cfg)
        ensure_ollama(cfg)
        ensure_kokoro_models(cfg)

        # 1. Script
        script_path = run_dir / "script.txt"
        if script_path.exists() and not cfg.force:
            script = script_path.read_text(encoding="utf-8")
        else:
            script = generate_script(cfg, topic)
            script_path.write_text(script, encoding="utf-8")
        manifest.script = script
        manifest.checkpoints["script"] = True
        save_manifest(run_dir, manifest)

        # 2. TTS
        audio_path = run_dir / "audio" / "narration.wav"
        if audio_path.exists() and not cfg.force:
            duration = sf.info(str(audio_path)).duration
        else:
            duration = generate_narration(cfg, script, audio_path)
        manifest.audio_path = str(audio_path)
        manifest.audio_duration = duration
        manifest.checkpoints["audio"] = True
        save_manifest(run_dir, manifest)

        # 3. Scenes
        scenes = plan_scenes(cfg, script, duration)
        manifest.scenes = [asdict(s) for s in scenes]
        manifest.checkpoints["scenes_planned"] = True
        save_manifest(run_dir, manifest)

        # 4. Images
        generate_all_images(cfg, scenes, run_dir, manifest)
        manifest.scenes = [asdict(s) for s in scenes]
        manifest.checkpoints["images"] = True
        save_manifest(run_dir, manifest)

        # 5. Subtitles
        cues = build_subtitles(script, duration, cfg.max_subtitle_words)
        for c in cues:
            if len(c.text.split()) > cfg.max_subtitle_words:
                raise RuntimeError(f"Subtitle exceeds {cfg.max_subtitle_words} words: {c.text}")
        ass_path = run_dir / "subtitles.ass"
        srt_path = run_dir / "captions.srt"
        write_ass(cues, ass_path, cfg.image_width, cfg.image_height)
        write_srt(cues, srt_path)
        manifest.subtitle_ass = str(ass_path)
        manifest.subtitle_srt = str(srt_path)
        manifest.checkpoints["subtitles"] = True
        save_manifest(run_dir, manifest)

        # 6. Thumbnail + SEO
        meta = generate_metadata(cfg, script, topic, duration)
        thumb_path = run_dir / "thumbnail.jpg"
        create_thumbnail(cfg, topic, meta.get("thumbnail_headline", topic[:40]),
                         thumb_path, seed=(cfg.seed or 42) + 99)
        (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        manifest.thumbnail_path = str(thumb_path)
        manifest.metadata = meta
        manifest.checkpoints["thumbnail_meta"] = True
        save_manifest(run_dir, manifest)

        # 7. Render
        video_path = run_dir / "video.mp4"
        vdur = render_video(cfg, scenes, audio_path, ass_path, video_path)
        if video_path.stat().st_size < 100_000:
            raise RuntimeError("Rendered video suspiciously small")
        manifest.video_path = str(video_path)
        manifest.video_duration = vdur
        manifest.checkpoints["render"] = True
        save_manifest(run_dir, manifest)

        # 8. YouTube (optional – failures do not block Release)
        try:
            yt_id = upload_to_youtube(cfg, video_path, thumb_path, meta)
            if yt_id:
                manifest.youtube_video_id = yt_id
                manifest.checkpoints["youtube"] = True
        except Exception as e:
            log.error("YouTube upload failed (continuing): %s", e)
            manifest.errors.append(f"youtube: {e}")
        save_manifest(run_dir, manifest)

        manifest.status = "success"
        save_manifest(run_dir, manifest)
        log.info("=== PIPELINE SUCCESS ===")
        log.info("Video: %s (%.1f s, %d scenes)", video_path, vdur, len(scenes))
        return 0

    except Exception as e:
        log.error("PIPELINE FAILED: %s", e)
        log.error(traceback.format_exc())
        manifest.status = "failed"
        manifest.errors.append(str(e))
        save_manifest(run_dir, manifest)
        return 1

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Stickman Finance YouTube Pipeline")
    p.add_argument("--mode", choices=["test", "full"], default="test")
    p.add_argument("--topic", default="")
    p.add_argument("--output-dir", default="output")
    p.add_argument("--enable-youtube", action="store_true")
    p.add_argument("--youtube-privacy", default="unlisted", choices=["private", "unlisted", "public"])
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--force", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()
    return Config(
        mode=args.mode, topic=args.topic, output_dir=Path(args.output_dir),
        enable_youtube=args.enable_youtube, youtube_privacy=args.youtube_privacy,
        log_level=args.log_level, force=args.force, seed=args.seed,
    )

if __name__ == "__main__":
    sys.exit(run_pipeline(parse_args()))
