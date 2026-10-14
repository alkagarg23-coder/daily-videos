#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stickman Finance Doodle Studio - headless pipeline for GitHub Actions.

Stages (each one is checkpointed in projects/<slug>/project.json and resumable):
  1. calibrate   measure real Piper speech speed (words/second)
  2. script      Ollama (Gemma) writes the narration section by section
  3. scenes      scene count is DYNAMIC (decided by the narration, never hard-coded)
  4. audio       Piper voices every sentence, exact durations from sample counts
  5. prompts     doodle prompts (style rules are hard-coded in this file)
  6. images      DreamShaper v8 on CPU, time-budget aware, honest fallback doodles
  7. subtitles   word timing from the real audio, 4-words-per-round by default
  8. thumbnail   1280x720 via Pillow
  9. render      per-scene motion + random transitions with FFmpeg, burned subtitles
 10. verify      ffprobe validation of the final MP4

Everything runs locally on the runner: no paid API, no API key, no cloud inference.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

T0 = time.time()
GAP = 0.28  # seconds of silence after every spoken sentence

STYLE_ANCHOR = (
    "Hand-drawn 2D doodle cartoon animation, flat colors, bold black outlines, "
    "slightly imperfect sketchy marker lines, educational YouTube explainer doodle style"
)
NEGATIVE_LOCK = (
    "no gradients, no shadows, no textures, no photorealism, no 3D, "
    "no realistic faces, no anime style"
)
NEGATIVE_EXTRA = (
    "gradient, shadow, texture, photo, photorealistic, 3d render, realistic face, anime, "
    "text, letters, words, watermark, signature, blurry, noisy, complex background, "
    "multiple panels, frame, border"
)

FINANCE_DNA = (
    "You write narration for a calm, highly intelligent, hand-drawn doodle YouTube explainer about "
    "macroeconomics, personal finance, market history and financial mechanics. Rules: "
    "Speak in second person only (you, your wallet, your savings, your bank account). Never use 'we' or 'I'. "
    "Rhythm: a short sentence. A short sentence. Then one longer sentence that builds structural depth. "
    "Then a short sentence again. Ask a question every four to six sentences. "
    "Every financial mechanism or piece of jargon must be decoded into plain English straight away with a real-world analogy. "
    "Never invent statistics, historical market figures, quotations or sources; when unsure, stay general. "
    "Never promise investment returns and never give personal investment advice. "
    "Output ONLY the narration as plain flowing paragraphs: no headings, no bullet points, no markdown, "
    "no stage directions, no emojis, no greeting."
)


# --------------------------------------------------------------------------- utils
def log(msg, level="INFO"):
    el = int(time.time() - T0)
    print(f"[{el // 3600:02d}:{el % 3600 // 60:02d}:{el % 60:02d}] {level:5s} {msg}", flush=True)


def warn(msg):
    log(msg, "WARN")


class PipelineError(RuntimeError):
    pass


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def load_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


def run(cmd, timeout=None, input_bytes=None, check=True, cwd=None):
    cmd = [str(c) for c in cmd]
    try:
        p = subprocess.run(cmd, input=input_bytes, capture_output=True, timeout=timeout, cwd=cwd)
    except subprocess.TimeoutExpired:
        raise PipelineError(f"Command timed out after {timeout}s: {' '.join(cmd[:4])} ...")
    if check and p.returncode != 0:
        err = p.stderr.decode("utf-8", "replace")[-1800:]
        raise PipelineError(f"Command failed ({p.returncode}): {' '.join(cmd[:5])} ...\n{err}")
    return p


def even(x):
    x = int(round(x))
    return x if x % 2 == 0 else x + 1


FFMPEG = None
FFPROBE = None


def find_ffmpeg():
    global FFMPEG, FFPROBE
    FFMPEG = shutil.which("ffmpeg")
    if not FFMPEG:
        try:
            import imageio_ffmpeg  # type: ignore
            FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            FFMPEG = None
    FFPROBE = shutil.which("ffprobe")
    if not FFMPEG:
        raise PipelineError("ffmpeg was not found. Install it (sudo apt-get install -y ffmpeg).")
    log(f"ffmpeg: {FFMPEG}")
    log(f"ffprobe: {FFPROBE or 'not found (using ffmpeg -i parsing)'}")


def ffmpeg_has_filter(name):
    p = run([FFMPEG, "-hide_banner", "-filters"], check=False)
    return re.search(r"\s" + re.escape(name) + r"\s", p.stdout.decode("utf-8", "replace")) is not None


def probe(path):
    path = str(path)
    info = {"duration": 0.0, "has_video": False, "has_audio": False, "width": 0, "height": 0, "nb_frames": 0}
    if FFPROBE:
        p = run([FFPROBE, "-v", "error", "-show_entries",
                 "format=duration:stream=codec_type,width,height,nb_frames", "-of", "json", path], check=False)
        if p.returncode == 0:
            try:
                j = json.loads(p.stdout or b"{}")
                for s in j.get("streams", []):
                    if s.get("codec_type") == "video":
                        info["has_video"] = True
                        info["width"] = int(s.get("width") or 0)
                        info["height"] = int(s.get("height") or 0)
                        try:
                            info["nb_frames"] = int(s.get("nb_frames") or 0)
                        except ValueError:
                            pass
                    elif s.get("codec_type") == "audio":
                        info["has_audio"] = True
                info["duration"] = float((j.get("format") or {}).get("duration") or 0.0)
                return info
            except Exception:
                pass
    p = run([FFMPEG, "-hide_banner", "-i", path], check=False)
    txt = p.stderr.decode("utf-8", "replace")
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", txt)
    if m:
        info["duration"] = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    info["has_video"] = re.search(r"Stream #.*Video:", txt) is not None
    info["has_audio"] = re.search(r"Stream #.*Audio:", txt) is not None
    m = re.search(r"Video:.*?,\s*(\d{2,5})x(\d{2,5})", txt)
    if m:
        info["width"], info["height"] = int(m.group(1)), int(m.group(2))
    return info


# --------------------------------------------------------------------------- config
def E(name, default):
    v = os.environ.get(name)
    return default if v in (None, "") else v


def truthy(v):
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Stickman Finance Doodle Studio (headless)")
    a = p.add_argument
    a("--topic", default=E("TOPIC", "auto"), help="finance topic, or 'auto' to let the pipeline choose one")
    a("--topic-history", default=E("TOPIC_HISTORY", "topic_history.json"))
    a("--audience", default=E("AUDIENCE", "everyday people with no finance background"))
    a("--tone", default=E("TONE", "calm, clear and quietly curious"))
    a("--minutes", type=float, default=float(E("MINUTES", 25)), help="target / maximum video length")
    a("--resolution", type=int, choices=[720, 1080], default=int(E("RESOLUTION", 720)))
    a("--fps", type=int, default=int(E("FPS", 24)))
    a("--piper", default=E("PIPER_BIN", "tools/piper/piper"))
    a("--voice-model", default=E("VOICE_MODEL", "tools/voices/en_US-lessac-medium.onnx"))
    a("--speed", type=float, default=float(E("SPEECH_SPEED", 1.0)))
    a("--volume", type=float, default=float(E("VOLUME", 1.0)))
    a("--animation", choices=["random", "cinematic", "minimal", "off"], default=E("ANIMATION", "random"))
    a("--transition", choices=["random", "crossfade", "cut", "cinematic", "off"], default=E("TRANSITION", "random"))
    a("--intensity", choices=["subtle", "medium", "strong"], default=E("INTENSITY", "medium"))
    a("--transition-seconds", type=float, default=float(E("TRANSITION_SECONDS", 0.5)))
    a("--seed", type=int, default=int(E("SEED", 0)))
    a("--no-avoid-repeat", action="store_true", default=truthy(E("NO_AVOID_REPEAT", "false")))
    a("--subtitle-mode", choices=["4", "1", "2", "sentence"], default=E("SUBTITLE_MODE", "4"))
    a("--scene-seconds", type=float, default=float(E("SCENE_SECONDS", 14)))
    a("--ollama-url", default=E("OLLAMA_URL", "http://127.0.0.1:11434"))
    a("--ollama-model", default=E("OLLAMA_MODEL", "gemma2:2b"))
    a("--sd-model", default=E("SD_MODEL", "Lykon/dreamshaper-8"))
    a("--sd-steps", type=int, default=int(E("SD_STEPS", 14)))
    a("--sd-min-steps", type=int, default=int(E("SD_MIN_STEPS", 6)))
    # DreamShaper draws at a very low resolution (fast on CPU); every image is then upscaled to image-width x image-height
    a("--sd-width", type=int, default=int(E("SD_WIDTH", 512)))
    a("--sd-height", type=int, default=int(E("SD_HEIGHT", 288)))
    a("--image-width", type=int, default=int(E("IMAGE_WIDTH", 1920)))
    a("--image-height", type=int, default=int(E("IMAGE_HEIGHT", 1080)))
    a("--guidance", type=float, default=float(E("SD_GUIDANCE", 6.0)))
    a("--require-sd", action="store_true", default=truthy(E("REQUIRE_SD", "false")))
    a("--budget-minutes", type=float, default=float(E("BUDGET_MINUTES", 300)))
    a("--workdir", default=E("WORKDIR", "projects"))
    a("--output", default=E("OUTPUT_DIR", "output"))
    a("--font-dir", default=E("FONT_DIR", "assets/fonts"))
    a("--smoke", action="store_true", default=truthy(E("SMOKE", "false")),
      help="fast self-test: built-in script, doodle fallback images, silent audio if Piper is missing")
    cfg = p.parse_args(argv)
    cfg.topic = re.sub(r"\s+", " ", cfg.topic or "").strip()
    cfg.auto_topic = cfg.topic.lower() in ("", "auto", "random", "auto-pick")
    cfg.sd_width = max(256, cfg.sd_width // 8 * 8)
    cfg.sd_height = max(256, cfg.sd_height // 8 * 8)
    cfg.img_w, cfg.img_h = even(cfg.image_width), even(cfg.image_height)
    cfg.H = cfg.resolution
    cfg.W = even(cfg.resolution * 16 / 9)
    cfg.transition_seconds = max(0.0, min(2.0, cfg.transition_seconds))
    cfg.minutes = max(0.2, cfg.minutes)
    return cfg


class Paths:
    def __init__(self, cfg, slug):
        self.proj = Path(cfg.workdir) / slug
        self.images = self.proj / "images"
        self.audio = self.proj / "audio"
        self.sent = self.audio / "sentences"
        self.subs = self.proj / "subs"
        self.render = self.proj / "render"
        self.pieces = self.render / "pieces"
        self.fonts = self.proj / "fonts"
        self.state = self.proj / "project.json"
        self.out = Path(cfg.output) / "videos"
        for d in (self.images, self.sent, self.subs, self.pieces, self.fonts, self.out):
            d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- text helpers
def clean_narration(text):
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    kept = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if re.match(r"^(#+\s|section\s*\d+|part\s*\d+|title\s*:|narrator\s*:|scene\s*\d+|\[.*\]$|\(.*\)$)", ln, re.I):
            continue
        if re.match(r"^(okay|ok|sure|certainly|here'?s|here is|of course)\b", ln, re.I) and len(ln.split()) < 25:
            continue
        if re.match(r"^(let me know|i hope|would you like)", ln, re.I):
            continue
        ln = re.sub(r"^\s*(?:[-*\u2022]|\d{1,2}[.)])\s+", "", ln)
        kept.append(ln)
    t = " ".join(kept)
    t = t.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
    t = t.replace("\u2014", ", ").replace("\u2013", ", ").replace("\u2026", ". ")
    t = re.sub(r"\[[^\]]*\]", " ", t)
    t = re.sub(r"[*_#`~|<>{}\\^=+]", " ", t)
    t = re.sub(r"\$\s?(\d[\d,\.]*)", r"\1 dollars", t)
    t = t.replace("$", " dollars ").replace("%", " percent").replace("&", " and ").replace("/", " ")
    t = re.sub(r"[()]", "", t)
    t = re.sub(r"[^A-Za-z0-9\s\.,\?!;:'\"\-]", " ", t)
    t = re.sub(r"\bWe\b", "You", t)
    t = re.sub(r"\bwe\b", "you", t)
    t = re.sub(r"\bOur\b", "Your", t)
    t = re.sub(r"\bour\b", "your", t)
    t = re.sub(r"\bourselves\b", "yourselves", t)
    t = re.sub(r"\.{2,}", ".", t)
    t = re.sub(r"\s+([.,?!;:])", r"\1", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def split_long(s, limit=38):
    words = s.split()
    if len(words) <= limit:
        return [s]
    parts = re.split(r"(?<=[,;:])\s+", s)
    out, cur = [], ""
    for part in parts:
        if cur and len((cur + " " + part).split()) > limit:
            out.append(cur)
            cur = part
        else:
            cur = (cur + " " + part).strip()
    if cur:
        out.append(cur)
    final = []
    for o in out:
        w = o.split()
        while len(w) > limit + 8:
            final.append(" ".join(w[:limit]))
            w = w[limit:]
        final.append(" ".join(w))
    res = []
    for x in final:
        x = x.strip()
        if not x:
            continue
        if not re.search(r"[.!?]$", x):
            x = x.rstrip(",;:") + "."
        res.append(x)
    return res


def split_sentences(text):
    raw = re.split(r"(?<=[.!?])\s+(?=[\"'A-Z0-9])", text)
    out = []
    for r in raw:
        r = r.strip().strip('"').strip()
        if len(re.sub(r"\W", "", r)) < 2:
            continue
        if not re.search(r"[.!?]$", r):
            r = r.rstrip(",;:") + "."
        out.extend(split_long(r))
    return out


def wcount(sentences):
    return sum(len(s.split()) for s in sentences)


# --------------------------------------------------------------------------- Ollama
class Ollama:
    def __init__(self, url, model):
        self.url = url.rstrip("/")
        self.model = model
        self.proc = None

    def alive(self):
        try:
            return requests.get(self.url + "/api/tags", timeout=3).status_code == 200
        except Exception:
            return False

    def start(self):
        if self.alive():
            return
        if not shutil.which("ollama"):
            raise PipelineError("Ollama is not installed and not reachable at " + self.url)
        log("Starting 'ollama serve' ...")
        self.proc = subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(60):
            if self.alive():
                return
            time.sleep(2)
        raise PipelineError("Ollama server did not start within 120 s")

    def installed(self):
        r = requests.get(self.url + "/api/tags", timeout=10).json()
        return [m.get("name", "") for m in r.get("models", [])]

    def ensure_model(self):
        names = self.installed()
        want = self.model
        if want in names or (":" not in want and any(n.startswith(want + ":") for n in names)):
            log(f"Ollama model ready: {want}")
            return
        if shutil.which("ollama"):
            log(f"Pulling {want} (one-time download) ...")
            run(["ollama", "pull", want], timeout=1800)
            if want in self.installed():
                return
        others = [n for n in self.installed() if n.lower().startswith("gemma")]
        if others:
            warn(f"{want} missing, using installed Gemma model {others[0]}")
            self.model = others[0]
            return
        raise PipelineError(f"Ollama model {want} is not available")

    def generate(self, prompt, system="", num_predict=400, temperature=0.7, tries=3):
        last = None
        for attempt in range(1, tries + 1):
            try:
                r = requests.post(
                    self.url + "/api/generate",
                    json={
                        "model": self.model, "prompt": prompt, "system": system, "stream": False,
                        "keep_alive": "30m",
                        "options": {"temperature": temperature, "top_p": 0.9, "repeat_penalty": 1.12,
                                    "num_predict": int(num_predict), "num_ctx": 3072},
                    },
                    timeout=(10, 900),
                )
                r.raise_for_status()
                txt = (r.json().get("response") or "").strip()
                if txt:
                    return txt
                last = "empty response"
            except Exception as e:  # noqa: BLE001
                last = str(e)
            warn(f"Ollama attempt {attempt}/{tries} failed: {last}")
            time.sleep(3 * attempt)
        raise PipelineError(f"Ollama failed after {tries} attempts: {last}")

    def unload(self):
        try:
            requests.post(self.url + "/api/generate", json={"model": self.model, "keep_alive": 0}, timeout=20)
        except Exception:
            pass


# --------------------------------------------------------------------------- Piper
class Piper:
    def __init__(self, cfg):
        self.exe = Path(cfg.piper)
        self.model = Path(cfg.voice_model)
        self.speed = max(0.5, min(2.0, cfg.speed))
        self.volume = cfg.volume
        self.available = self.exe.exists() and self.model.exists()
        self.rate = 22050
        cj = load_json(str(self.model) + ".json", {}) or {}
        try:
            self.rate = int(cj.get("audio", {}).get("sample_rate", 22050))
        except Exception:
            self.rate = 22050
        self.espeak = self.exe.parent / "espeak-ng-data"

    def synth(self, text, out_path):
        out_path = Path(out_path)
        cmd = [str(self.exe.resolve()), "--model", str(self.model.resolve()), "--output_file", str(out_path.resolve()),
               "--length_scale", f"{1.0 / self.speed:.3f}", "--sentence_silence", "0.0"]
        if self.espeak.exists():
            cmd += ["--espeak_data", str(self.espeak.resolve())]
        run(cmd, timeout=180, input_bytes=(text + "\n").encode("utf-8"))
        if not out_path.exists() or out_path.stat().st_size < 100:
            raise PipelineError("Piper produced no audio for: " + text[:60])


def read_wav(path):
    with wave.open(str(path), "rb") as w:
        return w.getnchannels(), w.getsampwidth(), w.getframerate(), w.readframes(w.getnframes())


def write_wav(path, rate, samples):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.asarray(samples, dtype=np.int16).tobytes())


def wav_samples(path, rate):
    """Return mono int16 samples at `rate`; convert with ffmpeg only if the file differs."""
    nch, sw, fr, raw = read_wav(path)
    if (nch, sw, fr) != (1, 2, rate):
        tmp = Path(str(path) + ".conv.wav")
        run([FFMPEG, "-y", "-v", "error", "-i", path, "-ar", rate, "-ac", 1, "-sample_fmt", "s16", tmp])
        os.replace(tmp, path)
        nch, sw, fr, raw = read_wav(path)
    return np.frombuffer(raw, dtype=np.int16)


def est_speech_seconds(text, speed):
    return max(0.8, len(text.split()) / (2.75 * speed))


def tts_sentence(cfg, piper, text, P, rate):
    """Voice one sentence. Cached by content. Returns (path, used_fallback)."""
    key = hashlib.sha1(f"{piper.model.name}|{piper.speed}|{text}".encode()).hexdigest()[:16]
    out = P.sent / f"{key}.wav"
    if out.exists() and out.stat().st_size > 100:
        return out, False
    if piper.available:
        last = None
        for attempt in range(3):
            try:
                piper.synth(text, out)
                return out, False
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(1 + attempt)
        warn(f"Piper failed for a sentence, using silent placeholder: {last}")
    n = int(est_speech_seconds(text, piper.speed) * rate)
    write_wav(out, rate, np.zeros(n, dtype=np.int16))
    return out, True


# --------------------------------------------------------------------------- analysis (mood / objects / labels)
MOOD_RULES = [
    ("orange", ["crash", "collapse", "panic", "crisis", "hyperinflation", "inflation", "bubble", "default",
                "bank run", "recession", "depression", "meltdown", "bankrupt"]),
    ("tan", ["ancient", "barter", "medieval", "roman", "empire", "merchant", "gold standard", "history",
             "historical", "coin", "centuries"]),
    ("blue", ["bank", "central bank", "corporat", "company", "federal reserve", "interest rate", "loan",
              "credit", "stock market", "investor", "mortgage"]),
]
RED_WORDS = ["panic", "crash", "collapse", "debt", "crisis", "loss", "lose", "broke", "fear", "hyperinflation",
             "default", "bankrupt", "poorer", "steal", "stolen", "shrink"]
GREEN_WORDS = ["profit", "wealth", "save", "saving", "growth", "grow", "earn", "invest", "gain", "rich", "compound"]
OBJECT_RULES = [
    (["coffee", "latte", "cafe"], "a big coffee cup", "coffee"),
    (["bank", "loan", "credit", "federal", "mortgage"], "an oversized bank building", "bank"),
    (["save", "saving", "piggy", "wallet", "deposit"], "a simple piggy bank", "piggy"),
    (["inflation", "price", "rate", "graph", "growth", "chart", "market", "stock"], "a basic graph with a giant arrow", "graph"),
    (["gold", "coin", "barter", "silver"], "a pile of big gold coins", "coins"),
    (["debt", "owe"], "a heavy chain tied to a big dollar sign", "dollar"),
]
LABEL_RULES = [
    ("hyperinflation", "HYPERINFLATION!"), ("inflation", "INFLATION?"), ("crash", "CRASH!"),
    ("collapse", "COLLAPSE!"), ("panic", "PANIC!"), ("debt", "DEBT!"), ("central bank", "CENTRAL BANK"),
    ("bank", "THE BANK"), ("saving", "YOUR SAVINGS"), ("coffee", "YOUR COFFEE"), ("gold", "GOLD"),
    ("interest", "INTEREST"), ("profit", "PROFIT!"), ("loan", "LOAN?"), ("price", "PRICES?"),
    ("wealth", "WEALTH"), ("money", "MONEY?"),
]


def has_word(text, kw):
    return re.search(r"\b" + re.escape(kw), text) is not None


def analyze(narration, title):
    t = (narration + " " + title).lower()
    mood = "white"
    for name, kws in MOOD_RULES:
        if any(has_word(t, k) for k in kws):
            mood = name
            break
    red = sum(1 for k in RED_WORDS if has_word(t, k))
    green = sum(1 for k in GREEN_WORDS if has_word(t, k))
    fill = "red" if red > green else ("green" if green > 0 else "white")
    objs, drawer = [], None
    for kws, phrase, dr in OBJECT_RULES:
        if any(has_word(t, k) for k in kws):
            objs.append(phrase)
            drawer = drawer or dr
        if len(objs) == 2:
            break
    if not objs:
        objs, drawer = ["a big dollar sign"], "dollar"
    label = None
    for kw, lab in LABEL_RULES:
        if has_word(t, kw):
            label = lab
            break
    emotion = {"red": "panic", "green": "happy", "white": "neutral"}[fill]
    return {"mood": mood, "fill": fill, "objects": objs, "drawer": drawer, "label": label, "emotion": emotion}


BG_PHRASE = {
    "white": "flat solid white background with a flat green strip along the bottom as ground",
    "orange": "flat solid orange background",
    "blue": "flat solid blue background",
    "tan": "flat solid tan background",
}
BG_RGB = {"white": (255, 255, 255), "orange": (255, 150, 40), "blue": (80, 140, 225), "tan": (222, 190, 140)}
FILL_RGB = {"red": (235, 64, 52), "green": (84, 200, 96), "white": (255, 255, 255)}
INK = (15, 15, 15)


def build_prompt(analysis, idea):
    fill_word = {"red": "red-filled", "green": "green-filled", "white": "white-filled"}[analysis["fill"]]
    objs = ", ".join(analysis["objects"])
    pos = (f"{STYLE_ANCHOR}, one simple stick figure with a large circular {fill_word} head, dot eyes, "
           f"thick expressive eyebrow lines, {idea}, {objs}, {BG_PHRASE[analysis['mood']]}, chunky simplified cartoon objects")
    neg = NEGATIVE_LOCK + ", " + NEGATIVE_EXTRA
    return pos, neg


# --------------------------------------------------------------------------- fonts / drawing
def find_font(cfg):
    cands = []
    fd = Path(cfg.font_dir)
    if fd.exists():
        cands += sorted(fd.glob("*.ttf")) + sorted(fd.glob("*.otf"))
    cands += [Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
              Path("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf")]
    for c in cands:
        if c.exists():
            return str(c)
    return None


def load_font(fp, size):
    size = max(8, int(size))
    if fp:
        try:
            return ImageFont.truetype(fp, size)
        except Exception:
            pass
    try:
        return ImageFont.load_default(size)
    except Exception:
        return ImageFont.load_default()


def text_center(d, xy, text, font, fill, stroke=0, stroke_fill=None):
    bb = d.textbbox((0, 0), text, font=font, stroke_width=stroke)
    w, h = bb[2] - bb[0], bb[3] - bb[1]
    d.text((xy[0] - w / 2 - bb[0], xy[1] - h / 2 - bb[1]), text, font=font, fill=fill,
           stroke_width=stroke, stroke_fill=stroke_fill)


def draw_arrow(d, p0, p1, color, width):
    d.line([p0, p1], fill=color, width=width)
    ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    L = width * 4
    a1, a2 = ang + math.radians(150), ang - math.radians(150)
    d.polygon([p1, (p1[0] + L * math.cos(a1), p1[1] + L * math.sin(a1)),
               (p1[0] + L * math.cos(a2), p1[1] + L * math.sin(a2))], fill=color)


def add_label(img, text, fp, seed):
    if not text:
        return img
    img = img.copy()
    d = ImageDraw.Draw(img)
    W, H = img.size
    f = load_font(fp, H * 0.085)
    bb = d.textbbox((0, 0), text, font=f)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    pad = int(H * 0.03)
    left = seed % 2 == 0
    x0 = int(W * 0.04) if left else W - tw - 2 * pad - int(W * 0.04)
    y0 = int(H * 0.05)
    ow = max(3, H // 100)
    d.rounded_rectangle([x0, y0, x0 + tw + 2 * pad, y0 + th + 2 * pad], radius=pad, fill=(255, 212, 0), outline=INK, width=ow)
    d.text((x0 + pad - bb[0], y0 + pad - bb[1]), text, font=f, fill=INK)
    sx, sy = x0 + (tw + 2 * pad) // 2, y0 + th + 2 * pad + 4
    ex = int(W * (0.5 + (0.1 if left else -0.1)))
    draw_arrow(d, (sx, sy), (ex, int(H * 0.42)), (255, 190, 0), max(4, H // 70))
    return img


def draw_stick(d, cx, gy, H, fill, emotion, rng):
    u = H / 400.0
    w = max(3, int(7 * u))
    r = int(46 * u)
    hip_y = gy - int(95 * u)
    neck_y = hip_y - int(100 * u)
    hc = neck_y - r
    d.line([(cx, neck_y), (cx, hip_y)], fill=INK, width=w)
    sway = rng.choice([-1, 1])
    if emotion == "panic":
        d.line([(cx, neck_y + 15 * u), (cx - 60 * u, neck_y - 35 * u)], fill=INK, width=w)
        d.line([(cx, neck_y + 15 * u), (cx + 60 * u, neck_y - 35 * u)], fill=INK, width=w)
    elif emotion == "happy":
        d.line([(cx, neck_y + 20 * u), (cx - 65 * u, neck_y - 10 * u)], fill=INK, width=w)
        d.line([(cx, neck_y + 20 * u), (cx + 65 * u, neck_y - 10 * u)], fill=INK, width=w)
    else:
        d.line([(cx, neck_y + 20 * u), (cx - 55 * u * sway, neck_y + 65 * u)], fill=INK, width=w)
        d.line([(cx, neck_y + 20 * u), (cx + 60 * u * sway, neck_y + 45 * u)], fill=INK, width=w)
    d.line([(cx, hip_y), (cx - 45 * u, gy)], fill=INK, width=w)
    d.line([(cx, hip_y), (cx + 45 * u, gy)], fill=INK, width=w)
    d.ellipse([cx - r, hc - r, cx + r, hc + r], fill=fill, outline=INK, width=w)
    ex, er = int(r * 0.4), max(3, int(5 * u))
    for s in (-1, 1):
        d.ellipse([cx + s * ex - er, hc - er, cx + s * ex + er, hc + er], fill=INK)
    by = hc - int(r * 0.35)
    for s in (-1, 1):
        if emotion == "panic":
            d.line([(cx + s * ex * 1.8, by - 9 * u), (cx + s * ex * 0.3, by + 5 * u)], fill=INK, width=w)
        elif emotion == "happy":
            d.line([(cx + s * ex * 1.8, by + 3 * u), (cx + s * ex * 0.3, by - 4 * u)], fill=INK, width=w)
        else:
            d.line([(cx + s * ex * 1.6, by), (cx + s * ex * 0.4, by)], fill=INK, width=w)
    my = hc + int(r * 0.45)
    if emotion == "happy":
        d.arc([cx - r * 0.5, my - r * 0.4, cx + r * 0.5, my + r * 0.3], 10, 170, fill=INK, width=w)
    elif emotion == "panic":
        d.ellipse([cx - r * 0.18, my - r * 0.1, cx + r * 0.18, my + r * 0.3], fill=INK)
    else:
        d.line([(cx - r * 0.3, my + 3 * u), (cx + r * 0.3, my + 3 * u)], fill=INK, width=w)


def obj_dollar(d, x, gy, H, rng, fp):
    r = int(H * 0.2)
    cy = gy - int(H * 0.3)
    d.ellipse([x - r, cy - r, x + r, cy + r], fill=(255, 212, 0), outline=INK, width=max(4, H // 70))
    text_center(d, (x, cy), "$", load_font(fp, r * 1.5), INK)


def obj_bank(d, x, gy, H, rng, fp):
    w, h = int(H * 0.55), int(H * 0.42)
    ow = max(3, H // 90)
    top = gy - h
    d.polygon([(x - w // 2 - 10, top + 30), (x, top - 20), (x + w // 2 + 10, top + 30)], fill=(235, 235, 235), outline=INK)
    d.rectangle([x - w // 2, top + 30, x + w // 2, gy], fill=(240, 240, 240), outline=INK, width=ow)
    for i in range(4):
        cx = x - w // 2 + int((i + 0.5) * w / 4)
        d.rectangle([cx - 10, top + 45, cx + 10, gy - 12], fill=(200, 200, 205), outline=INK, width=ow)
    text_center(d, (x, top + 12), "$", load_font(fp, 28), INK)


def obj_piggy(d, x, gy, H, rng, fp):
    ow = max(3, H // 90)
    bw, bh = int(H * 0.5), int(H * 0.32)
    cy = gy - int(H * 0.22)
    d.ellipse([x - bw // 2, cy - bh // 2, x + bw // 2, cy + bh // 2], fill=(255, 170, 190), outline=INK, width=ow)
    d.ellipse([x + bw // 2 - 20, cy - 12, x + bw // 2 + 25, cy + 22], fill=(255, 140, 170), outline=INK, width=ow)
    d.polygon([(x + 5, cy - bh // 2 + 8), (x + 35, cy - bh // 2 - 28), (x + 50, cy - bh // 2 + 12)], fill=(255, 140, 170), outline=INK)
    for dx in (-bw // 4, bw // 4):
        d.rectangle([x + dx - 10, cy + bh // 2 - 6, x + dx + 10, gy], fill=(255, 140, 170), outline=INK, width=ow)
    d.line([(x - 25, cy - bh // 2 + 14), (x + 5, cy - bh // 2 + 14)], fill=INK, width=ow + 1)
    d.ellipse([x + bw // 4 - 5, cy - 20, x + bw // 4 + 5, cy - 10], fill=INK)


def obj_graph(d, x, gy, H, rng, fp):
    ow = max(4, H // 70)
    w, h = int(H * 0.7), int(H * 0.45)
    x0, y0 = x - w // 2, gy - 10
    d.line([(x0, y0 - h), (x0, y0), (x0 + w, y0)], fill=INK, width=ow)
    up = rng.random() < 0.65
    pts = [(x0 + 15, y0 - 20), (x0 + w * 0.3, y0 - h * (0.35 if up else 0.7)), (x0 + w * 0.55, y0 - h * (0.25 if up else 0.5)),
           (x0 + w * 0.9, y0 - h * (0.9 if up else 0.12))]
    d.line(pts, fill=(220, 40, 40) if not up else (30, 150, 60), width=ow + 2, joint="curve")
    draw_arrow(d, pts[-2], pts[-1], (220, 40, 40) if not up else (30, 150, 60), ow + 2)


def obj_coffee(d, x, gy, H, rng, fp):
    ow = max(3, H // 90)
    w, h = int(H * 0.3), int(H * 0.32)
    top = gy - h
    d.rectangle([x - w // 2, top, x + w // 2, gy - 4], fill=(150, 95, 60), outline=INK, width=ow)
    d.ellipse([x + w // 2 - 6, top + 15, x + w // 2 + 38, top + h * 0.65], outline=INK, width=ow + 1)
    for dx in (-18, 0, 18):
        d.line([(x + dx, top - 8), (x + dx + 8, top - 28), (x + dx - 4, top - 46)], fill=INK, width=max(2, ow - 1))


def obj_coins(d, x, gy, H, rng, fp):
    ow = max(3, H // 90)
    for row in range(3):
        for i in range(3 - row):
            cx = x - 60 + i * 60 + row * 30
            cy = gy - 20 - row * 34
            d.ellipse([cx - 32, cy - 16, cx + 32, cy + 16], fill=(255, 212, 0), outline=INK, width=ow)


DRAWERS = {"dollar": obj_dollar, "bank": obj_bank, "piggy": obj_piggy, "graph": obj_graph,
           "coffee": obj_coffee, "coins": obj_coins}


def upscale_hd(img, W, H):
    """Upscale a low-resolution DreamShaper image to W x H (cover + centre crop), then restore crisp doodle lines."""
    scale = max(W / img.width, H / img.height)
    big = img.resize((max(W, int(round(img.width * scale))), max(H, int(round(img.height * scale)))), Image.LANCZOS)
    l, t = (big.width - W) // 2, (big.height - H) // 2
    big = big.crop((l, t, l + W, t + H))
    big = big.filter(ImageFilter.UnsharpMask(radius=2.2, percent=140, threshold=2))
    big = ImageEnhance.Color(big).enhance(1.15)
    return ImageEnhance.Contrast(big).enhance(1.08)


def draw_fallback(scene, W, H, seed, fp):
    """Procedural doodle used only when DreamShaper cannot run. Each scene gets its own drawing."""
    rng = random.Random(seed)
    info = scene["analysis"]
    img = Image.new("RGB", (W, H), BG_RGB[info["mood"]])
    d = ImageDraw.Draw(img)
    gy = H - int(H * 0.13)
    d.rectangle([0, gy, W, H], fill=(66, 170, 72))
    d.line([0, gy, W, gy], fill=INK, width=max(3, H // 90))
    fx = int(W * rng.choice([0.26, 0.32, 0.68, 0.74]))
    ox = W - fx
    draw_stick(d, fx, gy, H, FILL_RGB[info["fill"]], info["emotion"], rng)
    DRAWERS.get(info["drawer"] or "dollar", obj_dollar)(d, ox, gy, H, rng, fp)
    return add_label(img, info.get("label"), fp, seed)


# --------------------------------------------------------------------------- image engine
class SDEngine:
    def __init__(self, cfg):
        import torch
        from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

        self.torch = torch
        torch.set_num_threads(max(1, os.cpu_count() or 2))
        src = cfg.sd_model
        t0 = time.time()
        log(f"Loading image model {src} (first run downloads ~2 GB) ...")
        kw = dict(torch_dtype=torch.float32, safety_checker=None, requires_safety_checker=False)
        if os.path.isfile(src):
            pipe = StableDiffusionPipeline.from_single_file(src, **kw)
        else:
            try:
                pipe = StableDiffusionPipeline.from_pretrained(src, variant="fp16", **kw)
            except Exception as e:  # noqa: BLE001
                warn(f"fp16 variant unavailable ({str(e)[:120]}), loading default weights")
                pipe = StableDiffusionPipeline.from_pretrained(src, **kw)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="dpmsolver++")
        pipe.set_progress_bar_config(disable=True)
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
        self.pipe = pipe
        self.cfg = cfg
        log(f"Image model ready in {time.time() - t0:.0f}s")

    def generate(self, prompt, negative, seed, steps):
        torch = self.torch
        g = torch.Generator("cpu").manual_seed(int(seed) % (2 ** 31))
        with torch.inference_mode():
            out = self.pipe(prompt=prompt, negative_prompt=negative, num_inference_steps=int(steps),
                            guidance_scale=self.cfg.guidance, width=self.cfg.sd_width, height=self.cfg.sd_height,
                            generator=g)
        return out.images[0].convert("RGB")


# --------------------------------------------------------------------------- state
def stage_done(state, name):
    return bool(state["stages"].get(name))


def mark(P, state, name, **extra):
    state["stages"][name] = True
    state.update(extra)
    save_json(P.state, state)


# --------------------------------------------------------------------------- stage 1: calibrate
CAL_TEXT = [
    "Picture this. You tap your card to pay for a coffee.",
    "It takes two seconds. Nothing about it feels strange.",
    "But the hidden machinery behind that tiny payment is far stranger than you think.",
    "Every number on your screen is a promise written by a bank.",
    "So where does that promise come from, and who decides how much of it exists?",
]


def stage_calibrate(cfg, P, state, piper):
    if stage_done(state, "calibrate"):
        return state["wps"]
    wps = 2.3 * piper.speed
    if piper.available:
        try:
            total = 0.0
            words = 0
            cal = P.audio / "calibration"
            cal.mkdir(parents=True, exist_ok=True)
            for i, s in enumerate(CAL_TEXT):
                out = cal / f"c{i}.wav"
                piper.synth(s, out)
                n = len(wav_samples(out, piper.rate))
                total += n / piper.rate + GAP
                words += len(s.split())
            wps = words / total
            log(f"Calibrated speech speed: {wps:.2f} words/second (incl. pauses)")
        except Exception as e:  # noqa: BLE001
            warn(f"Calibration failed, using default speed estimate: {e}")
    else:
        warn("Piper not available - using default speed estimate")
    mark(P, state, "calibrate", wps=wps)
    return wps


# --------------------------------------------------------------------------- stage 2: script
DEEPER = ["What most people miss", "Who benefits and who pays", "How it echoes through history",
          "What it means for ordinary savers", "The trade-offs behind the scenes", "Where the system is fragile",
          "How the pressure shows up in daily prices", "Why the rules keep changing",
          "The incentives nobody talks about", "How small choices add up over years"]
GENERIC_TITLES = ["The hidden incentives", "What history shows", "Where the pressure builds", "The ripple effects",
                  "Who gains and who pays", "The trade-offs nobody mentions", "How it feels in daily life",
                  "The mechanism step by step", "Why it breaks", "What changes next"]


def smoke_sections(topic, target_words):
    base = [
        "You tap your card to pay for a coffee.", "It takes two seconds.",
        "But the hidden machinery behind that payment is far stranger than you think.",
        f"Behind it sits the story of {topic.lower()}.",
        "Think of money like a promise written on a very large piece of paper.",
        "When more promises are written, each one buys a little less.",
        "So who decides how many promises exist?",
        "A central bank does, and it moves levers that most people never see, much like a thermostat quietly steering the temperature of an entire house.",
        "Short sentence. Long consequences.",
        "Your savings feel solid, yet they sit inside this moving system.",
        "Does that mean you should panic?",
        "No, it means you should understand the machine.",
    ]
    mirror = ["History keeps repeating one simple lesson about trust and scarcity.",
              "Look at your own grocery receipt and you will see that lesson in action.",
              "This video is educational and not financial advice."]
    sections, words, i = [], 0, 0
    while words < max(60, target_words - 60):
        chunk = [base[(i + k) % len(base)] for k in range(8)]
        sections.append({"title": f"Part {len(sections) + 1}", "role": "hook" if not sections else "body", "sentences": chunk})
        words += wcount(chunk)
        i += 3
    sections.append({"title": "Modern mirror", "role": "mirror", "sentences": mirror})
    return sections


def llm_outline(llm, topic, n):
    titles = []
    for _ in range(3):
        have = "; ".join(titles)
        prompt = (f"Topic: {topic}\nGive {n - len(titles)} short, distinct section titles (max 7 words each) that walk "
                  f"through this topic step by step, from basic mechanics to deeper consequences."
                  + (f" Do not repeat these: {have}." if have else "")
                  + " One title per line, no numbering, no extra text.")
        try:
            txt = llm.generate(prompt, "You are a careful documentary outliner.", num_predict=160, temperature=0.6, tries=2)
        except PipelineError:
            break
        for ln in txt.splitlines():
            ln = re.sub(r"^\s*(?:[-*\u2022]|\d{1,2}[.)])\s*", "", ln).strip().strip('"*')
            ln = re.sub(r"[^A-Za-z0-9 ,:'\-?]", "", ln)
            if 2 <= len(ln.split()) <= 12 and ln.lower() not in [t.lower() for t in titles]:
                titles.append(ln)
        if len(titles) >= n:
            break
    k = 0
    while len(titles) < n:
        titles.append(GENERIC_TITLES[k % len(GENERIC_TITLES)] + (f" ({k // len(GENERIC_TITLES) + 2})" if k >= len(GENERIC_TITLES) else ""))
        k += 1
    return titles[:n]


def gen_section(llm, cfg, role, title, idx, total, prev_tail, words):
    if role == "hook":
        role_txt = ("Open with a relatable modern financial moment (for example buying a cup of coffee), then say clearly that the "
                    "hidden machinery behind that moment is far stranger than it looks, and reframe the concept.")
    elif role == "mirror":
        role_txt = ("This is the final section. Reflect the historical or systemic economic truth back onto something the viewer "
                    "experiences in their own financial life today. End with one memorable closing sentence and a brief reminder "
                    "that this video is educational and not financial advice.")
    else:
        role_txt = "Explain this part of the topic clearly. Introduce one new idea and decode any jargon with an analogy."
    prompt = (f"Video topic: {cfg.topic}\nAudience: {cfg.audience}\nTone: {cfg.tone}\n"
              f"Section {idx} of {total}: \"{title}\"\n{role_txt}\n"
              f"Write about {words} words.\n"
              + (f"The previous section ended with: \"{prev_tail}\"\nContinue naturally, without repeating it.\n" if prev_tail else "")
              + "Narration only.")
    best = []
    for attempt in range(3):
        txt = llm.generate(prompt, FINANCE_DNA, num_predict=int(words * 1.9) + 40, temperature=0.7)
        sents = split_sentences(clean_narration(txt))
        if wcount(sents) > wcount(best):
            best = sents
        if wcount(best) >= 0.55 * words:
            break
        warn(f"Section '{title}' too short ({wcount(sents)} words), retrying")
    if not best:
        raise PipelineError(f"Gemma returned no usable text for section '{title}'")
    if role != "mirror":
        while wcount(best) > 1.5 * words and len(best) > 3:
            best.pop()
    return best


def stage_script(cfg, P, state, wps, llm):
    if stage_done(state, "script") and state.get("sections"):
        return state["sections"]
    target = int(cfg.minutes * 60 * wps * 0.985)
    log(f"Target narration: ~{target} words for {cfg.minutes:g} minutes")
    if cfg.smoke:
        sections = smoke_sections(cfg.topic, target)
    else:
        hook_w, mirror_w, sec_w = 170, 140, 230
        n_body = max(1, round((target - hook_w - mirror_w) / sec_w))
        total = n_body + 2
        titles = llm_outline(llm, cfg.topic, n_body)
        sections = []
        t0 = time.time()
        sections.append({"title": "The hook", "role": "hook",
                         "sentences": gen_section(llm, cfg, "hook", f"The hook: {cfg.topic}", 1, total, "", hook_w)})
        log(f"Section 1/{total} done ({wcount(sections[-1]['sentences'])} words)")
        for i, t in enumerate(titles, start=2):
            tail = " ".join(sections[-1]["sentences"][-2:])
            sections.append({"title": t, "role": "body",
                             "sentences": gen_section(llm, cfg, "body", t, i, total, tail, sec_w)})
            log(f"Section {i}/{total} done ({wcount(sections[-1]['sentences'])} words, {time.time() - t0:.0f}s elapsed)")
        extra = 0
        while sum(wcount(s["sentences"]) for s in sections) + mirror_w < target * 0.99 and extra < 12:
            need = target - sum(wcount(s["sentences"]) for s in sections) - mirror_w
            if need < 70:
                break
            t = DEEPER[extra % len(DEEPER)]
            extra += 1
            tail = " ".join(sections[-1]["sentences"][-2:])
            sections.append({"title": t, "role": "body",
                             "sentences": gen_section(llm, cfg, "body", t, len(sections) + 1, total + extra, tail, int(min(sec_w, need)))})
            log(f"Extra section '{t}' added ({wcount(sections[-1]['sentences'])} words)")
        tail = " ".join(sections[-1]["sentences"][-2:])
        sections.append({"title": "Modern mirror", "role": "mirror",
                         "sentences": gen_section(llm, cfg, "mirror", "Modern mirror", total + extra, total + extra, tail, mirror_w)})
        llm.unload()
    # never exceed the time limit: trim middle sections' trailing sentences if the script is too long
    limit_words = int(cfg.minutes * 60 * wps * 0.995)
    while sum(wcount(s["sentences"]) for s in sections) > limit_words:
        mids = [s for s in sections[1:-1] if len(s["sentences"]) > 4]
        if not mids:
            break
        big = max(mids, key=lambda s: len(s["sentences"]))
        big["sentences"].pop()
        warn("Script above time limit - trimmed one sentence")
    total_words = sum(wcount(s["sentences"]) for s in sections)
    log(f"Script ready: {len(sections)} sections, {total_words} words (~{total_words / wps / 60:.1f} min)")
    save_json(P.proj / "script.json", {"topic": cfg.topic, "sections": sections})
    mark(P, state, "script", sections=sections)
    return sections


# --------------------------------------------------------------------------- stage 3: scenes (dynamic count)
def stage_scenes(cfg, P, state, wps):
    if stage_done(state, "scenes") and state.get("scenes"):
        return state["scenes"]
    target_words = max(12, int(cfg.scene_seconds * wps))
    scenes, cur, cur_titles = [], [], []

    def flush():
        nonlocal cur, cur_titles
        if cur:
            scenes.append({"section": cur_titles[0], "sentences": cur})
        cur, cur_titles = [], []

    for sec in state["sections"]:
        for s in sec["sentences"]:
            if not cur:
                cur_titles = [sec["title"]]
            cur.append(s)
            if wcount(cur) >= target_words:
                flush()
        if cur and wcount(cur) >= 0.45 * target_words:
            flush()
    flush()
    if len(scenes) > 1 and wcount(scenes[-1]["sentences"]) < 0.3 * target_words:
        last = scenes.pop()
        scenes[-1]["sentences"] += last["sentences"]
    for i, sc in enumerate(scenes, start=1):
        sc["id"] = i
        sc["narration"] = " ".join(sc["sentences"])
    log(f"Scene plan: {len(scenes)} scenes (decided by the narration, no fixed limit)")
    mark(P, state, "scenes", scenes=scenes)
    return scenes


# --------------------------------------------------------------------------- stage 4: audio
def stage_audio(cfg, P, state, piper):
    scenes = state["scenes"]
    narr = P.audio / "narration.wav"
    if stage_done(state, "audio") and narr.exists() and all("sent_timing" in s for s in scenes):
        return scenes
    if not piper.available and not cfg.smoke:
        raise PipelineError(f"Piper or its voice is missing (exe={piper.exe}, voice={piper.model}).")
    rate = piper.rate
    uniq = []
    for sc in scenes:
        for s in sc["sentences"]:
            if s not in uniq:
                uniq.append(s)
    log(f"Voicing {len(uniq)} sentences with Piper ...")
    paths, fallbacks = {}, 0
    workers = max(1, min(3, os.cpu_count() or 2))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(tts_sentence, cfg, piper, s, P, rate): s for s in uniq}
        for k, f in enumerate(as_completed(futs), start=1):
            s = futs[f]
            path, fb = f.result()
            paths[s] = path
            fallbacks += 1 if fb else 0
            if k % 50 == 0:
                log(f"  voiced {k}/{len(uniq)}")
    if fallbacks and not cfg.smoke and fallbacks > max(2, 0.1 * len(uniq)):
        raise PipelineError(f"Too many sentences failed in Piper ({fallbacks}/{len(uniq)}).")
    samples = {s: wav_samples(p, rate) for s, p in paths.items()}

    def scene_seconds(sc):
        return sum(len(samples[s]) / rate + GAP for s in sc["sentences"])

    limit = cfg.minutes * 60.0
    total = sum(scene_seconds(sc) for sc in scenes)
    removed = 0
    while total > limit and len(scenes) > 4:
        idx = min(len(scenes) - 3, max(1, int(len(scenes) * 0.6)))
        sc = scenes.pop(idx)
        total -= scene_seconds(sc)
        removed += 1
    if removed:
        warn(f"Narration exceeded {cfg.minutes:g} min - removed {removed} middle scene(s) to stay inside the limit")
        for i, sc in enumerate(scenes, start=1):
            sc["id"] = i
    gap = np.zeros(int(GAP * rate), dtype=np.int16)
    chunks, t = [], 0.0
    vol = max(0.1, min(2.0, cfg.volume))
    for sc in scenes:
        sc["start"] = t
        off = 0.0
        sc["sent_timing"] = []
        for s in sc["sentences"]:
            a = samples[s]
            if vol != 1.0:
                a = np.clip(a.astype(np.float32) * vol, -32768, 32767).astype(np.int16)
            d = len(a) / rate
            sc["sent_timing"].append({"text": s, "start": off, "dur": d})
            chunks += [a, gap]
            off += d + len(gap) / rate
        sc["duration"] = off
        sc["end"] = t + off
        t += off
    write_wav(narr, rate, np.concatenate(chunks))
    log(f"Narration: {len(scenes)} scenes, {t / 60:.2f} min ({t:.1f}s), limit {cfg.minutes:g} min")
    if t < cfg.minutes * 60 * 0.85:
        warn(f"Narration is shorter than 85% of the target ({t / 60:.1f} of {cfg.minutes:g} min)")
    mark(P, state, "audio", scenes=scenes, total_seconds=t, audio_rate=rate, tts_fallbacks=fallbacks)
    return scenes


# --------------------------------------------------------------------------- stage 5: prompts
def stage_prompts(cfg, P, state, llm):
    scenes = state["scenes"]
    if stage_done(state, "prompts") and all("visual_prompt" in s for s in scenes):
        return
    use_llm = not cfg.smoke
    for sc in scenes:
        sc["analysis"] = analyze(sc["narration"], sc["section"])
        idea = None
        if use_llm:
            try:
                q = (f"Narration: \"{sc['narration'][:420]}\"\n"
                     "Describe ONE simple hand-drawn doodle scene that illustrates this narration, in a single sentence of at most 22 words. "
                     "A stick figure with a big round head does something with one or two chunky finance objects "
                     "(dollar sign, piggy bank, bank building, graph with an arrow, coins). Mention the figure's emotion. "
                     "No text, no quotes. Output only that sentence.")
                txt = llm.generate(q, "You describe simple doodle illustrations.", num_predict=70, temperature=0.8, tries=2)
                txt = re.sub(r"[\"*\n]", " ", txt).strip()
                txt = re.sub(r"\s+", " ", txt)
                txt = split_sentences(txt)[0] if split_sentences(txt) else ""
                if 4 <= len(txt.split()) <= 40:
                    idea = txt.rstrip(".")
            except Exception as e:  # noqa: BLE001
                warn(f"Scene {sc['id']}: prompt idea fell back to template ({str(e)[:80]})")
        if not idea:
            mood_word = {"panic": "looking worried", "happy": "smiling happily", "neutral": "looking curious"}[sc["analysis"]["emotion"]]
            idea = f"a stick figure {mood_word} next to {sc['analysis']['objects'][0]}"
        sc["visual_prompt"], sc["negative_prompt"] = build_prompt(sc["analysis"], idea)
    if use_llm:
        llm.unload()
    mark(P, state, "prompts", scenes=scenes)


# --------------------------------------------------------------------------- stage 6: images
def image_ok(P, sc):
    p = P.images / f"scene_{sc['id']:04d}.png"
    return sc.get("image_source") in ("sd", "fallback") and p.exists() and p.stat().st_size > 1000


def stage_images(cfg, P, state, deadline, fp):
    scenes = state["scenes"]
    todo = [s for s in scenes if not image_ok(P, s)]
    log(f"Images: {len(scenes)} scenes, {len(todo)} to create, time left for images {max(0, deadline - time.time()) / 60:.0f} min")
    engine, sd_fail_reason = None, None
    if todo and not cfg.smoke:
        try:
            engine = SDEngine(cfg)
        except Exception as e:  # noqa: BLE001
            sd_fail_reason = f"image model could not load: {str(e)[:200]}"
            if cfg.require_sd:
                raise PipelineError(sd_fail_reason)
            warn(sd_fail_reason + " -> using clearly marked fallback doodles")
    elif cfg.smoke:
        sd_fail_reason = "smoke mode"
    sec_per_step, consecutive_fail = None, 0
    for k, sc in enumerate(todo):
        path = P.images / f"scene_{sc['id']:04d}.png"
        seed = state["seed"] + sc["id"] * 7919
        steps, img, why = cfg.sd_steps, None, None
        if engine is not None:
            remaining = deadline - time.time()
            allow = remaining / max(1, len(todo) - k)
            if remaining <= 0:
                why = "time budget exhausted"
            elif sec_per_step:
                fit = int((allow * 0.92 - 3.0) / sec_per_step)
                if fit < cfg.sd_min_steps:
                    why = f"time budget too tight ({allow:.0f}s per image)"
                else:
                    steps = min(cfg.sd_steps, fit)
            if why is None:
                for attempt in range(2):
                    try:
                        t = time.time()
                        raw = engine.generate(sc["visual_prompt"], sc["negative_prompt"], seed + attempt, steps)
                        dt = time.time() - t
                        sps = dt / steps
                        sec_per_step = sps if sec_per_step is None else 0.7 * sec_per_step + 0.3 * sps
                        img = add_label(upscale_hd(raw, cfg.img_w, cfg.img_h), sc["analysis"].get("label"), fp, seed)
                        consecutive_fail = 0
                        log(f"  image {k + 1}/{len(todo)} scene {sc['id']} ok: {dt:.0f}s, {steps} steps")
                        break
                    except Exception as e:  # noqa: BLE001
                        why = f"generation error: {str(e)[:150]}"
                        warn(f"Scene {sc['id']} attempt {attempt + 1} failed: {why}")
                        gc.collect()
                        time.sleep(2 * (attempt + 1))
                if img is None:
                    consecutive_fail += 1
                    if consecutive_fail >= 3 and not cfg.require_sd:
                        warn("3 image failures in a row - disabling the image model for the rest of the run")
                        engine = None
                        sd_fail_reason = why
                    elif cfg.require_sd:
                        raise PipelineError(f"Scene {sc['id']}: {why}")
        else:
            why = why or sd_fail_reason or "image model unavailable"
        if img is None:
            img = draw_fallback(sc, cfg.img_w, cfg.img_h, seed, fp)
            sc["image_source"] = "fallback"
            sc["fallback_reason"] = why or sd_fail_reason or "unknown"
            if cfg.smoke is False:
                warn(f"Scene {sc['id']}: FALLBACK doodle used ({sc['fallback_reason']})")
        else:
            sc["image_source"] = "sd"
            sc.pop("fallback_reason", None)
        img.save(path)
        sc["image"] = str(path)
        if (k + 1) % 5 == 0 or k == len(todo) - 1:
            save_json(P.state, state)
    n_fb = sum(1 for s in scenes if s.get("image_source") == "fallback")
    log(f"Images complete: {len(scenes) - n_fb} DreamShaper, {n_fb} fallback")
    engine = None
    gc.collect()
    mark(P, state, "images", scenes=scenes)


# --------------------------------------------------------------------------- stage 7: subtitles
def srt_time(t):
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def ass_time(t):
    cs = int(round(t * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def build_cues(scenes, mode):
    cues = []
    for sc in scenes:
        for st in sc["sent_timing"]:
            words = st["text"].split()
            if not words:
                continue
            n = len(words)
            weights = []
            for i, w in enumerate(words):
                wt = max(1, len(re.sub(r"\W", "", w)))
                if i < n - 1:
                    if w[-1] in ",;:":
                        wt += 2
                    elif w[-1] in ".?!":
                        wt += 3
                weights.append(wt)
            tot = float(sum(weights))
            t0 = sc["start"] + st["start"]
            starts, acc = [], 0.0
            for wt in weights:
                starts.append(t0 + st["dur"] * acc / tot)
                acc += wt
            g = n if mode == "sentence" else int(mode)
            groups = [(i, min(n, i + g)) for i in range(0, n, g)]
            for gi, (a, b) in enumerate(groups):
                s = starts[a]
                e = starts[groups[gi + 1][0]] if gi + 1 < len(groups) else t0 + st["dur"] + 0.12
                if e - s < 0.12:
                    e = s + 0.12
                cues.append((s, e, " ".join(words[a:b])))
    return cues


def stage_subtitles(cfg, P, state, fp):
    scenes = state["scenes"]
    cues = build_cues(scenes, cfg.subtitle_mode)
    # integrity: every narrated word must appear exactly once and in order
    spoken = " ".join(sc["narration"] for sc in scenes).split()
    shown = " ".join(c[2] for c in cues).split()
    if spoken != shown:
        raise PipelineError("Subtitle integrity check failed (words missing or reordered)")
    srt = []
    for i, (s, e, tx) in enumerate(cues, start=1):
        srt.append(f"{i}\n{srt_time(s)} --> {srt_time(e)}\n{tx.upper()}\n")
    (P.subs / "subs.srt").write_text("\n".join(srt), encoding="utf-8")
    font_name = "DejaVu Sans"
    if fp and Path(fp).exists():
        shutil.copy(fp, P.fonts / Path(fp).name)
        font_name = "Permanent Marker" if "marker" in Path(fp).name.lower() else ("DejaVu Sans" if "dejavu" in Path(fp).name.lower() else Path(fp).stem)
    H, W = cfg.H, cfg.W
    size = int(H * 0.085)
    hdr = ("[Script Info]\nScriptType: v4.00+\nPlayResX: %d\nPlayResY: %d\nWrapStyle: 2\nScaledBorderAndShadow: yes\n\n"
           "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, "
           "Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
           "MarginR, MarginV, Encoding\n"
           "Style: Default,%s,%d,&H00FFFFFF,&H00FFFFFF,&H00000000,&H96000000,-1,0,0,0,100,100,1,0,1,%d,%d,2,%d,%d,%d,1\n\n"
           "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
           % (W, H, font_name, size, max(3, size // 10), max(1, size // 25), int(W * 0.06), int(W * 0.06), int(H * 0.07)))
    lines = [hdr]
    for s, e, tx in cues:
        tx = tx.upper().replace("\\", "").replace("{", "").replace("}", "")
        lines.append(f"Dialogue: 0,{ass_time(s)},{ass_time(e)},Default,,0,0,0,,{tx}\n")
    (P.subs / "subs.ass").write_text("".join(lines), encoding="utf-8")
    log(f"Subtitles: {len(cues)} cues in mode '{cfg.subtitle_mode}'")
    mark(P, state, "subtitles", cue_count=len(cues))


# --------------------------------------------------------------------------- stage 8: thumbnail
HOOK_TABLE = [("inflation", "STOLEN?"), ("bank", "BANK CRASH?"), ("debt", "DEBT TRAP!"), ("gold", "GOLD LIES?"),
              ("crash", "IT CRASHED!"), ("central", "WHO PRINTS IT?"), ("money", "YOUR MONEY?"),
              ("interest", "HIDDEN COST?"), ("tax", "WHO PAYS?"), ("saving", "SAFE?"), ("housing", "TRAPPED?")]
THUMB_STYLES = [
    {"fill": (255, 221, 0), "stroke": (0, 0, 0), "band": None},
    {"fill": (255, 255, 255), "stroke": (200, 20, 20), "band": (0, 0, 0, 150)},
    {"fill": (255, 85, 60), "stroke": (255, 255, 255), "band": None},
    {"fill": (120, 255, 140), "stroke": (0, 0, 0), "band": (0, 0, 0, 120)},
    {"fill": (255, 221, 0), "stroke": (30, 30, 130), "band": None},
]


def thumbnail_hook(cfg, llm, rng):
    t = cfg.topic.lower()
    cands = []
    if not cfg.smoke:
        try:
            txt = llm.generate(
                f"Write 6 different punchy YouTube thumbnail hooks (1 to 3 words each, ALL CAPS, may end with ? or !) for a calm "
                f"educational finance video titled '{cfg.topic}'. No numbers, no promises of profit. One hook per line, no numbering.",
                "You write punchy thumbnail text.", num_predict=70, temperature=0.9, tries=2)
            for ln in txt.splitlines():
                ln = re.sub(r"^\s*(?:[-*•]|\d{1,2}[.)])\s*", "", ln)
                ln = re.sub(r"[^A-Za-z0-9 ?!']", "", ln).strip().upper()
                if 1 <= len(ln.split()) <= 3 and 3 <= len(ln) <= 20 and not re.search(r"\d|GUARANTEE|GET RICH|PROFIT", ln):
                    cands.append(ln)
        except Exception:
            pass
    for k, v in HOOK_TABLE:
        if k in t:
            cands.append(v)
    if not cands:
        cands = [" ".join(cfg.topic.upper().split()[:3])[:20] or "WAIT... WHAT?"]
    return rng.choice(cands)


def stage_thumbnail(cfg, P, state, llm, fp):
    rng = random.Random(state["seed"] + 31337)  # different layout / colours / crop for every run
    scenes = state["scenes"]
    pool = [s for s in scenes if s.get("image_source") == "sd"] or scenes
    dramatic = [s for s in pool if s["analysis"]["mood"] == "orange" or s["analysis"]["fill"] == "red"]
    pick = rng.choice(dramatic or pool)
    base = Image.open(pick["image"]).convert("RGB")
    W, H = 1280, 720
    scale = max(W / base.width, H / base.height) * rng.uniform(1.0, 1.3)
    base = base.resize((int(base.width * scale) + 1, int(base.height * scale) + 1), Image.LANCZOS)
    l = int((base.width - W) * rng.uniform(0.2, 0.8))
    t = int((base.height - H) * rng.uniform(0.2, 0.8))
    img = base.crop((l, t, l + W, t + H)).convert("RGBA")
    style = rng.choice(THUMB_STYLES)
    layout = rng.choice(["left", "right", "top", "bottom"])
    hook = thumbnail_hook(cfg, llm, rng)
    words = hook.split()
    lines = words if len(words) <= 2 else [" ".join(words[:2]), " ".join(words[2:])]
    tmp = ImageDraw.Draw(img)
    max_w = (W - 160) if layout in ("top", "bottom") else int(W * 0.58)
    size = 200
    while size > 60:
        f = load_font(fp, size)
        sw = max(4, size // 12)
        bw = max(tmp.textbbox((0, 0), ln, font=f, stroke_width=sw)[2] for ln in lines)
        if bw <= max_w and len(lines) * size * 1.1 <= H * 0.55:
            break
        size -= 8
    f = load_font(fp, size)
    sw = max(4, size // 12)
    bw = max(tmp.textbbox((0, 0), ln, font=f, stroke_width=sw)[2] for ln in lines)
    bh = int(len(lines) * size * 1.1)
    x = {"left": 70, "right": W - 70 - bw, "top": (W - bw) // 2, "bottom": (W - bw) // 2}[layout]
    y = {"left": (H - bh) // 2, "right": (H - bh) // 2, "top": 40, "bottom": H - bh - 50}[layout]
    if style["band"]:
        ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(ov).rounded_rectangle([x - 30, y - 15, x + bw + 30, y + bh + 15], radius=30, fill=style["band"])
        img = Image.alpha_composite(img, ov)
    d = ImageDraw.Draw(img)
    yy = y
    for ln in lines:
        d.text((x, yy), ln, font=f, fill=style["fill"], stroke_width=sw, stroke_fill=style["stroke"])
        yy += int(size * 1.1)
    if rng.random() < 0.55:  # big arrow pointing from the text toward the picture
        sx = x + bw + 15 if layout == "left" else (x - 15 if layout == "right" else W // 2)
        sy = y + bh // 2 if layout in ("left", "right") else (y + bh + 15 if layout == "top" else y - 15)
        ex = int(W * (0.72 if layout == "left" else 0.28)) if layout in ("left", "right") else W // 2 + rng.choice([-250, 250])
        ey = H // 2 if layout in ("left", "right") else (int(H * 0.62) if layout == "top" else int(H * 0.38))
        if math.hypot(ex - sx, ey - sy) > 120:
            draw_arrow(d, (sx, sy), (ex, ey), (230, 30, 30), 16)
    d.rectangle([0, 0, W - 1, H - 1], outline=rng.choice([(0, 0, 0), (255, 255, 255)]), width=10)
    out = P.proj / "thumbnail.png"
    img.convert("RGB").save(out)
    log(f"Thumbnail: '{hook}' (layout {layout}, scene {pick['id']})")
    mark(P, state, "thumbnail", thumbnail_text=hook)


# --------------------------------------------------------------------------- topic choice, SEO, history
BUILTIN_TOPICS = [
    "Why Inflation Steals Your Wealth", "How Central Banks Control Money", "How Banks Create Money From Nothing",
    "What Really Happens In A Bank Run", "Why Your Savings Lose Value Every Year", "How Interest Rates Move The Whole Economy",
    "The Hidden Cost Of Credit Card Debt", "How Compound Interest Works For And Against You", "Why Houses Became So Expensive",
    "What A Recession Actually Is", "How Hyperinflation Destroys A Currency", "The Story Of The Gold Standard",
    "How The Stock Market Really Works", "Why Stock Market Bubbles Keep Happening", "What Is The Federal Reserve And What Does It Do",
    "How Taxes Quietly Shape Your Paycheck", "Why Prices Never Go Back Down", "How Money Evolved From Barter To Bitcoin",
    "The Real Reason Banks Charge Overdraft Fees", "How Mortgages Really Work", "What Happens When A Country Defaults On Its Debt",
    "Why Some Countries Print Money And Others Cannot", "How Index Funds Changed Investing", "What Is Quantitative Easing In Plain English",
    "How Credit Scores Control Your Life", "The Hidden Machinery Behind Every Card Payment", "Why The Rich Get Richer Faster",
    "How Insurance Companies Make Money", "What Is Fractional Reserve Banking", "How Exchange Rates Decide What You Pay",
    "The 2008 Financial Crisis Explained Simply", "Why Student Loans Grew So Large", "How Supply And Demand Set Every Price",
    "What Is Opportunity Cost And Why It Rules Your Choices", "How Pensions Work And Why They Struggle",
    "Why Banks Fail And Who Pays", "How Governments Borrow Money", "What Is Stagflation And Why Is It So Feared",
    "How Currency Collapses Begin", "Why Your Rent Keeps Going Up", "How Gold Became Money", "The Psychology Behind Money Mistakes",
    "How Venture Capital And Startups Really Work", "Why The Dollar Is Used Around The World", "How Wages And Prices Chase Each Other",
    "What Is A Budget Deficit And Why Does It Matter", "How Dividends And Buybacks Actually Work", "The Truth About Get Rich Quick Schemes",
    "How Trade And Tariffs Change What You Pay", "What Is An Emergency Fund And Why It Works",
]
STOPWORDS = {"the", "a", "an", "and", "of", "to", "in", "on", "for", "is", "are", "why", "how", "what", "your", "you", "it",
             "that", "this", "with", "does", "do", "its", "from", "real", "really", "explained", "simply"}


def topic_tokens(s):
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOPWORDS and len(w) > 2}


def too_similar(a, b):
    ta, tb = topic_tokens(a), topic_tokens(b)
    if not ta or not tb:
        return a.lower() == b.lower()
    return len(ta & tb) / len(ta | tb) >= 0.6


def choose_topic(cfg, llm, history):
    rng = random.Random()  # truly random every run
    used = [h.get("topic", "") for h in history if isinstance(h, dict)]
    ideas = []
    if not cfg.smoke:
        try:
            txt = llm.generate(
                "Suggest 12 distinct YouTube video topics for a calm, educational channel about macroeconomics, personal finance, "
                "market history and how money mechanics work (banking, inflation, currencies, debt, investing basics). "
                "Each topic is a short title of 4 to 9 words that sparks curiosity. One per line, no numbering, no extra text. "
                + (f"Avoid these already-made topics: {'; '.join(used[-15:])}." if used else ""),
                "You are a creative YouTube strategist for an educational finance channel.", num_predict=260, temperature=1.0, tries=2)
            for ln in txt.splitlines():
                ln = re.sub(r"^\s*(?:[-*•]|\d{1,2}[.)])\s*", "", ln).strip().strip('"*')
                ln = re.sub(r"[^A-Za-z0-9 ,'\-?]", "", ln).strip()
                if 3 <= len(ln.split()) <= 11 and 15 <= len(ln) <= 80:
                    ideas.append(ln)
        except Exception as e:  # noqa: BLE001
            warn(f"Topic brainstorm failed, using the built-in topic list: {str(e)[:100]}")
    fresh = [t for t in ideas + BUILTIN_TOPICS if not any(too_similar(t, u) for u in used)]
    if fresh:
        # half the time prefer the fresh AI idea, otherwise a curated topic
        ai = [t for t in fresh if t in ideas]
        cur = [t for t in fresh if t in BUILTIN_TOPICS]
        pool = ai if (ai and (not cur or rng.random() < 0.5)) else (cur or fresh)
        return rng.choice(pool)
    warn("Every known topic was already used - re-using the one made longest ago")
    order = {u: i for i, u in enumerate(used)}
    return min(BUILTIN_TOPICS, key=lambda t: order.get(t, -1))


def record_topic(cfg, state, history, pend_path):
    entry = {"topic": cfg.topic, "title": (state.get("seo") or {}).get("title"),
             "date": time.strftime("%Y-%m-%d"), "minutes": cfg.minutes}
    try:
        history = [h for h in history if isinstance(h, dict)] + [entry]
        save_json(cfg.topic_history, history[-500:])
        if cfg.auto_topic:
            save_json(pend_path, {"topic": cfg.topic, "minutes": cfg.minutes, "done": True})
    except Exception as e:  # noqa: BLE001
        warn(f"Could not update topic history: {e}")


def fmt_ts(t):
    t = int(t)
    return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60}:{t % 60:02d}"


def build_chapters(scenes):
    chap, last_sec = [], None
    for sc in scenes:
        sec = sc["section"]
        if sec == last_sec:
            continue
        last_sec = sec
        title = sec
        if not chap:
            title = "Intro"
        elif sec.lower().startswith("modern mirror"):
            title = "What this means for you"
        title = re.sub(r"^The hook:.*", "Intro", title)[:60]
        if chap and sc["start"] - chap[-1][0] < 10:
            continue
        chap.append((sc["start"], title))
    return chap if len(chap) >= 3 else []


def score_title(t, kw):
    n, s = len(t), 0
    s += 3 if 35 <= n <= 65 else (1 if n <= 70 else -3)
    s += min(3, sum(1 for k in kw if k in t.lower()))
    if re.search(r"\b(why|how|what|secret|hidden|really|actually|truth|nobody|explained|machinery)\b", t, re.I):
        s += 1
    if re.search(r"guarantee|get rich|millionaire|100%|free money|!!!", t, re.I):
        s -= 5
    if sum(1 for c in t if c.isupper()) / max(1, n) > 0.5:
        s -= 3
    return s


def clip_words(t, n):
    if len(t) <= n:
        return t
    return t[:n].rsplit(" ", 1)[0].rstrip(",:;-")


def stage_seo(cfg, P, state, llm):
    """Title, description (with chapters), tags. Written for search + click-through, without false promises."""
    rng = random.Random(state["seed"] + 4242)
    topic = cfg.topic
    kw = sorted(topic_tokens(topic))
    cands = []
    if not cfg.smoke:
        try:
            txt = llm.generate(
                f"Write 6 different YouTube titles (35 to 62 characters each) for an educational doodle-animation video about: {topic}. "
                "Use curiosity and plain words, include the main topic keyword, no clickbait lies, no ALL CAPS shouting, no promises of profit. "
                "One title per line, no numbering, no quotes.", "You are a YouTube title writer for an educational finance channel.",
                num_predict=140, temperature=0.9, tries=2)
            for ln in txt.splitlines():
                ln = re.sub(r"^\s*(?:[-*•]|\d{1,2}[.)])\s*", "", ln).strip().strip('"*')
                ln = re.sub(r"[^A-Za-z0-9 ,:'\-?!]", "", ln).strip()
                if 4 <= len(ln.split()) <= 13:
                    cands.append(clip_words(ln, 70))
        except Exception as e:  # noqa: BLE001
            warn(f"SEO title brainstorm failed, using templates: {str(e)[:100]}")
    T = topic.strip().rstrip("?.")
    cands += [clip_words(x, 70) for x in (
        f"{T}: What Nobody Explains", f"The Hidden Machinery Behind {T}", f"{T} Explained Simply",
        f"{T} (And What It Means For Your Wallet)", f"The Truth About {T}")]
    scored = sorted(((score_title(c, kw), c) for c in cands), key=lambda x: -x[0])
    best = scored[0][0]
    title = rng.choice([c for s, c in scored if s >= best - 1][:3])
    desc_hook = None
    if not cfg.smoke:
        try:
            desc_hook = llm.generate(
                f"Write a 2 sentence YouTube description opening for an educational doodle video titled '{title}' about {topic}. "
                "Plain words, second person, no promises, no hashtags. Output only the two sentences.",
                "You write concise YouTube descriptions.", num_predict=90, temperature=0.7, tries=2)
            desc_hook = " ".join(split_sentences(clean_narration(desc_hook))[:2])
        except Exception:
            desc_hook = None
    if not desc_hook or len(desc_hook.split()) < 8:
        desc_hook = (f"{T}, explained with simple hand-drawn doodles. In plain English, you will see how it actually works "
                     "and how it quietly touches your own money.")
    chapters = build_chapters(state["scenes"])
    tag_pool = [topic.lower(), "personal finance", "economics", "money explained", "how money works", "financial education",
                "doodle animation", "whiteboard animation", "macroeconomics", "finance for beginners"] + kw
    tags, total_len = [], 0
    for tg in tag_pool:
        tg = tg.strip()
        if tg and tg not in tags and total_len + len(tg) + 1 <= 450:
            tags.append(tg)
            total_len += len(tg) + 1
    ordered = [w for w in re.findall(r"[a-z0-9]+", topic.lower()) if w in kw and not w.endswith("ly")]
    ordered = list(dict.fromkeys(sorted(ordered, key=len, reverse=True)))[:2]
    hashtags = " ".join("#" + w for w in ordered + ["finance", "economics"])
    description = desc_hook + "\n\n"
    if chapters:
        description += "\n".join(f"{fmt_ts(t)} {name}" for t, name in chapters) + "\n\n"
    description += ("This video is for education only and is not financial advice. "
                    "Always do your own research or speak to a qualified professional.\n\n" + hashtags)
    seo = {"title": title, "description": description, "tags": tags, "thumbnail_text": state.get("thumbnail_text"),
           "alternative_titles": [c for s, c in scored[:5] if c != title]}
    state["seo"] = seo
    slug = state["slug"]
    save_json(P.out / f"{slug}_seo.json", seo)
    (P.out / f"{slug}_seo.txt").write_text(
        f"TITLE:\n{title}\n\nTHUMBNAIL TEXT:\n{seo['thumbnail_text']}\n\nDESCRIPTION:\n{description}\n\nTAGS:\n{', '.join(tags)}\n",
        encoding="utf-8")
    log(f"SEO title: {title}")
    if not cfg.smoke:
        llm.unload()
    mark(P, state, "seo", seo=seo)


# --------------------------------------------------------------------------- stage 9: render
EFFECTS_ALL = ["zoom_in", "zoom_out", "pan_left", "pan_right", "pan_up", "pan_down", "zoom_pan", "track"]
INTENSITY_K = {"subtle": 0.06, "medium": 0.12, "strong": 0.22}


def effect_params(name, k, rng):
    if name == "static":
        return dict(z0=1.0, z1=1.0, fx0=.5, fx1=.5, fy0=.5, fy1=.5)
    c = dict(fx0=.5, fx1=.5, fy0=.5, fy1=.5)
    if name == "zoom_in":
        return dict(z0=1.0, z1=1 + k, **c)
    if name == "zoom_out":
        return dict(z0=1 + k, z1=1.0, **c)
    z = 1 + k
    if name == "pan_left":
        return dict(z0=z, z1=z, fx0=1.0, fx1=0.0, fy0=.5, fy1=.5)
    if name == "pan_right":
        return dict(z0=z, z1=z, fx0=0.0, fx1=1.0, fy0=.5, fy1=.5)
    if name == "pan_up":
        return dict(z0=z, z1=z, fx0=.5, fx1=.5, fy0=1.0, fy1=0.0)
    if name == "pan_down":
        return dict(z0=z, z1=z, fx0=.5, fx1=.5, fy0=0.0, fy1=1.0)
    if name == "zoom_pan":
        a, b = (0.2, 0.8) if rng.random() < 0.5 else (0.8, 0.2)
        return dict(z0=1.0, z1=z, fx0=a, fx1=b, fy0=.35, fy1=.65)
    a, b = (0.0, 1.0) if rng.random() < 0.5 else (1.0, 0.0)  # tracking shot
    return dict(z0=1 + 0.8 * k, z1=1 + k, fx0=a, fx1=b, fy0=.55, fy1=.45)


def choose_effects(cfg, scenes, rng):
    k = INTENSITY_K[cfg.intensity]
    if cfg.animation == "off":
        pool = ["static"]
    elif cfg.animation == "minimal":
        pool, k = ["zoom_in", "zoom_out"], k * 0.6
    elif cfg.animation == "cinematic":
        pool, k = ["zoom_in", "zoom_out", "track", "zoom_pan", "pan_left", "pan_right"], k * 1.25
    else:
        pool = EFFECTS_ALL
    prev = None
    for sc in scenes:
        opts = [p for p in pool if p != prev] if (len(pool) > 1 and not cfg.no_avoid_repeat) else pool
        name = rng.choice(opts)
        prev = name
        sc["animation"] = name
        sc["animation_params"] = effect_params(name, k, rng)


def choose_transitions(cfg, scenes, rng, xfade_ok):
    mode = cfg.transition
    if not xfade_ok and mode not in ("cut", "off"):
        warn("This FFmpeg has no 'xfade' filter - falling back to direct cuts")
        mode = "cut"
    if mode in ("cut", "off"):
        pool = ["cut"]
    elif mode == "crossfade":
        pool = ["fade"]
    elif mode == "cinematic":
        pool = ["fade", "fadeblack", "dissolve"]
    else:
        pool = ["fade", "dissolve", "fadeblack", "slideleft", "slideright", "fade", "dissolve", "cut"]
    prev = None
    for i, sc in enumerate(scenes):
        if i == len(scenes) - 1:
            sc["transition_out"] = "none"
            continue
        opts = [p for p in pool if p != prev] if (len(set(pool)) > 1 and not cfg.no_avoid_repeat) else pool
        name = rng.choice(opts)
        prev = name
        sc["transition_out"] = name


def motion_chain(eff, c, dur, fps, W, H, d_frames):
    P_ = f"min(1,max(0,({c:.6f}+on/{fps})/{dur:.6f}))"
    z = f"({eff['z0']:.4f}+({eff['z1']:.4f}-{eff['z0']:.4f})*{P_})"
    x = f"(iw-iw/{z})*({eff['fx0']:.4f}+({eff['fx1']:.4f}-{eff['fx0']:.4f})*{P_})"
    y = f"(ih-ih/{z})*({eff['fy0']:.4f}+({eff['fy1']:.4f}-{eff['fy0']:.4f})*{P_})"
    W2, H2 = even(W * 1.5), even(H * 1.5)
    return (f"scale={W2}:{H2}:force_original_aspect_ratio=increase:flags=lanczos,crop={W2}:{H2},"
            f"zoompan=z='{z}':x='{x}':y='{y}':d={d_frames}:s={W}x{H}:fps={fps},format=yuv420p")


def enc_args(fps):
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-bf", "0",
            "-g", str(fps * 2), "-threads", "2", "-an", "-movflags", "+faststart"]


def render_body(cfg, sc, f0, n, out):
    fps = cfg.fps
    c = f0 / fps - sc["start"]
    vf = motion_chain(sc["animation_params"], c, sc["duration"], fps, cfg.W, cfg.H, n)
    run([FFMPEG, "-y", "-v", "error", "-i", sc["image"], "-vf", vf, "-frames:v", n] + enc_args(fps) + [out], timeout=900)


def render_bridge(cfg, a, b, f0, n, name, out):
    fps = cfg.fps
    ca = motion_chain(a["animation_params"], f0 / fps - a["start"], a["duration"], fps, cfg.W, cfg.H, n + 1)
    cb = motion_chain(b["animation_params"], f0 / fps - b["start"], b["duration"], fps, cfg.W, cfg.H, n + 1)
    fc = (f"[0:v]{ca}[a];[1:v]{cb}[b];[a][b]xfade=transition={name}:duration={n / fps:.4f}:offset=0,format=yuv420p[v]")
    run([FFMPEG, "-y", "-v", "error", "-i", a["image"], "-i", b["image"], "-filter_complex", fc, "-map", "[v]",
         "-frames:v", n] + enc_args(fps) + [out], timeout=900)


def plan_render(cfg, scenes, total):
    fps = cfg.fps
    F = lambda t: int(round(t * fps))  # noqa: E731
    total_frames = F(total)
    pieces, cursor, n = [], 0, len(scenes)
    for i, sc in enumerate(scenes):
        if i < n - 1:
            nxt = scenes[i + 1]
            B = sc["end"]
            name = sc.get("transition_out", "cut")
            h = min(cfg.transition_seconds / 2.0, 0.4 * sc["duration"], 0.4 * nxt["duration"])
            bf0, bf1 = F(B - h), F(B + h)
            if name != "cut" and bf1 - bf0 >= 2 and bf0 >= cursor:
                if bf0 > cursor:
                    pieces.append({"type": "body", "scene": i, "f0": cursor, "n": bf0 - cursor})
                pieces.append({"type": "bridge", "a": i, "b": i + 1, "f0": bf0, "n": bf1 - bf0, "name": name})
                cursor = bf1
            else:
                sc["transition_out"] = "cut"
                end = F(B)
                if end > cursor:
                    pieces.append({"type": "body", "scene": i, "f0": cursor, "n": end - cursor})
                    cursor = end
        else:
            if total_frames > cursor:
                pieces.append({"type": "body", "scene": i, "f0": cursor, "n": total_frames - cursor})
                cursor = total_frames
    return pieces, total_frames


def stage_render(cfg, P, state):
    scenes = state["scenes"]
    total = state["total_seconds"]
    rng = random.Random(state["seed"])
    xfade_ok = ffmpeg_has_filter("xfade")
    choose_effects(cfg, scenes, rng)
    choose_transitions(cfg, scenes, rng, xfade_ok)
    pieces, total_frames = plan_render(cfg, scenes, total)
    sig = hashlib.sha1(json.dumps([pieces, cfg.W, cfg.H, cfg.fps, [(s["animation"], s["animation_params"], s["image"]) for s in scenes]],
                                  sort_keys=True).encode()).hexdigest()
    sigf = P.render / "sig.txt"
    if not (sigf.exists() and sigf.read_text() == sig):
        shutil.rmtree(P.pieces, ignore_errors=True)
        P.pieces.mkdir(parents=True, exist_ok=True)
        sigf.write_text(sig)
    log(f"Rendering {len(pieces)} clips for {len(scenes)} scenes ({total_frames} frames @ {cfg.fps} fps, {cfg.W}x{cfg.H})")
    fallback_cuts = []

    def job(idx):
        pc = pieces[idx]
        out = P.pieces / f"p_{idx:05d}.mp4"
        if out.exists() and out.stat().st_size > 500:
            return idx
        tmp = P.pieces / f"p_{idx:05d}.part.mp4"
        if pc["type"] == "body":
            render_body(cfg, scenes[pc["scene"]], pc["f0"], pc["n"], tmp)
        else:
            a, b = scenes[pc["a"]], scenes[pc["b"]]
            try:
                render_bridge(cfg, a, b, pc["f0"], pc["n"], pc["name"], tmp)
            except Exception as e:  # noqa: BLE001
                warn(f"Transition '{pc['name']}' failed at boundary {a['id']}->{b['id']}, using a direct cut: {str(e)[:150]}")
                fallback_cuts.append(idx)
                half = pc["n"] // 2
                t1, t2 = P.pieces / f"p_{idx:05d}.a.mp4", P.pieces / f"p_{idx:05d}.b.mp4"
                render_body(cfg, a, pc["f0"], half, t1)
                render_body(cfg, b, pc["f0"] + half, pc["n"] - half, t2)
                lst = P.pieces / f"p_{idx:05d}.txt"
                lst.write_text(f"file '{t1.resolve()}'\nfile '{t2.resolve()}'\n")
                run([FFMPEG, "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", tmp])
        os.replace(tmp, out)
        return idx

    workers = max(1, (os.cpu_count() or 2) // 2)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(job, i) for i in range(len(pieces))]
        for f in as_completed(futs):
            f.result()
            done += 1
            if done % 25 == 0 or done == len(pieces):
                log(f"  clips {done}/{len(pieces)}")
    lst = P.render / "concat.txt"
    lst.write_text("".join(f"file '{(P.pieces / f'p_{i:05d}.mp4').resolve()}'\n" for i in range(len(pieces))))
    raw = P.render / "video_raw.mp4"
    run([FFMPEG, "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", raw], timeout=1800)
    info = probe(raw)
    exp = total_frames / cfg.fps
    log(f"Concatenated visuals: {info['duration']:.2f}s (expected {exp:.2f}s)")
    state["transition_fallbacks"] = len(fallback_cuts)
    mark(P, state, "render", total_frames=total_frames)
    return raw


def stage_final(cfg, P, state, raw, slug):
    total = state["total_seconds"]
    narr = P.audio / "narration.wav"
    out_tmp = P.render / "final.mp4"
    base = ["-y", "-v", "error", "-i", raw.relative_to(P.proj).as_posix(), "-i", narr.relative_to(P.proj).as_posix()]
    tail = ["-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p",
            "-r", str(cfg.fps), "-c:a", "aac", "-b:a", "160k", "-ar", "44100", "-movflags", "+faststart",
            "-t", f"{total:.3f}", out_tmp.relative_to(P.proj).as_posix()]
    vf_pad = f"tpad=stop_mode=clone:stop_duration=1"  # safety: never let video end before audio
    burned = True
    try:
        run([FFMPEG] + base + ["-vf", f"{vf_pad},ass=subs/subs.ass:fontsdir=fonts"] + tail, cwd=P.proj, timeout=7200)
    except PipelineError as e:
        warn(f"Burning subtitles failed ({str(e)[:200]}). Rendering without burned-in subtitles; SRT/ASS are kept next to the MP4.")
        burned = False
        run([FFMPEG] + base + ["-vf", vf_pad] + tail, cwd=P.proj, timeout=7200)
    info = probe(out_tmp)
    if not (info["has_video"] and info["has_audio"]):
        raise PipelineError("Final MP4 is missing a video or audio stream")
    if abs(info["duration"] - total) > 1.0:
        raise PipelineError(f"Final duration {info['duration']:.2f}s differs from narration {total:.2f}s")
    limit = cfg.minutes * 60
    if info["duration"] > limit + 0.5:
        warn(f"Final video ({info['duration']:.1f}s) is above the {limit:.0f}s limit")
    final = P.out / f"{slug}.mp4"
    shutil.move(str(out_tmp), final)
    shutil.copy(P.subs / "subs.srt", P.out / f"{slug}.srt")
    shutil.copy(P.subs / "subs.ass", P.out / f"{slug}.ass")
    if (P.proj / "thumbnail.png").exists():
        shutil.copy(P.proj / "thumbnail.png", P.out / f"{slug}_thumbnail.png")
    if (P.proj / "script.json").exists():
        shutil.copy(P.proj / "script.json", P.out / f"{slug}_script.json")
    scenes = state["scenes"]
    report = {
        "topic": cfg.topic, "title": (state.get("seo") or {}).get("title") or cfg.topic,
        "thumbnail_text": state.get("thumbnail_text"), "file": str(final), "size_mb": round(final.stat().st_size / 1e6, 1),
        "duration_seconds": round(info["duration"], 2), "limit_seconds": limit,
        "resolution": f"{info['width']}x{info['height']}", "scenes": len(scenes),
        "images_dreamshaper": sum(1 for s in scenes if s.get("image_source") == "sd"),
        "images_fallback": sum(1 for s in scenes if s.get("image_source") == "fallback"),
        "tts_silent_fallbacks": state.get("tts_fallbacks", 0),
        "transition_fallbacks": state.get("transition_fallbacks", 0),
        "subtitles_burned": burned, "pipeline_minutes": round((time.time() - T0) / 60, 1),
        "scene_manifest": [{k: s.get(k) for k in ("id", "start", "end", "duration", "animation", "transition_out", "image", "image_source")}
                           for s in scenes],
    }
    save_json(P.out / f"{slug}_report.json", report)
    mark(P, state, "final", report={k: v for k, v in report.items() if k != "scene_manifest"})
    return report


# --------------------------------------------------------------------------- main
def write_summary(report):
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    md = ["## Doodle video finished", "",
          f"- **YouTube title:** {report.get('title')}",
          f"- **Topic:** {report['topic']}",
          f"- **Thumbnail text:** {report.get('thumbnail_text')}",
          f"- **Length:** {report['duration_seconds'] / 60:.1f} min (limit {report['limit_seconds'] / 60:.0f} min)",
          f"- **Scenes:** {report['scenes']} (DreamShaper {report['images_dreamshaper']}, fallback {report['images_fallback']})",
          f"- **Size:** {report['size_mb']} MB, {report['resolution']}",
          f"- **Pipeline time:** {report['pipeline_minutes']} min", ""]
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n".join(md) + "\n")
    except Exception:
        pass


def main(argv=None):
    cfg = parse_args(argv)
    find_ffmpeg()
    llm = Ollama(cfg.ollama_url, cfg.ollama_model)
    if not cfg.smoke:
        llm.start()
        llm.ensure_model()
    history = load_json(cfg.topic_history, []) or []
    Path(cfg.workdir).mkdir(parents=True, exist_ok=True)
    pend_path = Path(cfg.workdir) / "_auto_topic.json"
    if cfg.auto_topic:
        pend = load_json(pend_path, None)
        if pend and not pend.get("done") and pend.get("topic") and pend.get("minutes") == cfg.minutes:
            cfg.topic = pend["topic"]
            log(f"Resuming auto-selected topic: {cfg.topic}")
        else:
            cfg.topic = choose_topic(cfg, llm, history)
            save_json(pend_path, {"topic": cfg.topic, "minutes": cfg.minutes, "done": False})
            log(f"Auto-selected topic: {cfg.topic}")
    slug = re.sub(r"[^a-z0-9]+", "-", cfg.topic.lower()).strip("-")[:48] + f"-{cfg.minutes:g}m"
    if cfg.smoke:
        slug += "-smoke"
    P = Paths(cfg, slug)
    state = load_json(P.state, None) or {"version": 1, "slug": slug, "stages": {}}
    if not state.get("seed"):
        state["seed"] = cfg.seed or int(time.time()) % 1000000
    state["config"] = {k: v for k, v in vars(cfg).items()}
    save_json(P.state, state)
    log(f"Project '{slug}' | {cfg.minutes:g} min | {cfg.H}p | seed {state['seed']} | smoke={cfg.smoke}")

    fp = find_font(cfg)
    log(f"Font: {fp or 'PIL default'}")
    budget_s = cfg.budget_minutes * 60
    video_s = cfg.minutes * 60
    cores = os.cpu_count() or 2
    res_factor = 1.0 if cfg.H <= 720 else 2.3
    core_factor = 1.0 if cores >= 4 else 2.0
    # time kept free for subtitles, thumbnail, SEO, render and final encode (deliberately pessimistic)
    reserve = 900 + video_s * 2.2 * res_factor * core_factor
    image_deadline = T0 + budget_s - reserve
    log(f"{cores} CPU cores. Time budget {cfg.budget_minutes:g} min; images must finish by +{(image_deadline - T0) / 60:.0f} min "
        f"(reserve {reserve / 60:.0f} min for render)")

    piper = Piper(cfg)
    log(f"Piper: {'ready' if piper.available else 'NOT available'} ({piper.exe}, {piper.model.name})")

    wps = stage_calibrate(cfg, P, state, piper)
    stage_script(cfg, P, state, wps, llm)
    stage_scenes(cfg, P, state, wps)
    stage_audio(cfg, P, state, piper)
    stage_prompts(cfg, P, state, llm)
    stage_images(cfg, P, state, image_deadline, fp)
    stage_subtitles(cfg, P, state, fp)
    stage_thumbnail(cfg, P, state, llm, fp)
    stage_seo(cfg, P, state, llm)
    left = T0 + budget_s - time.time()
    need = 300 + state["total_seconds"] * 1.0 * res_factor * core_factor
    if left < need and not (cfg.animation == "off" and cfg.transition in ("cut", "off")):
        warn(f"Only {left / 60:.0f} min left for rendering (~{need / 60:.0f} min needed): switching to the light render "
             "(no scene motion, direct cuts) to finish inside the time limit")
        cfg.animation, cfg.transition = "off", "cut"
    raw = stage_render(cfg, P, state)
    report = stage_final(cfg, P, state, raw, slug)
    record_topic(cfg, state, history, pend_path)
    save_json(Path(cfg.output) / "latest.json", {"topic": cfg.topic, "title": report["title"], "slug": slug,
                                                  "video": report["file"], "seo_file": str(P.out / f"{slug}_seo.txt")})
    write_summary(report)
    log(f"DONE: {report['file']} ({report['duration_seconds'] / 60:.1f} min, {report['scenes']} scenes, "
        f"{report['size_mb']} MB, {report['pipeline_minutes']} min pipeline time)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PipelineError as e:
        log(str(e), "ERROR")
        sys.exit(2)
    except KeyboardInterrupt:
        sys.exit(130)
