#!/usr/bin/env python3
"""Local-model stickman finance documentary builder for GitHub Actions.

No external AI API keys are required. Text generation, TTS and image generation
run locally on the Actions runner after public model weights are downloaded.
A normal GitHub-hosted Linux runner has no GPU, so SD-Turbo inference is CPU-based.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import requests
import soundfile as sf
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "studio_work"
STATE_DIR = ROOT / ".studio_state"
OUTPUT_DIR = ROOT / "output"
CHECKPOINT = WORK / "checkpoint.json"
SAMPLE_RATE = 24000
FPS = 30
WIDTH, HEIGHT = 1920, 1080
IMAGE_INTERVAL = 5.0
VIDEO_SECONDS_FULL = 30 * 60
VIDEO_SECONDS_TEST = 60
TEXT_MODEL_ID = os.environ.get("TEXT_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
IMAGE_MODEL_ID = os.environ.get("IMAGE_MODEL_ID", "stabilityai/sdxl-turbo")
VOICE_NAME = "am_puck"
NEGATIVE_PROMPT = "colored skin on stickman, realistic human, 3d render, blurry, watermark, text, deformed limbs, low quality"

TOPICS = [
    "The Quiet Math Behind Lifestyle Inflation",
    "Why High Earners Still Feel Broke",
    "The Psychology of Debt and Minimum Payments",
    "How Compound Growth Rewards Patience",
    "The Hidden Cost of Buying Status",
    "Why Emergency Funds Create Real Freedom",
    "The Wealth Gap Between Income and Assets",
    "How Inflation Changes Everyday Decisions",
    "Why Your First Investment Habit Matters",
    "The Behavioral Traps Behind Panic Selling",
    "How Subscription Creep Drains Wealth",
    "The Difference Between Looking Rich and Being Rich",
    "Why Cash Flow Matters More Than Appearances",
    "The Psychology of Spending After a Pay Raise",
    "How High-Interest Debt Compounds Against You",
    "Why Small Fees Can Become Big Financial Leaks",
    "The Real Trade-Off Between Renting and Buying",
    "How Opportunity Cost Shapes Your Future",
    "Why Financial Advice Often Ignores Human Behavior",
    "The Wealth-Building Power of Boring Consistency",
    "How Consumer Credit Changes Spending Decisions",
    "Why a Bigger Salary Does Not Guarantee Security",
    "The Economics of Convenience Spending",
    "How Risk and Time Horizon Work Together",
    "Why Diversification Is a Behavior Strategy Too",
    "The Hidden Price of Buy Now Pay Later",
    "How Marketing Turns Wants Into Needs",
    "Why Financial Goals Need a Time Frame",
    "The Difference Between Saving and Investing",
    "How Lifestyle Choices Affect Financial Optionality",
    "Why Financial Stress Can Narrow Your Choices",
    "The Economics of Owning Productive Assets",
    "How Paycheck Timing Shapes Money Decisions",
    "Why a Budget Is a Feedback System, Not a Punishment",
    "The Psychology of Loss Aversion in Investing",
    "How Inflation and Raises Interact Over Time",
    "Why Financial Resilience Beats Perfect Forecasts",
    "The Hidden Cost of Car Payments",
    "How Social Comparison Distorts Money Goals",
    "Why the First Ten Thousand Dollars Feel Different",
    "The Risk of Confusing Luck With Skill",
    "How Cash Reserves Change Negotiating Power",
    "Why Debt-Free Does Not Automatically Mean Wealthy",
    "The Economics of Time, Energy, and Money",
    "How Financial Habits Become Automatic",
    "Why Wealth Is Often Built Through Avoiding Mistakes",
    "The Power and Limits of Compound Interest",
    "How Financial Incentives Shape Everyday Life",
    "Why Diversified Income Is Not the Same as Easy Money",
    "The Psychology of Spending Windfalls",
    "How to Think Clearly During a Market Crash",
    "Why Financial Independence Is About Options",
    "The Difference Between Net Worth and Cash Flow",
    "How Debt Changes Future Freedom",
    "Why a Good Plan Includes Uncertainty",
    "The Hidden Economics of Free Trials",
    "How Small Recurring Choices Shape Long-Term Wealth",
    "Why Simple Financial Systems Beat Motivation",
    "The Trade-Off Between Convenience and Control",
    "How Money Stories Influence Risk Taking",
    "Why Protecting Your Downside Matters",
    "The Economics of Waiting Before You Buy",
    "How a Financial Cushion Changes Career Choices",
    "Why Wealth Is a System, Not a Single Trick",
    "The Psychology of Delayed Gratification",
    "How to Separate Market Noise From Long-Term Decisions",
    "Why a High Savings Rate Creates Options",
    "The Hidden Cost of Financial Overconfidence",
    "How Fees, Taxes, and Inflation Interact",
    "Why Your Money Plan Must Fit Your Real Life",
]

CHAPTER_SPECS = [
    ("The Moment Money Starts Controlling the Story", "Open with an ordinary money decision and expose the hidden trade-off behind it."),
    ("The Numbers Beneath the Feeling", "Explain cash flow, compounding, purchasing power, and why arithmetic beats impressions."),
    ("The Brain Wants Comfort Now", "Explore present bias, reward loops, social comparison, and emotional spending."),
    ("Income Is Not the Same as Wealth", "Separate wages, savings, productive assets, liabilities, liquidity, and net worth."),
    ("Debt Turns Tomorrow Into a Bill", "Explain interest, minimum payments, opportunity cost, and the value of reading terms."),
    ("The Quiet Leak of Lifestyle Inflation", "Show how fixed commitments can absorb raises and reduce flexibility."),
    ("Time Is a Financial Multiplier", "Explain time horizon, consistent contributions, uncertainty, and the limits of projections."),
    ("Risk Is More Than a Number", "Discuss concentration, volatility, liquidity, job risk, and matching risk to capacity."),
    ("The Systems That Make Good Choices Easier", "Explain budgets, automatic transfers, emergency reserves, and simple feedback loops."),
    ("The Stories That Sell You Things", "Analyze status signaling, urgency, anchoring, defaults, and marketing incentives."),
    ("Build a Plan That Survives Real Life", "Offer practical reflection questions and risk-aware steps without promises or stock picks."),
    ("The Real Measure of Financial Freedom", "Bring the themes together around resilience, choice, time, and realistic expectations."),
]

TAG_LIST = [
    "personal finance", "financial education", "wealth building", "money psychology", "investing basics",
    "financial literacy", "compound interest", "debt payoff", "budgeting tips", "money habits",
    "wealth mindset", "saving money", "behavioral finance", "stock market basics", "passive income basics",
    "financial freedom", "inflation explained", "cash flow", "net worth", "credit card debt",
    "emergency fund", "lifestyle inflation", "long term investing", "money management", "economics explained",
    "financial independence", "spending habits", "risk management", "assets and liabilities", "smart money",
    "financial mistakes", "investing psychology", "wealth gap", "money documentary", "stickman finance",
]

FALLBACK_SENTENCE_PATTERNS = [
    "A useful way to study {topic} is to separate the visible choice from the incentives working behind it.",
    "When you look only at the price today, you can miss the time, flexibility, and future options that the decision consumes.",
    "The first question is not whether a choice looks impressive, but whether it supports the life you are trying to build.",
    "A simple example makes the trade-off clearer: money assigned to one purpose cannot be used for another purpose at the same time.",
    "That does not make every purchase wrong; it means every purchase has a cost beyond the number printed on the receipt.",
    "Your future self inherits both the benefits and the commitments created by decisions that feel small in the present.",
    "The arithmetic is often straightforward, but emotions can make a short-term reward feel more important than a long-term outcome.",
    "You can reduce that conflict by deciding your rules before the pressure, advertising, or urgency arrives.",
    "A useful financial system makes the preferred choice easier to repeat and the risky choice harder to make impulsively.",
    "There is no universal number that works for everyone, because income, dependents, location, health, and job stability differ.",
    "Treat any projection as a scenario rather than a promise, and ask what happens if returns are lower or expenses are higher.",
    "The point is not to predict every surprise; the point is to make sure one surprise does not destroy the whole plan.",
    "A longer time horizon can help with uncertainty, but it does not remove risk or guarantee a positive outcome.",
    "When you compare options, include fees, taxes, interest, maintenance, time, and the possibility that your plans may change.",
    "A decision that preserves flexibility can be valuable even when it does not produce the largest result in an optimistic forecast.",
    "Notice how the decision feels before you make it, because urgency and embarrassment can be signals that deserve a pause.",
    "The best next step is usually one you can explain clearly, afford under reasonable assumptions, and maintain consistently.",
    "Small improvements become meaningful when they reduce repeated mistakes instead of depending on constant willpower.",
    "Compare the outcome with your own goals rather than with a highlight reel built from somebody else's spending.",
    "A resilient plan leaves room for ordinary life, including repairs, family needs, career changes, and periods of uncertainty.",
    "If a financial offer depends on you ignoring the terms, rushing, or assuming the best possible outcome, slow down and examine it.",
    "Financial progress is usually easier to measure through habits and balance-sheet changes than through a single exciting moment.",
    "Your goal is not to eliminate every trade-off; it is to choose trade-offs deliberately rather than by default.",
    "The lesson becomes practical when you turn it into one measurable decision that can be reviewed next month.",
]


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temp_path.replace(path)


def load_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def run_checked(command: list[str], *, cwd: Path | None = None, capture: bool = False) -> subprocess.CompletedProcess[str]:
    log("RUN: " + " ".join(str(part) for part in command[:8]) + (" ..." if len(command) > 8 else ""))
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


def assert_tools() -> None:
    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing:
        raise RuntimeError(f"Required system tools are missing: {', '.join(missing)}. The workflow must install ffmpeg.")


def normalize_text(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"^\s{0,3}#{1,6}\s+.*$", " ", text, flags=re.M)
    text = re.sub(r"^\s*\d+[.)]\s+", "", text, flags=re.M)
    text = re.sub(r"\*\*(.*?)\*\*|__(.*?)__", lambda m: m.group(1) or m.group(2), text)
    text = re.sub(r"[*_`#]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    # Preserve punctuation while splitting ordinary documentary prose into TTS units.
    pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9“\"'])", text)
    result: list[str] = []
    for item in pieces:
        item = item.strip()
        if item:
            if item[-1] not in ".!?…":
                item += "."
            result.append(item)
    return result


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w’'-]+\b", text, flags=re.UNICODE))


def github_release_history() -> list[dict[str, Any]]:
    repo = os.environ.get("GITHUB_REPOSITORY", "").strip()
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not repo or not token:
        return []
    headers = {"Accept": "application/vnd.github+json", "Authorization": f"Bearer {token}", "X-GitHub-Api-Version": "2022-11-28"}
    releases: list[dict[str, Any]] = []
    for page in range(1, 4):
        try:
            response = requests.get(f"https://api.github.com/repos/{repo}/releases", headers=headers, params={"per_page": 100, "page": page}, timeout=20)
            if response.status_code == 404:
                break
            response.raise_for_status()
            batch = response.json()
            if not batch:
                break
            releases.extend(batch)
            if len(batch) < 100:
                break
        except requests.RequestException as exc:
            log(f"Release history lookup failed; local topic history will be used: {exc}")
            break
    return releases


def select_topic() -> str:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    local_history = load_json(STATE_DIR / "topics.json", {"topics": []})
    used = {str(item).strip().casefold() for item in local_history.get("topics", [])}
    for release in github_release_history():
        title = str(release.get("name") or "")
        body = str(release.get("body") or "")
        for candidate in (title, body):
            match = re.search(r"(?im)^\s*topic\s*:\s*(.+?)\s*$", candidate)
            if match:
                used.add(match.group(1).strip().casefold())
        for topic in TOPICS:
            if topic.casefold() in title.casefold():
                used.add(topic.casefold())
    available = [topic for topic in TOPICS if topic.casefold() not in used]
    if available:
        chosen = random.SystemRandom().choice(available)
    else:
        # Preserve uniqueness after the base pool is consumed by adding a fresh angle marker.
        base = random.SystemRandom().choice(TOPICS)
        nonce = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC") + " " + uuid.uuid4().hex[:6]
        chosen = f"{base} — a new case study ({nonce})"
    known = list(local_history.get("topics", []))
    known.append(chosen)
    atomic_json(STATE_DIR / "topics.json", {"topics": known[-1000:]})
    return chosen


def local_text_model() -> tuple[Any, Any, Any] | tuple[None, None, None]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(max(1, min(2, os.cpu_count() or 1)))
        log(f"Loading local text model: {TEXT_MODEL_ID}")
        tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_ID, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(
            TEXT_MODEL_ID,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        )
        model.to("cpu")
        model.eval()
        return torch, tokenizer, model
    except Exception as exc:
        log(f"Local text model unavailable; using built-in finance-writing framework instead: {exc}")
        return None, None, None


def ask_local_writer(torch: Any, tokenizer: Any, model: Any, prompt: str, max_new_tokens: int) -> str:
    messages = [
        {"role": "system", "content": "You write careful, engaging financial-education documentaries. Avoid fabricated statistics, guaranteed returns, stock picks, and personalized financial advice. Use direct second-person narration, natural spoken English, and complete sentences. Do not use lists."},
        {"role": "user", "content": prompt},
    ]
    try:
        encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        attention_mask = torch.ones_like(encoded)
        with torch.inference_mode():
            generated = model.generate(
                input_ids=encoded,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.78,
                top_p=0.9,
                repetition_penalty=1.12,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(generated[0][encoded.shape[-1]:], skip_special_tokens=True)
        return normalize_text(text)
    except Exception as exc:
        log(f"Chapter generation failed and will use the built-in writer for this chapter: {exc}")
        return ""


def fallback_sentences(topic: str, chapter_title: str, chapter_index: int) -> list[str]:
    sentences: list[str] = []
    for offset, pattern in enumerate(FALLBACK_SENTENCE_PATTERNS):
        sentence = pattern.format(topic=topic, chapter_title=chapter_title)
        if offset % 5 == 0:
            sentence = sentence.replace("the decision", f"the decision in {chapter_title.lower()}")
        if offset % 7 == 0:
            sentence += f" In this chapter, you are examining {chapter_title.lower()}, not searching for a shortcut."
        sentences.append(sentence)
    # Chapter-specific analytical bridge makes fallback passages less generic.
    bridges = [
        f"For {chapter_title.lower()}, the practical test is to ask what the choice changes today and what it commits you to tomorrow.",
        f"Return to {topic.lower()} and compare a best-case story with a less comfortable but plausible scenario.",
        "A sound decision should remain understandable after the excitement of the moment has passed.",
        "Write down the assumption that matters most, then ask how the result changes when that assumption is wrong.",
        "You do not need a perfect forecast to notice a fee, a recurring obligation, or a risk you cannot afford.",
        "The useful habit is to review the evidence, make one deliberate choice, and measure the result without shaming yourself.",
        "When two options look similar, preserving liquidity and flexibility may matter more than optimizing a spreadsheet by a tiny amount.",
        "This is why a personal plan must fit your actual responsibilities rather than an idealized example from the internet.",
    ]
    sentences.extend(bridges)
    random.Random(chapter_index + len(topic)).shuffle(sentences)
    return sentences


def fit_chapter(text: str, topic: str, title: str, index: int, target_words: int) -> str:
    candidate = split_sentences(text)
    fillers = fallback_sentences(topic, title, index)
    filler_idx = 0
    current_words = sum(word_count(s) for s in candidate)
    while current_words < target_words:
        sentence = fillers[filler_idx % len(fillers)]
        filler_idx += 1
        candidate.append(sentence)
        current_words += word_count(sentence)
    # Keep complete sentences where possible; one final sentence can put the chapter a little over target.
    kept: list[str] = []
    total = 0
    for sentence in candidate:
        size = word_count(sentence)
        if total >= target_words and total + size > target_words + 14:
            break
        kept.append(sentence)
        total += size
        if total >= target_words:
            break
    while total < target_words:
        sentence = fillers[filler_idx % len(fillers)]
        filler_idx += 1
        needed = target_words - total
        tokens = sentence.split()
        if len(tokens) > needed and needed >= 7:
            sentence = " ".join(tokens[:needed]).rstrip(",;:") + "."
        kept.append(sentence)
        total += word_count(sentence)
    return " ".join(kept)


def generate_script(topic: str, test_run: bool) -> dict[str, Any]:
    torch, tokenizer, model = local_text_model()
    chapters: list[dict[str, Any]] = []
    if test_run:
        title = "A One-Minute Money Lesson"
        prompt = (
            f"Write about 145 words for a one-minute stickman finance documentary about {topic}. "
            "Open with a vivid everyday money moment, explain one mechanism, and finish with one practical question. "
            "Use second-person 'you', natural spoken English, no headings, no bullet points, no fabricated statistics."
        )
        generated = ask_local_writer(torch, tokenizer, model, prompt, 250) if model is not None else ""
        text = fit_chapter(generated, topic, title, 0, 135)
        chapters.append({"title": title, "text": text})
    else:
        for index, (title, angle) in enumerate(CHAPTER_SPECS):
            prompt = (
                f"Write approximately 365 words for chapter {index + 1} of a 12-chapter finance documentary.\n"
                f"Main topic: {topic}.\nChapter title: {title}.\nChapter focus: {angle}\n"
                "Use natural, engaging, second-person narration addressed to 'you'. Include a concrete everyday example, explain the financial mechanism, and identify a limitation or risk. "
                "Use complete spoken sentences and paragraphs, no title, no bullets, no numbered list, no markdown. "
                "Do not invent statistics, present projections as guarantees, recommend specific securities, or claim that a tactic always works. Avoid repeating the chapter opening."
            )
            generated = ask_local_writer(torch, tokenizer, model, prompt, 650) if model is not None else ""
            text = fit_chapter(generated, topic, title, index, 365)
            chapters.append({"title": title, "text": text})
            log(f"Prepared chapter {index + 1}/{len(CHAPTER_SPECS)} ({word_count(text)} words): {title}")
    if model is not None:
        del model, tokenizer, torch
        gc.collect()
    full_text = "\n\n".join(f"{chapter['title']}.\n{chapter['text']}" for chapter in chapters)
    return {"topic": topic, "chapters": chapters, "script_text": full_text, "word_count": word_count(full_text)}


def flatten_sentences(script_data: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for chapter_index, chapter in enumerate(script_data["chapters"]):
        for sentence in split_sentences(chapter["text"]):
            result.append({"chapter_index": chapter_index, "chapter_title": chapter["title"], "text": sentence})
    return result


def audio_array_from_kokoro(kpipeline: Any, sentence: str) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for item in kpipeline(sentence, voice=VOICE_NAME, speed=1.0):
        audio = item[2] if isinstance(item, (tuple, list)) and len(item) >= 3 else item
        if hasattr(audio, "detach"):
            audio = audio.detach().cpu().numpy()
        array = np.asarray(audio, dtype=np.float32).reshape(-1)
        if array.size:
            chunks.append(array)
    if not chunks:
        raise RuntimeError(f"Kokoro returned no audio for sentence: {sentence[:90]}")
    return np.concatenate(chunks).astype(np.float32, copy=False)


def atempo_chain(speed: float) -> str:
    # FFmpeg atempo supports 0.5..2.0 per stage; chain factors if necessary.
    factors: list[float] = []
    remaining = speed
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    factors.append(remaining)
    return ",".join(f"atempo={factor:.8f}" for factor in factors)


def generate_narration(script_data: dict[str, Any], target_duration: float) -> list[dict[str, Any]]:
    audio_dir = WORK / "audio"
    segment_dir = audio_dir / "segments"
    segment_dir.mkdir(parents=True, exist_ok=True)
    narration_path = audio_dir / "narration.wav"
    timing_path = WORK / "sentence_timings.json"
    state = load_json(CHECKPOINT, {}) or {}
    if narration_path.exists() and timing_path.exists() and state.get("stage") in {"audio_generating", "audio_ready", "images_generating", "images_ready", "clips_generating", "clips_ready", "rendered"}:
        log("Reusing completed narration and sentence timings from checkpoint.")
        return load_json(timing_path, [])

    try:
        import torch
        from kokoro import KPipeline
        torch.set_num_threads(max(1, min(2, os.cpu_count() or 1)))
        log("Loading local Kokoro-82M voice engine.")
        kpipeline = KPipeline(lang_code="a")
    except Exception as exc:
        raise RuntimeError(f"Could not initialize Kokoro local TTS. Ensure kokoro, misaki[en], espeak-ng and model downloads are available: {exc}") from exc

    sentences = flatten_sentences(script_data)
    pad_samples = int(0.25 * SAMPLE_RATE)
    pad = np.zeros(pad_samples, dtype=np.float32)
    segment_arrays: list[np.ndarray] = []
    timings_raw: list[dict[str, Any]] = []
    sample_cursor = 0
    for index, item in enumerate(sentences):
        segment_path = segment_dir / f"sentence_{index:05d}.wav"
        if segment_path.exists():
            audio, sr = sf.read(segment_path, dtype="float32", always_2d=False)
            if sr != SAMPLE_RATE:
                segment_path.unlink(missing_ok=True)
                audio = audio_array_from_kokoro(kpipeline, item["text"])
                sf.write(segment_path, audio, SAMPLE_RATE, subtype="PCM_16")
            else:
                audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        else:
            audio = audio_array_from_kokoro(kpipeline, item["text"])
            sf.write(segment_path, audio, SAMPLE_RATE, subtype="PCM_16")
        segment_arrays.append(audio)
        start = sample_cursor / SAMPLE_RATE
        speech_end = (sample_cursor + len(audio)) / SAMPLE_RATE
        segment_arrays.append(pad)
        sample_cursor += len(audio) + pad_samples
        timings_raw.append({
            "index": index,
            "chapter_index": item["chapter_index"],
            "chapter_title": item["chapter_title"],
            "text": item["text"],
            "raw_start": start,
            "raw_speech_end": speech_end,
            "raw_end": sample_cursor / SAMPLE_RATE,
            "raw_duration": len(audio) / SAMPLE_RATE + 0.25,
            "audio_samples": int(len(audio)),
        })
        if index % 20 == 0 or index == len(sentences) - 1:
            log(f"Narration sentence {index + 1}/{len(sentences)}")
            atomic_json(CHECKPOINT, {**state, "stage": "audio_generating", "audio_sentence_index": index + 1, "sentence_count": len(sentences)})
        del audio
        if index % 30 == 0:
            gc.collect()

    raw_audio_path = audio_dir / "narration_raw.wav"
    full_audio = np.concatenate(segment_arrays).astype(np.float32, copy=False)
    sf.write(raw_audio_path, full_audio, SAMPLE_RATE, subtype="PCM_16")
    raw_duration = len(full_audio) / SAMPLE_RATE
    del full_audio, segment_arrays, kpipeline
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    speed = max(0.5, min(2.0, raw_duration / target_duration))
    scale = 1.0 / speed
    audio_filter = f"{atempo_chain(speed)},apad=whole_dur={target_duration:.3f},atrim=duration={target_duration:.3f},asetpts=N/SR/TB"
    run_checked(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(raw_audio_path), "-af", audio_filter, "-ar", str(SAMPLE_RATE), "-ac", "1", str(narration_path)])
    raw_audio_path.unlink(missing_ok=True)

    timings: list[dict[str, Any]] = []
    for record in timings_raw:
        start = record["raw_start"] * scale
        speech_end = record["raw_speech_end"] * scale
        end = record["raw_end"] * scale
        # Keep all event positions inside the exact target track duration.
        start = max(0.0, min(target_duration, start))
        speech_end = max(start, min(target_duration, speech_end))
        end = max(speech_end, min(target_duration, end))
        timings.append({
            "index": record["index"],
            "chapter_index": record["chapter_index"],
            "chapter_title": record["chapter_title"],
            "text": record["text"],
            "start_time": round(start, 6),
            "speech_end_time": round(speech_end, 6),
            "end_time": round(end, 6),
            "duration": round(end - start, 6),
            "audio_samples": record["audio_samples"],
        })
    atomic_json(timing_path, timings)
    shutil.rmtree(segment_dir, ignore_errors=True)
    atomic_json(CHECKPOINT, {**state, "stage": "audio_ready", "audio_sentence_index": len(sentences), "sentence_count": len(sentences), "narration_seconds": target_duration})
    log(f"Saved narration.wav at {target_duration:.3f}s; audio timing scale={scale:.5f}.")
    return timings


def visual_description(text: str, topic: str, chapter_title: str, index: int) -> str:
    blob = f"{text} {topic} {chapter_title}".casefold()
    choices = [
        (("debt", "credit", "loan", "interest", "payment"), "a red debt meter, a credit card, red downward arrows, and a worried stickman checking bills"),
        (("inflation", "price", "cost", "grocery", "rent"), "colorful grocery prices rising beside a rent bill, coins, and a vivid price chart"),
        (("stock", "market", "share", "invest", "crash", "portfolio"), "a glowing emerald stock chart, colorful candlesticks, gold coins, and a market dashboard"),
        (("compound", "growth", "time", "interest", "future"), "a staircase of shiny gold coins growing toward an emerald upward graph and a calendar"),
        (("budget", "spending", "purchase", "shopping", "consumer"), "a colorful shopping cart, a wallet, receipts, and bright comparison price tags"),
        (("bank", "cash", "saving", "emergency", "reserve"), "a modern colorful bank, a safe, neatly stacked cash, and a glowing green safety shield"),
        (("asset", "wealth", "net worth", "income", "cash flow"), "colorful asset icons, gold coins, a rising chart, a home, and a simple balance sheet"),
        (("risk", "uncertainty", "volatility", "diversif", "resilien"), "a balance scale, diversified colorful asset symbols, a safety net, and a choppy market chart"),
        (("marketing", "status", "social", "brand", "luxury"), "a colorful luxury storefront, a sports car, shopping bags, and a stickman comparing choices"),
        (("time", "career", "option", "freedom", "choice"), "a vibrant road splitting toward a home, education, free time, and a growing savings jar"),
    ]
    for keywords, description in choices:
        if any(word in blob for word in keywords):
            return description
    options = [
        "a colorful city skyline, glowing green charts, golden coins, and a simple financial decision board",
        "a bright modern room with bills, savings jars, gold coins, and a rising emerald arrow",
        "a vibrant financial control panel with colorful graphs, a piggy bank, and a stack of coins",
        "a colorful crossroads surrounded by financial choices, green charts, and shiny gold coins",
    ]
    return options[index % len(options)]


def stickman_prompt(description: str, extra: str = "") -> str:
    return (
        "2D flat vector financial illustration, a strictly black and white minimalist stickman character, "
        "pure white head and body fill #FFFFFF, bold clean black ink outlines #000000, no colored skin, "
        f"surrounded by vibrant colorful {description}, glowing emerald green stock market graphs, shiny gold coins, "
        "rich colorful modern environment, high contrast, clean vector art, crisp edges, wide 16:9 composition, "
        "no letters, no logos, no watermark. " + extra
    )


def choose_font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/custom/Anton-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def fallback_vector_art(description: str, seed: int, width: int = 512, height: int = 288) -> Image.Image:
    rng = random.Random(seed)
    palettes = [(12, 33, 66), (26, 23, 72), (7, 56, 48), (67, 27, 25), (14, 44, 55)]
    bg = palettes[seed % len(palettes)]
    image = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(image)
    # Glowing atmosphere, skyline, and chart elements.
    for i in range(13):
        x = rng.randint(-40, width - 10)
        y = rng.randint(-30, height - 20)
        r = rng.randint(15, 65)
        color = rng.choice([(15, 110, 88), (20, 85, 150), (122, 74, 21), (113, 35, 88), (32, 125, 50)])
        draw.ellipse((x-r, y-r, x+r, y+r), fill=color)
    for x in range(0, width, 40):
        h = rng.randint(35, 95)
        draw.rectangle((x, height-h, x+25, height), fill=rng.choice([(16, 56, 95), (20, 65, 83), (36, 43, 76)]), outline=(36, 99, 121), width=1)
        for wy in range(height-h+8, height-5, 12):
            draw.rectangle((x+5, wy, x+10, wy+4), fill=(240, 180, 55))
    # Green graph and bright gold objects.
    points = [(35, 190), (86, 172), (128, 185), (177, 130), (226, 150), (280, 93), (328, 115), (386, 61), (460, 48)]
    draw.line(points, fill=(38, 255, 111), width=4, joint="curve")
    for px, py in points[::2]:
        draw.ellipse((px-4, py-4, px+4, py+4), fill=(255, 245, 110), outline=(0, 0, 0), width=1)
    for cx, cy, radius in [(width-52, height-49, 19), (width-91, height-41, 15), (width-123, height-32, 12)]:
        draw.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill=(255, 193, 31), outline=(120, 69, 0), width=3)
        draw.arc((cx-radius+4, cy-radius+4, cx+radius-4, cy+radius-4), 20, 300, fill=(255, 231, 112), width=2)
    # Minimalist stickman: white fill/body, black outlines only.
    head_x, head_y = int(width*0.28), int(height*0.42)
    hr = 25
    draw.ellipse((head_x-hr, head_y-hr, head_x+hr, head_y+hr), fill=(255, 255, 255), outline=(0, 0, 0), width=5)
    draw.ellipse((head_x-9, head_y-4, head_x-4, head_y+1), fill=(0, 0, 0))
    draw.ellipse((head_x+5, head_y-4, head_x+10, head_y+1), fill=(0, 0, 0))
    draw.arc((head_x-8, head_y+1, head_x+9, head_y+15), 5, 170, fill=(0, 0, 0), width=2)
    draw.line((head_x, head_y+25, head_x, head_y+91), fill=(0, 0, 0), width=10)
    draw.line((head_x, head_y+35, head_x-31, head_y+62), fill=(0, 0, 0), width=8)
    draw.line((head_x-31, head_y+62, head_x-47, head_y+47), fill=(0, 0, 0), width=8)
    draw.line((head_x, head_y+35, head_x+34, head_y+57), fill=(0, 0, 0), width=8)
    draw.line((head_x+34, head_y+57, head_x+52, head_y+37), fill=(0, 0, 0), width=8)
    draw.line((head_x, head_y+91, head_x-25, head_y+131), fill=(0, 0, 0), width=9)
    draw.line((head_x, head_y+91, head_x+28, head_y+131), fill=(0, 0, 0), width=9)
    draw.line((head_x-25, head_y+131, head_x-39, head_y+131), fill=(255, 255, 255), width=8)
    draw.line((head_x+28, head_y+131, head_x+43, head_y+131), fill=(255, 255, 255), width=8)
    # Small descriptor strip, used as a visual accent rather than readable generated text.
    draw.rounded_rectangle((8, 8, width-8, 35), radius=8, fill=(0, 0, 0))
    draw.rectangle((13, 14, 28, 29), fill=(255, 190, 34))
    return image


def upscale_save(image: Image.Image, path: Path, *, png: bool = False) -> None:
    image = image.convert("RGB").resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
    image = image.filter(ImageFilter.UnsharpMask(radius=1.35, percent=125, threshold=3))
    path.parent.mkdir(parents=True, exist_ok=True)
    if png:
        image.save(path, format="PNG", optimize=True)
    else:
        image.save(path, format="JPEG", quality=91, optimize=True, progressive=True)


def load_image_pipeline() -> tuple[Any, Any, str]:
    try:
        import torch
        from diffusers import AutoPipelineForText2Image
        torch.set_num_threads(max(1, min(2, os.cpu_count() or 1)))
        has_gpu = torch.cuda.is_available()
        device = "cuda" if has_gpu else "cpu"
        dtype = torch.float16 if has_gpu else torch.float32
        log(f"Loading local image model {IMAGE_MODEL_ID} on {device} ({dtype}).")
        pipe = AutoPipelineForText2Image.from_pretrained(
            IMAGE_MODEL_ID,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        try:
            pipe.enable_attention_slicing("max")
        except Exception:
            try:
                pipe.enable_attention_slicing()
            except Exception:
                pass
        try:
            pipe.enable_vae_tiling()
        except Exception:
            pass
        pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        return torch, pipe, device
    except Exception as exc:
        log(f"Local diffusion model could not load; Pillow vector illustrations will be used to finish safely: {exc}")
        return None, None, "pillow-fallback"


def make_model_image(torch: Any, pipe: Any, device: str, prompt: str, seed: int) -> Image.Image:
    generator = None
    try:
        if torch is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
        if "turbo" in IMAGE_MODEL_ID.casefold():
            inference_steps = 1
            guidance = 0.0
            negative = None
        else:
            # Tiny-SD is a compact public diffusion model rather than a turbo one-step model.
            inference_steps = 4
            guidance = 5.0
            negative = NEGATIVE_PROMPT
        result = pipe(
            prompt=prompt,
            negative_prompt=negative,
            width=512,
            height=288,
            num_inference_steps=inference_steps,
            guidance_scale=guidance,
            generator=generator,
        )
        if not result.images:
            raise RuntimeError("Image model returned an empty image list")
        return result.images[0].convert("RGB")
    except Exception as exc:
        log(f"Image inference failed for one scene; using illustrated fallback for this scene: {exc}")
        return fallback_vector_art(prompt[:180], seed)


def map_visual_windows(timings: list[dict[str, Any]], script_data: dict[str, Any], target_duration: float) -> list[dict[str, Any]]:
    scene_count = int(math.ceil(target_duration / IMAGE_INTERVAL))
    windows: list[dict[str, Any]] = []
    for index in range(scene_count):
        start = index * IMAGE_INTERVAL
        end = min(target_duration, (index + 1) * IMAGE_INTERVAL)
        overlapping = [record for record in timings if record["speech_end_time"] > start and record["start_time"] < end]
        if overlapping:
            chosen = max(overlapping, key=lambda rec: max(0.0, min(end, rec["speech_end_time"]) - max(start, rec["start_time"])))
        else:
            chosen = timings[min(len(timings) - 1, int(start / max(target_duration, 1.0) * len(timings)))]
        description = visual_description(chosen["text"], script_data["topic"], chosen["chapter_title"], index)
        windows.append({
            "index": index,
            "start_time": round(start, 3),
            "end_time": round(end, 3),
            "chapter_title": chosen["chapter_title"],
            "source_sentence": chosen["text"],
            "visual_description": description,
            "prompt": stickman_prompt(description),
        })
    return windows


def generate_images(windows: list[dict[str, Any]], script_data: dict[str, Any]) -> tuple[list[Path], str]:
    image_dir = WORK / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    state = load_json(CHECKPOINT, {}) or {}
    missing = [window for window in windows if not (image_dir / f"scene_{window['index']:04d}.jpg").exists()]
    torch = pipe = None
    device = "pillow-fallback"
    backend = state.get("image_backend", "")
    if missing:
        torch, pipe, device = load_image_pipeline()
        backend = IMAGE_MODEL_ID if pipe is not None else "Pillow vector fallback (local model unavailable)"
        total = len(windows)
        for count, window in enumerate(windows, 1):
            image_path = image_dir / f"scene_{window['index']:04d}.jpg"
            if image_path.exists():
                continue
            seed = (int(time.time()) + window["index"] * 7919) % 2_147_483_647
            if pipe is not None:
                image = make_model_image(torch, pipe, device, window["prompt"], seed)
            else:
                image = fallback_vector_art(window["visual_description"], seed)
            upscale_save(image, image_path)
            del image
            atomic_json(CHECKPOINT, {**state, "stage": "images_generating", "image_count": count, "scene_count": total, "image_backend": backend})
            if count % 5 == 0 or count == total:
                log(f"Scene images: {count}/{total}")
            gc.collect()
            try:
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        if pipe is not None:
            del pipe
        if torch is not None:
            del torch
        gc.collect()
    paths = [image_dir / f"scene_{window['index']:04d}.jpg" for window in windows]
    if not all(path.exists() for path in paths):
        raise RuntimeError("One or more scene images are missing after the image generation stage.")
    atomic_json(WORK / "scene_prompts.json", windows)
    state = load_json(CHECKPOINT, {}) or {}
    atomic_json(CHECKPOINT, {**state, "stage": "images_ready", "image_count": len(paths), "scene_count": len(paths), "image_backend": backend or "reused cached images"})
    return paths, backend or "reused cached images"


def create_thumbnail(script_data: dict[str, Any], titles: list[str]) -> Path:
    release_dir = WORK / "release"
    release_dir.mkdir(parents=True, exist_ok=True)
    thumb_path = release_dir / "thumbnail.png"
    if thumb_path.exists() and thumb_path.stat().st_size > 20_000:
        return thumb_path
    torch, pipe, device = load_image_pipeline()
    description = "extreme colorful wealth imagery, a towering pile of gold coins, an emerald rocket-shaped growth arrow, a vivid red debt meter, luxury buildings, and dramatic financial chart shapes"
    prompt = stickman_prompt(description, "Eye-catching documentary thumbnail composition, clear center-left black-and-white stickman, vibrant colors, cinematic contrast, no text.")
    seed = random.SystemRandom().randrange(1, 2_147_483_647)
    if pipe is not None:
        image = make_model_image(torch, pipe, device, prompt, seed)
        del pipe
        del torch
        gc.collect()
    else:
        image = fallback_vector_art(description, seed)
    image = image.convert("RGB").resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
    image = image.filter(ImageFilter.UnsharpMask(radius=1.5, percent=140, threshold=3))
    draw = ImageDraw.Draw(image, "RGBA")
    # Strong editable thumbnail typography, not text baked by the image model.
    topic_words = re.findall(r"[A-Za-z0-9]+", script_data["topic"].upper())
    compact = " ".join(topic_words[:3]) if topic_words else "MONEY MINDSET"
    if len(compact) > 19:
        compact = " ".join(topic_words[:2])
    box = (65, 735, 1170, 1010)
    draw.rounded_rectangle(box, radius=26, fill=(0, 0, 0, 195), outline=(255, 202, 40, 255), width=8)
    font = choose_font(106, bold=True)
    words = compact.split()
    lines: list[str] = []
    line = ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if draw.textbbox((0, 0), candidate, font=font, stroke_width=3)[2] > 1030 and line:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    if len(lines) > 2:
        lines = lines[:2]
    y = 765 if len(lines) == 2 else 820
    for line in lines:
        draw.text((100, y), line, font=font, fill=(255, 218, 70, 255), stroke_width=7, stroke_fill=(0, 0, 0, 255))
        y += 115
    image.save(thumb_path, format="PNG", optimize=True)
    return thumb_path


def ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    total_cs = int(round(seconds * 100))
    hours, remainder = divmod(total_cs, 360000)
    minutes, remainder = divmod(remainder, 6000)
    whole_seconds, centiseconds = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{whole_seconds:02d}.{centiseconds:02d}"


def ass_escape(text: str) -> str:
    text = text.replace("\\", "")
    text = text.replace("{", "(").replace("}", ")")
    return text


def subtitle_word(text: str) -> str:
    clean = ass_escape(text)
    key = re.sub(r"[^A-Z]", "", clean.upper())
    important = {"MONEY", "WEALTH", "DEBT", "BILLIONS", "CRASH", "PROFIT", "SECRET", "ASSET", "ASSETS", "RISK", "INFLATION", "COMPOUND", "INTEREST", "CASHFLOW", "FREEDOM", "INCOME", "INVESTING", "SAVING"}
    if key in important:
        return r"{\c&H0000D7FF&}" + clean + r"{\c&H00FFFFFF&}"
    return clean


def create_ass_subtitles(timings: list[dict[str, Any]]) -> Path:
    path = WORK / "subtitles.ass"
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        f"PlayResX: {WIDTH}",
        f"PlayResY: {HEIGHT}",
        "YCbCr Matrix: TV.709",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        "Style: Default,Anton,86,&H00FFFFFF,&H0000D7FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,6,3,2,100,100,85,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    animation = r"{\fscx65\fscy65\t(0,90,\fscx108\fscy108)\t(90,150,\fscx100\fscy100)}"
    for record in timings:
        tokens = record["text"].split()
        if not tokens:
            continue
        chunks = [tokens[i:i+4] for i in range(0, len(tokens), 4)]
        weights = [sum(max(1, len(re.sub(r"\W", "", word))) for word in chunk) for chunk in chunks]
        weight_sum = max(1, sum(weights))
        start = float(record["start_time"])
        speech_end = float(record["speech_end_time"])
        speech_duration = max(0.08, speech_end - start)
        cursor = start
        cumulative_weight = 0
        for chunk_index, chunk in enumerate(chunks):
            cumulative_weight += weights[chunk_index]
            if chunk_index == len(chunks) - 1:
                end = speech_end
            else:
                end = start + speech_duration * cumulative_weight / weight_sum
            end = max(cursor + 0.04, min(speech_end, end))
            content = " ".join(subtitle_word(word) for word in chunk)
            lines.append(f"Dialogue: 0,{ass_time(cursor)},{ass_time(end)},Default,,0,0,0,,{animation}{content}")
            cursor = end
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    return path


def make_titles(topic: str) -> list[str]:
    seeds = [
        f"{topic}: The Money Mistake Most People Miss",
        f"The Hidden Financial Cost of {topic}",
        f"What {topic} Can Teach You About Building Wealth",
        f"The Psychology Behind {topic} (Explained Simply)",
        f"{topic}: The Quiet Choice That Changes Your Money",
    ]
    return list(dict.fromkeys(seeds))[:5]


def format_chapter_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def build_chapter_times(chapters: list[dict[str, Any]], timings: list[dict[str, Any]]) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    for index, chapter in enumerate(chapters):
        first = next((item for item in timings if item["chapter_index"] == index), None)
        if first is not None:
            result.append((format_chapter_timestamp(first["start_time"]), chapter["title"]))
    return result


def create_seo(script_data: dict[str, Any], timings: list[dict[str, Any]], target_duration: float, test_run: bool, image_backend: str) -> tuple[Path, Path]:
    release_dir = WORK / "release"
    release_dir.mkdir(parents=True, exist_ok=True)
    seo_path = release_dir / "youtube_seo.txt"
    notes_path = release_dir / "release_notes.md"
    titles = make_titles(script_data["topic"])
    chapter_times = build_chapter_times(script_data["chapters"], timings)
    overview = (
        f"What if the biggest change in your finances did not begin with a bigger paycheck, but with seeing {script_data['topic'].lower()} more clearly? "
        "This documentary breaks the subject into practical mechanisms, behavioral patterns, trade-offs, and questions you can use to review your own decisions. "
        "It is educational content, not personalized financial, tax, or investment advice. Outcomes vary, and no strategy guarantees returns."
    )
    description = (
        f"{overview}\n\nIn this episode, you will explore the numbers beneath everyday money choices, the psychology that can distort them, "
        "and the habits that can make a financial plan more resilient. The goal is not a secret shortcut; it is a clearer way to reason about risk, time, costs, and options.\n\n"
        "CHAPTERS\n" + "\n".join(f"{stamp} {title}" for stamp, title in chapter_times) +
        "\n\nDisclaimer: This video is for general education only and does not constitute financial, investment, legal, or tax advice. Research independently and consider your circumstances and risk tolerance."
    )
    seo_text = (
        "5 TITLE OPTIONS\n" + "\n".join(f"{i}. {title}" for i, title in enumerate(titles, 1)) +
        "\n\nFULL YOUTUBE DESCRIPTION\n" + description +
        "\n\n35 TAGS\n" + ", ".join(TAG_LIST) +
        f"\n\nVIDEO METADATA\nTopic: {script_data['topic']}\nMode: {'1-minute test' if test_run else '30-minute documentary'}\n"
        f"Target duration: {target_duration:.3f} seconds\nScript words: {script_data['word_count']}\nImage backend: {image_backend}\n"
    )
    seo_path.write_text(seo_text, encoding="utf-8")
    notes_path.write_text(
        f"# Stickman Finance Documentary\n\nTopic: {script_data['topic']}\n\n"
        f"Mode: {'1-minute TEST render' if test_run else '30-minute full render'}\n\n"
        f"Duration: {format_chapter_timestamp(target_duration)}\n\n"
        f"Script words: {script_data['word_count']}\n\n"
        f"Image backend: {image_backend}\n\n"
        "The episode uses locally-run model inference where the public model weights are available. Word-subtitle boundaries are estimated proportionally within each Kokoro sentence; they are not forced-aligned phoneme timings.\n",
        encoding="utf-8",
    )
    return seo_path, notes_path


def render_video(image_paths: list[Path], narration_path: Path, subtitle_path: Path, target_duration: float) -> Path:
    """Render a static 5-second-per-image slideshow. Only ASS subtitles animate."""
    if not image_paths:
        raise RuntimeError("No scene images were generated.")
    release_dir = WORK / "release"
    release_dir.mkdir(parents=True, exist_ok=True)
    final_path = release_dir / "final_documentary.mp4"
    state = load_json(CHECKPOINT, {}) or {}
    if final_path.exists() and final_path.stat().st_size > 10_000:
        try:
            existing_duration = check_ffprobe_duration(final_path)
        except Exception:
            existing_duration = 0.0
        if abs(existing_duration - target_duration) <= 0.5:
            log("Reusing a valid final rendered MP4 from checkpoint.")
            atomic_json(CHECKPOINT, {**state, "stage": "rendered", "video_seconds": existing_duration})
            return final_path

    # FFmpeg concat demuxer holds each still image for exactly five seconds.
    # No zoompan, pan, fade, xfade, or other image/scene animation is applied.
    concat_path = WORK / "still_images.txt"
    concat_path.parent.mkdir(parents=True, exist_ok=True)
    scene_duration = IMAGE_INTERVAL
    with concat_path.open("w", encoding="utf-8") as handle:
        for image_path in image_paths:
            escaped = str(image_path.resolve()).replace("\\", "/").replace("'", "'\\''")
            handle.write(f"file '{escaped}'\n")
            handle.write(f"duration {scene_duration:.6f}\n")
        # concat demuxer needs a final repeated file to honor the final duration.
        escaped = str(image_paths[-1].resolve()).replace("\\", "/").replace("'", "'\\''")
        handle.write(f"file '{escaped}'\n")

    subtitle_file = str(subtitle_path.resolve()).replace("\\", "/").replace("'", "\\'")
    font_dir = "/usr/share/fonts/truetype/custom" if Path("/usr/share/fonts/truetype/custom").exists() else "/usr/share/fonts/truetype/dejavu"
    subtitle_filter = f"ass=filename='{subtitle_file}':fontsdir='{font_dir}'"
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "concat", "-safe", "0", "-i", str(concat_path),
        "-i", str(narration_path),
        "-vf", f"fps={FPS},scale={WIDTH}:{HEIGHT}:flags=lanczos,{subtitle_filter},format=yuv420p",
        "-map", "0:v:0", "-map", "1:a:0",
        "-t", f"{target_duration:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", str(SAMPLE_RATE),
        "-movflags", "+faststart", "-threads", "2", str(final_path),
    ]
    log(f"Rendering {len(image_paths)} static images at {IMAGE_INTERVAL:g}s each; only subtitles are animated.")
    run_checked(command, cwd=WORK)
    if not final_path.exists() or final_path.stat().st_size < 10_000:
        raise RuntimeError("FFmpeg did not produce a valid final_documentary.mp4.")
    atomic_json(CHECKPOINT, {**(load_json(CHECKPOINT, {}) or {}), "stage": "rendered", "scene_count": len(image_paths), "video_seconds": target_duration})
    return final_path


def ensure_release_outputs(script_data: dict[str, Any], target_duration: float, test_run: bool, image_backend: str, timings: list[dict[str, Any]]) -> dict[str, Path]:
    release_dir = WORK / "release"
    release_dir.mkdir(parents=True, exist_ok=True)
    thumb = create_thumbnail(script_data, make_titles(script_data["topic"]))
    seo, notes = create_seo(script_data, timings, target_duration, test_run, image_backend)
    final_video = release_dir / "final_documentary.mp4"
    for path in (final_video, thumb, seo, notes):
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Expected release asset missing or empty: {path}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in (final_video, thumb, seo, notes):
        shutil.copy2(path, OUTPUT_DIR / path.name)
    return {"video": final_video, "thumbnail": thumb, "seo": seo, "notes": notes}


def set_github_output(name: str, value: str) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return
    delimiter = f"END_{uuid.uuid4().hex}"
    with open(output_path, "a", encoding="utf-8") as handle:
        handle.write(f"{name}<<{delimiter}\n{value}\n{delimiter}\n")


def mark_published() -> int:
    state = load_json(CHECKPOINT, {})
    if not state:
        log("No checkpoint found; nothing to mark as published.")
        return 0
    state["stage"] = "complete"
    state["published_at"] = datetime.now(timezone.utc).isoformat()
    atomic_json(CHECKPOINT, state)
    for child in list(WORK.iterdir()) if WORK.exists() else []:
        if child.name == "checkpoint.json":
            continue
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            try:
                child.unlink()
            except OSError:
                pass
    log("Release upload succeeded; removed temporary audio, images, clips and intermediate files.")
    return 0


def reset_work() -> None:
    if WORK.exists():
        shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)


def check_ffprobe_duration(path: Path) -> float:
    result = run_checked([
        "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)
    ], capture=True)
    try:
        return float(result.stdout.strip())
    except (AttributeError, ValueError):
        return 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a 1-minute test or 30-minute stickman finance documentary.")
    parser.add_argument("--test-run", action="store_true", help="Build exactly a one-minute test video instead of a full documentary.")
    parser.add_argument("--mark-published", action="store_true", help="Mark the current build as published and delete temporary files.")
    args = parser.parse_args()

    if args.mark_published:
        return mark_published()

    assert_tools()
    random.seed()
    mode = "test" if args.test_run else "full"
    target_duration = float(VIDEO_SECONDS_TEST if args.test_run else VIDEO_SECONDS_FULL)
    existing = load_json(CHECKPOINT, {})
    can_resume = bool(existing and existing.get("stage") != "complete" and existing.get("mode") == mode and existing.get("episode_id"))
    if can_resume:
        state = existing
        topic = str(state.get("topic") or select_topic())
        log(f"Resuming checkpoint for {state.get('episode_id')} at stage {state.get('stage')}.")
    else:
        reset_work()
        topic = select_topic()
        episode_id = datetime.now(timezone.utc).strftime("ep_%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6]
        state = {"mode": mode, "topic": topic, "episode_id": episode_id, "stage": "started", "started_at": datetime.now(timezone.utc).isoformat()}
        atomic_json(CHECKPOINT, state)
        log(f"Starting new {mode} episode: {topic}")

    try:
        script_path = WORK / "script.json"
        if script_path.exists() and (load_json(CHECKPOINT, {}) or {}).get("stage") in {"script_ready", "audio_generating", "audio_ready", "images_generating", "images_ready", "clips_generating", "clips_ready", "rendered"}:
            script_data = load_json(script_path)
            log("Reusing script.json from checkpoint.")
        else:
            script_data = generate_script(topic, args.test_run)
            atomic_json(script_path, script_data)
            (WORK / "script.txt").write_text(script_data["script_text"], encoding="utf-8")
            state = {**(load_json(CHECKPOINT, {}) or state), "stage": "script_ready", "script_words": script_data["word_count"]}
            atomic_json(CHECKPOINT, state)
        if not script_data or not script_data.get("chapters"):
            raise RuntimeError("Script generation did not create any chapters.")

        timings = generate_narration(script_data, target_duration)
        windows = map_visual_windows(timings, script_data, target_duration)
        image_paths, image_backend = generate_images(windows, script_data)
        subtitle_path = create_ass_subtitles(timings)
        thumb_path = WORK / "release" / "thumbnail.png"
        if not thumb_path.exists() or thumb_path.stat().st_size < 20_000:
            # Reuse the loaded scene images; image model is reloaded only for a missing thumbnail.
            create_thumbnail(script_data, make_titles(topic))
        video_path = render_video(image_paths, WORK / "audio" / "narration.wav", subtitle_path, target_duration)
        assets = ensure_release_outputs(script_data, target_duration, args.test_run, image_backend, timings)
        duration = check_ffprobe_duration(assets["video"])
        if abs(duration - target_duration) > 0.5:
            raise RuntimeError(f"Final MP4 duration is {duration:.3f}s; expected {target_duration:.3f}s.")

        release_name = f"{'[TEST] ' if args.test_run else ''}Stickman Finance — {topic}"
        set_github_output("release_name", release_name)
        set_github_output("topic", topic)
        set_github_output("video_seconds", f"{duration:.3f}")
        state = {
            **(load_json(CHECKPOINT, {}) or state),
            "stage": "rendered",
            "mode": mode,
            "topic": topic,
            "release_name": release_name,
            "scene_count": len(image_paths),
            "script_words": script_data["word_count"],
            "video_seconds": duration,
            "image_backend": image_backend,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_json(CHECKPOINT, state)
        log(f"Build ready: {assets['video']} ({duration:.3f}s), {len(image_paths)} images, {script_data['word_count']} script words.")
        log("GitHub Actions will publish the video, thumbnail and SEO file as a GitHub Release.")
        return 0
    except Exception as exc:
        current = load_json(CHECKPOINT, {}) or state
        current["last_error"] = str(exc)
        current["last_error_at"] = datetime.now(timezone.utc).isoformat()
        current["last_traceback"] = traceback.format_exc()[-12000:]
        atomic_json(CHECKPOINT, current)
        log(f"PIPELINE FAILED: {exc}")
        traceback.print_exc()
        return 1
    finally:
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
