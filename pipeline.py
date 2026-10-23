#!/usr/bin/env python3
"""Two-file Stickman Finance documentary pipeline.

All AI inference is local: Ollama/Gemma, Diffusers/SDXL-Turbo, and Kokoro ONNX.
Run `python pipeline.py --help` for configuration and `python pipeline.py --self-test`
for dependency-light unit tests. The GitHub workflow installs runtime dependencies.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import logging
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import unicodedata
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import requests

APP_VERSION = "1.0.0"
KOKORO_MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/kokoro-v1.0.onnx"
KOKORO_VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/voices-v1.0.bin"
KOKORO_SAMPLE_RATE = 24_000
IMAGE_MODEL_DEFAULT = "stabilityai/sdxl-turbo"
OLLAMA_MODEL_DEFAULT = "gemma2:2b"
VOICE_DEFAULT = "am_puck"
SCENE_SECONDS = 5
MAX_SUBTITLE_WORDS = 4
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
WORD_RE = re.compile(r"\b[\w]+(?:['’][\w]+)*(?:-[\w]+)*\b", re.UNICODE)
SLUG_RE = re.compile(r"[^a-z0-9]+")
LOGGER = logging.getLogger("stickman_finance")


class PipelineError(RuntimeError):
    """Actionable, user-facing pipeline error."""


def setup_logging(output_dir: Path | None = None, level: str = "INFO") -> None:
    LOGGER.setLevel(getattr(logging, level.upper(), logging.INFO))
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    LOGGER.addHandler(stream)
    if output_dir is not None:
        logs = output_dir / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(logs / "pipeline.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        LOGGER.addHandler(file_handler)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp_path.replace(path)


def sanitize_slug(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii").lower()
    slug = SLUG_RE.sub("-", normalized).strip("-")
    return (slug[:72].rstrip("-") or "stickman-finance")


def word_tokens(text: str) -> list[str]:
    return WORD_RE.findall(text)


def word_count(text: str) -> int:
    return len(word_tokens(text))


def chunks_of_words(text: str, maximum: int = MAX_SUBTITLE_WORDS) -> list[str]:
    if maximum != MAX_SUBTITLE_WORDS:
        # Four is a product requirement, not a tunable preference.
        raise ValueError("Subtitle maximum is fixed at four words.")
    raw_tokens = text.split()
    return [" ".join(raw_tokens[index:index + maximum]) for index in range(0, len(raw_tokens), maximum)]


def visible_word_count(text: str) -> int:
    return word_count(re.sub(r"\{[^}]*\}", "", text))


def timestamp_ass(seconds: float) -> str:
    centiseconds = max(0, int(round(seconds * 100)))
    hours, remainder = divmod(centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    whole_seconds, centi = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{whole_seconds:02d}.{centi:02d}"


def timestamp_srt(seconds: float) -> str:
    milliseconds = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milli = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{milli:03d}"


def validate_cues(cues: Sequence[dict[str, Any]], audio_duration: float) -> None:
    if not cues:
        raise PipelineError("Subtitle timing produced zero cues; refusing to render an uncaptioned video.")
    previous_start = -1.0
    previous_end = 0.0
    all_caption_words = 0
    for index, cue in enumerate(cues, 1):
        text = str(cue.get("text", "")).strip()
        start = float(cue.get("start", -1))
        end = float(cue.get("end", -1))
        count = visible_word_count(text)
        if not text or count == 0:
            raise PipelineError(f"Subtitle cue {index} has no visible words.")
        if count > MAX_SUBTITLE_WORDS:
            raise PipelineError(f"Subtitle cue {index} contains {count} words; the maximum is four.")
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= audio_duration + 0.08):
            raise PipelineError(f"Subtitle cue {index} has invalid timing {start:.3f}–{end:.3f}s for {audio_duration:.3f}s audio.")
        if start < previous_start or start < previous_end - 0.04:
            raise PipelineError(f"Subtitle cue {index} is out of order or overlaps an earlier cue.")
        previous_start, previous_end = start, end
        all_caption_words += count
    if previous_end > audio_duration + 0.08:
        raise PipelineError("Final subtitle cue extends beyond the audio stream.")


def validate_scene_plan(scenes: Sequence[dict[str, Any]], expected_count: int | None = None) -> None:
    if expected_count is not None and len(scenes) != expected_count:
        raise PipelineError(f"Scene plan contains {len(scenes)} scenes; expected {expected_count}.")
    if not scenes:
        raise PipelineError("Scene planner returned no scenes.")
    ids: set[str] = set()
    numbers: set[int] = set()
    for scene in scenes:
        sid = str(scene.get("scene_id", ""))
        number = int(scene.get("scene_number", 0))
        prompt = str(scene.get("image_prompt", "")).strip()
        narration = str(scene.get("narration", "")).strip()
        start = float(scene.get("start_seconds", -1))
        end = float(scene.get("end_seconds", -1))
        if not sid or sid in ids:
            raise PipelineError(f"Scene ID is blank or duplicated: {sid!r}")
        if number < 1 or number in numbers:
            raise PipelineError(f"Scene number is invalid or duplicated: {number}")
        if not prompt or not narration:
            raise PipelineError(f"Scene {number} is missing its image prompt or narration reference.")
        if start < 0 or end <= start:
            raise PipelineError(f"Scene {number} has invalid time range {start}–{end}.")
        ids.add(sid)
        numbers.add(number)
    # Deliberately no upper scene-count bound: 127, 360, or more are allowed.
    if sorted(numbers) != list(range(1, len(scenes) + 1)):
        raise PipelineError("Scene numbers must be contiguous from 1 to the planned scene count.")


def validate_metadata(metadata: dict[str, Any]) -> None:
    title = str(metadata.get("title", "")).strip()
    description = str(metadata.get("description", "")).strip()
    tags = metadata.get("tags", [])
    hashtags = metadata.get("hashtags", [])
    if not title or len(title) > 100:
        raise PipelineError("Metadata title must contain 1–100 characters.")
    if len(description) > 5000:
        raise PipelineError("Metadata description exceeds 5,000 characters.")
    if not isinstance(tags, list) or not all(isinstance(tag, str) and tag.strip() for tag in tags):
        raise PipelineError("Metadata tags must be a list of non-empty strings.")
    if sum(len(tag) for tag in tags) + max(0, len(tags) - 1) > 450:
        raise PipelineError("Metadata tags exceed the conservative 450-character budget.")
    if not isinstance(hashtags, list) or not all(isinstance(tag, str) and tag.startswith("#") for tag in hashtags):
        raise PipelineError("Metadata hashtags must be a list of strings beginning with #.")
    if str(metadata.get("category_id", "27")) not in {str(i) for i in range(1, 45)}:
        raise PipelineError("Metadata category_id must be a YouTube category ID.")
    if metadata.get("language", "en") not in {"en", "en-US", "en-GB"}:
        raise PipelineError("This pipeline is configured for English metadata.")


def retry_call(operation: Callable[[], Any], attempts: int = 3, initial_delay: float = 0.1,
               sleep: Callable[[float], None] = time.sleep,
               retry_if: Callable[[Exception], bool] | None = None) -> Any:
    """Bounded retry helper, also exercised by built-in tests."""
    if attempts < 1:
        raise ValueError("attempts must be at least one")
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as exc:  # noqa: BLE001 - retry wrapper must classify failures
            last_error = exc
            if attempt + 1 >= attempts or (retry_if is not None and not retry_if(exc)):
                raise
            sleep(initial_delay * (2 ** attempt))
    assert last_error is not None
    raise last_error


def retryable_http_error(exc: Exception) -> bool:
    if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
        return True
    if isinstance(exc, PipelineError) and "Incomplete model download" in str(exc):
        return True
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return exc.response.status_code in {408, 425, 429, 500, 502, 503, 504}
    return False


def download_file(url: str, destination: Path, min_bytes: int, timeout: tuple[int, int] = (20, 180)) -> None:
    """Download to a temporary path, validate size, and atomically replace destination."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size >= min_bytes:
        return
    part = destination.with_suffix(destination.suffix + ".part")
    part.unlink(missing_ok=True)

    def perform() -> None:
        part.unlink(missing_ok=True)
        with requests.get(url, stream=True, timeout=timeout, headers={"User-Agent": f"stickman-finance/{APP_VERSION}"}) as response:
            response.raise_for_status()
            expected = response.headers.get("Content-Length")
            expected_len = int(expected) if expected and expected.isdigit() else None
            written = 0
            with part.open("wb") as handle:
                for block in response.iter_content(chunk_size=1024 * 1024):
                    if block:
                        handle.write(block)
                        written += len(block)
            if written < min_bytes or (expected_len is not None and written != expected_len):
                raise PipelineError(f"Incomplete model download from {url.rsplit('/', 1)[-1]}: received {written} bytes.")
        part.replace(destination)

    try:
        retry_call(perform, attempts=4, initial_delay=2.0, retry_if=retryable_http_error)
    except Exception:
        part.unlink(missing_ok=True)
        raise


def request_json(url: str, payload: dict[str, Any], timeout: tuple[int, int] = (10, 600)) -> dict[str, Any]:
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise PipelineError(f"Unexpected JSON response from local model endpoint {url}.")
    return data


def ollama_base_url() -> str:
    return os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/").replace("0.0.0.0", "127.0.0.1")


def ensure_ollama(model_name: str, timeout_seconds: int = 120) -> None:
    base = ollama_base_url()
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            response = requests.get(base + "/api/tags", timeout=5)
            response.raise_for_status()
            break
        except Exception as exc:  # noqa: BLE001 - bounded health-poll retry
            if time.monotonic() >= deadline:
                raise PipelineError(f"Ollama is not healthy at {base}. Start `ollama serve` and pull `{model_name}`. Details: {type(exc).__name__}") from exc
            time.sleep(2)
    data = response.json()
    models = data.get("models", []) if isinstance(data, dict) else []
    present = any(str(item.get("name", "")) == model_name or str(item.get("model", "")) == model_name for item in models if isinstance(item, dict))
    if present:
        return
    LOGGER.warning("Ollama model %s is not present; downloading via the local Ollama service.", model_name)
    response = requests.post(base + "/api/pull", json={"name": model_name, "stream": True}, stream=True, timeout=(10, 1800))
    response.raise_for_status()
    last_progress = 0.0
    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if item.get("error"):
            raise PipelineError(f"Ollama could not pull {model_name}: {item['error']}")
        now = time.monotonic()
        if now - last_progress > 15:
            status = item.get("status", "pulling model")
            LOGGER.info("Ollama model setup: %s", status)
            last_progress = now
        if item.get("status") == "success":
            return
    # A pull may finish without a final status line on a proxy; verify tags again.
    tags = requests.get(base + "/api/tags", timeout=10)
    tags.raise_for_status()
    models = tags.json().get("models", [])
    if not any(str(item.get("name", "")) == model_name or str(item.get("model", "")) == model_name for item in models):
        raise PipelineError(f"Ollama pull ended without confirming model `{model_name}` is installed.")


def ollama_generate(prompt: str, model_name: str, num_predict: int = 1200, timeout: int = 900) -> str:
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "keep_alive": 0,
        "options": {"temperature": 0.45, "num_ctx": 8192, "num_predict": num_predict, "repeat_penalty": 1.08},
    }
    result = request_json(ollama_base_url() + "/api/generate", payload, timeout=(15, timeout))
    response = result.get("response")
    if not isinstance(response, str) or not response.strip():
        raise PipelineError("Ollama returned an empty response. Check Gemma memory/context settings and Ollama logs.")
    return response.strip()


def extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise PipelineError("Local language model did not return a JSON object.")
        try:
            parsed = json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError as exc:
            raise PipelineError(f"Local language model returned malformed JSON: {exc.msg} at character {exc.pos}.") from exc
    if not isinstance(parsed, dict):
        raise PipelineError("Local language model JSON must be an object.")
    return parsed


def ollama_json(prompt: str, model_name: str, num_predict: int = 1200, repair_attempts: int = 1) -> dict[str, Any]:
    first_error: Exception | None = None
    current_prompt = prompt + "\nReturn only one valid JSON object, with no markdown fences or commentary."
    for attempt in range(repair_attempts + 1):
        raw = ollama_generate(current_prompt, model_name, num_predict=num_predict)
        try:
            return extract_json(raw)
        except PipelineError as exc:
            first_error = exc
            current_prompt = (
                "Your previous answer was invalid JSON. Correct it now. Output only a syntactically valid JSON object "
                "matching the original schema. Do not add markdown.\nOriginal task:\n" + prompt + "\nInvalid answer:\n" + raw[:5000]
            )
            LOGGER.warning("Local model JSON validation failed; bounded repair attempt %d/%d.", attempt + 1, repair_attempts)
    raise PipelineError(f"Local model JSON validation failed after repair: {first_error}")


def finance_style_prompt(extra: str = "") -> str:
    return (
        "Hand-drawn 2D stickman finance editorial illustration, clean bold black outlines, simple white stickman "
        "with round head and expressive simple face, tasteful flat colors, polished modern explainer-video artwork, "
        "strong readable silhouette, meaningful financial props, uncluttered composition, balanced negative space, "
        "high contrast, crisp shapes, consistent channel art direction, no legible text, no letters, no watermark, "
        "no photorealism, no 3D render. " + extra
    )


def choose_topic(model_name: str, requested: str) -> str:
    if requested and requested.strip().lower() not in {"auto", "automatic", "random"}:
        return requested.strip()
    result = ollama_json(
        "Choose one specific, evergreen, educational personal-finance documentary topic for a US audience. "
        "Avoid personalized investment recommendations, predictions, current tax figures, unsupported statistics, "
        "and guaranteed outcomes. Return JSON {\"topic\": \"...\"}.",
        model_name, num_predict=120,
    )
    topic = str(result.get("topic", "")).strip()
    if not topic or len(topic) > 160:
        raise PipelineError("Ollama returned an invalid topic. Pass --topic to choose one manually.")
    return topic


def generate_test_sections(model_name: str, topic: str) -> list[dict[str, str]]:
    prompt = f"""Write a spoken English narration for a roughly one-minute Stickman Finance explainer about: {topic!r}.
Audience: adults in the United States. Deliver 105–125 spoken words, not a full documentary. Have a hook, explain one useful concept in plain English, use at most one clearly hypothetical example, mention one caveat, and end naturally. Avoid greetings, headings in the narration, stage directions, unsupported statistics, claims about current laws/taxes/markets, personalized financial advice, and promised returns. Do not pad length. Return JSON {{\"heading\": \"short chapter name\", \"narration\": \"spoken words only\"}}."""
    for attempt in range(3):
        result = ollama_json(prompt, model_name, num_predict=260)
        narration = str(result.get("narration", "")).strip()
        heading = str(result.get("heading", "The key idea")).strip() or "The key idea"
        if 80 <= word_count(narration) <= 155:
            return [{"heading": heading, "narration": narration}]
        LOGGER.warning("Test script had %s words; asking the local model for a better one (attempt %d/3).", word_count(narration), attempt + 1)
        prompt = f"Rewrite this narration for {topic!r} to contain 105–125 spoken words while keeping the facts cautious. Return JSON {{\"heading\": str, \"narration\": str}}. Previous draft: {narration[:6000]}"
    raise PipelineError("Could not produce a valid short Test Run narration after three bounded attempts.")


def generate_full_sections(model_name: str, topic: str) -> list[dict[str, str]]:
    outline = ollama_json(
        f"Create an 8-chapter documentary outline for an educational Stickman Finance video for a US audience on {topic!r}. "
        "Use a story arc: opening hook; core problem; underlying concept; worked hypothetical example; trade-offs and counterarguments; "
        "common mistakes; practical decision framework; conclusion. Avoid current rates, tax thresholds, market predictions, made-up data, "
        "investment promises, and personal advice. Return JSON {\"chapters\":[{\"heading\":str,\"focus\":str}, ...]} with exactly 8 distinct chapters.",
        model_name, num_predict=450,
    )
    chapters = outline.get("chapters")
    if not isinstance(chapters, list) or len(chapters) != 8:
        raise PipelineError("Ollama outline validation failed: expected exactly eight distinct chapter ideas.")
    section_specs: list[dict[str, str]] = []
    for item in chapters:
        if not isinstance(item, dict) or not str(item.get("heading", "")).strip() or not str(item.get("focus", "")).strip():
            raise PipelineError("Ollama outline contains a chapter without a heading or focus.")
        section_specs.append({"heading": str(item["heading"]).strip(), "focus": str(item["focus"]).strip()})
    sections: list[dict[str, str]] = []
    for index, spec in enumerate(section_specs, 1):
        prompt = f"""Write chapter {index} of an approximately 30-minute English Stickman Finance documentary for a US audience.
Topic: {topic!r}
Chapter heading (production-only; do NOT speak it): {spec['heading']!r}
Chapter focus: {spec['focus']}
Write about 500–600 spoken words with clear examples, varied sentences, and meaningful transitions. This is one part of a larger documentary: do not repeat the opening hook, do not announce the chapter number, and do not conclude the entire video unless this is chapter 8. Do not use headings, bullets, parenthetical stage directions, or production notes in the narration. Do not invent statistics, studies, citations, laws, tax thresholds, current market data, or guaranteed investment results. Use clearly hypothetical numbers only for simple arithmetic examples, label them as hypothetical, and explain trade-offs/counterarguments where relevant. Explain general concepts rather than giving personalized financial or investment advice. Return JSON {{\"narration\": \"spoken narration only\"}}."""
        generated = ollama_json(prompt, model_name, num_predict=1150)
        narration = str(generated.get("narration", "")).strip()
        if word_count(narration) < 360:
            repair_prompt = f"Expand and rewrite the following chapter to 500–600 useful spoken words without padding or repeating ideas. Preserve caution: do not invent statistics, citations, regulations, or returns. Return JSON {{\"narration\": str}}. Heading: {spec['heading']}. Draft: {narration[:7000]}"
            repaired = ollama_json(repair_prompt, model_name, num_predict=1150)
            narration = str(repaired.get("narration", "")).strip()
        if word_count(narration) < 320:
            raise PipelineError(f"Ollama chapter {index} is too short ({word_count(narration)} words) after repair; not proceeding with an under-length documentary.")
        sections.append({"heading": spec["heading"], "narration": narration})
        LOGGER.info("Script chapter %d/8 generated: %d words.", index, word_count(narration))
    total = sum(word_count(s["narration"]) for s in sections)
    if total < 3_900:
        # Add one non-repetitive chapter specifically about a practical scenario / counterargument.
        extra = ollama_json(
            f"For the ongoing 30-minute documentary about {topic!r}, propose one additional distinct practical chapter that does not repeat any of these existing headings: {', '.join(s['heading'] for s in sections)}. Return JSON {{\"heading\":str,\"focus\":str}}.",
            model_name, num_predict=220,
        )
        heading, focus = str(extra.get("heading", "A practical decision framework")).strip(), str(extra.get("focus", "A careful, hypothetical decision scenario.")).strip()
        generated = ollama_json(
            f"Write a standalone additional chapter for a US audience about {topic!r}. Heading (not spoken): {heading}. Focus: {focus}. Write 450–550 useful words, no filler or repetition. No current statistics, laws, tax thresholds, fabricated citations, guaranteed returns or personalized advice. Return JSON {{\"narration\": str}}.",
            model_name, num_predict=1150,
        )
        narration = str(generated.get("narration", "")).strip()
        if word_count(narration) < 300:
            raise PipelineError(f"Full script reached only {total} words, and the extra chapter was also too short. Reduce generation limits only after checking Ollama logs.")
        sections.append({"heading": heading, "narration": narration})
        total += word_count(narration)
    if total < 3_900:
        raise PipelineError(f"Full script is only {total} words; expected approximately 4,200–4,500. Refusing to label a short script a 30-minute documentary.")
    return sections


def plain_script(sections: Sequence[dict[str, str]]) -> str:
    # Chapter headings remain production metadata and are intentionally not spoken.
    return "\n\n".join(str(section["narration"]).strip() for section in sections).strip()


def ensure_kokoro_files(model_path: Path, voices_path: Path) -> None:
    # Official upstream release documents these file names and the 0.6.x timing-capable model graph.
    download_file(KOKORO_MODEL_URL, model_path, min_bytes=100 * 1024 * 1024)
    download_file(KOKORO_VOICES_URL, voices_path, min_bytes=10 * 1024 * 1024)


def load_kokoro(model_path: Path, voices_path: Path) -> Any:
    try:
        from kokoro_onnx import Kokoro
    except ImportError as exc:
        raise PipelineError("Missing `kokoro-onnx`. Run the GitHub Actions workflow or install the package listed in the workflow's pip step.") from exc
    try:
        kokoro = Kokoro(str(model_path), str(voices_path))
    except Exception as exc:  # noqa: BLE001 - enrich model-load diagnostics
        raise PipelineError(f"Kokoro ONNX failed to load model/voices ({type(exc).__name__}). Verify the downloaded files and available RAM.") from exc
    available = kokoro.get_voices() if hasattr(kokoro, "get_voices") else []
    if VOICE_DEFAULT not in available:
        try:
            kokoro.voices.close()
        except Exception:
            pass
        raise PipelineError(f"Required Kokoro voice `{VOICE_DEFAULT}` is unavailable. Available voices: {', '.join(available) or 'unknown'}; no silent voice substitution was made.")
    if not getattr(kokoro, "has_timings", False):
        LOGGER.warning("Kokoro graph has no duration output. Caption timing will use measured TTS segment durations as a fallback.")
    return kokoro


def extract_timing_field(item: Any, names: Sequence[str], default: Any = None) -> Any:
    for name in names:
        if isinstance(item, dict) and name in item:
            return item[name]
        if hasattr(item, name):
            return getattr(item, name)
    return default


def timed_items_to_word_timing(timings: Sequence[Any], original_words: Sequence[str]) -> list[dict[str, Any]] | None:
    """Accept only plausible word-level timings; never fabricate word times from character counts."""
    if not timings or not original_words:
        return None
    normalized_original = [re.sub(r"[^a-z0-9']", "", w.lower().replace("’", "'")) for w in original_words]
    parsed: list[dict[str, Any]] = []
    for item in timings:
        text_value = extract_timing_field(item, ("text", "word", "token"), "")
        start = extract_timing_field(item, ("start", "start_time", "begin"), None)
        end = extract_timing_field(item, ("end", "end_time", "stop"), None)
        if start is None or end is None:
            return None
        tokens = word_tokens(str(text_value))
        if len(tokens) != 1:
            # Punctuation/phoneme or grouped timing output isn't guaranteed to be word-aligned.
            return None
        parsed.append({"word": tokens[0], "start": float(start), "end": float(end)})
    if len(parsed) != len(original_words):
        return None
    normalized_timed = [re.sub(r"[^a-z0-9']", "", entry["word"].lower().replace("’", "'")) for entry in parsed]
    if not normalized_original or not normalized_timed:
        return None
    match_rate = sum(left == right for left, right in zip(normalized_original, normalized_timed)) / max(len(normalized_original), len(normalized_timed))
    if match_rate < 0.88:
        return None
    for entry in parsed:
        if not (math.isfinite(entry["start"]) and math.isfinite(entry["end"]) and entry["start"] >= 0 and entry["end"] > entry["start"]):
            return None
    return parsed


def make_cues_from_word_times(original_words: Sequence[str], timed_words: Sequence[dict[str, Any]], audio_duration: float) -> list[dict[str, Any]]:
    cues: list[dict[str, Any]] = []
    for index in range(0, len(original_words), MAX_SUBTITLE_WORDS):
        last_index = min(index + MAX_SUBTITLE_WORDS, len(original_words))
        start = float(timed_words[index]["start"])
        end = float(timed_words[last_index - 1]["end"])
        start = max(0.0, min(start, audio_duration))
        end = max(start + 0.04, min(end, audio_duration))
        cues.append({"text": " ".join(original_words[index:last_index]), "start": start, "end": end,
                     "word_start_index": index, "word_end_index": last_index})
    # Timing outputs can overlap slightly on phonetic boundaries; enforce legal, non-overlapping subtitle cues.
    for i in range(1, len(cues)):
        if cues[i]["start"] < cues[i - 1]["end"]:
            boundary = max(cues[i - 1]["start"] + 0.02, cues[i]["start"])
            cues[i - 1]["end"] = min(cues[i - 1]["end"], boundary)
            cues[i]["start"] = max(cues[i]["start"], cues[i - 1]["end"])
        if cues[i]["end"] <= cues[i]["start"]:
            cues[i]["end"] = min(audio_duration, cues[i]["start"] + 0.04)
    return cues


def synthesize_with_kokoro(script: str, sections: Sequence[dict[str, str]], tts: Any, audio_path: Path) -> tuple[float, list[dict[str, Any]], str]:
    import numpy as np
    import soundfile as sf

    audio_path.parent.mkdir(parents=True, exist_ok=True)
    words = script.split()
    if not words:
        raise PipelineError("Script is empty; refusing to synthesize zero-duration audio.")
    start_wall = time.monotonic()
    timings: Sequence[Any] = []
    samples = None
    sample_rate = None
    timing_method = "kokoro-create-timed"
    try:
        if hasattr(tts, "create_timed") and getattr(tts, "has_timings", False):
            samples, sample_rate, timings = tts.create_timed(script, voice=VOICE_DEFAULT, speed=1.0,
                                                              lang="en-us", continuous=True,
                                                              sentence_pause=0.22, clause_pause=0.10)
        else:
            raise RuntimeError("selected model does not expose phoneme duration output")
    except Exception as exc:  # noqa: BLE001 - deliberate tested fallback with measured duration per cue
        LOGGER.warning("Kokoro's timed whole-script path failed (%s); using measured four-word TTS segments. Speech prosody can be less smooth in this fallback.", type(exc).__name__)
        samples = None

    if samples is not None and sample_rate is not None:
        samples = np.asarray(samples, dtype=np.float32).ravel()
        if int(sample_rate) != KOKORO_SAMPLE_RATE:
            raise PipelineError(f"Kokoro returned {sample_rate} Hz, expected {KOKORO_SAMPLE_RATE} Hz. Refusing to mislabel audio.")
        audio_duration = len(samples) / int(sample_rate)
        timed_words = timed_items_to_word_timing(timings, words)
        if timed_words is not None:
            cues = make_cues_from_word_times(words, timed_words, audio_duration)
            timing_method = "kokoro-onnx-duration-timestamps"
        else:
            # Don't guess word timing. Re-synthesize the exact caption phrases and use each measured segment's duration.
            LOGGER.warning("Kokoro timing output did not validate as one timing per source word; re-synthesizing measured four-word segments for reliable cue boundaries.")
            samples = None

    if samples is None:
        timing_method = "measured-kokoro-four-word-segments"
        segment_arrays: list[Any] = []
        cues = []
        current_sample = 0
        silence = np.zeros(int(KOKORO_SAMPLE_RATE * 0.055), dtype=np.float32)
        for index in range(0, len(words), MAX_SUBTITLE_WORDS):
            end_index = min(index + MAX_SUBTITLE_WORDS, len(words))
            phrase = " ".join(words[index:end_index])
            try:
                chunk, chunk_rate = tts.create(phrase, voice=VOICE_DEFAULT, speed=1.0, lang="en-us")
            except Exception as exc:  # noqa: BLE001 - attach concise segment diagnosis
                raise PipelineError(f"Kokoro failed on narration words {index + 1}–{end_index}: {type(exc).__name__}.") from exc
            if int(chunk_rate) != KOKORO_SAMPLE_RATE:
                raise PipelineError(f"Kokoro returned {chunk_rate} Hz, expected {KOKORO_SAMPLE_RATE} Hz.")
            chunk = np.asarray(chunk, dtype=np.float32).ravel()
            if len(chunk) == 0:
                raise PipelineError(f"Kokoro produced an empty speech segment for words {index + 1}–{end_index}.")
            start_sec = current_sample / KOKORO_SAMPLE_RATE
            segment_arrays.append(chunk)
            current_sample += len(chunk)
            end_sec = current_sample / KOKORO_SAMPLE_RATE
            cues.append({"text": phrase, "start": start_sec, "end": end_sec,
                         "word_start_index": index, "word_end_index": end_index})
            segment_arrays.append(silence)
            current_sample += len(silence)
        if not segment_arrays:
            raise PipelineError("Kokoro fallback produced no audio segments.")
        samples = np.concatenate(segment_arrays).astype(np.float32, copy=False)
        audio_duration = len(samples) / KOKORO_SAMPLE_RATE
    if not math.isfinite(audio_duration) or audio_duration <= 0 or len(samples) < KOKORO_SAMPLE_RATE:
        raise PipelineError("Kokoro generated invalid audio (empty or under one second).")
    sf.write(str(audio_path), samples, KOKORO_SAMPLE_RATE, subtype="PCM_16", format="WAV")
    info = sf.info(str(audio_path))
    if info.samplerate != KOKORO_SAMPLE_RATE or info.frames == 0 or info.channels not in {1, 2}:
        raise PipelineError(f"Generated WAV failed validation: {info}.")
    # Clip only any tiny rounding excess from model token timing; speech is not truncated.
    adjusted: list[dict[str, Any]] = []
    for cue in cues:
        start = max(0.0, float(cue["start"]))
        end = min(audio_duration, float(cue["end"]))
        if end <= start:
            continue
        adjusted.append({**cue, "start": start, "end": end})
    validate_cues(adjusted, audio_duration)
    elapsed = time.monotonic() - start_wall
    LOGGER.info("Kokoro narration: %.2f seconds at %d Hz; timing method=%s; synthesis wall time=%.1fs.", audio_duration, info.samplerate, timing_method, elapsed)
    return audio_duration, adjusted, timing_method


def generate_scene_prompts(model_name: str, topic: str, captions: Sequence[dict[str, Any]], scene_count: int) -> list[dict[str, Any]]:
    scenes: list[dict[str, Any]] = []
    batch_size = 12  # prompt batch size only; not a scene cap
    for offset in range(0, scene_count, batch_size):
        current_count = min(batch_size, scene_count - offset)
        batch_requests = []
        for local_idx in range(current_count):
            number = offset + local_idx + 1
            start_sec = (number - 1) * SCENE_SECONDS
            end_sec = number * SCENE_SECONDS
            relevant = [str(cue["text"]) for cue in captions if float(cue["end"]) > start_sec and float(cue["start"]) < end_sec]
            if not relevant:
                if captions:
                    nearest = min(captions, key=lambda cue: abs(float(cue["start"]) - min(start_sec, max(0.0, float(captions[-1]["end"]) - 0.1))))
                    relevant = [str(nearest["text"])]
                else:
                    relevant = [topic]
            narration = " ".join(relevant)
            batch_requests.append({"scene_number": number, "narration": narration[:500]})
        prompt = f"""Create distinct scene-specific text-to-image prompts for an educational Stickman Finance documentary about {topic!r}.
The scenes follow the supplied narration. For each item, produce one unique visual prompt that precisely conveys that sentence's main financial concept. Vary camera composition, stickman pose, props, background, framing, visual metaphor, foreground/background placement, and colors across scenes while keeping a coherent simple 2D editorial stickman style. Do not add readable text, numbers, labels, logos, brand names or watermark to images. Do not put multiple panels or a collage in the image. Each prompt needs a different visual composition and must not merely repeat a generic centered stickman.
Return JSON {{\"scenes\":[{{\"scene_number\": integer, \"image_prompt\": string, \"negative_prompt\": string}}, ...]}}. Return exactly {current_count} records, for the input scene numbers and in the same order.\nInput scenes:\n{json.dumps(batch_requests, ensure_ascii=False)}"""
        result = ollama_json(prompt, model_name, num_predict=max(500, current_count * 100))
        items = result.get("scenes")
        if not isinstance(items, list) or len(items) != current_count:
            raise PipelineError(f"Scene-planner batch {offset + 1}–{offset + current_count} returned {len(items) if isinstance(items, list) else 'invalid'} prompts; expected {current_count}. Refusing to omit scene prompts.")
        for local_idx, item in enumerate(items):
            number = offset + local_idx + 1
            expected_number = int(item.get("scene_number", -1)) if isinstance(item, dict) else -1
            image_prompt = str(item.get("image_prompt", "")).strip() if isinstance(item, dict) else ""
            if expected_number != number or not image_prompt:
                raise PipelineError(f"Scene planner returned an invalid record for scene {number}.")
            narration = batch_requests[local_idx]["narration"]
            start_sec = (number - 1) * SCENE_SECONDS
            seed = random.SystemRandom().randint(0, 2**31 - 1)
            prompt_hash = sha256_text(image_prompt + "\n" + narration + "\n" + str(seed))[:10]
            scenes.append({
                "scene_id": f"scene-{number:04d}-{prompt_hash}",
                "scene_number": number,
                "narration": narration,
                "image_prompt": image_prompt,
                "negative_prompt": str(item.get("negative_prompt", ""))[:500] if isinstance(item, dict) else "",
                "start_seconds": float(start_sec),
                "end_seconds": float(start_sec + SCENE_SECONDS),
                "output_image_path": f"scenes/original/scene_{number:04d}_{prompt_hash}.png",
                "processed_image_path": f"scenes/processed/scene_{number:04d}_{prompt_hash}.jpg",
                "seed": seed,
                "generation_metadata": {"model": IMAGE_MODEL_DEFAULT, "attempts": 0, "prompt_sha256": sha256_text(image_prompt)},
                "validation_status": "planned",
            })
    validate_scene_plan(scenes, scene_count)
    return scenes


def image_dhash(image: Any) -> int:
    from PIL import Image
    grayscale = image.convert("L").resize((9, 8))
    pixels = list(grayscale.getdata())
    bits = 0
    for y in range(8):
        for x in range(8):
            bits = (bits << 1) | int(pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return bits


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def image_cache_root() -> Path:
    return Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))).expanduser()


def available_ram_gb() -> float | None:
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 ** 3)
    except Exception:
        try:
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 ** 2)
        except Exception:
            return None
    return None


def load_diffusion_pipeline(model_name: str) -> tuple[Any, Any, str]:
    try:
        import torch
        from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
    except ImportError as exc:
        raise PipelineError("Missing torch/diffusers/transformers dependencies. Run the workflow's pinned install step.") from exc
    cpu_threads = max(1, min(4, os.cpu_count() or 2))
    try:
        torch.set_num_threads(cpu_threads)
    except Exception:
        pass
    has_cuda = bool(torch.cuda.is_available())
    device = "cuda" if has_cuda else "cpu"
    dtype = torch.float16 if has_cuda else torch.float32
    ram_free_gb = available_ram_gb()
    if not has_cuda and ram_free_gb is not None:
        min_ram = float(os.environ.get("SDXL_MIN_AVAILABLE_RAM_GB", "9.0"))
        allow_low_ram = os.environ.get("SDXL_ALLOW_LOW_RAM_CPU", "false").lower() in {"1", "true", "yes"}
        if ram_free_gb < min_ram and not allow_low_ram:
            raise PipelineError(f"CPU-only SDXL-Turbo was not loaded: only {ram_free_gb:.1f} GB RAM is available; this float32 path requires at least {min_ram:.1f} GB available RAM as a conservative safety gate. Standard GitHub-hosted runners and 8 GB PCs may not meet this requirement. Use a CUDA-capable runner with sufficient VRAM or explicitly set SDXL_ALLOW_LOW_RAM_CPU=true to attempt at your own risk.")
        if ram_free_gb < min_ram:
            LOGGER.warning("Explicit low-RAM CPU override enabled: %.1f GB available. OOM or very slow inference is possible.", ram_free_gb)
    LOGGER.info("Loading %s locally on %s (dtype=%s, available RAM=%s GB). Cache: %s", model_name, device, dtype,
                f"{ram_free_gb:.1f}" if ram_free_gb is not None else "unknown", image_cache_root())
    try:
        pipe = DiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=dtype,
            use_safetensors=True,
            cache_dir=str(image_cache_root()),
            local_files_only=False,
        )
        try:
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
        except Exception as exc:  # avoid hiding an incompatible required scheduler adjustment
            raise PipelineError(f"Could not configure SDXL-Turbo's trailing timestep scheduler: {type(exc).__name__}.") from exc
        pipe.to(device)
        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=True)
        return pipe, torch, device
    except PipelineError:
        raise
    except Exception as exc:  # noqa: BLE001 - add diagnostic without massive stack dump
        raise PipelineError(f"SDXL-Turbo failed to load on {device} ({type(exc).__name__}). CPU runners can run out of RAM/disk; check model cache and PyTorch compatibility.") from exc


def run_image_generation(pipe: Any, torch: Any, device: str, prompt: str, seed: int, steps: int | None = None) -> Any:
    # SDXL-Turbo's model card recommends 1–4 steps and guidance_scale=0.0; use 1 CPU / 2 GPU steps.
    inference_steps = steps or (2 if device == "cuda" else 1)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    with torch.inference_mode():
        result = pipe(prompt=prompt, width=512, height=512, num_inference_steps=inference_steps,
                      guidance_scale=0.0, generator=generator)
    if not getattr(result, "images", None):
        raise PipelineError("SDXL-Turbo returned no image.")
    return result.images[0].convert("RGB")


def image_to_video_frame(image: Any, output_path: Path) -> None:
    from PIL import Image
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source = image.convert("RGB")
    target_ratio = VIDEO_WIDTH / VIDEO_HEIGHT
    src_ratio = source.width / source.height
    if src_ratio > target_ratio:
        new_height = VIDEO_HEIGHT
        new_width = max(VIDEO_WIDTH, int(round(new_height * src_ratio)))
    else:
        new_width = VIDEO_WIDTH
        new_height = max(VIDEO_HEIGHT, int(round(new_width / src_ratio)))
    resized = source.resize((new_width, new_height), Image.Resampling.LANCZOS)
    left = (new_width - VIDEO_WIDTH) // 2
    top = (new_height - VIDEO_HEIGHT) // 2
    frame = resized.crop((left, top, left + VIDEO_WIDTH, top + VIDEO_HEIGHT)).convert("RGB")
    frame.save(output_path, format="JPEG", quality=91, optimize=True, subsampling=0)
    with Image.open(output_path) as verified:
        verified.verify()
    with Image.open(output_path) as verified:
        if verified.size != (VIDEO_WIDTH, VIDEO_HEIGHT) or verified.mode not in {"RGB", "RGBA"}:
            raise PipelineError(f"Processed frame validation failed: {output_path}.")


def generate_scene_images(scenes: list[dict[str, Any]], output_dir: Path, manifest: dict[str, Any], manifest_path: Path,
                          model_name: str, allow_slow_cpu_full: bool, topic: str,
                          thumbnail_path: Path, thumbnail_headline: str) -> tuple[float, str]:
    from PIL import Image
    recent_hashes: list[int] = []
    for scene in scenes:
        processed_path = output_dir / scene["processed_image_path"]
        original_path = output_dir / scene["output_image_path"]
        if scene.get("validation_status") == "generated" and processed_path.is_file() and original_path.is_file():
            expected_hash = scene.get("processed_sha256")
            if expected_hash and sha256_file(processed_path) == expected_hash:
                try:
                    with Image.open(processed_path) as image:
                        image.verify()
                    with Image.open(original_path) as image:
                        image.verify()
                    with Image.open(processed_path) as image:
                        recent_hashes.append(image_dhash(image))
                        recent_hashes = recent_hashes[-12:]
                    continue
                except Exception:
                    LOGGER.warning("A checkpointed scene image failed validation; regenerating scene %d.", scene["scene_number"])
            else:
                LOGGER.warning("A checkpointed scene hash mismatched; regenerating scene %d.", scene["scene_number"])
        if scene.get("validation_status") != "generated":
            # Same run folder + prompt-hash filename means these can only be interrupted files for this scene.
            # Delete invalid partials and regenerate; never reuse them as a successful checkpoint.
            if original_path.exists() or processed_path.exists():
                LOGGER.warning("Removing incomplete/uncheckpointed image partial for scene %d before safe regeneration.", scene["scene_number"])
                original_path.unlink(missing_ok=True)
                processed_path.unlink(missing_ok=True)
    pending = [scene for scene in scenes if not (scene.get("validation_status") == "generated" and
               (output_dir / scene["processed_image_path"]).is_file() and (output_dir / scene["output_image_path"]).is_file() and
               scene.get("processed_sha256") == sha256_file(output_dir / scene["processed_image_path"]))]
    if not pending:
        LOGGER.info("All %d image assets passed checkpoint hashes; reusing them.", len(scenes))
        if not thumbnail_path.is_file() or manifest.get("thumbnail", {}).get("sha256") != (sha256_file(thumbnail_path) if thumbnail_path.is_file() else None):
            pipe, torch, device = load_diffusion_pipeline(model_name)
            try:
                create_thumbnail(pipe, torch, device, topic, thumbnail_headline, thumbnail_path,
                                 random.SystemRandom().randint(0, 2**31 - 1), manifest, manifest_path)
            finally:
                del pipe
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
            return 0.0, device
        return 0.0, "cached"
    pipe, torch, device = load_diffusion_pipeline(model_name)
    wall_total = 0.0
    per_image_seconds: list[float] = []
    try:
        for scene in pending:
            original_path = output_dir / scene["output_image_path"]
            processed_path = output_dir / scene["processed_image_path"]
            original_path.parent.mkdir(parents=True, exist_ok=True)
            processed_path.parent.mkdir(parents=True, exist_ok=True)
            last_error: Exception | None = None
            accepted = None
            accepted_hash = None
            for attempt in range(3):
                prompt = finance_style_prompt(scene["image_prompt"])
                if attempt:
                    prompt += f" Alternate composition attempt {attempt}: different camera framing, pose, props and color arrangement."
                started = time.monotonic()
                try:
                    image = run_image_generation(pipe, torch, device, prompt, int(scene["seed"]) + attempt * 104_729)
                    image_hash = image_dhash(image)
                    if any(hamming_distance(image_hash, old_hash) <= 3 for old_hash in recent_hashes[-12:]):
                        last_error = PipelineError("Generated frame is perceptually too similar to one of the previous 12 frames.")
                        LOGGER.warning("Scene %d looks too similar to a recent image; bounded retry %d/2.", scene["scene_number"], attempt + 1)
                        continue
                    image.save(original_path, format="PNG", optimize=True)
                    image_to_video_frame(image, processed_path)
                    accepted = image
                    accepted_hash = image_hash
                    per_image_seconds.append(time.monotonic() - started)
                    wall_total += per_image_seconds[-1]
                    scene["generation_metadata"]["attempts"] = attempt + 1
                    scene["generation_metadata"]["device"] = device
                    scene["generation_metadata"]["inference_steps"] = (2 if device == "cuda" else 1)
                    scene["generation_metadata"]["generated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
                    scene["validation_status"] = "generated"
                    scene["original_sha256"] = sha256_file(original_path)
                    scene["processed_sha256"] = sha256_file(processed_path)
                    scene["perceptual_hash"] = f"{image_hash:016x}"
                    recent_hashes.append(accepted_hash)
                    recent_hashes[:] = recent_hashes[-12:]
                    atomic_json(manifest_path, manifest)
                    LOGGER.info("Image %d/%d completed (%.1fs on %s).", scene["scene_number"], len(scenes), per_image_seconds[-1], device)
                    if device == "cpu" and manifest.get("mode") == "full" and len(per_image_seconds) >= 1:
                        measured = sum(per_image_seconds) / len(per_image_seconds)
                        projected_hours = measured * len(pending) / 3600.0
                        if projected_hours > float(os.environ.get("FULL_RUN_MAX_CPU_IMAGE_HOURS", "5.0")) and not allow_slow_cpu_full:
                            raise PipelineError(
                                f"CPU-only SDXL-Turbo measured {measured:.1f}s per image, projecting about {projected_hours:.1f} hours for remaining image generation, before model download/TTS/render. "
                                "To avoid a likely six-hour GitHub-hosted job timeout, this run stopped with a valid checkpoint. Use a CUDA-enabled self-hosted runner, or explicitly set allow_slow_cpu_full=true in workflow dispatch to accept the risk. No scenes were silently skipped."
                            )
                    break
                except PipelineError:
                    raise
                except Exception as exc:  # noqa: BLE001 - bounded retry per scene
                    last_error = exc
                    LOGGER.warning("Scene %d generation attempt %d/3 failed (%s).", scene["scene_number"], attempt + 1, type(exc).__name__)
                    if attempt < 2:
                        time.sleep(1.0 + attempt)
            if accepted is None:
                raise PipelineError(f"Failed to generate scene {scene['scene_number']} after three attempts. Last error: {type(last_error).__name__ if last_error else 'unknown'}.")
        if (not thumbnail_path.is_file()) or manifest.get("thumbnail", {}).get("sha256") != (sha256_file(thumbnail_path) if thumbnail_path.is_file() else None):
            create_thumbnail(pipe, torch, device, topic, thumbnail_headline, thumbnail_path,
                             random.SystemRandom().randint(0, 2**31 - 1), manifest, manifest_path)
    finally:
        del pipe
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    mean_s = (sum(per_image_seconds) / len(per_image_seconds)) if per_image_seconds else 0.0
    LOGGER.info("Images generated: %d/%d; image inference wall time %.1fs; measured mean %.1fs/image.",
                sum(s.get("validation_status") == "generated" for s in scenes), len(scenes), wall_total, mean_s)
    return wall_total, device


def find_font_path() -> str | None:
    candidates = [
        os.environ.get("STICKMAN_FONT"),
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/Arial.ttf",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    return None


def create_thumbnail(pipe: Any, torch: Any, device: str, topic: str, headline: str, output_path: Path,
                     seed: int, manifest: dict[str, Any], manifest_path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont
    prompt = finance_style_prompt(
        f"YouTube documentary thumbnail illustration about {topic!r}; a dramatic, off-center stickman character and one large unmistakable relevant financial object; "
        "landscape composition intended for a 16:9 crop; leave uncluttered negative space on the left for a headline; vivid contrasting but tasteful palette."
    )
    image = run_image_generation(pipe, torch, device, prompt, seed, steps=(2 if device == "cuda" else 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_to_video_frame(image, output_path.with_name("thumbnail-base.jpg"))
    with Image.open(output_path.with_name("thumbnail-base.jpg")) as base:
        canvas = base.convert("RGB").resize((1280, 720), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(canvas, "RGBA")
    font_path = find_font_path()
    clean_headline = " ".join(str(headline).upper().split())[:65] or "MONEY HABITS THAT MATTER"
    words = clean_headline.split()
    lines: list[str] = []
    current = ""
    for word in words:
        proposed = (current + " " + word).strip()
        if len(proposed) > 16 and current:
            lines.append(current)
            current = word
        else:
            current = proposed
    if current:
        lines.append(current)
    lines = lines[:3]
    if font_path:
        font_size = 104
        while font_size > 56:
            font = ImageFont.truetype(font_path, font_size)
            widths = [draw.textbbox((0, 0), line, font=font, stroke_width=4)[2] for line in lines]
            if max(widths, default=0) <= 700 and font_size * len(lines) * 1.15 <= 560:
                break
            font_size -= 4
    else:
        font = ImageFont.load_default()
    # Semi-transparent dark panel + strong stroke keep text readable over generated art.
    panel_width = min(790, max(600, max(draw.textbbox((0, 0), line, font=font, stroke_width=3)[2] for line in lines) + 80))
    draw.rounded_rectangle((22, 70, 22 + panel_width, 650), radius=28, fill=(5, 12, 23, 190), outline=(255, 205, 55, 230), width=4)
    line_height = max(60, int(getattr(font, "size", 60) * 1.15))
    total_height = line_height * len(lines)
    y = max(85, int((720 - total_height) / 2))
    for line in lines:
        draw.text((56, y), line, font=font, fill=(255, 225, 65, 255), stroke_width=4, stroke_fill=(0, 0, 0, 255))
        y += line_height
    canvas.save(output_path, "JPEG", quality=93, optimize=True)
    (output_path.with_name("thumbnail-base.jpg")).unlink(missing_ok=True)
    with Image.open(output_path) as check:
        check.verify()
    with Image.open(output_path) as check:
        if check.size != (1280, 720) or check.format != "JPEG" or output_path.stat().st_size > 50 * 1024 * 1024:
            raise PipelineError("Thumbnail failed output dimension/format/size validation.")
    manifest["thumbnail"] = {"status": "generated", "path": output_path.name, "sha256": sha256_file(output_path), "seed": seed}
    atomic_json(manifest_path, manifest)


def write_subtitles(cues: Sequence[dict[str, Any]], output_dir: Path, title: str, audio_duration: float) -> tuple[Path, Path]:
    validate_cues(cues, audio_duration)
    ass_path = output_dir / "subtitles.ass"
    srt_path = output_dir / "captions.srt"
    safe_title = title.replace("\n", " ")[:100]
    ass_lines = [
        "[Script Info]", f"Title: {safe_title}", "ScriptType: v4.00+", "WrapStyle: 2", "ScaledBorderAndShadow: yes",
        "PlayResX: 1920", "PlayResY: 1080", "YCbCr Matrix: TV.709", "", "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        "Style: Default,DejaVu Sans,56,&H00FFFFFF,&H0000FFFF,&H00101010,&H78000000,-1,0,0,0,100,100,0,0,1,4,2,2,100,100,100,1",
        "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    srt_blocks: list[str] = []
    for index, cue in enumerate(cues, 1):
        text = str(cue["text"]).replace("{", "(").replace("}", ")").replace("\n", " ").strip()
        start, end = float(cue["start"]), float(cue["end"])
        # Subtle pop/bounce applies to subtitle glyphs only; still images remain static.
        animated_text = r"{\an2\pos(960,965)\fscx78\fscy78\t(0,100,\fscx112\fscy112)\t(100,190,\fscx100\fscy100)}" + text
        ass_lines.append(f"Dialogue: 0,{timestamp_ass(start)},{timestamp_ass(end)},Default,,0,0,0,,{animated_text}")
        srt_blocks.append(f"{index}\n{timestamp_srt(start)} --> {timestamp_srt(end)}\n{text}")
    ass_path.write_text("\n".join(ass_lines) + "\n", encoding="utf-8")
    srt_path.write_text("\n\n".join(srt_blocks) + "\n", encoding="utf-8")
    validate_subtitle_files(cues, ass_path, srt_path, audio_duration)
    return ass_path, srt_path


def validate_subtitle_files(cues: Sequence[dict[str, Any]], ass_path: Path, srt_path: Path, audio_duration: float) -> None:
    validate_cues(cues, audio_duration)
    if not ass_path.is_file() or "[Events]" not in ass_path.read_text(encoding="utf-8"):
        raise PipelineError("ASS subtitle file failed structural validation.")
    srt = srt_path.read_text(encoding="utf-8")
    blocks = [block for block in srt.strip().split("\n\n") if block.strip()]
    if len(blocks) != len(cues):
        raise PipelineError(f"SRT has {len(blocks)} blocks but {len(cues)} subtitle cues.")
    for index, block in enumerate(blocks, 1):
        lines = block.splitlines()
        if len(lines) < 3 or lines[0].strip() != str(index):
            raise PipelineError(f"SRT block {index} is malformed.")
        if visible_word_count(" ".join(lines[2:])) > MAX_SUBTITLE_WORDS:
            raise PipelineError(f"SRT block {index} violates the four-word maximum.")


def ffmpeg_binary() -> str:
    binary = shutil.which(os.environ.get("FFMPEG_BINARY", "ffmpeg"))
    if not binary:
        raise PipelineError("FFmpeg is missing. The GitHub workflow installs it with apt; install FFmpeg and libass locally.")
    return binary


def ffprobe_binary() -> str:
    binary = shutil.which(os.environ.get("FFPROBE_BINARY", "ffprobe"))
    if not binary:
        raise PipelineError("FFprobe is missing. Install the FFmpeg package that includes ffprobe.")
    return binary


def run_cmd(command: Sequence[str], timeout: int = 300, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(list(command), cwd=str(cwd) if cwd else None, check=True, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise PipelineError(f"Command timed out after {timeout}s: {Path(command[0]).name} {command[1] if len(command) > 1 else ''}") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()[-2500:]
        raise PipelineError(f"Command failed ({exc.returncode}): {' '.join(str(part) for part in command[:3])}; output: {detail}") from exc


def validate_render_tools() -> dict[str, str]:
    ffmpeg = ffmpeg_binary()
    ffprobe = ffprobe_binary()
    version = run_cmd([ffmpeg, "-version"], timeout=20).stdout.splitlines()[0]
    filters = run_cmd([ffmpeg, "-hide_banner", "-filters"], timeout=20).stdout
    if not re.search(r"\bsubtitles\b", filters):
        raise PipelineError("This FFmpeg build has no `subtitles` filter (libass). Install an FFmpeg build compiled with libass.")
    probe_version = run_cmd([ffprobe, "-version"], timeout=20).stdout.splitlines()[0]
    return {"ffmpeg": version, "ffprobe": probe_version, "libass_subtitles": "available"}


def probe_media(video_path: Path) -> dict[str, Any]:
    if not video_path.is_file() or video_path.stat().st_size < 10_000:
        raise PipelineError("Rendered video is missing or implausibly small; refusing to publish it.")
    raw = run_cmd([ffprobe_binary(), "-v", "error", "-show_entries", "format=duration,size:stream=codec_type,codec_name,width,height,sample_rate,channels", "-of", "json", str(video_path)], timeout=60).stdout
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PipelineError("FFprobe returned invalid JSON for the rendered MP4.") from exc
    streams = data.get("streams", [])
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if not video_stream or not audio_stream:
        raise PipelineError("Rendered MP4 must contain both video and audio streams.")
    if video_stream.get("codec_name") != "h264" or (video_stream.get("width"), video_stream.get("height")) != (1920, 1080):
        raise PipelineError(f"Unexpected video stream: {video_stream}; expected H.264 at 1920x1080.")
    if audio_stream.get("codec_name") != "aac":
        raise PipelineError(f"Unexpected audio codec: {audio_stream.get('codec_name')}; expected AAC.")
    duration = float(data.get("format", {}).get("duration", 0))
    if duration <= 0:
        raise PipelineError("FFprobe reports a zero/invalid MP4 duration.")
    return {"duration_seconds": duration, "size_bytes": video_path.stat().st_size, "video_codec": video_stream["codec_name"],
            "audio_codec": audio_stream["codec_name"], "width": video_stream["width"], "height": video_stream["height"],
            "audio_sample_rate": audio_stream.get("sample_rate"), "audio_channels": audio_stream.get("channels")}


def create_concat_file(scenes: Sequence[dict[str, Any]], output_dir: Path) -> Path:
    concat_path = output_dir / "scenes_concat.txt"
    rows = ["ffconcat version 1.0"]
    for scene in scenes:
        frame = (output_dir / scene["processed_image_path"]).resolve()
        if not frame.is_file():
            raise PipelineError(f"Scene {scene['scene_number']} processed image is missing: {frame}")
        # FFconcat uses single-quoted paths; escape backslashes and embedded single quotes.
        escaped = str(frame).replace("\\", "/").replace("'", "'\\''")
        rows.append(f"file '{escaped}'")
        rows.append(f"duration {SCENE_SECONDS}")
    # The concat demuxer applies the last duration only when another packet follows.
    last = (output_dir / scenes[-1]["processed_image_path"]).resolve()
    escaped_last = str(last).replace("\\", "/").replace("'", "'\\''")
    rows.append(f"file '{escaped_last}'")
    concat_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return concat_path


def render_video(scenes: Sequence[dict[str, Any]], audio_path: Path, ass_path: Path, output_dir: Path,
                 expected_audio_duration: float) -> dict[str, Any]:
    tools = validate_render_tools()
    concat_file = create_concat_file(scenes, output_dir)
    video_path = output_dir / "video.mp4"
    temp_video = output_dir / "video.rendering.mp4"
    temp_video.unlink(missing_ok=True)
    video_duration = len(scenes) * SCENE_SECONDS
    if expected_audio_duration > video_duration + 0.03:
        raise PipelineError(f"Narration is {expected_audio_duration:.2f}s but the planned still-image timeline is only {video_duration}s. Will not cut off narration; re-plan the scene count.")
    escaped_ass = str(ass_path.resolve()).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
    command = [
        ffmpeg_binary(), "-hide_banner", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file),
        "-i", str(audio_path), "-vf", f"subtitles=filename='{escaped_ass}'", "-r", "30", "-t", f"{video_duration:.3f}",
        "-c:v", "libx264", "-preset", os.environ.get("FFMPEG_PRESET", "veryfast"), "-crf", os.environ.get("FFMPEG_CRF", "27"),
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-af", "apad", "-movflags", "+faststart", str(temp_video),
    ]
    LOGGER.info("Rendering static hard-cut slideshow: %d scenes × %ds = %ds; no image motion/transitions.", len(scenes), SCENE_SECONDS, video_duration)
    run_cmd(command, timeout=int(os.environ.get("FFMPEG_RENDER_TIMEOUT_SECONDS", "18000")))
    info = probe_media(temp_video)
    if abs(info["duration_seconds"] - video_duration) > 0.35:
        raise PipelineError(f"Rendered MP4 duration {info['duration_seconds']:.2f}s differs from the expected {video_duration}s scene timeline.")
    # Decode all streams; a zero FFmpeg exit without decodability is not accepted as proof of a valid video.
    run_cmd([ffmpeg_binary(), "-hide_banner", "-v", "error", "-xerror", "-i", str(temp_video), "-f", "null", "-"], timeout=int(os.environ.get("FFMPEG_DECODE_TIMEOUT_SECONDS", "18000")))
    temp_video.replace(video_path)
    info = probe_media(video_path)
    info["ffmpeg_version"] = tools["ffmpeg"]
    info["ffprobe_version"] = tools["ffprobe"]
    info["decode_validation"] = "full decode passed"
    return info


def chapters_from_sections(sections: Sequence[dict[str, str]], cues: Sequence[dict[str, Any]], video_duration: float) -> list[dict[str, Any]]:
    total_words = sum(word_count(section["narration"]) for section in sections)
    if not sections or not cues:
        return [{"timestamp": "00:00", "start_seconds": 0, "title": "The key idea"}]
    offsets: list[tuple[int, str]] = []
    offset = 0
    for section in sections:
        offsets.append((offset, str(section["heading"]).strip()[:80] or "Chapter"))
        # Cue word indices follow whitespace-separated source tokens, not regex tokenization.
        offset += len(str(section["narration"]).split())
    def time_at_word(word_index: int) -> float:
        for cue in cues:
            if int(cue.get("word_start_index", 0)) <= word_index < int(cue.get("word_end_index", 0)):
                return float(cue["start"])
        return float(cues[-1]["end"])
    raw = [{"start_seconds": 0.0, "title": offsets[0][1]}]
    for word_index, title in offsets[1:]:
        start = time_at_word(word_index)
        if start < video_duration - 10:
            raw.append({"start_seconds": start, "title": title})
    # YouTube chapter requirements: 00:00 start, at least 3 chapters, each >=10 seconds.
    unique: list[dict[str, Any]] = []
    for chapter in sorted(raw, key=lambda item: item["start_seconds"]):
        if unique and (chapter["start_seconds"] - unique[-1]["start_seconds"] < 10 or chapter["title"].lower() == unique[-1]["title"].lower()):
            continue
        unique.append(chapter)
    if len(unique) < 3 and video_duration >= 35:
        # Use real cue boundaries near thirds rather than text-length interpolation.
        targets = [video_duration / 3, 2 * video_duration / 3]
        for target in targets:
            cue = min(cues, key=lambda item: abs(float(item["start"]) - target))
            candidate = {"start_seconds": float(cue["start"]), "title": "A closer look" if target < video_duration / 2 else "Practical takeaway"}
            if candidate["start_seconds"] >= 10 and candidate["start_seconds"] < video_duration - 10 and all(abs(candidate["start_seconds"] - c["start_seconds"]) >= 10 for c in unique):
                unique.append(candidate)
        unique.sort(key=lambda item: item["start_seconds"])
    def display_time(value: float) -> str:
        minutes, seconds = divmod(max(0, int(value)), 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"
    return [{"timestamp": display_time(chapter["start_seconds"]), "start_seconds": round(chapter["start_seconds"], 3), "title": chapter["title"]} for chapter in unique]


def generate_metadata(model_name: str, topic: str, sections: Sequence[dict[str, str]], script: str,
                      duration: float, video_duration: float, cues: Sequence[dict[str, Any]]) -> dict[str, Any]:
    summary_text = script[:9000]
    prompt = f"""Create truthful YouTube SEO metadata for an English Stickman Finance documentary aimed at US adults.
Topic: {topic!r}
Actual narration duration seconds: {duration:.2f}
Narration (for accuracy; don't invent additional content): {summary_text}
Return JSON keys: title, description, tags (array of strings), hashtags (array of strings), category_id (string; use \"27\" for Education), language (use \"en\"), thumbnail_headline (2–5 simple words, no fake figures or promises), video_slug.
Title max 100 characters; description under 4,000 characters and include a short educational-not-personal-financial-advice note. Tags relevant only, <=450 characters total. No fabricated sources, current facts, guaranteed returns, clickbait or keyword stuffing. Slug must be lowercase ASCII words separated by hyphens. Do not include guessed chapters; the system calculates chapter timestamps after rendering."""
    raw = ollama_json(prompt, model_name, num_predict=650)
    title = str(raw.get("title", "")).strip()[:100]
    description = str(raw.get("description", "")).strip()
    tags = raw.get("tags", [])
    hashtags = raw.get("hashtags", [])
    if not isinstance(tags, list):
        tags = []
    if not isinstance(hashtags, list):
        hashtags = []
    normalized_tags = []
    for tag in tags:
        tag = re.sub(r"\s+", " ", str(tag)).strip()
        if tag and tag.lower() not in {item.lower() for item in normalized_tags}:
            normalized_tags.append(tag[:50])
    while sum(map(len, normalized_tags)) + max(0, len(normalized_tags) - 1) > 450 and normalized_tags:
        normalized_tags.pop()
    clean_hashtags = []
    for tag in hashtags:
        clean = "#" + re.sub(r"[^A-Za-z0-9_]", "", str(tag).lstrip("#"))
        if len(clean) > 1 and clean.lower() not in {item.lower() for item in clean_hashtags}:
            clean_hashtags.append(clean[:60])
    slug = sanitize_slug(str(raw.get("video_slug", "")) or topic)
    chapters = chapters_from_sections(sections, cues, video_duration)
    chapter_lines = "\n".join(f"{item['timestamp']} {item['title']}" for item in chapters)
    if chapter_lines:
        description = description.rstrip() + "\n\nChapters\n" + chapter_lines
    if clean_hashtags:
        description += "\n\n" + " ".join(clean_hashtags[:5])
    if not title:
        title = f"{topic}: The Money Mechanics Explained"[:100]
    if not description:
        description = f"An educational stickman-finance documentary about {topic}. This is general education, not individualized financial advice.\n\nChapters\n{chapter_lines}"
    metadata = {
        "title": title,
        "description": description[:5000],
        "tags": normalized_tags,
        "hashtags": clean_hashtags,
        "chapters": chapters,
        "category_id": str(raw.get("category_id", "27")) if str(raw.get("category_id", "27")).isdigit() else "27",
        "language": "en",
        "thumbnail_headline": str(raw.get("thumbnail_headline", "Money Habits Explained")).strip()[:80],
        "video_slug": slug,
        "actual_duration_seconds": round(duration, 3),
        "rendered_video_duration_seconds": round(video_duration, 3),
        "topic": topic,
        "disclaimer": "General educational information only; not individualized financial, tax, or investment advice.",
        "factuality_note": "The generation prompt forbids unsupported statistics, fabricated citations, current-law specifics and guaranteed outcomes. Review financial claims and hypothetical examples before publishing.",
    }
    validate_metadata(metadata)
    return metadata


def maybe_upload_youtube(video_path: Path, thumbnail_path: Path, metadata: dict[str, Any], manifest: dict[str, Any], manifest_path: Path, mode: str) -> None:
    requested = os.environ.get("UPLOAD_TO_YOUTUBE", "false").strip().lower() in {"1", "true", "yes", "on"}
    test_allowed = os.environ.get("TEST_UPLOAD_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
    if not requested:
        manifest["youtube_upload"] = {"status": "skipped", "reason": "UPLOAD_TO_YOUTUBE is not enabled"}
        atomic_json(manifest_path, manifest)
        LOGGER.info("YouTube upload skipped by configuration. GitHub Release publication remains independent.")
        return
    if mode == "test" and not test_allowed:
        manifest["youtube_upload"] = {"status": "skipped", "reason": "Test uploads require TEST_UPLOAD_ENABLED=true"}
        atomic_json(manifest_path, manifest)
        LOGGER.warning("Test Run YouTube upload was not attempted; set TEST_UPLOAD_ENABLED=true explicitly to allow an unlisted/private test upload.")
        return
    privacy = os.environ.get("YOUTUBE_PRIVACY_STATUS", "unlisted").strip().lower()
    if privacy not in {"private", "unlisted", "public"}:
        manifest["youtube_upload"] = {"status": "failed", "reason": "YOUTUBE_PRIVACY_STATUS must be private, unlisted or public"}
        atomic_json(manifest_path, manifest)
        LOGGER.error("YouTube upload skipped: invalid privacy status. Release publishing will continue.")
        return
    if mode == "test" and privacy == "public":
        manifest["youtube_upload"] = {"status": "failed", "reason": "Test Run forbids public uploads"}
        atomic_json(manifest_path, manifest)
        LOGGER.error("Test Run will not upload publicly. Select private/unlisted for a Test Run; GitHub Release publishing will continue.")
        return
    client_id = os.environ.get("YOUTUBE_CLIENT_ID", "").strip()
    client_secret = os.environ.get("YOUTUBE_CLIENT_SECRET", "").strip()
    refresh_token = os.environ.get("YOUTUBE_REFRESH_TOKEN", "").strip()
    missing = [name for name, value in (("YOUTUBE_CLIENT_ID", client_id), ("YOUTUBE_CLIENT_SECRET", client_secret), ("YOUTUBE_REFRESH_TOKEN", refresh_token)) if not value]
    if missing:
        manifest["youtube_upload"] = {"status": "skipped", "reason": "Missing required GitHub Actions secrets: " + ", ".join(missing)}
        atomic_json(manifest_path, manifest)
        LOGGER.warning("YouTube upload skipped because OAuth secret(s) are missing (%s). The completed video will still be released on GitHub.", ", ".join(missing))
        return
    yt_state = manifest.setdefault("youtube_upload", {})
    video_id = yt_state.get("video_id")
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
        from googleapiclient.errors import HttpError
        from googleapiclient.http import MediaFileUpload
    except ImportError as exc:
        manifest["youtube_upload"] = {"status": "skipped", "reason": "Google API client libraries are unavailable; release fallback will be used."}
        atomic_json(manifest_path, manifest)
        LOGGER.warning("YouTube API library unavailable (%s). GitHub Release is the fallback.", type(exc).__name__)
        return
    try:
        credentials = Credentials(token=None, refresh_token=refresh_token, token_uri="https://oauth2.googleapis.com/token",
                                  client_id=client_id, client_secret=client_secret,
                                  scopes=["https://www.googleapis.com/auth/youtube.upload"])
        credentials.refresh(Request())
        youtube = build("youtube", "v3", credentials=credentials, cache_discovery=False)
        if not video_id:
            body = {
                "snippet": {"title": metadata["title"], "description": metadata["description"], "tags": metadata["tags"],
                            "categoryId": metadata["category_id"], "defaultLanguage": metadata["language"]},
                "status": {"privacyStatus": privacy, "selfDeclaredMadeForKids": False},
            }
            media = MediaFileUpload(str(video_path), mimetype="video/mp4", chunksize=8 * 1024 * 1024, resumable=True)
            request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
            response = None
            attempts = 0
            while response is None:
                try:
                    status, response = request.next_chunk(num_retries=3)
                    if status:
                        LOGGER.info("YouTube upload progress: %.1f%%", status.progress() * 100)
                except HttpError as exc:
                    status_code = getattr(exc.resp, "status", 0)
                    if status_code in {400, 401, 403, 404}:
                        raise
                    attempts += 1
                    if attempts >= 4:
                        raise
                    time.sleep(min(2 ** attempts, 30))
            video_id = response.get("id") if isinstance(response, dict) else None
            if not video_id:
                raise PipelineError("YouTube upload response did not contain a video ID.")
            # Persist ID immediately; future retries only set the thumbnail, not upload again.
            yt_state.update({"status": "video_uploaded", "video_id": video_id, "url": f"https://www.youtube.com/watch?v={video_id}",
                             "privacy_status": privacy, "uploaded_at": dt.datetime.now(dt.timezone.utc).isoformat()})
            atomic_json(manifest_path, manifest)
            LOGGER.info("YouTube acknowledged uploaded video ID %s.", video_id)
        else:
            LOGGER.info("Checkpoint contains YouTube video ID %s; skipping duplicate video upload and retrying thumbnail only.", video_id)
        thumbnail_media = MediaFileUpload(str(thumbnail_path), mimetype="image/jpeg", resumable=False)
        youtube.thumbnails().set(videoId=video_id, media_body=thumbnail_media).execute()
        yt_state.update({"status": "uploaded", "video_id": video_id, "url": f"https://www.youtube.com/watch?v={video_id}",
                         "thumbnail_status": "set", "privacy_status": privacy})
        atomic_json(manifest_path, manifest)
        LOGGER.info("YouTube upload complete: %s", yt_state["url"])
    except Exception as exc:  # noqa: BLE001 - YouTube errors must not destroy completed video/release fallback
        safe_reason = f"{type(exc).__name__}: {str(exc)[:500]}"
        safe_reason = re.sub(r"(?i)(access_token|refresh_token|client_secret|authorization)\s*[:=]\s*[^\s,;&]+", r"\1=[REDACTED]", safe_reason)
        if video_id:
            yt_state.update({"status": "video_uploaded_thumbnail_failed", "video_id": video_id, "url": f"https://www.youtube.com/watch?v={video_id}", "error": safe_reason})
            LOGGER.error("YouTube video upload was acknowledged but thumbnail update failed; saved video ID and will retry only thumbnail later. GitHub Release will still publish the assets.")
        else:
            yt_state.update({"status": "failed", "error": safe_reason})
            LOGGER.error("YouTube upload failed (%s). The validated video will still be published as a GitHub Release.", type(exc).__name__)
        atomic_json(manifest_path, manifest)


def load_or_initialize_manifest(output_dir: Path, manifest_path: Path, run_id: str, mode: str, topic: str) -> dict[str, Any]:
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise PipelineError(f"Checkpoint manifest is unreadable: {manifest_path}. Preserve a copy before manual repair.") from exc
        if manifest.get("run_id") != run_id or manifest.get("mode") != mode or manifest.get("topic") != topic:
            raise PipelineError("Existing output folder belongs to a different run ID, mode or topic; refusing to overwrite it. Choose a new --run-id.")
        return manifest
    output_dir.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(output_dir).free
    return {
        "schema_version": 1,
        "pipeline_version": APP_VERSION,
        "run_id": run_id,
        "mode": mode,
        "topic": topic,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "stages": {},
        "scene_count_target": 12 if mode == "test" else 360,
        "image_interval_seconds": SCENE_SECONDS,
        "subtitle_word_limit": MAX_SUBTITLE_WORDS,
        "kokoro_voice": VOICE_DEFAULT,
        "kokoro_sample_rate_expected": KOKORO_SAMPLE_RATE,
        "disk_free_bytes_at_start": free_bytes,
        "youtube_upload": {"status": "not_attempted"},
    }


def save_script_and_sections(output_dir: Path, manifest: dict[str, Any], manifest_path: Path,
                             topic: str, mode: str, model_name: str) -> tuple[str, list[dict[str, str]]]:
    script_path = output_dir / "script.txt"
    existing_sections = manifest.get("sections")
    existing_script_hash = manifest.get("script_sha256")
    if script_path.is_file() and isinstance(existing_sections, list) and existing_script_hash:
        script = script_path.read_text(encoding="utf-8").strip()
        if sha256_text(script) == existing_script_hash and all(isinstance(item, dict) and item.get("heading") and item.get("narration") for item in existing_sections):
            LOGGER.info("Reusing validated checkpointed narration (%d words).", word_count(script))
            return script, existing_sections
    if mode == "test":
        sections = generate_test_sections(model_name, topic)
    else:
        sections = generate_full_sections(model_name, topic)
    script = plain_script(sections)
    script_path.write_text(script + "\n", encoding="utf-8")
    manifest["sections"] = sections
    manifest["script_sha256"] = sha256_text(script)
    manifest["script_word_count"] = word_count(script)
    manifest["stages"]["script"] = {"status": "complete", "word_count": word_count(script), "sha256": sha256_text(script)}
    atomic_json(manifest_path, manifest)
    LOGGER.info("Script generated locally with %d spoken words across %d chapters.", word_count(script), len(sections))
    return script, sections


def synthesize_or_resume_audio(script: str, sections: Sequence[dict[str, str]], output_dir: Path, manifest: dict[str, Any], manifest_path: Path) -> tuple[float, list[dict[str, Any]], str]:
    import soundfile as sf
    audio_path = output_dir / "audio" / "narration.wav"
    existing = manifest.get("audio", {})
    if audio_path.is_file() and existing.get("script_sha256") == sha256_text(script) and isinstance(manifest.get("subtitle_cues"), list):
        try:
            info = sf.info(str(audio_path))
            duration = float(info.duration)
            cues = manifest["subtitle_cues"]
            validate_cues(cues, duration)
            if info.samplerate == KOKORO_SAMPLE_RATE and duration > 0:
                LOGGER.info("Reusing validated audio checkpoint (%.1fs, %d cues).", duration, len(cues))
                return duration, cues, str(existing.get("timing_method", "checkpoint"))
        except Exception as exc:  # noqa: BLE001 - invalid checkpoint gets regenerated
            LOGGER.warning("Audio checkpoint failed validation (%s); regenerating narration.", type(exc).__name__)
    model_path = output_dir / "models" / "kokoro-v1.0.onnx"
    voices_path = output_dir / "models" / "voices-v1.0.bin"
    cache_dir = Path(os.environ.get("MODEL_DIR", str(Path.home() / ".cache" / "stickman-finance-models"))).expanduser()
    model_path = cache_dir / model_path.name
    voices_path = cache_dir / voices_path.name
    ensure_kokoro_files(model_path, voices_path)
    tts = load_kokoro(model_path, voices_path)
    try:
        # Test selected voice and the real model/sample rate before doing long-form narration.
        sample_audio, sample_rate = tts.create("This is a short local voice check.", voice=VOICE_DEFAULT, speed=1.0, lang="en-us")
        if int(sample_rate) != KOKORO_SAMPLE_RATE or len(sample_audio) < 1000:
            raise PipelineError(f"Kokoro preflight sample was invalid: {sample_rate} Hz or too few samples.")
        duration, cues, timing_method = synthesize_with_kokoro(script, sections, tts, audio_path)
    finally:
        try:
            tts.voices.close()
        except Exception:
            pass
        del tts
    manifest["subtitle_cues"] = cues
    manifest["audio"] = {"status": "complete", "path": "audio/narration.wav", "duration_seconds": round(duration, 4),
                         "sample_rate": KOKORO_SAMPLE_RATE, "script_sha256": sha256_text(script),
                         "sha256": sha256_file(audio_path), "timing_method": timing_method}
    manifest["stages"]["narration"] = {"status": "complete", "duration_seconds": round(duration, 4), "timing_method": timing_method}
    atomic_json(manifest_path, manifest)
    return duration, cues, timing_method


def adjust_script_duration_if_needed(script: str, sections: list[dict[str, str]], topic: str, mode: str,
                                     model_name: str, duration: float, manifest: dict[str, Any], manifest_path: Path) -> tuple[str, list[dict[str, str]]]:
    if mode == "test":
        if duration > 60.05:
            raise PipelineError(f"Test narration measured {duration:.1f}s, exceeding its 60s / 12-scene timeline. Refusing to cut off speech; use a new run ID so the local model can regenerate a shorter script.")
        if duration < 38:
            raise PipelineError(f"Test narration remained unusually short at {duration:.1f}s after two bounded revisions. Use a new run ID or inspect the local Ollama script-generation logs.")
        if duration < 45:
            LOGGER.warning("Test narration is %.1fs; the 60-second video still has exactly 12 five-second scenes.", duration)
        return script, sections
    minimum = float(os.environ.get("FULL_MIN_NARRATION_SECONDS", "1500"))
    maximum = float(os.environ.get("FULL_MAX_NARRATION_SECONDS", "1980"))
    if minimum <= duration <= maximum:
        return script, sections
    raise PipelineError(
        f"Measured narration duration is {duration / 60:.1f} minutes, outside the configured full-run window of {minimum/60:.0f}–{maximum/60:.0f} minutes. "
        "No speech was cut or time-stretched. Adjust the generated chapter length/pacing or set FULL_MIN_NARRATION_SECONDS/FULL_MAX_NARRATION_SECONDS deliberately, then resume with a new run ID if script changes."
    )


def write_validation_report(output_dir: Path, manifest: dict[str, Any], manifest_path: Path, status: str, error: str | None = None) -> None:
    report = {
        "status": status,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_id": manifest.get("run_id"),
        "mode": manifest.get("mode"),
        "topic": manifest.get("topic"),
        "scene_count_target": manifest.get("scene_count_target"),
        "scene_count_completed": sum(scene.get("validation_status") == "generated" for scene in manifest.get("scenes", []) if isinstance(scene, dict)),
        "audio": manifest.get("audio"),
        "render": manifest.get("render"),
        "youtube_upload": manifest.get("youtube_upload"),
        "error": error,
    }
    atomic_json(output_dir / "validation_report.json", report)
    manifest["validation_report"] = "validation_report.json"
    if status != "complete":
        manifest.setdefault("stages", {})["pipeline"] = {"status": status, "error": error}
    atomic_json(manifest_path, manifest)


def run_preflight(mode: str, output_dir: Path, model_name: str, image_model: str, model_dir: Path) -> dict[str, Any]:
    if sys.version_info < (3, 11):
        raise PipelineError(f"Python 3.11+ is required; this interpreter is {platform.python_version()}.")
    if mode not in {"test", "full"}:
        raise PipelineError("Mode must be test or full.")
    for directory in (output_dir, output_dir / "logs", output_dir / "audio", output_dir / "scenes" / "original", output_dir / "scenes" / "processed"):
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / ".write-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    free_gb = shutil.disk_usage(output_dir).free / (1024**3)
    if mode == "full" and free_gb < float(os.environ.get("MIN_FREE_DISK_GB_FULL", "12")):
        raise PipelineError(f"Full Run requires at least {os.environ.get('MIN_FREE_DISK_GB_FULL', '12')} GB free as a conservative budget for model cache/output; currently {free_gb:.1f} GB are free. Clear disk or use a larger runner.")
    if mode == "test" and free_gb < 8:
        raise PipelineError(f"Test Run needs a conservative 8 GB free for model weights and intermediates; currently {free_gb:.1f} GB are free.")
    ffmpeg_info = validate_render_tools()
    ensure_ollama(model_name)
    LOGGER.info("Ollama health and model availability passed for %s.", model_name)
    import requests as _requests
    _requests.get(ollama_base_url() + "/api/tags", timeout=10).raise_for_status()
    # Preflight also confirms local text-model inference, not merely an HTTP health endpoint.
    _ = ollama_generate('Return only JSON: {"ok": true, "purpose": "local inference health check"}', model_name, num_predict=40, timeout=180)
    # Load SDXL and render a disposable local 512x512 sample before writing a large scene plan.
    # This intentionally catches missing weights, insufficient RAM/VRAM, scheduler issues and broken Diffusers installs early.
    sample_pipe, sample_torch, sample_device = load_diffusion_pipeline(image_model)
    try:
        sample_image = run_image_generation(sample_pipe, sample_torch, sample_device,
                                             finance_style_prompt("Preflight validation: simple editorial stickman beside a savings jar, flat 2D illustration, no text"),
                                             seed=24681357, steps=1)
        if sample_image.size != (512, 512) or sample_image.mode != "RGB":
            raise PipelineError(f"SDXL-Turbo preflight image invalid: {sample_image.mode} {sample_image.size}.")
        from PIL import Image
        sample_path = output_dir / ".preflight-image.jpg"
        image_to_video_frame(sample_image, sample_path)
        with Image.open(sample_path) as check:
            check.verify()
        sample_path.unlink(missing_ok=True)
    finally:
        del sample_pipe
        try:
            if sample_torch.cuda.is_available():
                sample_torch.cuda.empty_cache()
        except Exception:
            pass
    return {"python": platform.python_version(), "ffmpeg": ffmpeg_info, "ollama_model": model_name, "image_model": image_model,
            "free_disk_gb": round(free_gb, 2), "model_cache": str(model_dir), "local_text_inference": "passed",
            "sdxl_turbo_sample": "passed", "sdxl_device": sample_device}


def save_updated_script(output_dir: Path, sections: list[dict[str, str]], manifest: dict[str, Any], manifest_path: Path) -> str:
    script = plain_script(sections)
    (output_dir / "script.txt").write_text(script + "\n", encoding="utf-8")
    manifest["sections"] = sections
    manifest["script_sha256"] = sha256_text(script)
    manifest["script_word_count"] = word_count(script)
    manifest["subtitle_cues"] = None
    manifest["audio"] = {}
    manifest.pop("scenes", None)
    manifest.pop("metadata_input_sha256", None)
    manifest.pop("metadata_sha256", None)
    manifest["stages"]["script"] = {"status": "complete", "word_count": word_count(script), "sha256": sha256_text(script), "revision": int(manifest.get("script_revision", 0)) + 1}
    manifest["script_revision"] = int(manifest.get("script_revision", 0)) + 1
    # Previous audio is no longer a valid checkpoint because it was synthesized from another script.
    (output_dir / "audio" / "narration.wav").unlink(missing_ok=True)
    (output_dir / "subtitles.ass").unlink(missing_ok=True)
    (output_dir / "captions.srt").unlink(missing_ok=True)
    (output_dir / "video.mp4").unlink(missing_ok=True)
    manifest.pop("render", None)
    atomic_json(manifest_path, manifest)
    return script


def revise_script_for_duration(model_name: str, mode: str, topic: str, sections: list[dict[str, str]],
                               duration: float) -> list[dict[str, str]]:
    if mode == "test":
        desired = "90–105 spoken words" if duration > 60 else "115–130 spoken words"
        prompt = f"Rewrite this short English voiceover about {topic!r} so Kokoro's audio target is about 45–58 seconds. Target {desired}; keep one hook, one clear explanatory point, one cautious/hypothetical example at most and a natural close. Do not add statistics, invented sources, current tax/rate facts or investment guarantees. Keep it natural; no heading in spoken text. Return JSON {{\"heading\": str, \"narration\": str}}. Current narration: {sections[0]['narration'][:5000]}"
        revised = ollama_json(prompt, model_name, num_predict=280)
        narration = str(revised.get("narration", "")).strip()
        if not 70 <= word_count(narration) <= 145:
            raise PipelineError("The bounded Test Run script revision returned an implausible word count.")
        return [{"heading": str(revised.get("heading", sections[0]["heading"])).strip(), "narration": narration}]
    if duration < float(os.environ.get("FULL_MIN_NARRATION_SECONDS", "1500")):
        headings = ", ".join(section["heading"] for section in sections)
        result = ollama_json(
            f"Add a new distinct, non-repetitive 450–550 word chapter to the educational Stickman Finance documentary about {topic!r}. It must extend, not recap, these existing chapter headings: {headings}. Focus on a useful hypothetical scenario, trade-offs and counterarguments. No unsupported statistics, fabricated citations, current tax/regulatory specifics or guaranteed outcomes. Return JSON {{\"heading\": str, \"narration\": str}}.",
            model_name, num_predict=1150)
        narration = str(result.get("narration", "")).strip()
        if word_count(narration) < 300:
            raise PipelineError("Local model could not add a sufficiently substantial chapter to reach the full duration target.")
        return [*sections, {"heading": str(result.get("heading", "A practical scenario")).strip(), "narration": narration}]
    revised_sections: list[dict[str, str]] = []
    for section in sections:
        result = ollama_json(
            f"Edit this documentary chapter to about 450–500 spoken words, preserving its core explanation, examples, important caveats and chapter focus. Remove redundancy, do not delete important reasoning, do not invent claims or sources. Return JSON {{\"narration\": str}}. Chapter heading (not spoken): {section['heading']}. Text: {section['narration'][:7000]}",
            model_name, num_predict=1050)
        narration = str(result.get("narration", "")).strip()
        if word_count(narration) < 350:
            raise PipelineError("Local model's duration correction cut a chapter too short; refusing to discard the original silently.")
        revised_sections.append({"heading": section["heading"], "narration": narration})
    return revised_sections


def run_pipeline(args: argparse.Namespace) -> Path:
    mode = args.mode
    model_name = os.environ.get("OLLAMA_MODEL", OLLAMA_MODEL_DEFAULT).strip() or OLLAMA_MODEL_DEFAULT
    image_model = os.environ.get("IMAGE_MODEL", IMAGE_MODEL_DEFAULT).strip() or IMAGE_MODEL_DEFAULT
    model_dir = Path(os.environ.get("MODEL_DIR", str(Path.home() / ".cache" / "stickman-finance-models"))).expanduser()
    run_id = args.run_id or os.environ.get("PIPELINE_RUN_ID") or os.environ.get("GITHUB_RUN_ID") or dt.datetime.now().strftime("local-%Y%m%d-%H%M%S")
    topic = args.topic or os.environ.get("VIDEO_TOPIC", "automatic")
    if len(topic) > 180:
        raise PipelineError("Topic must be 180 characters or fewer.")
    output_root = Path(args.output_root).expanduser().resolve()
    setup_logging(None, os.environ.get("LOG_LEVEL", "INFO"))
    # On a workflow retry, resolve the prior generated topic before selecting a new automatic topic.
    cached_topic = None
    if topic.strip().lower() in {"auto", "automatic", "random"} and output_root.exists():
        suffix = "-" + sanitize_slug(run_id)[:40]
        for candidate in output_root.glob("*" + suffix):
            candidate_manifest = candidate / "manifest.json"
            if candidate_manifest.is_file():
                try:
                    candidate_data = json.loads(candidate_manifest.read_text(encoding="utf-8"))
                    if candidate_data.get("run_id") == run_id and candidate_data.get("mode") == mode:
                        cached_topic = str(candidate_data.get("topic", "")).strip()
                        if cached_topic:
                            break
                except (OSError, json.JSONDecodeError):
                    continue
    if cached_topic:
        effective_topic = cached_topic
    else:
        ensure_ollama(model_name)
        effective_topic = choose_topic(model_name, topic)
    slug = sanitize_slug(effective_topic)
    output_dir = output_root / f"{slug}-{sanitize_slug(run_id)[:40]}"
    manifest_path = output_dir / "manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir, os.environ.get("LOG_LEVEL", "INFO"))
    manifest = load_or_initialize_manifest(output_dir, manifest_path, run_id, mode, effective_topic)
    manifest["topic"] = effective_topic
    manifest["topic_slug"] = slug
    manifest["stage_started_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    atomic_json(manifest_path, manifest)
    started = time.monotonic()
    try:
        preflight = run_preflight(mode, output_dir, model_name, image_model, model_dir)
        manifest["preflight"] = preflight
        manifest["stages"]["preflight"] = {"status": "complete", "details": preflight}
        atomic_json(manifest_path, manifest)

        script, sections = save_script_and_sections(output_dir, manifest, manifest_path, effective_topic, mode, model_name)
        audio_duration, cues, timing_method = synthesize_or_resume_audio(script, sections, output_dir, manifest, manifest_path)

        # Measure actual Kokoro audio and make at most two bounded script revisions. Never cut speech or stretch audio.
        for revision_attempt in range(3):
            full_min = float(os.environ.get("FULL_MIN_NARRATION_SECONDS", "1500"))
            full_max = float(os.environ.get("FULL_MAX_NARRATION_SECONDS", "1980"))
            out_of_range = ((mode == "test" and not 45.0 <= audio_duration <= 60.05)
                            or (mode == "full" and not full_min <= audio_duration <= full_max))
            if not out_of_range or revision_attempt >= 2:
                break
            LOGGER.warning("Measured narration duration %.1fs is outside the target; attempting bounded local script revision %d/2.", audio_duration, revision_attempt + 1)
            sections = revise_script_for_duration(model_name, mode, effective_topic, sections, audio_duration)
            script = save_updated_script(output_dir, sections, manifest, manifest_path)
            audio_duration, cues, timing_method = synthesize_or_resume_audio(script, sections, output_dir, manifest, manifest_path)

        script, sections = adjust_script_duration_if_needed(script, sections, effective_topic, mode, model_name,
                                                            audio_duration, manifest, manifest_path)
        video_duration = 60.0 if mode == "test" else math.ceil(audio_duration / SCENE_SECONDS) * SCENE_SECONDS
        scene_count = 12 if mode == "test" else math.ceil(audio_duration / SCENE_SECONDS)
        if mode == "full" and scene_count != 360:
            LOGGER.warning("Measured narration needs %d scenes at five seconds each, differing from the 360-scene target by %+d. No speech is cut.", scene_count, scene_count - 360)
        if audio_duration > video_duration + 0.03:
            raise PipelineError(f"Audio duration {audio_duration:.2f}s exceeds the {video_duration:.2f}s timeline; refusing to truncate speech.")
        manifest["actual_narration_duration_seconds"] = round(audio_duration, 3)
        manifest["video_duration_target_seconds"] = video_duration
        manifest["scene_count_target"] = scene_count
        manifest["timing_method"] = timing_method

        metadata_path = output_dir / "metadata.json"
        metadata_key = sha256_text(json.dumps({"script": sha256_text(script), "duration": round(audio_duration, 3),
                                                "video_duration": video_duration, "topic": effective_topic}, sort_keys=True))
        if (metadata_path.is_file() and manifest.get("metadata_input_sha256") == metadata_key):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                validate_metadata(metadata)
                LOGGER.info("Reusing validated metadata checkpoint.")
            except Exception:
                metadata = generate_metadata(model_name, effective_topic, sections, script, audio_duration, video_duration, cues)
        else:
            metadata = generate_metadata(model_name, effective_topic, sections, script, audio_duration, video_duration, cues)
        metadata["actual_duration_seconds"] = round(audio_duration, 3)
        metadata["rendered_video_duration_seconds"] = round(video_duration, 3)
        metadata["chapters"] = chapters_from_sections(sections, cues, video_duration)
        chapter_lines = "\n".join(f"{chapter['timestamp']} {chapter['title']}" for chapter in metadata["chapters"])
        desc = str(metadata.get("description", "")).split("\n\nChapters\n", 1)[0]
        hashtags_line = " ".join(metadata.get("hashtags", [])[:5])
        metadata["description"] = (desc + ("\n\nChapters\n" + chapter_lines if chapter_lines else "")
                                   + ("\n\n" + hashtags_line if hashtags_line else ""))[:5000]
        validate_metadata(metadata)
        atomic_json(metadata_path, metadata)
        manifest["metadata_input_sha256"] = metadata_key
        manifest["metadata_sha256"] = sha256_file(metadata_path)
        manifest["stages"]["metadata"] = {"status": "complete", "path": "metadata.json"}
        atomic_json(manifest_path, manifest)

        scene_cache_is_valid = isinstance(manifest.get("scenes"), list) and len(manifest["scenes"]) == scene_count
        if scene_cache_is_valid:
            try:
                validate_scene_plan(manifest["scenes"], scene_count)
                for scene in manifest["scenes"]:
                    if not scene.get("image_prompt"):
                        raise PipelineError("cached scene lacks a prompt")
            except Exception:
                scene_cache_is_valid = False
        if scene_cache_is_valid:
            scenes = manifest["scenes"]
            LOGGER.info("Reusing validated checkpointed scene plan with %d scenes.", len(scenes))
        else:
            LOGGER.info("Planning every required image prompt dynamically: %d scenes (batch size 12; no maximum scene cap).", scene_count)
            scenes = generate_scene_prompts(model_name, effective_topic, cues, scene_count)
            manifest["scenes"] = scenes
            manifest["stages"]["scene_plan"] = {"status": "complete", "scene_count": len(scenes)}
            atomic_json(manifest_path, manifest)
        validate_scene_plan(scenes, scene_count)

        thumb_path = output_dir / "thumbnail.jpg"
        image_seconds, device = generate_scene_images(
            scenes, output_dir, manifest, manifest_path, image_model, bool(args.allow_slow_cpu_full), effective_topic,
            thumb_path, metadata["thumbnail_headline"])
        manifest["stages"]["images"] = {"status": "complete", "scene_count": len(scenes), "generation_wall_seconds": round(image_seconds, 3), "device": device}
        manifest["scene_count_generated"] = sum(scene.get("validation_status") == "generated" for scene in scenes)
        if not thumb_path.is_file() or manifest.get("thumbnail", {}).get("sha256") != sha256_file(thumb_path):
            raise PipelineError("Thumbnail generation did not produce a validated, topic-specific JPEG.")
        manifest["stages"]["thumbnail"] = {"status": "complete", "path": "thumbnail.jpg"}
        atomic_json(manifest_path, manifest)

        ass_path, srt_path = write_subtitles(cues, output_dir, metadata["title"], audio_duration)
        manifest["stages"]["subtitles"] = {"status": "complete", "cue_count": len(cues), "word_limit": MAX_SUBTITLE_WORDS}
        atomic_json(manifest_path, manifest)

        video_path = output_dir / "video.mp4"
        render_input_hash = sha256_text(json.dumps({
            "scene_hashes": [scene.get("processed_sha256", "") for scene in scenes],
            "audio_sha256": sha256_file(output_dir / "audio" / "narration.wav"),
            "ass_sha256": sha256_file(ass_path), "crf": os.environ.get("FFMPEG_CRF", "27"),
            "preset": os.environ.get("FFMPEG_PRESET", "veryfast"), "duration": video_duration,
        }, sort_keys=True))
        cached_render_valid = (video_path.is_file() and manifest.get("render", {}).get("sha256") == sha256_file(video_path)
                               and manifest.get("render", {}).get("inputs_sha256") == render_input_hash)
        if cached_render_valid:
            try:
                render_info = probe_media(video_path)
                if abs(render_info["duration_seconds"] - video_duration) > 0.35:
                    cached_render_valid = False
            except Exception:
                cached_render_valid = False
        if not cached_render_valid:
            render_info = render_video(scenes, output_dir / "audio" / "narration.wav", ass_path, output_dir, audio_duration)
            manifest["render"] = {**render_info, "path": "video.mp4", "sha256": sha256_file(video_path), "inputs_sha256": render_input_hash, "validated_at": dt.datetime.now(dt.timezone.utc).isoformat()}
        else:
            LOGGER.info("Reusing validated MP4 checkpoint.")
        manifest["stages"]["render"] = {"status": "complete", "duration_seconds": manifest["render"]["duration_seconds"], "path": "video.mp4"}
        # Rebuild chapter timestamps against cue boundaries and the duration measured from the real MP4.
        measured_video_duration = float(manifest["render"]["duration_seconds"])
        metadata["rendered_video_duration_seconds"] = round(measured_video_duration, 3)
        metadata["chapters"] = chapters_from_sections(sections, cues, measured_video_duration)
        chapter_lines = "\n".join(f"{chapter['timestamp']} {chapter['title']}" for chapter in metadata["chapters"])
        description_without_chapters = str(metadata.get("description", "")).split("\n\nChapters\n", 1)[0]
        hashtags_line = " ".join(metadata.get("hashtags", [])[:5])
        metadata["description"] = (description_without_chapters
                                   + ("\n\nChapters\n" + chapter_lines if chapter_lines else "")
                                   + ("\n\n" + hashtags_line if hashtags_line else ""))[:5000]
        validate_metadata(metadata)
        atomic_json(output_dir / "metadata.json", metadata)
        manifest["metadata_sha256"] = sha256_file(output_dir / "metadata.json")
        manifest["final_files"] = {name: {"path": name, "sha256": sha256_file(output_dir / name), "size_bytes": (output_dir / name).stat().st_size}
                                    for name in ("video.mp4", "thumbnail.jpg", "metadata.json", "script.txt", "subtitles.ass", "captions.srt")}
        atomic_json(manifest_path, manifest)

        maybe_upload_youtube(video_path, thumb_path, metadata, manifest, manifest_path, mode)
        # Publish status is set by the workflow after `gh release create/upload` succeeds.
        manifest["stages"]["local_pipeline"] = {"status": "complete", "elapsed_seconds": round(time.monotonic() - started, 2)}
        manifest["rendered_video_duration_seconds"] = manifest["render"]["duration_seconds"]
        manifest["completed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        atomic_json(manifest_path, manifest)
        write_validation_report(output_dir, manifest, manifest_path, "complete")
        LOGGER.info("PIPELINE COMPLETE: %s", video_path)
        LOGGER.info("Video: %.2f min; scenes: %d; subtitles: %d cues; local pipeline wall time: %.1f min.",
                    manifest["render"]["duration_seconds"] / 60, len(scenes), len(cues), (time.monotonic() - started) / 60)
        LOGGER.info("GitHub Release is performed by .github/workflows/finance-video.yml; Test Run release is a prerelease.")
        return output_dir
    except Exception as exc:
        reason = f"{type(exc).__name__}: {str(exc)[:1000]}"
        LOGGER.error("PIPELINE FAILED: %s", reason)
        LOGGER.debug("Traceback follows:\n%s", traceback.format_exc())
        write_validation_report(output_dir, manifest, manifest_path, "failed", reason)
        raise


def _self_test() -> int:
    import unittest

    class PipelineUnitTests(unittest.TestCase):
        def test_slug_safety(self) -> None:
            self.assertEqual(sanitize_slug("The Hidden Cost: Credit Cards!"), "the-hidden-cost-credit-cards")
            self.assertNotIn("/", sanitize_slug("../../evil; rm -rf"))

        def test_word_chunks_max_four(self) -> None:
            text = "Small habits can compound over time and change financial outcomes."
            chunks = chunks_of_words(text)
            self.assertEqual([visible_word_count(chunk) for chunk in chunks], [4, 4, 2])
            self.assertTrue(all(visible_word_count(chunk) <= 4 for chunk in chunks))
            with self.assertRaises(ValueError):
                chunks_of_words(text, 5)

        def test_test_scene_count_is_12_and_full_is_360(self) -> None:
            self.assertEqual(60 // SCENE_SECONDS, 12)
            self.assertEqual(1800 // SCENE_SECONDS, 360)
            # Planner/validator deliberately accepts beyond 15 scenes and has no fixed upper bound.
            scenes = [{"scene_id": f"s-{i}", "scene_number": i, "image_prompt": f"prompt {i}", "narration": f"line {i}",
                       "start_seconds": (i - 1) * 5, "end_seconds": i * 5} for i in range(1, 128)]
            validate_scene_plan(scenes, 127)
            self.assertEqual(len(scenes), 127)

        def test_full_plan_can_represent_360_scenes(self) -> None:
            scenes = [{"scene_id": f"s-{i}", "scene_number": i, "image_prompt": f"prompt {i}", "narration": f"line {i}",
                       "start_seconds": (i - 1) * 5, "end_seconds": i * 5} for i in range(1, 361)]
            validate_scene_plan(scenes, 360)

        def test_subtitle_timestamps_reject_overrun(self) -> None:
            valid = [{"text": "BUILD WEALTH SLOWLY", "start": 0.0, "end": 1.2, "word_start_index": 0, "word_end_index": 3}]
            validate_cues(valid, 2.0)
            with self.assertRaises(PipelineError):
                validate_cues([{**valid[0], "text": "ONE TWO THREE FOUR FIVE"}], 2.0)
            with self.assertRaises(PipelineError):
                validate_cues([{**valid[0], "end": 2.1}], 2.0)

        def test_metadata_limits(self) -> None:
            metadata = {"title": "Finance basics", "description": "Educational only.", "tags": ["finance", "budget"],
                        "hashtags": ["#finance"], "category_id": "27", "language": "en"}
            validate_metadata(metadata)
            with self.assertRaises(PipelineError):
                validate_metadata({**metadata, "title": "x" * 101})

        def test_retry_is_bounded(self) -> None:
            calls = {"count": 0}
            def broken() -> None:
                calls["count"] += 1
                raise ValueError("no")
            with self.assertRaises(ValueError):
                retry_call(broken, attempts=3, initial_delay=0, sleep=lambda _: None)
            self.assertEqual(calls["count"], 3)

        def test_manifest_resume_conflict(self) -> None:
            with tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp)
                manifest_path = base / "manifest.json"
                manifest_path.write_text(json.dumps({"run_id": "one", "mode": "test", "topic": "A"}), encoding="utf-8")
                manifest = load_or_initialize_manifest(base, manifest_path, "one", "test", "A")
                self.assertEqual(manifest["run_id"], "one")
                with self.assertRaises(PipelineError):
                    load_or_initialize_manifest(base, manifest_path, "two", "test", "A")

        def test_missing_youtube_credentials_keep_local_bundle_and_skip_upload(self) -> None:
            from unittest.mock import patch
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                manifest_path = root / "manifest.json"
                manifest = {"youtube_upload": {"status": "not_attempted"}}
                with patch.dict(os.environ, {"UPLOAD_TO_YOUTUBE": "true", "YOUTUBE_PRIVACY_STATUS": "unlisted"}, clear=False):
                    for key in ("YOUTUBE_CLIENT_ID", "YOUTUBE_CLIENT_SECRET", "YOUTUBE_REFRESH_TOKEN"):
                        os.environ.pop(key, None)
                    maybe_upload_youtube(root / "video.mp4", root / "thumbnail.jpg", {"title": "Safe test", "description": "x", "tags": [], "category_id": "27", "language": "en"}, manifest, manifest_path, "full")
                self.assertEqual(manifest["youtube_upload"]["status"], "skipped")
                self.assertTrue(manifest_path.is_file())

        def test_tiny_corrupt_video_is_rejected_before_publish(self) -> None:
            with tempfile.TemporaryDirectory() as tmp:
                bad_video = Path(tmp) / "bad.mp4"
                bad_video.write_bytes(b"not a video")
                with self.assertRaises(PipelineError):
                    probe_media(bad_video)

        def test_chapter_timestamps_use_measured_cue_boundaries(self) -> None:
            sections = [{"heading": "Intro", "narration": "one two three four"},
                        {"heading": "Example", "narration": "five six seven eight"},
                        {"heading": "Wrap up", "narration": "nine ten eleven twelve"}]
            cues = [
                {"text": "one two three four", "start": 0, "end": 2, "word_start_index": 0, "word_end_index": 4},
                {"text": "five six seven eight", "start": 20, "end": 22, "word_start_index": 4, "word_end_index": 8},
                {"text": "nine ten eleven twelve", "start": 40, "end": 42, "word_start_index": 8, "word_end_index": 12},
            ]
            chapters = chapters_from_sections(sections, cues, 60)
            self.assertEqual(chapters[0]["timestamp"], "00:00")
            self.assertTrue(any(ch["timestamp"] == "00:20" for ch in chapters))

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PipelineUnitTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def validate_existing_bundle(output_dir: Path, expected_mode: str) -> int:
    """Validate a bundle restored from a failed-release workflow artifact for release-only retry."""
    setup_logging(output_dir, os.environ.get("LOG_LEVEL", "INFO"))
    manifest_path = output_dir / "manifest.json"
    required = ["video.mp4", "thumbnail.jpg", "metadata.json", "script.txt", "subtitles.ass", "captions.srt", "validation_report.json"]
    missing = [name for name in required if not (output_dir / name).is_file() or (output_dir / name).stat().st_size == 0]
    if missing:
        raise PipelineError("Cannot resume release-only publication; artifact is missing: " + ", ".join(missing))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("mode") != expected_mode:
        raise PipelineError(f"Restored release bundle mode is {manifest.get('mode')!r}, but workflow selected {expected_mode!r}.")
    report = json.loads((output_dir / "validation_report.json").read_text(encoding="utf-8"))
    if report.get("status") != "complete":
        raise PipelineError("Restored artifact does not contain a completed validation report.")
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    validate_metadata(metadata)
    video_path = output_dir / "video.mp4"
    render = probe_media(video_path)
    if manifest.get("render", {}).get("sha256") and sha256_file(video_path) != manifest["render"]["sha256"]:
        raise PipelineError("Restored MP4 hash does not match its manifest; refusing to publish.")
    from PIL import Image
    with Image.open(output_dir / "thumbnail.jpg") as thumb:
        thumb.verify()
    with Image.open(output_dir / "thumbnail.jpg") as thumb:
        if thumb.size != (1280, 720) or thumb.format != "JPEG":
            raise PipelineError("Restored thumbnail is not a 1280x720 JPEG.")
    audio_duration = float(manifest.get("audio", {}).get("duration_seconds", 0))
    cues = manifest.get("subtitle_cues", [])
    if audio_duration <= 0 or not cues:
        raise PipelineError("Restored manifest is missing audio timing/caption checkpoints.")
    validate_subtitle_files(cues, output_dir / "subtitles.ass", output_dir / "captions.srt", audio_duration)
    run_cmd([ffmpeg_binary(), "-hide_banner", "-v", "error", "-xerror", "-i", str(video_path), "-f", "null", "-"], timeout=int(os.environ.get("FFMPEG_DECODE_TIMEOUT_SECONDS", "18000")))
    manifest["release_retry_validation"] = {"status": "passed", "validated_at": dt.datetime.now(dt.timezone.utc).isoformat(), "video": render}
    atomic_json(manifest_path, manifest)
    upload_requested = os.environ.get("UPLOAD_TO_YOUTUBE", "false").strip().lower() in {"1", "true", "yes", "on"}
    yt_state = manifest.get("youtube_upload", {}) if isinstance(manifest.get("youtube_upload", {}), dict) else {}
    if upload_requested and yt_state.get("status") == "video_uploaded_thumbnail_failed":
        # `maybe_upload_youtube` sees the stored video ID and retries thumbnail.set only.
        LOGGER.info("Retrying YouTube thumbnail only for the persisted video ID; duplicate video upload is disabled.")
        maybe_upload_youtube(video_path, output_dir / "thumbnail.jpg", metadata, manifest, manifest_path, expected_mode)
    LOGGER.info("Restored bundle passed validation; GitHub Release retry is safe. Video %.2f minutes.", render["duration_seconds"] / 60)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local-only Stickman Finance video pipeline; exactly these two source files are required.")
    parser.add_argument("--mode", choices=("test", "full"), default=os.environ.get("PIPELINE_MODE", "test"),
                        help="test = 60-second video with exactly 12 planned five-second scenes; full = approximately 30 minutes")
    parser.add_argument("--topic", default=os.environ.get("VIDEO_TOPIC", "automatic"), help="Specific topic, or automatic")
    parser.add_argument("--run-id", default=None, help="Stable identifier used for checkpoint/resume; defaults to workflow run ID/local timestamp")
    parser.add_argument("--output-root", default=os.environ.get("OUTPUT_ROOT", "output"), help="Output root directory (default: output)")
    parser.add_argument("--allow-slow-cpu-full", action="store_true", help="Accept a potentially >5-hour CPU-only full image-generation estimate")
    parser.add_argument("--self-test", action="store_true", help="Run dependency-light built-in unit tests and exit")
    parser.add_argument("--preflight-only", action="store_true", help="Check local services/tools and exit; does not generate a video")
    parser.add_argument("--help-files", action="store_true", help="Print the setup checklist and exit")
    parser.add_argument("--validate-existing", default=None, help="Validate a previously rendered release bundle without downloading/running AI models")
    return parser


def print_setup() -> None:
    print("""STICKMAN FINANCE PIPELINE — ONLY TWO REPOSITORY SOURCE FILES
  pipeline.py
  .github/workflows/finance-video.yml

GitHub Actions installs the pinned runtime packages and OS dependencies; no requirements.txt or README is required.
Required local services/models: Ollama + gemma2:2b; local Diffusers SDXL-Turbo weights; Kokoro ONNX model-files-v1.1.
Kokoro files are cached under ~/.cache/stickman-finance-models by default (override MODEL_DIR).
Workflow secrets (optional YouTube upload): YOUTUBE_CLIENT_ID, YOUTUBE_CLIENT_SECRET, YOUTUBE_REFRESH_TOKEN.
GITHUB_TOKEN is provided by Actions and is used for GitHub Release creation; never hardcode it.
Run `python pipeline.py --self-test`, then `python pipeline.py --mode test --topic "The hidden cost of lifestyle inflation"`.
Use the repository Actions tab -> Run workflow -> TEST RUN — 1 minute first. The release is marked prerelease.
Full Run: select FULL RUN — approximately 30 minutes. Free hosted CPU runners are unlikely to generate 360 SDXL-Turbo images within a reasonable budget.
For release fallback, YouTube upload failures/missing OAuth only update the manifest; the workflow still releases a validated video.
License caveat: review Stability AI's current SDXL-Turbo license before commercial/monetized use.
""")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        return _self_test()
    if args.help_files:
        print_setup()
        return 0
    try:
        if args.validate_existing:
            return validate_existing_bundle(Path(args.validate_existing).expanduser().resolve(), args.mode)
        output_root = Path(args.output_root).expanduser().resolve()
        if args.preflight_only:
            setup_logging(None, os.environ.get("LOG_LEVEL", "INFO"))
            model_name = os.environ.get("OLLAMA_MODEL", OLLAMA_MODEL_DEFAULT)
            image_model = os.environ.get("IMAGE_MODEL", IMAGE_MODEL_DEFAULT)
            model_dir = Path(os.environ.get("MODEL_DIR", str(Path.home() / ".cache" / "stickman-finance-models"))).expanduser()
            details = run_preflight(args.mode, output_root / "preflight", model_name, image_model, model_dir)
            print(json.dumps(details, indent=2))
            return 0
        output_dir = run_pipeline(args)
        print(f"OUTPUT_DIR={output_dir}")
        return 0
    except KeyboardInterrupt:
        LOGGER.error("Interrupted by user; checkpoints remain on disk.")
        return 130
    except Exception as exc:
        if not LOGGER.handlers:
            setup_logging(None, os.environ.get("LOG_LEVEL", "INFO"))
        LOGGER.error("Fatal: %s", exc)
        LOGGER.debug("Traceback:\n%s", traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
