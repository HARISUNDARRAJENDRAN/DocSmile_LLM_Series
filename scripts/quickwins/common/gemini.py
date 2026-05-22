"""Multi-key, rate-limited Gemini client for batch topic-tagging & dialogue cleanup.

Quotas per key (gemini-3.1-flash-lite free tier as of 2026-05):
  15 RPM, 250k TPM, 500 RPD.

This client:
  * rotates across N keys (one logical key per worker slot),
  * enforces conservative 12 RPM and 240k TPM per key,
  * tracks per-key RPD with disk-persisted counters,
  * exponential backoff on 429/5xx,
  * disk-caches every (prompt -> response) by SHA1 to avoid repeat calls.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import deque
from pathlib import Path

from google import genai
from google.genai import types as gtypes

from .io import CACHE_DIR, LOG_DIR, load_env_keys

MODEL_NAME = "gemini-3.1-flash-lite"
RPM_LIMIT = 12          # conservative against 15 RPM ceiling
TPM_LIMIT = 240_000     # conservative against 250k TPM ceiling
RPD_LIMIT = 500
WINDOW_SEC = 60

CACHE_FILE = CACHE_DIR / "gemini_cache.jsonl"
RPD_FILE = LOG_DIR / "gemini_rpd.json"


def _cache_key(prompt: str, system: str | None) -> str:
    h = hashlib.sha1()
    h.update((system or "").encode("utf-8"))
    h.update(b"\n--\n")
    h.update(prompt.encode("utf-8"))
    return h.hexdigest()


class _DiskCache:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self._lock = threading.Lock()
        self._mem: dict[str, str] = {}
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                self._mem[rec["k"]] = rec["v"]
            except Exception:
                continue

    def get(self, k: str) -> str | None:
        return self._mem.get(k)

    def put(self, k: str, v: str) -> None:
        with self._lock:
            self._mem[k] = v
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"k": k, "v": v}) + "\n")


class _KeyState:
    def __init__(self, idx: int, key: str):
        self.idx = idx
        self.key = key
        self.client = genai.Client(api_key=key)
        self.lock = threading.Lock()
        self.calls: deque[float] = deque()        # timestamps within last 60s
        self.tokens: deque[tuple[float, int]] = deque()  # (ts, tokens) within 60s
        self.rpd = 0
        self.suspended = False  # set True permanently on a 403 PERMISSION_DENIED

    def _trim(self, now: float) -> None:
        while self.calls and now - self.calls[0] > WINDOW_SEC:
            self.calls.popleft()
        while self.tokens and now - self.tokens[0][0] > WINDOW_SEC:
            self.tokens.popleft()

    def _tpm_used(self) -> int:
        return sum(t for _, t in self.tokens)

    def can_serve(self, est_tokens: int, now: float) -> tuple[bool, float]:
        if self.suspended:
            return False, 24 * 3600.0
        self._trim(now)
        if self.rpd >= RPD_LIMIT:
            return False, 24 * 3600.0  # parked for the day
        if len(self.calls) >= RPM_LIMIT:
            return False, WINDOW_SEC - (now - self.calls[0]) + 0.1
        if self._tpm_used() + est_tokens > TPM_LIMIT:
            return False, WINDOW_SEC - (now - self.tokens[0][0]) + 0.1
        return True, 0.0

    def record(self, tokens: int) -> None:
        now = time.time()
        self.calls.append(now)
        self.tokens.append((now, tokens))
        self.rpd += 1


class GeminiPool:
    def __init__(self, keys: list[str] | None = None, files: list[str] | None = None):
        if keys is None:
            keys = load_env_keys(files=files)
        if not keys:
            raise RuntimeError("No GEMINI_API_KEY* values found")
        self.states = [_KeyState(i, k) for i, k in enumerate(keys)]
        self._rr_lock = threading.Lock()
        self._rr = 0
        self.cache = _DiskCache(CACHE_FILE)
        self._load_rpd()
        self.log_path = LOG_DIR / "gemini.log"

    # ---- persistence of RPD counters ----
    def _load_rpd(self) -> None:
        if RPD_FILE.exists():
            try:
                data = json.loads(RPD_FILE.read_text())
                day = data.get("day")
                today = time.strftime("%Y-%m-%d")
                if day == today:
                    for i, v in enumerate(data.get("counts", [])):
                        if i < len(self.states):
                            self.states[i].rpd = int(v)
            except Exception:
                pass

    def _save_rpd(self) -> None:
        try:
            RPD_FILE.write_text(json.dumps({
                "day": time.strftime("%Y-%m-%d"),
                "counts": [s.rpd for s in self.states],
            }))
        except Exception:
            pass

    # ---- core call ----
    def generate(self, prompt: str, *, system: str | None = None,
                 est_tokens: int = 4000, max_tries: int = 5) -> str:
        """Cache-first generate. Blocks until a key has capacity."""
        key = _cache_key(prompt, system)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        tries = 0
        backoff = 1.0
        while tries < max_tries:
            state = self._pick(est_tokens)
            if state is None:
                # All keys saturated or exhausted; small wait then retry
                time.sleep(2.0)
                continue
            try:
                cfg = gtypes.GenerateContentConfig(
                    system_instruction=system,
                    temperature=0.2,
                    max_output_tokens=2048,
                    response_mime_type="application/json",
                )
                resp = state.client.models.generate_content(
                    model=MODEL_NAME, contents=prompt, config=cfg,
                )
                state.record(est_tokens)
                self._save_rpd()
                out = (resp.text or "").strip()
                if out:
                    self.cache.put(key, out)
                return out
            except Exception as e:
                tries += 1
                msg = str(e)
                self._log(f"key{state.idx} err: {msg[:200]}")
                if "403" in msg or "PERMISSION_DENIED" in msg.upper() or "API_KEY_INVALID" in msg.upper() or "suspended" in msg.lower():
                    # Permanently park this key for the session.
                    with state.lock:
                        state.suspended = True
                    self._log(f"key{state.idx} SUSPENDED — permanently parked")
                    # Continue immediately; do not count this as a try.
                    tries -= 1
                    continue
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg.upper():
                    # park this key for a minute, try another
                    with state.lock:
                        state.calls.extend([time.time()] * RPM_LIMIT)
                    time.sleep(min(60.0, backoff))
                else:
                    time.sleep(min(30.0, backoff))
                backoff *= 2
        # Check if all keys are suspended — fail fast with a clear message
        live = [s for s in self.states if not s.suspended]
        if not live:
            raise RuntimeError("All Gemini keys are suspended (403 PERMISSION_DENIED)")
        raise RuntimeError("Gemini generate failed after retries")

    def _pick(self, est_tokens: int) -> _KeyState | None:
        now = time.time()
        with self._rr_lock:
            order = list(range(self._rr, len(self.states))) + list(range(0, self._rr))
            self._rr = (self._rr + 1) % len(self.states)
        soonest_wait = None
        for i in order:
            s = self.states[i]
            ok, wait = s.can_serve(est_tokens, now)
            if ok:
                return s
            if wait < 24 * 3600 and (soonest_wait is None or wait < soonest_wait):
                soonest_wait = wait
        if soonest_wait is None:
            return None  # all parked-for-day
        time.sleep(min(soonest_wait, 30.0))
        return None

    def _log(self, msg: str) -> None:
        try:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
        except Exception:
            pass
