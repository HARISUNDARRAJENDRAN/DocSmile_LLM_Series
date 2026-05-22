#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

import aiohttp
from dotenv import load_dotenv


def load_keys(env_path: Path) -> list[tuple[str, str]]:
    load_dotenv(env_path)
    keys: list[tuple[str, str]] = []
    seen: set[str] = set()
    for name in ["GEMINI_API_KEY", "GOOGLE_API_KEY", *[f"GEMINI_API_KEY{i}" for i in range(1, 80)]]:
        value = os.environ.get(name, "").strip()
        if value and value not in seen:
            keys.append((name, value))
            seen.add(value)
    return keys


async def check_one(session: aiohttp.ClientSession, model: str, name: str, key: str) -> dict:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    payload = {
        "contents": [{"parts": [{"text": "Reply with exactly: ok"}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 8},
    }
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=45)) as resp:
            body = await resp.text()
            parsed = None
            try:
                parsed = json.loads(body)
            except Exception:
                parsed = {"raw": body[:1000]}
            error = parsed.get("error") if isinstance(parsed, dict) else None
            return {
                "key_name": name,
                "status": resp.status,
                "error_code": error.get("code") if error else None,
                "error_status": error.get("status") if error else None,
                "error_message": error.get("message") if error else None,
                "ok_text": (
                    parsed.get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text")
                    if isinstance(parsed, dict) and parsed.get("candidates")
                    else None
                ),
            }
    except Exception as exc:
        return {"key_name": name, "status": "client_error", "error_message": f"{type(exc).__name__}: {exc}"}


async def main() -> None:
    parser = argparse.ArgumentParser(description="Check Gemini API keys and print exact API errors.")
    parser.add_argument("--model", default="gemini-3.1-flash-lite")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    keys = load_keys(root / ".env")
    if args.limit:
        keys = keys[: args.limit]
    async with aiohttp.ClientSession() as session:
        rows = await asyncio.gather(*(check_one(session, args.model, name, key) for name, key in keys))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
