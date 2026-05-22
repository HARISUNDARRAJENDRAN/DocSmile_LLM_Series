"""ChatDoctor HealthCareMagic-100k + iCliniq, regex-filtered to dental content.

Filtering uses scripts.quickwins.common.dental_filter.is_dental over question+answer.
"""
from __future__ import annotations

from typing import Iterator

from datasets import load_dataset

from ..common.dental_filter import is_dental
from ..common.schema import make_sft

HCM_REPO = "lavita/ChatDoctor-HealthCareMagic-100k"
ICL_REPO = "lavita/ChatDoctor-iCliniq"


def _yield_from(ds, source_name: str) -> Iterator[dict]:
    for row in ds:
        q = (row.get("input") or row.get("question") or row.get("instruction") or "").strip()
        a = (row.get("output") or row.get("answer") or row.get("response") or "").strip()
        if not q or not a:
            continue
        if not is_dental(q, a):
            continue
        rec = make_sft(q, a, source_name, "Dental Consultation")
        if rec:
            yield rec


def load() -> Iterator[dict]:
    try:
        hcm = load_dataset(HCM_REPO, split="train")
        yield from _yield_from(hcm, "chatdoctor_healthcaremagic_dental")
    except Exception as e:
        print(f"[chatdoctor] HCM load failed: {e}")
    try:
        icl = load_dataset(ICL_REPO, split="train")
        yield from _yield_from(icl, "chatdoctor_icliniq_dental")
    except Exception as e:
        print(f"[chatdoctor] iCliniq load failed: {e}")
