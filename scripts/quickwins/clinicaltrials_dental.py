"""ClinicalTrials.gov dental study scraper (API v2)."""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "cpt_prepared" / "clinicaltrials_dental"
LOG_DIR = ROOT / "logs" / "clinicaltrials"
for d in (OUT_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

API = "https://clinicaltrials.gov/api/v2/studies"

# Broad dental query — covers the major dental specialties.
QUERY = (
    "AREA[ConditionSearch] (dental OR dentistry OR tooth OR teeth OR periodontal "
    "OR periodontitis OR gingivitis OR endodont* OR orthodont* OR prosthodont* "
    "OR \"dental caries\" OR \"oral health\" OR \"oral cancer\" OR \"oral surgery\" "
    "OR \"dental implant\" OR malocclusion OR \"temporomandibular\" OR pulpitis "
    "OR \"mouth disease\" OR \"tooth loss\" OR \"dental pain\" OR halitosis "
    "OR xerostomia OR bruxism)"
)

FIELDS = [
    "NCTId", "BriefTitle", "OfficialTitle", "OverallStatus",
    "BriefSummary", "DetailedDescription",
    "Condition", "Keyword", "InterventionName", "InterventionDescription",
    "EligibilityCriteria", "Gender", "MinimumAge", "MaximumAge",
    "StudyType", "Phase", "EnrollmentCount", "PrimaryOutcomeMeasure",
    "PrimaryOutcomeDescription", "SecondaryOutcomeMeasure",
    "StartDate", "CompletionDate",
]


def _http_get(url: str, max_retries: int = 6) -> bytes:
    backoff = 1.0
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return resp.read()
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(backoff)
            backoff = min(backoff * 2, 30.0)
    raise RuntimeError("unreachable")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument("--max-pages", type=int, default=100)
    parser.add_argument("--out", default="clinicaltrials_dental.jsonl")
    args = parser.parse_args(argv)

    out_path = OUT_DIR / args.out
    state_path = OUT_DIR / "state.json"

    next_token = None
    page = 0
    n_written = 0
    if state_path.exists():
        st = json.loads(state_path.read_text())
        next_token = st.get("nextPageToken")
        page = st.get("page", 0)
        n_written = st.get("n_written", 0)
        print(f"[ct.gov] resume page={page} n={n_written}")

    with out_path.open("a", encoding="utf-8") as f_out:
        while page < args.max_pages:
            params = {
                "query.cond": QUERY,
                "pageSize": str(args.page_size),
                "format": "json",
            }
            if next_token:
                params["pageToken"] = next_token
            url = f"{API}?{urllib.parse.urlencode(params)}"
            data = json.loads(_http_get(url).decode("utf-8"))
            studies = data.get("studies", [])
            if not studies:
                print("[ct.gov] no more studies; done.")
                break
            for s in studies:
                f_out.write(json.dumps(s, ensure_ascii=False) + "\n")
            n_written += len(studies)
            page += 1
            next_token = data.get("nextPageToken")
            print(f"[ct.gov] page {page} +{len(studies)} (total={n_written}) token={'yes' if next_token else 'no'}")
            state_path.write_text(json.dumps({
                "nextPageToken": next_token, "page": page, "n_written": n_written,
            }))
            if not next_token:
                print("[ct.gov] no nextPageToken; done.")
                break
            time.sleep(0.3)
    print(f"[ct.gov] DONE. total={n_written} -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
