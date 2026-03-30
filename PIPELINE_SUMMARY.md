# DocSmile — Chat Summary & Final Data Pipeline

This document consolidates the technical decisions and workflow agreed for **DocSmile**: a series of dental-domain models combining **continued pre-training (CPT)** on text with a future **vision-language (VLM)** track for diagrams and clinical images.

---

## 1. Project goals (from `plan.md`)

- **Text LLM:** Domain adaptation via **CPT** on cleaned dental textbook and educational text.
- **VLM (later):** Interpret anatomical diagrams, radiographs, and procedural illustrations with **cross-modal** alignment to text.
- **Outputs:** Concept explanation, clinical-style reasoning, and multimodal understanding unified under one product vision.

---

## 2. Corpus inventory & prioritization

- **Source list:** `Dental_rec_list.xlsx` — **634** PDFs (metadata: file name, Drive link, date, size).
- **Estimated scale:** On the order of **~340k pages** total (user estimate); use for capacity planning, not as a hard guarantee until measured per PDF.
- **Priority artifact:** `books_cpt_priority.csv` — each row has `process_order`, `tier` (**S / A / B / C / D**), `score`, `category`, and notes.
  - **Tier S:** Ingest first — flagship references (e.g. major periodontics, oral pathology, OMS, endodontics, pharmacology, prosthodontics standards).
  - **Tier A–B:** Strong curriculum coverage.
  - **Tier C–D:** Supporting, review-style, or lower CPT signal — process after higher tiers or deprioritize for token budget.
- **Redundancy:** Automated clusters in `books_redundancy_clusters.json` (e.g. multiple editions of the same title). For CPT, prefer **newest edition** per title family when content overlaps.
- **Gaps:** Very few **public health** titles in the folder tags; consider adding **dental public health**, **ethics/jurisprudence**, and **imaging depth** if missing from “Miscellaneous” or elsewhere.

---

## 3. Text extraction: Docling vs Marker (conclusion)

| Topic | Conclusion |
|--------|------------|
| **GPU efficiency (published head-to-head)** | On comparable setups, **Docling** was faster than **Marker** in the Docling technical report’s L4 benchmark; **Marker** is very slow on **CPU-only** runs — use **GPU** for either tool at book scale. |
| **Cost** | Both are **open source**; cost is mostly **GPU hours** on cloud (e.g. GCP) plus optional **Marker LLM** API if enabled. |
| **Markdown / CPT suitability** | **Docling:** Strong on **structure and tables**, **extractive** behavior, good for **training corpora**. **Marker:** Often **polished** Markdown; optional **LLM** pass can improve hard pages but adds cost and governance. **Decision:** **Docling** as the primary extractor for this pipeline. |
| **Edge cases (e.g. chemistry atlases)** | Docling uses **layout + OCR + table models**; it does **not** “understand” chemistry. Dense periodic tables and orbital diagrams may be **partially** tabular/OCR’d; **non-text graphics** should be handled in the **VLM / image** track, not expected as perfect Markdown. |

---

## 4. GCP budget & GPU choice (preprocessing)

- **Budget guideline:** On the order of **~$300** for preprocessing (order-of-magnitude; confirm with a short pilot).
- **Recommended accelerator:** **NVIDIA L4** on **G2** instances (`g2-standard-*`), which matches Docling’s public L4 benchmarks and gives enough VRAM for **batched OCR/layout** tuning.
- **Rough throughput (literature, not your PDFs):** Docling ~**0.5 s/page** class speeds on L4 in benchmarks — use a **500–1000 page pilot** on GCP to extrapolate **GPU-hours** and cost.
- **Sharding:** Use **one Docling worker per GPU** to start (`CUDA_VISIBLE_DEVICES` per process). **G2** shapes scale to **multiple L4s per VM** (e.g. 2 / 4 / 8 GPUs); beyond that, add **more VMs** and a shared queue (GCS + task queue).

References: [GCP accelerator-optimized pricing](https://cloud.google.com/products/compute/pricing/accelerator-optimized), [GCP GPU add-on pricing](https://cloud.google.com/compute/gpus-pricing), [Docling technical report](https://arxiv.org/html/2408.09869v4).

---

## 5. Final pipeline — text-only CPT corpus (agreed)

This is the **go-to** production path for **text-only** continued pre-training data.

### 5.1 Inputs

- **634** scan-ready (or text-layer) **PDFs** stored in cloud storage (e.g. **GCS**), aligned with `Dental_rec_list.xlsx` / Drive exports.

### 5.2 Compute

- **GCP VM:** **G2** + **NVIDIA L4**.
- **Parallelism:** **Shard** work by **book/PDF** across **workers**; **one worker per GPU** initially. Scale **GPUs** (and VMs) until the queue finishes within schedule/budget.

### 5.3 Extraction engine

- **Docling** with:
  - **OCR on** (scanned pages).
  - **Table structure on** (tabular *text* for CPT).
  - **Picture / VLM enrichment off** for this stage (`generate_picture_images` off, no picture description, no embedded images in Markdown).
- **Markdown export:** **Text-only** — no figure embeds, no base64 images (e.g. `ImageRefMode.PLACEHOLDER` with empty placeholder, or equivalent serialization so figures do not pollute `.md`).
- **Rationale:** CPT pass uses **clean text**; **images** are intentionally **out of scope** for this artifact.

### 5.4 Outputs

- **Per-book (or per-chunk) `.md` files** under a prefix such as `gs://…/text-md/`.
- **Manifest** (e.g. `manifest.jsonl`): `source_id`, `gcs_uri`, `sha256`, `tier`, `page_count` if available, **status**, **timestamps** — supports **idempotent** retries and audits.

### 5.5 Images (explicitly separate)

Text-only Markdown **does not** replace a vision dataset. Planned options (pick one or combine):

| Plan | Description |
|------|-------------|
| **A — Manual / semi-manual** | High-quality labels on a **small gold set** (e.g. Tier S pages or sampled figures). |
| **B — Caption / heuristic pairs** | Pair **figure crops** with **nearby captions** from text for weak supervision. |
| **C — Model-assisted + review** | VLM or classifier proposes labels/descriptions; **experts** approve. |
| **D — Defer** | Ship **text CPT + RAG** first; add multimodal when budget and labels exist. |

---

## 6. Artifacts in this repo (reference)

| File | Purpose |
|------|---------|
| `plan.md` | Original product vision. |
| `Dental_rec_list.xlsx` | Master inventory of PDFs. |
| `books_cpt_priority.csv` | Processing order and tier for CPT sourcing. |
| `books_flat.json` / `books_redundancy_clusters.json` | Parsed metadata and duplicate hints. |
| `analyze_books.py` / `priority_tag_books.py` | Scripts used to generate priority and redundancy outputs (reproducible). |

---

## 7. Next implementation steps (checklist)

1. **Pilot:** 500–1000 pages on **one L4 G2** VM — measure **wall-clock**, **VRAM**, and **output quality**.
2. **Lock Docling config** for text-only MD (pipeline options + serializer) and **document** the exact Python version and Docling version.
3. **Wire GCS** input/output + **manifest** writes after each successful PDF.
4. **Scale** workers/GPUs; use **Spot** if preemption-tolerant with checkpointing.
5. **Kick off Tier S → A** first using `books_cpt_priority.csv`.
6. **Parallel track (optional):** define **image** scope (Plan A–D) and tooling (storage layout, labeling UI).

---

*This summary reflects the engineering decisions from the design conversation and is the single place to onboard teammates on the agreed preprocessing and multimodal split.*
