import argparse
import html
import json
import re
from pathlib import Path


MD_IMAGE_RE = re.compile(r"!\[.*?\]\(.*?\)")
MD_LINK_RE = re.compile(r"\[(.*?)\]\((.*?)\)")
TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
TABLE_DIV_RE = re.compile(r"^\s*\|?\s*-{2,}(\s*\|\s*-{2,})+\s*\|?\s*$")
CODE_FENCE_RE = re.compile(r"^\s*```")
HTML_TAG_RE = re.compile(r"<[^>]+>")


def _is_table_line(line: str) -> bool:
    if TABLE_DIV_RE.match(line):
        return True
    if TABLE_ROW_RE.match(line) and line.count("|") >= 2:
        return True
    return False


def clean_markdown(text: str, keep_figures: bool = False) -> str:
    lines = text.splitlines()
    cleaned_lines = []
    in_code_block = False

    for raw in lines:
        line = raw.rstrip("\n")

        if CODE_FENCE_RE.match(line):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue

        stripped = line.strip()
        if stripped == "":
            cleaned_lines.append("")
            continue

        if stripped.startswith("<!--") and stripped.endswith("-->"):
            continue

        lowered = stripped.lower()
        if "terms and conditions" in lowered:
            continue
        if "all rights reserved" in lowered:
            continue
        if "copyright" in lowered:
            continue

        if _is_table_line(stripped):
            continue

        if not keep_figures and re.match(r"^\s*(figure|fig\.|table)\b", stripped, re.IGNORECASE):
            continue

        # Remove markdown images and links
        line = MD_IMAGE_RE.sub("", line)
        line = MD_LINK_RE.sub(r"\1", line)

        # Remove HTML tags and unescape entities
        line = HTML_TAG_RE.sub("", line)
        line = html.unescape(line)

        # Remove bare URLs but keep surrounding text
        line = re.sub(r"https?://\S+|www\.\S+", "", line)

        # Strip heading markers
        heading_match = re.match(r"^\s*#{1,6}\s+(.*)$", line)
        if heading_match:
            heading_text = heading_match.group(1).strip()
            if heading_text:
                if cleaned_lines and cleaned_lines[-1] != "":
                    cleaned_lines.append("")
                cleaned_lines.append(heading_text)
                cleaned_lines.append("")
            continue

        # List bullets
        bullet_match = re.match(r"^\s*[-*+]\s+\[[ xX]\]\s+(.*)$", line)
        if bullet_match:
            item = bullet_match.group(1).strip()
            if item:
                cleaned_lines.append(item)
                cleaned_lines.append("")
            continue

        bullet_match = re.match(r"^\s*[-*+]\s+(.*)$", line)
        if bullet_match:
            item = bullet_match.group(1).strip()
            if item:
                cleaned_lines.append(item)
                cleaned_lines.append("")
            continue

        # Blockquotes
        line = re.sub(r"^\s*>\s+", "", line)

        # Emphasis markers
        line = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
        line = re.sub(r"__(.*?)__", r"\1", line)
        line = re.sub(r"\*(.*?)\*", r"\1", line)
        line = re.sub(r"_(.*?)_", r"\1", line)

        line = line.replace("\t", " ")
        line = re.sub(r"\s+", " ", line).strip()

        if line.isdigit():
            continue

        if line:
            cleaned_lines.append(line)

    # Collapse multiple blank lines
    collapsed = []
    last_blank = False
    for line in cleaned_lines:
        is_blank = line == ""
        if is_blank and last_blank:
            continue
        collapsed.append(line)
        last_blank = is_blank

    # Join lines into paragraphs
    paragraphs = []
    buf = []
    for line in collapsed:
        if line == "":
            if buf:
                paragraphs.append(" ".join(buf))
                buf = []
            continue
        buf.append(line)
    if buf:
        paragraphs.append(" ".join(buf))

    return "\n\n".join(paragraphs).strip()


def chunk_text(text: str, chunk_words: int, overlap_words: int):
    words = text.split()
    if not words:
        return []

    chunks = []
    step = max(1, chunk_words - overlap_words)
    for start in range(0, len(words), step):
        end = start + chunk_words
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break
    return chunks


def validate_outputs(texts, output_dir: Path, chunk_jsonl: Path):
    stats = {
        "files": 0,
        "empty_outputs": [],
        "warnings": [],
        "total_input_chars": 0,
        "total_output_chars": 0,
    }

    for src_path, src_text, clean_text in texts:
        stats["files"] += 1
        stats["total_input_chars"] += len(src_text)
        stats["total_output_chars"] += len(clean_text)

        if not clean_text:
            stats["empty_outputs"].append(src_path.name)
            continue

        if "```" in clean_text or "<!--" in clean_text or "![" in clean_text:
            stats["warnings"].append(f"Residual markdown in {src_path.name}")

        if len(clean_text) < 200:
            stats["warnings"].append(f"Very short output in {src_path.name}")

        ratio = len(clean_text) / max(1, len(src_text))
        if ratio < 0.2:
            stats["warnings"].append(f"Low retained ratio in {src_path.name}: {ratio:.2f}")

    stats_path = output_dir / "validation.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=True), encoding="utf-8")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Prepare CPT-ready text from Markdown.")
    parser.add_argument("--input-dir", required=True, help="Directory with .md files")
    parser.add_argument("--output-dir", required=True, help="Directory for cleaned .txt files")
    parser.add_argument("--chunk-jsonl", required=True, help="Path for chunk JSONL output")
    parser.add_argument("--chunk-words", type=int, default=800, help="Words per chunk")
    parser.add_argument("--overlap-words", type=int, default=120, help="Overlap words between chunks")
    parser.add_argument("--keep-figures", action="store_true", help="Keep figure/table captions")
    parser.add_argument("--max-files", type=int, default=0, help="Limit number of files for a test run")
    parser.add_argument("--validate", action="store_true", help="Write validation.json report")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    chunk_jsonl = Path(args.chunk_jsonl)

    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_jsonl.parent.mkdir(parents=True, exist_ok=True)

    md_files = sorted([p for p in input_dir.glob("*.md") if p.is_file()])
    if args.max_files > 0:
        md_files = md_files[: args.max_files]

    texts_for_validation = []
    total_chunks = 0

    with chunk_jsonl.open("w", encoding="utf-8") as jf:
        for md_path in md_files:
            src_text = md_path.read_text(encoding="utf-8", errors="ignore")
            clean_text = clean_markdown(src_text, keep_figures=args.keep_figures)

            texts_for_validation.append((md_path, src_text, clean_text))

            if not clean_text:
                continue

            txt_path = output_dir / (md_path.stem + ".txt")
            txt_path.write_text(clean_text, encoding="utf-8")

            chunks = chunk_text(clean_text, args.chunk_words, args.overlap_words)
            for idx, chunk in enumerate(chunks):
                record = {
                    "id": f"{md_path.stem}_{idx:04d}",
                    "source": md_path.stem,
                    "chunk_index": idx,
                    "word_count": len(chunk.split()),
                    "text": chunk,
                }
                jf.write(json.dumps(record, ensure_ascii=True) + "\n")
                total_chunks += 1

    if args.validate:
        validate_outputs(texts_for_validation, output_dir, chunk_jsonl)

    print(f"Processed files: {len(md_files)}")
    print(f"Total chunks: {total_chunks}")
    print(f"Cleaned text output: {output_dir}")
    print(f"Chunk JSONL: {chunk_jsonl}")
    if args.validate:
        print(f"Validation report: {output_dir / 'validation.json'}")


if __name__ == "__main__":
    main()
