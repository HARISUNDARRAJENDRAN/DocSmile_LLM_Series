"""Analyze CPT dataset characteristics for training planning."""
import json
from pathlib import Path

def analyze_file(path):
    """Analyze a JSONL file and return statistics."""
    total_words = 0
    total_chars = 0
    chunks = 0

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                chunks += 1
                total_words += data.get('word_count', 0)
                total_chars += len(data.get('text', ''))

    return chunks, total_words, total_chars

def main():
    selective = Path('cpt_prepared/selective_cpt_chunks.jsonl')
    core = Path('cpt_prepared/core_cpt_chunks.jsonl')

    sel_chunks, sel_words, sel_chars = analyze_file(selective)
    core_chunks, core_words, core_chars = analyze_file(core)

    total_chunks = sel_chunks + core_chunks
    total_words = sel_words + core_words
    total_chars = sel_chars + core_chars

    print('='*60)
    print('CPT Dataset Analysis')
    print('='*60)
    print(f'\nSelective CPT:')
    print(f'  Chunks: {sel_chunks:,}')
    print(f'  Words: {sel_words:,}')
    print(f'  Characters: {sel_chars:,}')
    print(f'  Avg words/chunk: {sel_words/sel_chunks:.0f}')

    print(f'\nCore CPT:')
    print(f'  Chunks: {core_chunks:,}')
    print(f'  Words: {core_words:,}')
    print(f'  Characters: {core_chars:,}')
    print(f'  Avg words/chunk: {core_words/core_chunks:.0f}')

    print(f'\nTotal Dataset:')
    print(f'  Chunks: {total_chunks:,}')
    print(f'  Words: {total_words:,}')
    print(f'  Characters: {total_chars:,}')
    print(f'  Avg words/chunk: {total_words/total_chunks:.0f}')

    # Estimate tokens (rough: 1 token ≈ 4 chars for English)
    est_tokens = total_chars / 4
    print(f'  Estimated tokens: {est_tokens:,.0f}')

    # Training estimates
    print(f'\nTraining Estimates (1 epoch):')
    print(f'  At 2048 context: ~{total_chunks:,} samples')
    print(f'  At 4096 context: ~{total_chunks//2:,} samples')
    print(f'  At 8192 context: ~{total_chunks//4:,} samples')
    print('='*60)

if __name__ == '__main__':
    main()
