import json
from collections import Counter, defaultdict
import random

def analyze_sft_dataset(file_path, sample_size=1000):
    """Analyze SFT dataset for diversity metrics"""

    topics = []
    sources = []
    question_types = []
    answer_lengths = []
    question_lengths = []

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total_samples = len(lines)
    sample_indices = random.sample(range(total_samples), min(sample_size, total_samples))

    for idx in sample_indices:
        try:
            data = json.loads(lines[idx])

            if 'topic' in data:
                topics.append(data['topic'])
            if 'source' in data:
                sources.append(data['source'])

            if 'question' in data:
                question = data['question']
                question_lengths.append(len(question.split()))

                if question.startswith('exam_reasoning'):
                    question_types.append('exam_reasoning')
                elif 'explain' in question.lower() or 'describe' in question.lower():
                    question_types.append('explanation')
                elif 'what' in question.lower() or 'how' in question.lower():
                    question_types.append('factual')
                elif 'compare' in question.lower() or 'differ' in question.lower():
                    question_types.append('comparison')
                else:
                    question_types.append('other')

            if 'answer' in data:
                answer_lengths.append(len(data['answer'].split()))

        except json.JSONDecodeError:
            continue

    return {
        'total_samples': total_samples,
        'analyzed_samples': len(sample_indices),
        'topics': Counter(topics),
        'sources': Counter(sources),
        'question_types': Counter(question_types),
        'avg_question_length': sum(question_lengths) / len(question_lengths) if question_lengths else 0,
        'avg_answer_length': sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0,
        'unique_topics': len(set(topics)),
        'unique_sources': len(set(sources))
    }

def analyze_dpo_dataset(file_path, sample_size=1000):
    """Analyze DPO dataset for diversity metrics"""

    topics = []
    sources = []
    prompt_types = []
    chosen_lengths = []
    rejected_lengths = []
    prompt_lengths = []

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total_samples = len(lines)
    sample_indices = random.sample(range(total_samples), min(sample_size, total_samples))

    for idx in sample_indices:
        try:
            data = json.loads(lines[idx])

            if 'topic' in data:
                topics.append(data['topic'])
            if 'source' in data:
                sources.append(data['source'])

            if 'prompt' in data:
                prompt = data['prompt']
                prompt_lengths.append(len(prompt.split()))

                if 'explain' in prompt.lower() or 'describe' in prompt.lower():
                    prompt_types.append('explanation')
                elif 'why' in prompt.lower():
                    prompt_types.append('reasoning')
                elif 'compare' in prompt.lower() or 'differ' in prompt.lower():
                    prompt_types.append('comparison')
                else:
                    prompt_types.append('other')

            if 'chosen' in data:
                chosen_lengths.append(len(data['chosen'].split()))
            if 'rejected' in data:
                rejected_lengths.append(len(data['rejected'].split()))

        except json.JSONDecodeError:
            continue

    return {
        'total_samples': total_samples,
        'analyzed_samples': len(sample_indices),
        'topics': Counter(topics),
        'sources': Counter(sources),
        'prompt_types': Counter(prompt_types),
        'avg_prompt_length': sum(prompt_lengths) / len(prompt_lengths) if prompt_lengths else 0,
        'avg_chosen_length': sum(chosen_lengths) / len(chosen_lengths) if chosen_lengths else 0,
        'avg_rejected_length': sum(rejected_lengths) / len(rejected_lengths) if rejected_lengths else 0,
        'unique_topics': len(set(topics)),
        'unique_sources': len(set(sources))
    }

def print_analysis(name, results):
    """Pretty print analysis results"""
    print(f"\n{'='*80}")
    print(f"{name} ANALYSIS")
    print(f"{'='*80}")
    print(f"Total Samples: {results['total_samples']:,}")
    print(f"Analyzed Samples: {results['analyzed_samples']:,}")
    print(f"Unique Topics: {results['unique_topics']}")
    print(f"Unique Sources: {results['unique_sources']}")

    if 'avg_question_length' in results:
        print(f"\nAverage Question Length: {results['avg_question_length']:.1f} words")
        print(f"Average Answer Length: {results['avg_answer_length']:.1f} words")
    else:
        print(f"\nAverage Prompt Length: {results['avg_prompt_length']:.1f} words")
        print(f"Average Chosen Length: {results['avg_chosen_length']:.1f} words")
        print(f"Average Rejected Length: {results['avg_rejected_length']:.1f} words")

    print(f"\n--- Top 10 Topics ---")
    for topic, count in results['topics'].most_common(10):
        print(f"  {topic}: {count} ({count/results['analyzed_samples']*100:.1f}%)")

    print(f"\n--- Top 10 Sources ---")
    for source, count in results['sources'].most_common(10):
        source_short = source.replace('__lib_book_', '').replace('.md', '')[:60]
        print(f"  {source_short}: {count}")

    if 'question_types' in results:
        print(f"\n--- Question Types ---")
        for qtype, count in results['question_types'].most_common():
            print(f"  {qtype}: {count} ({count/results['analyzed_samples']*100:.1f}%)")
    else:
        print(f"\n--- Prompt Types ---")
        for ptype, count in results['prompt_types'].most_common():
            print(f"  {ptype}: {count} ({count/results['analyzed_samples']*100:.1f}%)")

def assess_diversity(sft_results, dpo_results):
    """Provide diversity assessment and recommendations"""
    print(f"\n{'='*80}")
    print("DIVERSITY ASSESSMENT & RECOMMENDATIONS")
    print(f"{'='*80}")

    print("\n[+] STRENGTHS:")

    if sft_results['total_samples'] >= 25000:
        print(f"  * Strong SFT dataset size: {sft_results['total_samples']:,} samples")

    if dpo_results['total_samples'] >= 8000:
        print(f"  * Good DPO dataset size: {dpo_results['total_samples']:,} samples")

    if sft_results['unique_sources'] >= 20:
        print(f"  * Excellent source diversity: {sft_results['unique_sources']} unique books")

    if sft_results['unique_topics'] >= 30:
        print(f"  * Strong topic diversity: {sft_results['unique_topics']} unique topics")

    print("\n[!] AREAS TO CONSIDER:")

    if sft_results['unique_topics'] < 20:
        print(f"  * Limited topic diversity ({sft_results['unique_topics']} topics)")
        print("    -> Consider adding more specialized dental topics")

    if dpo_results['unique_topics'] < 15:
        print(f"  * DPO topic diversity could be improved ({dpo_results['unique_topics']} topics)")

    top_topic_pct = sft_results['topics'].most_common(1)[0][1] / sft_results['analyzed_samples'] * 100
    if top_topic_pct > 30:
        print(f"  * Top topic dominates {top_topic_pct:.1f}% of dataset")
        print("    -> Consider balancing topic distribution")

    print("\n[IMAGE] VISUAL ENCODER INTEGRATION PLAN:")
    print("  * Your textbook images will add crucial multimodal diversity")
    print("  * Image-text pairs will cover:")
    print("    - Anatomical diagrams and dental structures")
    print("    - Clinical procedure illustrations")
    print("    - Radiographic interpretations")
    print("    - Pathology visualizations")
    print("  * Recommendation: Aim for 15-20k image-text pairs minimum")

    print("\n[VERDICT] OVERALL ASSESSMENT:")
    if (sft_results['total_samples'] >= 25000 and
        dpo_results['total_samples'] >= 8000 and
        sft_results['unique_sources'] >= 15):
        print("  [OK] Your current text datasets are SUFFICIENT for initial training")
        print("  [OK] Adding visual encoder with textbook images will provide excellent diversity")
        print("  [OK] Proceed with CPT -> SFT -> DPO -> VLM pipeline as planned")
    else:
        print("  [WARN] Consider augmenting datasets before proceeding")
        print("  -> Target: 30k+ SFT, 10k+ DPO for optimal results")

if __name__ == "__main__":
    print("Analyzing DocSmile Training Datasets...")
    print("This may take a few minutes...\n")

    sft_results = analyze_sft_dataset('rl_prepared/rl_sft.jsonl', sample_size=2000)
    print_analysis("SFT DATASET", sft_results)

    dpo_results = analyze_dpo_dataset('rl_prepared/rl_dpo.jsonl', sample_size=1000)
    print_analysis("DPO DATASET", dpo_results)

    assess_diversity(sft_results, dpo_results)

    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"{'='*80}\n")
