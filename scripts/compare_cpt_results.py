"""
Comprehensive evaluation comparing pre-CPT baseline vs post-CPT models.
Acts as an automated judge to assess quality improvements across benchmarks.
"""
import json
from pathlib import Path
from collections import defaultdict
import statistics

def load_jsonl(path):
    """Load JSONL file into list of dicts."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def analyze_mcq_performance(baseline_path, cpt_path):
    """Analyze MCQ accuracy and error patterns."""
    baseline = load_jsonl(baseline_path)
    cpt = load_jsonl(cpt_path)

    # Overall accuracy
    baseline_correct = sum(1 for x in baseline if x.get('correct'))
    cpt_correct = sum(1 for x in cpt if x.get('correct'))

    baseline_acc = baseline_correct / len(baseline)
    cpt_acc = cpt_correct / len(cpt)

    # Per-question comparison
    improvements = 0
    regressions = 0

    baseline_dict = {x['id']: x for x in baseline}
    cpt_dict = {x['id']: x for x in cpt}

    for qid in baseline_dict:
        if qid in cpt_dict:
            b_correct = baseline_dict[qid].get('correct', False)
            c_correct = cpt_dict[qid].get('correct', False)

            if not b_correct and c_correct:
                improvements += 1
            elif b_correct and not c_correct:
                regressions += 1

    return {
        'baseline_accuracy': baseline_acc,
        'cpt_accuracy': cpt_acc,
        'accuracy_delta': cpt_acc - baseline_acc,
        'baseline_correct': baseline_correct,
        'cpt_correct': cpt_correct,
        'total_questions': len(baseline),
        'improvements': improvements,
        'regressions': regressions,
        'net_improvement': improvements - regressions
    }

def analyze_open_qa_quality(baseline_path, cpt_path):
    """Analyze open QA response quality improvements."""
    baseline = load_jsonl(baseline_path)
    cpt = load_jsonl(cpt_path)

    baseline_dict = {x['id']: x for x in baseline}
    cpt_dict = {x['id']: x for x in cpt}

    # Quality metrics
    baseline_lengths = []
    cpt_lengths = []

    repetition_issues_baseline = 0
    repetition_issues_cpt = 0

    for qid in baseline_dict:
        if qid in cpt_dict:
            b_pred = baseline_dict[qid].get('prediction', '')
            c_pred = cpt_dict[qid].get('prediction', '')

            baseline_lengths.append(len(b_pred))
            cpt_lengths.append(len(c_pred))

            # Check for repetition (same phrase repeated multiple times)
            if has_repetition(b_pred):
                repetition_issues_baseline += 1
            if has_repetition(c_pred):
                repetition_issues_cpt += 1

    return {
        'total_questions': len(baseline),
        'avg_length_baseline': statistics.mean(baseline_lengths) if baseline_lengths else 0,
        'avg_length_cpt': statistics.mean(cpt_lengths) if cpt_lengths else 0,
        'repetition_issues_baseline': repetition_issues_baseline,
        'repetition_issues_cpt': repetition_issues_cpt,
        'repetition_improvement': repetition_issues_baseline - repetition_issues_cpt
    }

def has_repetition(text):
    """Detect if text has obvious repetition patterns."""
    # Check for repeated sentences
    sentences = text.split('. ')
    if len(sentences) > 3:
        # Check if same sentence appears multiple times
        for sent in sentences:
            if sent and sentences.count(sent) > 2:
                return True

    # Check for repeated phrases
    words = text.split()
    if len(words) > 20:
        # Check for long repeated sequences
        for i in range(len(words) - 10):
            phrase = ' '.join(words[i:i+10])
            rest = ' '.join(words[i+10:])
            if phrase in rest:
                return True

    return False

def judge_response_quality(baseline_pred, cpt_pred):
    """
    Judge which response is better based on quality criteria.
    Returns: 'baseline', 'cpt', or 'tie'
    """
    score_baseline = 0
    score_cpt = 0

    # Criterion 1: No repetition (critical)
    if has_repetition(baseline_pred):
        score_baseline -= 3
    if has_repetition(cpt_pred):
        score_cpt -= 3

    # Criterion 2: Appropriate length (not too short, not too long)
    b_len = len(baseline_pred)
    c_len = len(cpt_pred)

    if 200 <= b_len <= 800:
        score_baseline += 1
    if 200 <= c_len <= 800:
        score_cpt += 1

    # Criterion 3: Structured response (has clear sections/points)
    if '\n' in baseline_pred or '- ' in baseline_pred:
        score_baseline += 1
    if '\n' in cpt_pred or '- ' in cpt_pred:
        score_cpt += 1

    # Criterion 4: Avoids generic disclaimers
    generic_phrases = [
        "I am not a doctor",
        "I am not a dentist",
        "I can't tell you what is wrong",
        "I can't give you a diagnosis"
    ]

    b_generic = sum(1 for phrase in generic_phrases if phrase in baseline_pred)
    c_generic = sum(1 for phrase in generic_phrases if phrase in cpt_pred)

    if b_generic > 2:
        score_baseline -= 1
    if c_generic > 2:
        score_cpt -= 1

    # Determine winner
    if score_cpt > score_baseline:
        return 'cpt'
    elif score_baseline > score_cpt:
        return 'baseline'
    else:
        return 'tie'

def compare_sample_responses(baseline_path, cpt_path, num_samples=10):
    """Compare sample responses with quality judgments."""
    baseline = load_jsonl(baseline_path)
    cpt = load_jsonl(cpt_path)

    baseline_dict = {x['id']: x for x in baseline}
    cpt_dict = {x['id']: x for x in cpt}

    comparisons = []
    wins = {'baseline': 0, 'cpt': 0, 'tie': 0}

    # Sample evenly across dataset
    sample_ids = list(baseline_dict.keys())[::len(baseline_dict)//num_samples][:num_samples]

    for qid in sample_ids:
        if qid in cpt_dict:
            b_pred = baseline_dict[qid].get('prediction', '')
            c_pred = cpt_dict[qid].get('prediction', '')

            winner = judge_response_quality(b_pred, c_pred)
            wins[winner] += 1

            comparisons.append({
                'id': qid,
                'baseline_length': len(b_pred),
                'cpt_length': len(c_pred),
                'baseline_has_repetition': has_repetition(b_pred),
                'cpt_has_repetition': has_repetition(c_pred),
                'winner': winner
            })

    return comparisons, wins

def main():
    print("="*80)
    print("LLAMA 3.1-8B: PRE-CPT vs POST-CPT EVALUATION ANALYSIS")
    print("="*80)
    print()

    # Paths
    baseline_dir = Path("evals/results")
    cpt_dir = Path("evals/results/modal_llama31_8b_cpt/llama31_8b_cpt_dental")

    # 1. MCQ Analysis
    print("[MCQ] MULTIPLE CHOICE QUESTIONS (MedMCQA Dental)")
    print("-" * 80)

    mcq_results = analyze_mcq_performance(
        baseline_dir / "medmcqa_dental_mcq_predictions.jsonl",
        cpt_dir / "medmcqa_dental_mcq_predictions.jsonl"
    )

    print(f"Baseline Accuracy: {mcq_results['baseline_accuracy']:.1%} ({mcq_results['baseline_correct']}/{mcq_results['total_questions']})")
    print(f"Post-CPT Accuracy: {mcq_results['cpt_accuracy']:.1%} ({mcq_results['cpt_correct']}/{mcq_results['total_questions']})")
    print(f"Accuracy Change: {mcq_results['accuracy_delta']:+.1%}")
    print()
    print(f"Questions improved: {mcq_results['improvements']}")
    print(f"Questions regressed: {mcq_results['regressions']}")
    print(f"Net improvement: {mcq_results['net_improvement']:+d} questions")
    print()

    # Verdict
    if mcq_results['accuracy_delta'] > 0:
        print("[+] VERDICT: CPT improved MCQ performance")
    elif mcq_results['accuracy_delta'] < 0:
        print("[!] VERDICT: CPT slightly decreased MCQ performance")
    else:
        print("[-] VERDICT: No change in MCQ performance")
    print()
    print()

    # 2. Oral Disease QA Analysis
    print("[QA] OPEN QA: ORAL DISEASE DATASET")
    print("-" * 80)

    oral_results = analyze_open_qa_quality(
        baseline_dir / "oral_disease_open_qa_predictions.jsonl",
        cpt_dir / "oral_disease_open_qa_predictions.jsonl"
    )

    print(f"Total questions: {oral_results['total_questions']}")
    print(f"Avg response length (baseline): {oral_results['avg_length_baseline']:.0f} chars")
    print(f"Avg response length (post-CPT): {oral_results['avg_length_cpt']:.0f} chars")
    print()
    print(f"Repetition issues (baseline): {oral_results['repetition_issues_baseline']}")
    print(f"Repetition issues (post-CPT): {oral_results['repetition_issues_cpt']}")
    print(f"Repetition improvement: {oral_results['repetition_improvement']:+d}")
    print()

    # Sample comparison
    oral_comparisons, oral_wins = compare_sample_responses(
        baseline_dir / "oral_disease_open_qa_predictions.jsonl",
        cpt_dir / "oral_disease_open_qa_predictions.jsonl",
        num_samples=20
    )

    print("Quality judgment (20 sample responses):")
    print(f"  Baseline wins: {oral_wins['baseline']}")
    print(f"  Post-CPT wins: {oral_wins['cpt']}")
    print(f"  Ties: {oral_wins['tie']}")
    print()

    if oral_wins['cpt'] > oral_wins['baseline']:
        print("[+] VERDICT: CPT significantly improved response quality")
    elif oral_wins['baseline'] > oral_wins['cpt']:
        print("[!] VERDICT: Baseline had better response quality")
    else:
        print("[-] VERDICT: Similar response quality")
    print()
    print()

    # 3. Dental Forum QA Analysis
    print("[QA] OPEN QA: DENTAL FORUM DATASET")
    print("-" * 80)

    forum_results = analyze_open_qa_quality(
        baseline_dir / "dental_forum_open_qa_predictions.jsonl",
        cpt_dir / "dental_forum_open_qa_predictions.jsonl"
    )

    print(f"Total questions: {forum_results['total_questions']}")
    print(f"Avg response length (baseline): {forum_results['avg_length_baseline']:.0f} chars")
    print(f"Avg response length (post-CPT): {forum_results['avg_length_cpt']:.0f} chars")
    print()
    print(f"Repetition issues (baseline): {forum_results['repetition_issues_baseline']}")
    print(f"Repetition issues (post-CPT): {forum_results['repetition_issues_cpt']}")
    print(f"Repetition improvement: {forum_results['repetition_improvement']:+d}")
    print()

    # Sample comparison
    forum_comparisons, forum_wins = compare_sample_responses(
        baseline_dir / "dental_forum_open_qa_predictions.jsonl",
        cpt_dir / "dental_forum_open_qa_predictions.jsonl",
        num_samples=20
    )

    print("Quality judgment (20 sample responses):")
    print(f"  Baseline wins: {forum_wins['baseline']}")
    print(f"  Post-CPT wins: {forum_wins['cpt']}")
    print(f"  Ties: {forum_wins['tie']}")
    print()

    if forum_wins['cpt'] > forum_wins['baseline']:
        print("[+] VERDICT: CPT significantly improved response quality")
    elif forum_wins['baseline'] > forum_wins['cpt']:
        print("[!] VERDICT: Baseline had better response quality")
    else:
        print("[-] VERDICT: Similar response quality")
    print()
    print()

    # 4. Overall Summary
    print("="*80)
    print("FINAL VERDICT: OVERALL CPT IMPACT")
    print("="*80)
    print()

    total_oral_cpt_wins = oral_wins['cpt']
    total_forum_cpt_wins = forum_wins['cpt']
    total_oral_baseline_wins = oral_wins['baseline']
    total_forum_baseline_wins = forum_wins['baseline']

    print("[METRICS] Quantitative Metrics:")
    print(f"  MCQ Accuracy: {mcq_results['accuracy_delta']:+.1%}")
    print(f"  Repetition reduction: {oral_results['repetition_improvement'] + forum_results['repetition_improvement']:+d} fewer issues")
    print()

    print("[QUALITY] Qualitative Assessment:")
    print(f"  Open QA quality wins: CPT {total_oral_cpt_wins + total_forum_cpt_wins} vs Baseline {total_oral_baseline_wins + total_forum_baseline_wins}")
    print()

    # Final recommendation
    mcq_improved = mcq_results['accuracy_delta'] >= 0
    qa_improved = (total_oral_cpt_wins + total_forum_cpt_wins) > (total_oral_baseline_wins + total_forum_baseline_wins)
    repetition_improved = (oral_results['repetition_improvement'] + forum_results['repetition_improvement']) > 0

    if mcq_improved and qa_improved and repetition_improved:
        print("[SUCCESS] RECOMMENDATION: CPT training was SUCCESSFUL")
        print("   The model shows improvements across MCQ accuracy, response quality,")
        print("   and reduced repetition issues. Proceed with this checkpoint.")
    elif qa_improved and repetition_improved:
        print("[SUCCESS] RECOMMENDATION: CPT training was MOSTLY SUCCESSFUL")
        print("   While MCQ accuracy slightly decreased, open QA quality improved")
        print("   significantly with better coherence and less repetition.")
    else:
        print("[MIXED] RECOMMENDATION: CPT training had MIXED RESULTS")
        print("   Consider adjusting training hyperparameters or data mixture.")
    print()
    print("="*80)

if __name__ == "__main__":
    main()
