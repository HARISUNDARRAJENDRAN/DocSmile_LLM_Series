# DocSmile – A Series of Dental LLMs

**DocSmile** is a series of State-of-the-Art models designed to advance intelligent assistance in dentistry through a specialized series of fine-tuned large language models. The primary objective of the system is to bridge the gap between unstructured medical knowledge and actionable clinical insight.

## Core Components
The system comprises two main models:

1.  **Domain-Adapted Text-Based LLM:** Focused on comprehensive understanding and reasoning over dental literature. This is achieved through **Continued Pre-Training (CPT)**.
2.  **Vision-Language Model (VLM):** Capable of interpreting and contextualizing visual data, such as anatomical diagrams, clinical images, and procedural illustrations.

---

## Data Pipeline & Methodology
To ensure high-quality outputs, the project utilizes a structured data pipeline that preprocesses and standardizes dental textbooks and educational resources:

*   **Textual Data:** Cleaned and normalized for consistency to ensure high-quality corpora.
*   **Visual Data:** Images are enriched with semantic annotations and labels corresponding to textual sections, enabling **cross-modal learning**.
*   **Integration:** The pipeline facilitates the association of visual patterns with underlying biomedical concepts.

---

## Key Capabilities
*   **Concept Explanation & Reasoning:** The text-based LLM is fine-tuned to support clinical reasoning and query-based knowledge retrieval.
*   **Visual Interpretation:** The VLM identifies structures and assists in visual reasoning tasks by interpreting complex diagrams.
*   **Multimodal Understanding:** Together, the models form a unified system that can process both text and imagery simultaneously.

---

## Future Applications & Extensibility
DocSmile is designed with extensibility in mind, supporting integration with:
*   **Interactive Educational Tools:** Enhancing learning for students and professionals.
*   **Clinical Decision-Support Systems:** Providing real-time insights for dental practitioners.
*   **Next-Gen AI Healthcare:** Laying the groundwork for advanced, AI-assisted healthcare applications.

---

## RL Dataset Generation (SFT + DPO)
Use [scripts/build_rl_datasets_gemini.py](scripts/build_rl_datasets_gemini.py) to convert Markdown under the rl folder into high-quality QLoRA SFT and DPO datasets using Gemini.

Outputs include SFT and DPO JSONL files plus progress and state files in the output directory you specify.

Example run:

```powershell
python scripts/build_rl_datasets_gemini.py --input-dir rl --output-dir rl_prepared --model gemini-3.1-flash-lite-preview --continue-on-error
```

Live progress (reuses the watcher):

```powershell
python scripts/watch_clean_progress.py --output-dir rl_prepared
```

