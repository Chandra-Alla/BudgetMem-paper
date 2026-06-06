# BudgetMem: Training-Free Selective Memory for Cost-Efficient Long-Context Processing

This repository contains the code, experiments, and results for the paper:

**BudgetMem: Training-Free Selective Memory for Cost-Efficient Long-Context Processing in Language Models**
Chandra Vamsi Krishna Alla, Harish Naidu Gaddam, Manohar Kommi
*University of Texas at Arlington*

## Overview

BudgetMem is a training-free architecture that selectively retains only high-salience content from long documents under explicit memory budgets. Instead of compressing text at the token level using learned models (like LLMLingua), BudgetMem operates at the **chunk level**: it splits a document into paragraph-sized pieces, scores each with simple interpretable features (entity density, TF-IDF, position bias, numerical density, discourse markers, question presence), and discards low-salience chunks before retrieval.

### Key Results

| Benchmark | Baseline F1 | BudgetMem F1 | LLMLingua-2 F1 | Storage Saved |
|---|---|---|---|---|
| Structured papers (5K-10K tokens) | 0.855 | **0.859** | 0.532 | **70%** |
| Qasper (real NeurIPS papers) | 0.179 | 0.165 | 0.175 | 70% |
| NarrativeQA (50K-100K tokens) | 0.0471 | 0.0351 | 0.0402 | 70% |
| SQuAD (short) | 0.801 | 0.723 | -- | 15% |

The full pipeline runs on a $10/month Google Colab instance with a single NVIDIA T4 GPU. No GPU or training is needed for the compression step.

### Cross-Model Validation

To confirm the result is not specific to one model, we replicate the medium-document comparison on **Qwen2.5-7B-Instruct** — a different scale (7B vs 3B) and family (Qwen vs Llama). The ranking holds:

| Method | Llama-3.2-3B | Qwen2.5-7B |
|---|---|---|
| Baseline RAG | 0.855 | 0.900 |
| LLMLingua-2 | 0.532 | 0.773 |
| **BudgetMem** | **0.859** | **0.960** |

BudgetMem keeps its lead over LLMLingua-2 and matches or exceeds the all-chunks baseline at both scales.

## Repository Structure

```
.
├── README.md                          # This file
├── LICENSE                            # MIT License
├── paper/
│   └── budgetmem_paper.pdf            # The paper (PDF)
├── notebooks/                         # Reproduction notebooks
│   ├── 01_main_experiments.ipynb      # SQuAD + medium-doc baseline experiments
│   ├── 02_narrativeqa_experiments.ipynb  # NarrativeQA with 3 seeds
│   ├── 03_llmlingua_baseline.ipynb    # LLMLingua-2 comparison
│   ├── 04_qasper_and_ablation.ipynb   # Qasper benchmark + per-feature ablation
│   ├── 05_three_seed_and_budget_sensitivity.ipynb  # Final polish experiments
│   └── 06_cross_model_7b_validation.ipynb  # Qwen2.5-7B cross-model check
├── results/                           # Raw experiment outputs (JSON)
│   ├── ALL_REVISION_RESULTS.json
│   ├── FINAL_POLISH_RESULTS.json
│   ├── narrativeqa_final_results.json
│   ├── llmlingua_results.json
│   └── cross_model_7b_summary.json
└── figures/                           # Paper figures
    ├── figure1_budget_sensitivity.png
    ├── figure2_length_scaling.png
    └── figure3_baseline_comparison.png
```

## Reproducing the Experiments

All notebooks are designed to run on Google Colab with a free or Pro tier (T4 GPU, 15GB VRAM).

### Requirements

- Python 3.10+
- HuggingFace account with access to `meta-llama/Llama-3.2-3B-Instruct` (the cross-model notebook uses `Qwen/Qwen2.5-7B-Instruct`, which is open access — no gating)
- Google Colab Pro recommended (~$10/month) for sustained T4 access

### Steps

1. **Clone this repo:**
   ```bash
   git clone https://github.com/Chandra-Alla/BudgetMem-paper.git
   ```

2. **Open notebooks in Colab:** Upload any of the `.ipynb` files in `notebooks/` to Google Colab, or open them directly via the Colab File menu.

3. **Run in order:**
   - Start with `01_main_experiments.ipynb` for SQuAD and medium-doc results
   - `02_narrativeqa_experiments.ipynb` for NarrativeQA with 3 seeds
   - `03_llmlingua_baseline.ipynb` for the LLMLingua-2 comparison
   - `04_qasper_and_ablation.ipynb` for the Qasper benchmark and per-feature ablation
   - `05_three_seed_and_budget_sensitivity.ipynb` for the final polish experiments
   - `06_cross_model_7b_validation.ipynb` for the Qwen2.5-7B cross-model validation

4. **Authenticate with HuggingFace** when prompted (paste your HF token).

5. **Results** will be saved as JSON files matching those in `results/`.

## Method Summary

```
Document D (50K-100K tokens)
        |
        v
Chunking (150-token chunks, 30-token overlap)
        |
        v
Salience Scoring (6 interpretable features, weighted sum)
        |
        v
Top-30% Selection (discard 70% of chunks)
        |
        v
BM25 Indexing
        |
        v
Query-time Retrieval (top-3 chunks)
        |
        v
LLM Answer Generation (Llama-3.2-3B-Instruct)
```

The six features and their weights:
- Entity density (0.20)
- TF-IDF importance (0.20)
- Position bias (0.15) — U-shaped, favors document beginning and end
- Numerical density (0.15)
- Discourse markers (0.10)
- Question presence (0.10)

Per-feature ablation reveals that **discourse markers** (-27.8% F1 when removed) and **numerical density** (-19.1%) drive most of the scoring signal on academic text.

## Citation

```bibtex
@inproceedings{alla2026budgetmem,
  title={BudgetMem: Training-Free Selective Memory for Cost-Efficient Long-Context Processing in Language Models},
  author={Alla, Chandra Vamsi Krishna and Gaddam, Harish Naidu and Kommi, Manohar},
  booktitle={Proceedings of the IEEE Annual Ubiquitous Computing, Electronics \& Mobile Communication Conference (UEMCON)},
  year={2026}
}
```

## License

This project is released under the MIT License — see the [LICENSE](LICENSE) file for details.

## Contact

For questions about the code or the paper, open a GitHub issue or contact:
- Chandra Vamsi Krishna Alla — cka0054@mavs.uta.edu
