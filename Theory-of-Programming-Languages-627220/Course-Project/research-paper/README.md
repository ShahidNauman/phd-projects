# Springer LaTeX Project — Explainable Sarcasm Detection

## Project Structure

```
research-paper/
├── main.tex                    ← Master document
├── references.bib              ← BibTeX database (all references)
├── sn-jnl.cls                  ← Springer class file (DOWNLOAD REQUIRED)
├── sn-mathphys-num.bst         ← Springer bibliography style (DOWNLOAD REQUIRED)
├── README.md                   ← This file
│
├── sections/
│   ├── 01_abstract.tex
│   ├── 02_introduction.tex
│   ├── 03_related_work.tex
│   ├── 04_methodology.tex      ← Contains TikZ pipeline diagram
│   ├── 05_mathematical_formulation.tex
│   ├── 06_algorithms.tex       ← Algorithms 1, 2, 3
│   ├── 07_experimental_setup.tex
│   ├── 08_results.tex          ← Tables 2–6, figures, ablation
│   ├── 09_discussion.tex
│   ├── 10_conclusion.tex
│   └── 11_future_work.tex
│
├── appendices/
│   └── nomenclature.tex        ← Symbol and abbreviation table
│
├── figures/                    ← Place PDF/PNG figure files here
├── tables/
├── algorithms/
├── equations/
└── deep-research-report.md     ← Original source report
```

## How to Compile

### Option A: Overleaf (Recommended)

1. **Download the Springer Nature LaTeX template** from:
   https://www.springernature.com/gp/authors/campaigns/latex-author-support
   
2. Extract `sn-jnl.cls` and `sn-mathphys-num.bst` from the downloaded ZIP.

3. Upload this entire project folder to Overleaf.

4. Place `sn-jnl.cls` and `sn-mathphys-num.bst` in the project root
   (same directory as `main.tex`).

5. Set `main.tex` as the main document in Overleaf settings.

6. Compile with **pdfLaTeX**.

### Option B: Local Compilation

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Figure Placeholders

The following figures are referenced in the paper but require external files:

| Figure | Filename | Description |
|--------|----------|-------------|
| Fig. 1 | *(TikZ inline)* | Proposed pipeline — generated automatically |
| Fig. 2 | `figures/overall_architecture.pdf` | Overall system architecture |
| Fig. 3 | `figures/training_pipeline.pdf` | Training pipeline flowchart |
| Fig. 4 | `figures/inference_pipeline.pdf` | Inference + explanation flowchart |
| Fig. 5 | `figures/confusion_matrix_roberta_base.pdf` | Confusion matrix (RoBERTa-base) |
| Fig. 6 | `figures/roc_curves.pdf` | ROC curves for all models |
| Fig. 7 | `figures/precision_recall_comparison.pdf` | Precision/Recall comparison |

**To compile without these files**, comment out the corresponding
`\includegraphics` lines or create blank placeholder PDFs.

## Notes

- **Author information**: Replace placeholder author names and affiliations
  in `main.tex` before submission.
- **Fig. 1 (TikZ)**: The proposed inference pipeline is rendered inline
  using TikZ in `sections/04_methodology.tex`. No external file is needed.
- All tables use `booktabs` formatting (no vertical lines).
- All equations are numbered and cross-referenced.
- All algorithms use `algorithm`/`algpseudocode`.
